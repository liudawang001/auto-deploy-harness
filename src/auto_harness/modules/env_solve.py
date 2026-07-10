import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from auto_harness.env import CondaBackend, CondaEnvironmentParser
from auto_harness.models.result import StageResult


class LocalEnvironmentProbe:
    """Collects local environment facts needed for dependency solving."""

    def probe(self) -> Dict:
        cuda = self._cuda_info()
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system().lower(),
            "machine": platform.machine().lower(),
            "executables": {
                "python3": shutil.which("python3") or "",
                "nvcc": shutil.which("nvcc") or "",
                "nvidia-smi": shutil.which("nvidia-smi") or "",
            },
            "cuda": cuda,
        }

    def _cuda_info(self) -> Dict:
        env_version = os.environ.get("AUTO_HARNESS_CUDA_VERSION", "").strip()
        if env_version:
            return {"available": True, "version": env_version, "source": "AUTO_HARNESS_CUDA_VERSION"}
        smi_version = self._probe_nvidia_smi()
        if smi_version:
            return {"available": True, "version": smi_version, "source": "nvidia-smi"}
        nvcc_version = self._probe_nvcc()
        if nvcc_version:
            return {"available": True, "version": nvcc_version, "source": "nvcc"}
        return {"available": False, "version": "", "source": "none"}

    def _probe_nvidia_smi(self) -> str:
        if not shutil.which("nvidia-smi"):
            return ""
        output = self._run_probe(["nvidia-smi"])
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", output)
        return match.group(1) if match else ""

    def _probe_nvcc(self) -> str:
        if not shutil.which("nvcc"):
            return ""
        output = self._run_probe(["nvcc", "--version"])
        match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", output)
        return match.group(1) if match else ""

    def _run_probe(self, cmd: List[str]) -> str:
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        return "\n".join([completed.stdout or "", completed.stderr or ""])


class EnvSolveModule:
    """Builds a safer dependency installation plan without executing commands."""

    def __init__(
        self,
        local_environment: Optional[Dict] = None,
        probe: LocalEnvironmentProbe = None,
        env_backend: str = "auto",
        conda_envs_dir: str = ".conda/envs",
        conda_prefer_mamba: bool = True,
        conda_allowed_channels: List[str] = None,
        conda_python_default: str = "3.10",
        torch_cuda_preference: str = "auto",
    ) -> None:
        self.local_environment = local_environment
        self.probe = probe or LocalEnvironmentProbe()
        self.env_backend = env_backend
        self.conda_envs_dir = conda_envs_dir
        self.conda_prefer_mamba = conda_prefer_mamba
        self.conda_allowed_channels = conda_allowed_channels or ["defaults", "conda-forge", "pytorch", "nvidia", "fastai"]
        self.conda_python_default = conda_python_default
        self.torch_cuda_preference = torch_cuda_preference

    def solve(self, repo_dir: Path, analysis: Dict, resource_plan: Dict, stage_hints: Dict = None) -> StageResult:
        requirements = self._read_requirements(repo_dir)
        frameworks = set(analysis.get("frameworks") or [])
        base_plan = [list(cmd) for cmd in analysis.get("install_plan") or []]
        local_environment = dict(self.local_environment or self.probe.probe())
        conda_file = CondaEnvironmentParser().parse_repo(repo_dir, default_python=self.conda_python_default)
        if (conda_file.get("torch") or {}).get("requires_torch"):
            frameworks.add("torch")
            requirements = requirements + [dep for dep in (conda_file.get("conda_dependencies") or []) if str(dep).startswith(("pytorch", "torch", "torchvision", "torchaudio"))]
            requirements = requirements + [dep for dep in (conda_file.get("pip_dependencies") or []) if str(dep).startswith(("torch", "torchvision", "torchaudio"))]
        constraints, reasons = self._constraints(requirements, frameworks)
        # Apply plan hints: prefer_constraints from LLM plan
        hints = stage_hints or {}
        prefer_constraints = hints.get("prefer_constraints", [])
        for hint_constraint in prefer_constraints:
            if hint_constraint and hint_constraint not in constraints:
                # Validate hint format: must be package+version_spec
                if self._valid_hint_constraint(hint_constraint):
                    constraints.append(hint_constraint)
                    reasons.append("plan hint: %s" % hint_constraint)
        environment_strategy = self._environment_strategy(analysis, conda_file)
        torch_solution = self._torch_solution(requirements, frameworks, resource_plan, local_environment, base_plan)
        gpu_package_matrix = self._gpu_package_matrix(requirements, frameworks, resource_plan, local_environment, torch_solution)
        install_plan = self._apply_constraints(base_plan, constraints)
        install_plan = self._apply_torch_solution(install_plan, torch_solution)
        risk_reasons = self._risk_reasons(requirements, frameworks, resource_plan, torch_solution, gpu_package_matrix)
        solved_analysis = dict(analysis)
        solved_analysis["install_plan"] = install_plan
        backend = self._selected_backend(environment_strategy, conda_file)
        env_solution = {
            "backend": backend,
            "python": environment_strategy.get("python") or conda_file.get("python") or self._python_choice(resource_plan),
            "python_range": resource_plan.get("python_range", "unknown"),
            "local_environment": local_environment,
            "constraints": constraints,
            "constraint_reasons": reasons,
            "torch_variant": torch_solution.get("selected", {}).get("variant") or resource_plan.get("torch_variant", ""),
            "torch_solution": torch_solution,
            "gpu_package_matrix": gpu_package_matrix,
            "gpu_required": bool(resource_plan.get("gpu_required")),
            "risk_reasons": risk_reasons,
            "environment_strategy": environment_strategy,
            "conda_file": conda_file,
        }
        if backend in ("conda", "mamba"):
            conda_plan = CondaBackend(
                backend=backend,
                envs_dir=self.conda_envs_dir,
                prefer_mamba=self.conda_prefer_mamba,
                allowed_channels=self.conda_allowed_channels,
                default_python=self.conda_python_default,
            )
            spec = conda_plan.build_spec(repo_dir, env_solution, conda_file=conda_file if conda_file.get("found") else {})
            plan = conda_plan.command_plan(spec, pip_plan=install_plan)
            env_solution.update({
                "conda": plan,
                "environment_prefix": plan["environment_prefix"],
                "environment_python": plan["environment_python"],
                "install_plan_effective": plan["commands"],
            })
        solved_analysis["env_solution"] = {
            **env_solution,
        }
        status = "passed" if install_plan or backend in ("conda", "mamba") else "uncertain"
        summary = "environment solution generated" if status == "passed" else "no install plan to solve"
        return StageResult(
            "env_solve",
            status,
            summary,
            {
                "backend": env_solution["backend"],
                "python": env_solution["python"],
                "install_plan": install_plan,
                "environment_strategy": environment_strategy,
                "conda_file": conda_file,
                "conda": env_solution.get("conda", {}),
                "constraints": constraints,
                "constraint_reasons": reasons,
                "local_environment": local_environment,
                "torch_solution": torch_solution,
                "gpu_package_matrix": gpu_package_matrix,
                "risk_reasons": risk_reasons,
                "analysis": solved_analysis,
            },
        )

    def _environment_strategy(self, analysis: Dict, conda_file: Dict) -> Dict:
        strategy = dict(analysis.get("environment_strategy") or {})
        if self.env_backend and self.env_backend != "auto":
            strategy["backend"] = self.env_backend
            strategy["source"] = "config"
        elif conda_file.get("found") and strategy.get("backend") in ("", None, "venv"):
            strategy.update({
                "backend": "conda",
                "preferred_tool": "mamba" if self.conda_prefer_mamba else "conda",
                "python": conda_file.get("python") or self.conda_python_default,
                "channels": conda_file.get("channels") or ["conda-forge"],
                "source": "deterministic_environment_yml",
                "confidence": 0.85,
                "reasons": ["environment.yml detected"],
            })
        strategy.setdefault("backend", "venv")
        strategy.setdefault("python", conda_file.get("python") or self.conda_python_default)
        strategy["channels"] = self._safe_channels(strategy.get("channels") or conda_file.get("channels") or [])
        return strategy

    def _selected_backend(self, strategy: Dict, conda_file: Dict) -> str:
        backend = str(strategy.get("backend") or "venv").lower()
        if backend == "auto":
            backend = "conda" if conda_file.get("found") else "venv"
        if backend == "local_venv":
            backend = "venv"
        if backend not in ("venv", "conda", "mamba"):
            backend = "venv"
        if conda_file.get("found") and self.env_backend == "auto" and backend == "venv":
            backend = "conda"
        return backend

    def _safe_channels(self, channels: List[str]) -> List[str]:
        allowed = set(self.conda_allowed_channels)
        result = []
        for channel in channels:
            item = str(channel).strip()
            if item in allowed and item not in result:
                result.append(item)
        return result

    def _read_requirements(self, repo_dir: Path) -> List[str]:
        path = repo_dir / "requirements.txt"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    def _constraints(self, requirements: List[str], frameworks: set) -> Tuple[List[str], List[str]]:
        constraints = []
        reasons = []
        req_text = "\n".join(requirements).lower()
        old_gradio = "gradio" in frameworks and not ("gradio>=" in req_text and "gradio>=4" in req_text)
        if old_gradio:
            if not self._has_package_constraint(requirements, "numpy", "<2"):
                constraints.append("numpy<2")
                reasons.append("old or unpinned gradio project detected; numpy 2.x can break legacy dependencies")
            if not self._has_package_constraint(requirements, "pydantic", "<2"):
                constraints.append("pydantic<2")
                reasons.append("old or unpinned gradio project detected; pydantic v2 can break legacy gradio stacks")
        if "opencv-python" in req_text and "opencv-python-headless" not in req_text:
            constraints.append("opencv-python-headless")
            reasons.append("headless deployment should prefer opencv-python-headless")
        return self._dedupe(constraints), self._dedupe(reasons)

    def _has_package_constraint(self, requirements: List[str], package: str, token: str) -> bool:
        prefix = package.lower()
        for requirement in requirements:
            normalized = requirement.lower().replace(" ", "")
            if normalized.startswith(prefix) and token in normalized:
                return True
        return False

    def _valid_hint_constraint(self, constraint: str) -> bool:
        """Validate hint constraint format: must be package+version_spec.

        Valid examples: pydantic<2, numpy<2, torch==2.3.0
        Invalid examples: <2pydantic, rm -rf /
        """
        import re
        # Must start with a letter or underscore (package name)
        # Followed by version spec (optional)
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_._-]*([<>=!~]+[a-zA-Z0-9._,]+)?$'
        return bool(re.match(pattern, constraint.strip()))

    def _apply_constraints(self, base_plan: List[List[str]], constraints: List[str]) -> List[List[str]]:
        if not constraints:
            return base_plan
        plan = [list(cmd) for cmd in base_plan]
        for index, cmd in enumerate(plan):
            if len(cmd) >= 4 and cmd[-2:] == ["-r", "requirements.txt"] and "pip" in cmd:
                plan[index] = cmd + constraints
                return plan
        if plan:
            python_cmd = ".venv/bin/python" if any(".venv" in " ".join(cmd) for cmd in plan) else "python3"
            plan.append([python_cmd, "-m", "pip", "install"] + constraints)
        return plan

    def _apply_torch_solution(self, install_plan: List[List[str]], torch_solution: Dict) -> List[List[str]]:
        selected = torch_solution.get("selected") or {}
        command = selected.get("command") or []
        if not command:
            return install_plan
        plan = [list(cmd) for cmd in install_plan]
        if command in plan:
            return plan
        insert_at = 1 if plan else 0
        for index, cmd in enumerate(plan):
            if "pip" in cmd and "install" in cmd and "--upgrade" in cmd and "pip" in cmd[-1:]:
                insert_at = index + 1
                break
            if "-m" in cmd and "venv" in cmd:
                insert_at = index + 1
        plan.insert(insert_at, list(command))
        return plan

    def _risk_reasons(self, requirements: List[str], frameworks: set, resource_plan: Dict, torch_solution: Dict, gpu_package_matrix: Dict) -> List[str]:
        reasons = []
        req_text = "\n".join(requirements).lower()
        if resource_plan.get("gpu_required"):
            reasons.append("GPU/CUDA signals detected; torch wheel variant must match local CUDA runtime")
        if torch_solution.get("selected", {}).get("variant") == "cpu" and resource_plan.get("gpu_required"):
            reasons.append("GPU was requested but no compatible local CUDA wheel was selected; CPU fallback is planned")
        if "torch" in frameworks and "torch" not in req_text:
            reasons.append("torch framework detected but requirements do not pin torch")
        if "flash-attn" in req_text:
            reasons.append("flash-attn may require CUDA toolkit and build isolation tuning")
            if torch_solution.get("selected", {}).get("variant") == "cpu":
                reasons.append("flash-attn is incompatible with the CPU torch fallback")
        if "bitsandbytes" in req_text:
            reasons.append("bitsandbytes requires compatible CUDA and platform support")
        for package in gpu_package_matrix.get("packages", []):
            if package.get("status") in ("blocked", "risky"):
                reasons.append("%s: %s" % (package["name"], "; ".join(package.get("reasons") or [])))
        return reasons

    def _gpu_package_matrix(self, requirements: List[str], frameworks: set, resource_plan: Dict, local_environment: Dict, torch_solution: Dict) -> Dict:
        package_names = ("xformers", "flash-attn", "flash_attn", "bitsandbytes", "triton", "deepspeed", "accelerate")
        declared = self._declared_packages(requirements)
        normalized_declared = set(declared)
        if "flash_attn" in normalized_declared:
            normalized_declared.add("flash-attn")
        if "torch" in frameworks and resource_plan.get("gpu_required"):
            normalized_declared.add("triton")
        packages = []
        for name in package_names:
            canonical = "flash-attn" if name == "flash_attn" else name
            if canonical not in normalized_declared and name not in normalized_declared:
                continue
            if canonical == "flash-attn" and any(item["name"] == "flash-attn" for item in packages):
                continue
            packages.append(self._gpu_package_rule(canonical, declared, local_environment, torch_solution, resource_plan))
        return {
            "python_version": local_environment.get("python_version", ""),
            "platform": local_environment.get("platform", ""),
            "machine": local_environment.get("machine", ""),
            "cuda": local_environment.get("cuda") or {},
            "torch_variant": torch_solution.get("selected", {}).get("variant", ""),
            "packages": packages,
        }

    def _declared_packages(self, requirements: List[str]) -> Dict[str, str]:
        packages = {}
        for requirement in requirements:
            name = re.split(r"[<>=~!;\[]", requirement.strip(), maxsplit=1)[0].strip().lower().replace("_", "-")
            if name:
                packages[name] = requirement.strip()
        return packages

    def _gpu_package_rule(self, name: str, declared: Dict[str, str], local_environment: Dict, torch_solution: Dict, resource_plan: Dict) -> Dict:
        platform_name = str(local_environment.get("platform") or "")
        machine = str(local_environment.get("machine") or "")
        python_version = str(local_environment.get("python_version") or "")
        cuda = local_environment.get("cuda") or {}
        cuda_available = bool(cuda.get("available"))
        torch_variant = str(torch_solution.get("selected", {}).get("variant") or "")
        reasons = []
        actions = []
        status = "compatible"
        if name in {"xformers", "flash-attn", "bitsandbytes", "deepspeed"} and (not cuda_available or torch_variant == "cpu"):
            status = "blocked"
            reasons.append("requires CUDA torch runtime; current selected torch variant is %s" % (torch_variant or "unknown"))
            actions.append("switch torch_solution to a CUDA wheel before installing %s" % name)
        if name in {"xformers", "flash-attn"} and platform_name not in {"linux"}:
            status = "blocked"
            reasons.append("%s is safest on Linux CUDA builds; current platform is %s" % (name, platform_name or "unknown"))
        if name == "bitsandbytes":
            if platform_name != "linux":
                status = "blocked"
                reasons.append("bitsandbytes production wheels are Linux-first; current platform is %s" % (platform_name or "unknown"))
            if machine not in {"x86_64", "amd64"}:
                status = "risky" if status == "compatible" else status
                reasons.append("bitsandbytes wheel support is limited on architecture %s" % (machine or "unknown"))
        if name == "flash-attn":
            if python_version and python_version not in {"3.9", "3.10", "3.11"}:
                status = "risky" if status == "compatible" else status
                reasons.append("flash-attn has frequent wheel/build gaps outside Python 3.9-3.11")
            actions.append("prefer --no-build-isolation only after torch CUDA wheel is installed")
        if name == "xformers":
            actions.append("pin xformers to a version matching the selected torch CUDA wheel")
        if name == "triton":
            if platform_name not in {"linux"}:
                status = "blocked"
                reasons.append("triton is not a stable runtime dependency on %s for this deployment path" % (platform_name or "unknown"))
            elif torch_variant == "cpu" and resource_plan.get("gpu_required"):
                status = "risky"
                reasons.append("triton was inferred for a GPU workload but torch selected CPU fallback")
            actions.append("let torch install the matching triton dependency unless the project pins it")
        if name == "deepspeed":
            if platform_name != "linux":
                status = "blocked"
                reasons.append("deepspeed production installs are Linux/CUDA oriented; current platform is %s" % (platform_name or "unknown"))
            actions.append("install deepspeed only after torch CUDA runtime is selected")
        if name == "accelerate":
            actions.append("accelerate is pure Python but should follow the selected torch runtime")
        if not reasons:
            reasons.append("%s appears compatible with the detected Python/CUDA/Torch envelope" % name)
        return {
            "name": name,
            "declared_requirement": declared.get(name, ""),
            "status": status,
            "torch_variant": torch_variant,
            "requires_cuda": name in {"xformers", "flash-attn", "bitsandbytes"},
            "reasons": reasons,
            "recommended_actions": actions,
        }

    def _torch_solution(self, requirements: List[str], frameworks: set, resource_plan: Dict, local_environment: Dict, base_plan: List[List[str]]) -> Dict:
        torch_requirements = self._torch_requirements(requirements)
        required = bool(torch_requirements or {"torch", "transformers", "diffusers"}.intersection(frameworks))
        solution = {
            "required": required,
            "declared_requirements": torch_requirements,
            "local_cuda_version": (local_environment.get("cuda") or {}).get("version", ""),
            "local_cuda_available": bool((local_environment.get("cuda") or {}).get("available")),
            "selected": {},
            "fallbacks": [],
            "notes": [],
        }
        if not required:
            return solution

        packages = self._torch_packages(torch_requirements)
        python_executable = self._plan_python_executable(resource_plan, base_plan)
        selected_variant = self._select_torch_variant(resource_plan, local_environment)
        selected = self._torch_install_option(selected_variant, packages, python_executable, primary=True)
        if selected:
            solution["selected"] = selected

        for variant in self._fallback_variants(selected_variant, resource_plan, local_environment):
            option = self._torch_install_option(variant, packages, python_executable, primary=False)
            if option:
                solution["fallbacks"].append(option)

        if selected_variant == "cpu" and resource_plan.get("gpu_required"):
            solution["notes"].append("local CUDA runtime was not detected or is unsupported; CPU torch fallback generated")
        elif selected_variant.startswith("cu"):
            solution["notes"].append(f"CUDA runtime maps to PyTorch wheel index {selected_variant}")
        if torch_requirements and not any(self._has_version_pin(req) for req in torch_requirements):
            solution["notes"].append("torch requirements are unpinned; preinstalling a wheel variant reduces accidental CPU/CUDA mismatch")
        return solution

    def _torch_requirements(self, requirements: List[str]) -> List[str]:
        packages = ("torch", "torchvision", "torchaudio")
        result = []
        for requirement in requirements:
            normalized = requirement.strip().lower()
            if any(normalized == package or normalized.startswith(package + spec) for package in packages for spec in ("==", ">=", "<=", "~=", ">", "<", "[", ";")):
                result.append(requirement.strip())
        return result

    def _torch_packages(self, torch_requirements: List[str]) -> List[str]:
        packages = []
        for requirement in torch_requirements:
            name = re.split(r"[<>=~!;\[]", requirement.strip(), maxsplit=1)[0].strip()
            if name in {"torch", "torchvision", "torchaudio"}:
                packages.append(name)
        if "torch" not in packages:
            packages.insert(0, "torch")
        return self._dedupe(packages)

    def _has_version_pin(self, requirement: str) -> bool:
        normalized = requirement.replace(" ", "")
        return any(token in normalized for token in ("==", "~=", ">=", "<=", ">", "<"))

    def _select_torch_variant(self, resource_plan: Dict, local_environment: Dict) -> str:
        if self.torch_cuda_preference in {"cpu", "cu118", "cu121"}:
            return self.torch_cuda_preference
        requested = str(resource_plan.get("torch_variant") or "").lower()
        if requested in {"cpu", "cu118", "cu121"}:
            return requested
        if requested.startswith("cu") and requested[2:].isdigit():
            return requested
        if not resource_plan.get("gpu_required"):
            return "cpu"
        cuda_version = str((local_environment.get("cuda") or {}).get("version") or "")
        return self._cuda_to_torch_variant(cuda_version) or "cpu"

    def _cuda_to_torch_variant(self, cuda_version: str) -> str:
        match = re.search(r"([0-9]+)(?:\.([0-9]+))?", cuda_version)
        if not match:
            return ""
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        version = major + minor / 10
        if version >= 12.1:
            return "cu121"
        if version >= 11.8:
            return "cu118"
        return ""

    def _torch_install_option(self, variant: str, packages: List[str], python_executable: str, primary: bool) -> Dict:
        if not variant:
            return {}
        index_url = self._torch_index_url(variant)
        if not index_url:
            return {}
        reason = "selected torch wheel variant" if primary else "fallback torch wheel variant"
        if variant == "cpu":
            reason += " for CPU-only execution"
        else:
            reason += f" for CUDA {variant[2]}.{variant[3:]}"
        return {
            "variant": variant,
            "index_url": index_url,
            "packages": packages,
            "command": [python_executable, "-m", "pip", "install"] + packages + ["--index-url", index_url],
            "reason": reason,
        }

    def _torch_index_url(self, variant: str) -> str:
        mapping = {
            "cpu": "https://download.pytorch.org/whl/cpu",
            "cu118": "https://download.pytorch.org/whl/cu118",
            "cu121": "https://download.pytorch.org/whl/cu121",
        }
        return mapping.get(variant, "")

    def _fallback_variants(self, selected_variant: str, resource_plan: Dict, local_environment: Dict) -> List[str]:
        variants = []
        if selected_variant.startswith("cu"):
            variants.append("cpu")
            if selected_variant == "cu121":
                variants.append("cu118")
        elif resource_plan.get("gpu_required") and (local_environment.get("cuda") or {}).get("available"):
            variants.extend(["cu121", "cu118"])
        if selected_variant != "cpu":
            variants.append("cpu")
        return [variant for variant in self._dedupe(variants) if variant != selected_variant]

    def _plan_python_executable(self, resource_plan: Dict, base_plan: List[List[str]]) -> str:
        for cmd in base_plan:
            if "-m" in cmd and "pip" in cmd:
                return cmd[0]
        choice = self._python_choice(resource_plan)
        if choice.startswith("3."):
            return f"python{choice}"
        return "python3"

    def _python_choice(self, resource_plan: Dict) -> str:
        python_range = str(resource_plan.get("python_range") or "")
        if "3.10" in python_range:
            return "3.10"
        if "3.11" in python_range:
            return "3.11"
        return "python3"

    def _dedupe(self, values: List[str]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
