import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, local_environment: Optional[Dict] = None, probe: LocalEnvironmentProbe = None) -> None:
        self.local_environment = local_environment
        self.probe = probe or LocalEnvironmentProbe()

    def solve(self, repo_dir: Path, analysis: Dict, resource_plan: Dict) -> StageResult:
        requirements = self._read_requirements(repo_dir)
        frameworks = set(analysis.get("frameworks") or [])
        base_plan = [list(cmd) for cmd in analysis.get("install_plan") or []]
        constraints, reasons = self._constraints(requirements, frameworks)
        local_environment = dict(self.local_environment or self.probe.probe())
        torch_solution = self._torch_solution(requirements, frameworks, resource_plan, local_environment, base_plan)
        install_plan = self._apply_constraints(base_plan, constraints)
        install_plan = self._apply_torch_solution(install_plan, torch_solution)
        risk_reasons = self._risk_reasons(requirements, frameworks, resource_plan, torch_solution)
        solved_analysis = dict(analysis)
        solved_analysis["install_plan"] = install_plan
        solved_analysis["env_solution"] = {
            "backend": "local_venv" if install_plan else "unknown",
            "python": self._python_choice(resource_plan),
            "python_range": resource_plan.get("python_range", "unknown"),
            "local_environment": local_environment,
            "constraints": constraints,
            "constraint_reasons": reasons,
            "torch_variant": torch_solution.get("selected", {}).get("variant") or resource_plan.get("torch_variant", ""),
            "torch_solution": torch_solution,
            "gpu_required": bool(resource_plan.get("gpu_required")),
            "risk_reasons": risk_reasons,
        }
        status = "passed" if install_plan else "uncertain"
        summary = "environment solution generated" if status == "passed" else "no install plan to solve"
        return StageResult(
            "env_solve",
            status,
            summary,
            {
                "backend": solved_analysis["env_solution"]["backend"],
                "python": solved_analysis["env_solution"]["python"],
                "install_plan": install_plan,
                "constraints": constraints,
                "constraint_reasons": reasons,
                "local_environment": local_environment,
                "torch_solution": torch_solution,
                "risk_reasons": risk_reasons,
                "analysis": solved_analysis,
            },
        )

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

    def _risk_reasons(self, requirements: List[str], frameworks: set, resource_plan: Dict, torch_solution: Dict) -> List[str]:
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
        return reasons

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
