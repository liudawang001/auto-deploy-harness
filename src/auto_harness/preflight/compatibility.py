"""Deterministic environment backend and GPU compatibility decisions."""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict

from auto_harness.preflight.schemas import EnvironmentCompatibilityDecision
from auto_harness.utils.files import safe_name


class EnvironmentCompatibilityResolver:
    def resolve(
        self,
        repo_dir: Path,
        analysis: Dict,
        resource_plan: Dict,
        conda_file: Dict,
        capabilities: Dict,
        inventory: Dict,
        config,
    ) -> Dict:
        repo_dir = Path(repo_dir).resolve()
        project_id = hashlib.sha256(str(repo_dir).encode("utf-8")).hexdigest()[:16]
        repo_fingerprint = self._repo_fingerprint(repo_dir, conda_file)
        runtimes = capabilities.get("environment_runtimes") or {}
        gpu = capabilities.get("gpu") or {}
        requested = str(getattr(config, "env_backend", "auto") or "auto").lower()
        python = str(conda_file.get("python") or getattr(config, "conda_python_default", "3.10"))
        reasons = []
        warnings = []
        if conda_file.get("rejected_channels"):
            return self._decision(
                status="blocked",
                action="block",
                backend=requested if requested != "auto" else "conda",
                reasons=["environment file contains channels outside the allowlist"],
                warnings=list(conda_file.get("rejected_channels") or []),
                project_id=project_id,
                repo_fingerprint=repo_fingerprint,
                python=python,
            )

        gpu_required = bool(resource_plan.get("gpu_required"))
        selected_gpu = self._select_gpu(
            gpu.get("devices") or [],
            int(resource_plan.get("min_gpu_memory_mb") or getattr(config, "min_gpu_memory_mb", 0) or 0),
        )
        if gpu_required and gpu.get("status") != "detected":
            if gpu.get("status") in ("permission_denied", "timeout", "parse_error", "probe_error"):
                return self._decision(
                    status="uncertain", action="request_approval", backend="venv",
                    reasons=["GPU probe did not produce a trustworthy result"],
                    warnings=list(gpu.get("errors") or []), project_id=project_id,
                    repo_fingerprint=repo_fingerprint, python=python,
                )
            if getattr(config, "conda_allow_cpu_fallback", False):
                warnings.append("required GPU unavailable; explicit CPU fallback selected")
            else:
                return self._decision(
                    status="blocked", action="block", backend="venv",
                    reasons=["GPU is required but no compatible device was detected"],
                    project_id=project_id, repo_fingerprint=repo_fingerprint, python=python,
                )
        if gpu_required and gpu.get("status") == "detected" and not selected_gpu:
            if getattr(config, "conda_allow_cpu_fallback", False):
                warnings.append("no GPU satisfies the minimum memory requirement; explicit CPU fallback selected")
            else:
                return self._decision(
                    status="blocked",
                    action="block",
                    backend="venv",
                    reasons=["no detected GPU satisfies the minimum memory requirement"],
                    project_id=project_id,
                    repo_fingerprint=repo_fingerprint,
                    python=python,
                )

        backend, runtime = self._select_backend(requested, conda_file, runtimes, config)
        conda_only = self._conda_only(conda_file)
        if backend == "venv" and conda_only:
            return self._decision(
                status="blocked",
                action="block",
                backend="venv",
                reasons=["Conda-only dependencies or channels cannot be represented by venv"],
                project_id=project_id,
                repo_fingerprint=repo_fingerprint,
                python=python,
            )
        if backend in ("conda", "mamba", "micromamba") and not runtime:
            if requested != "auto" or conda_only or not getattr(config, "conda_allow_venv_fallback", False):
                return self._decision(
                    status="blocked", action="block", backend=backend,
                    reasons=["requested Conda runtime is unavailable"],
                    project_id=project_id, repo_fingerprint=repo_fingerprint, python=python,
                )
            backend = "venv"
            reasons.append("Conda runtime unavailable; pure pip project falls back to venv")

        name = safe_name(str(conda_file.get("name") or repo_dir.name or "auto-harness"))
        envs_dir = Path(getattr(config, "conda_envs_dir", ".conda/envs"))
        if not envs_dir.is_absolute():
            envs_dir = repo_dir / envs_dir
        target_prefix = str((envs_dir / name).resolve()) if backend != "venv" else ""
        spec_hash = self._spec_hash(backend, python, conda_file, resource_plan)
        reuse = self._reuse_candidate(inventory, target_prefix, project_id, spec_hash)
        action = "reuse" if reuse and getattr(config, "conda_reuse_owned_env", True) else "create"
        if backend == "venv":
            action = "fallback_venv" if conda_file.get("found") and requested != "venv" else "create"
        reasons.extend(self._reasons(conda_file, backend, action, selected_gpu))
        torch_variant = self._torch_variant(resource_plan, selected_gpu, gpu, config)
        return EnvironmentCompatibilityDecision(
            status="allowed",
            backend=backend,
            tool=(runtime or {}).get("path", ""),
            action=action,
            target_prefix=target_prefix,
            selected_gpu_index=int(selected_gpu.get("index", -1)) if selected_gpu else -1,
            python=python,
            torch_variant=torch_variant,
            spec_hash=spec_hash,
            project_id=project_id,
            repo_fingerprint=repo_fingerprint,
            reuse_candidate=reuse,
            fallback="cpu" if gpu_required and not selected_gpu else "",
            reasons=reasons,
            warnings=warnings,
            policy_requirements=[
                "allow_dependency_install",
                "environment_prefix_owned",
                "channel_allowlist_passed",
            ] if backend != "venv" else [],
        ).to_dict()

    def _select_backend(self, requested, conda_file, runtimes, config):
        if requested in ("conda", "mamba", "micromamba"):
            runtime = runtimes.get(requested) or {}
            return requested, runtime if runtime.get("available") else {}
        if requested == "venv":
            return "venv", {}
        if conda_file.get("found"):
            if getattr(config, "conda_prefer_mamba", True) and (runtimes.get("mamba") or {}).get("available"):
                return "mamba", runtimes["mamba"]
            if (runtimes.get("conda") or {}).get("available"):
                return "conda", runtimes["conda"]
            if (runtimes.get("micromamba") or {}).get("available"):
                return "micromamba", runtimes["micromamba"]
            return "conda", {}
        return "venv", {}

    def _select_gpu(self, devices, minimum):
        candidates = [item for item in devices if int(item.get("memory_free_mb") or 0) >= minimum]
        candidates.sort(
            key=lambda item: (
                int(item.get("memory_free_mb") or 0),
                int(item.get("memory_total_mb") or 0),
                -int(item.get("index") or 0),
            ),
            reverse=True,
        )
        return candidates[0] if candidates else {}

    def _reuse_candidate(self, inventory, target, project_id, spec_hash):
        for env in inventory.get("environments") or []:
            if (
                env.get("prefix") == target
                and env.get("owned_by_harness")
                and env.get("owner_project_id") == project_id
                and env.get("spec_hash") == spec_hash
            ):
                return target
        return ""

    def _conda_only(self, conda_file):
        dependencies = [
            str(item).lower() for item in conda_file.get("conda_dependencies") or []
            if not str(item).lower().startswith("python=") and str(item).lower() != "pip"
        ]
        return bool(dependencies or conda_file.get("channels"))

    def _torch_variant(self, resource_plan, selected_gpu, gpu_info, config):
        if not selected_gpu:
            return "cpu"
        preferred = str(getattr(config, "torch_cuda_preference", "auto"))
        if preferred in ("cpu", "cu118", "cu121"):
            return preferred
        requested = str(resource_plan.get("torch_variant") or "").lower()
        if requested in ("cpu", "cu118", "cu121"):
            return requested
        return self._cuda_to_torch_variant(
            str(gpu_info.get("driver_cuda_version") or "")
        ) or "auto"

    @staticmethod
    def _cuda_to_torch_variant(cuda_version):
        match = re.match(r"^\s*(\d+)(?:\.(\d+))?", str(cuda_version))
        if not match:
            return ""
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        if (major, minor) >= (12, 1):
            return "cu121"
        if (major, minor) >= (11, 8):
            return "cu118"
        return ""

    def _reasons(self, conda_file, backend, action, gpu):
        reasons = []
        if conda_file.get("found"):
            reasons.append("environment.yml detected")
        reasons.append("selected environment backend: %s" % backend)
        reasons.append("environment action: %s" % action)
        if gpu:
            reasons.append("selected GPU index %s" % gpu.get("index"))
        return reasons

    def _spec_hash(self, backend, python, conda_file, resource_plan):
        payload = {
            "backend": backend,
            "python": python,
            "channels": conda_file.get("channels") or [],
            "conda_dependencies": conda_file.get("conda_dependencies") or [],
            "pip_dependencies": conda_file.get("pip_dependencies") or [],
            "gpu_required": bool(resource_plan.get("gpu_required")),
            "torch_variant": resource_plan.get("torch_variant", ""),
        }
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _repo_fingerprint(self, repo_dir, conda_file):
        digest = hashlib.sha256(str(repo_dir).encode("utf-8"))
        path = conda_file.get("path")
        if path and Path(path).exists():
            digest.update(Path(path).read_bytes())
        return "sha256:" + digest.hexdigest()

    def _decision(self, **values):
        return EnvironmentCompatibilityDecision(**values).to_dict()
