"""Safe parser for the explicit repository deployment contract."""

from pathlib import Path
from typing import Dict

import yaml

from auto_harness.command_auth.evidence import file_sha256, safe_repository_file
from auto_harness.deployment_contract.schema import (
    DeploymentContract,
    EnvironmentContract,
    SecurityContract,
    ServiceContract,
    VerifyContract,
)
from auto_harness.deployment_contract.validator import DeploymentContractValidator


class DeploymentContractParser:
    filename = "auto-deploy.yaml"

    def __init__(self, validator=None) -> None:
        self.validator = validator or DeploymentContractValidator()

    def parse_repo(self, repo_dir: Path) -> Dict:
        path = Path(repo_dir) / self.filename
        if not path.is_file() or path.is_symlink():
            return {"found": False, "valid": False, "path": self.filename}
        try:
            path = safe_repository_file(repo_dir, self.filename)
            raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="strict"))
            if not isinstance(raw, dict):
                raise ValueError("contract_root_not_object")
            contract = self._from_dict(raw, file_sha256(path))
            self.validator.validate(contract, raw=raw)
        except Exception as exc:  # parser errors are reported, never executed
            reason_code = getattr(exc, "reason_code", "invalid_contract:%s" % type(exc).__name__)
            return {
                "found": True,
                "valid": False,
                "path": self.filename,
                "reason_code": str(reason_code),
            }
        return {
            "found": True,
            "valid": True,
            "path": self.filename,
            "sha256": contract.sha256,
            "contract": contract,
        }

    def _from_dict(self, raw: Dict, sha256: str) -> DeploymentContract:
        project = raw.get("project") if isinstance(raw.get("project"), dict) else {}
        environment = raw.get("environment") if isinstance(raw.get("environment"), dict) else {}
        service = raw.get("service") if isinstance(raw.get("service"), dict) else {}
        verify = raw.get("verify") if isinstance(raw.get("verify"), dict) else {}
        security = raw.get("security") if isinstance(raw.get("security"), dict) else {}
        return DeploymentContract(
            schema_version=raw.get("schema_version"),
            workload_type=str(project.get("workload_type") or "service"),
            runtime_family=str(project.get("runtime_family") or ""),
            environment=EnvironmentContract(
                backend=str(environment.get("backend") or "venv"),
                python=str(environment.get("python") or ""),
                dependency_files=list(environment.get("dependency_files") or []),
                install_commands=list(environment.get("install_commands") or []),
            ),
            service=ServiceContract(
                command=list(service.get("command") or []) if isinstance(service.get("command"), list) else service.get("command"),
                cwd=str(service.get("cwd") or "."),
                host=str(service.get("host") or "0.0.0.0"),
                port=service.get("port", 0),
                startup_timeout_seconds=service.get("startup_timeout_seconds", 60),
                required_env_names=list(service.get("required_env_names") or []),
            ),
            verify=VerifyContract(
                protocol=str(verify.get("protocol") or "http"),
                request=dict(verify.get("request") or {}) if isinstance(verify.get("request"), dict) else verify.get("request"),
                success=dict(verify.get("success") or {}) if isinstance(verify.get("success"), dict) else verify.get("success"),
                timeout_seconds=verify.get("timeout_seconds", 20),
            ),
            security=SecurityContract(
                required_backend=str(security.get("required_backend") or "docker"),
                network_profile=str(security.get("network_profile") or "none"),
                allow_source_edit=security.get("allow_source_edit", False),
            ),
            sha256=sha256,
        )
