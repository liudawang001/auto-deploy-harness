"""Typed schema for ``auto-deploy.yaml`` version 1."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class EnvironmentContract:
    backend: str = "venv"
    python: str = ""
    dependency_files: List[str] = field(default_factory=list)
    install_commands: List[List[str]] = field(default_factory=list)


@dataclass
class ServiceContract:
    command: List[str] = field(default_factory=list)
    cwd: str = "."
    host: str = "0.0.0.0"
    port: int = 0
    startup_timeout_seconds: int = 60
    required_env_names: List[str] = field(default_factory=list)


@dataclass
class VerifyContract:
    protocol: str = "http"
    request: Dict[str, Any] = field(default_factory=dict)
    success: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 20


@dataclass
class SecurityContract:
    required_backend: str = "docker"
    network_profile: str = "none"
    allow_source_edit: bool = False


@dataclass
class DeploymentContract:
    schema_version: int
    workload_type: str = "service"
    runtime_family: str = ""
    environment: EnvironmentContract = field(default_factory=EnvironmentContract)
    service: ServiceContract = field(default_factory=ServiceContract)
    verify: VerifyContract = field(default_factory=VerifyContract)
    security: SecurityContract = field(default_factory=SecurityContract)
    path: str = "auto-deploy.yaml"
    sha256: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)
