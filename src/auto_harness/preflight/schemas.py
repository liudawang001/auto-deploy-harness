"""JSON-compatible preflight schemas."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class HostCapabilitySnapshot:
    schema_version: int = 1
    collected_at: str = ""
    host: Dict[str, str] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    environment_runtimes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CondaEnvironmentInventory:
    schema_version: int = 1
    tool: str = ""
    tool_path: str = ""
    root_prefix: str = ""
    active_prefix: str = ""
    environments: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentCompatibilityDecision:
    schema_version: int = 1
    status: str = "blocked"
    backend: str = "venv"
    tool: str = ""
    action: str = "block"
    target_prefix: str = ""
    selected_gpu_index: int = -1
    python: str = "3.10"
    torch_variant: str = "cpu"
    spec_hash: str = ""
    project_id: str = ""
    repo_fingerprint: str = ""
    reuse_candidate: str = ""
    fallback: str = ""
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    policy_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentPostcheckEvidence:
    schema_version: int = 1
    status: str = "failed"
    prefix: str = ""
    python: Dict[str, Any] = field(default_factory=dict)
    packages: Dict[str, Any] = field(default_factory=dict)
    gpu_runtime: Dict[str, Any] = field(default_factory=dict)
    spec_hash: str = ""
    evidence_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
