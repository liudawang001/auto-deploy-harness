from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EnvironmentSpec:
    backend: str = "venv"
    name: str = "auto-harness"
    prefix: str = ".conda/envs/auto-harness"
    python: str = "3.10"
    channels: List[str] = field(default_factory=list)
    conda_dependencies: List[str] = field(default_factory=list)
    pip_dependencies: List[str] = field(default_factory=list)
    torch: Dict = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    tool_path: str = ""
    action: str = "create"
    spec_hash: str = ""
    project_id: str = ""
    repo_fingerprint: str = ""
