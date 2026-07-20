from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectSpec:
    name: str
    repo_url: str
    branch: str = "main"
    description: str = ""


@dataclass
class RuntimePolicy:
    workspace_root: str
    timeout_minutes: int = 90
    allow_network: bool = True
    allow_gpu: bool = False
    allow_source_edit: bool = False
    allow_dependency_install: bool = False
    allow_service_start: bool = False
    max_agent_calls: int = 20


@dataclass
class TaskSpec:
    task_id: str
    project: ProjectSpec
    runtime: RuntimePolicy
    created_at: str
    source_report: Optional[str] = None
    controller: str = "legacy"

