"""Controller abstraction for deployment orchestration.

DeploymentController is the Protocol that both LegacyController and
LangGraphController implement. TaskRunner only constructs context,
selects a controller, and saves unified results.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class DeploymentContext:
    """Immutable input for a deployment controller."""
    task_id: str
    run_dir: str
    repo_dir: str
    dry_run: bool
    runtime_policy: Dict[str, Any]
    resume_input: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentResult:
    """Unified output from any deployment controller."""
    task_id: str
    status: str
    stop_reason: str
    controller: str
    verify_status: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class DeploymentController(Protocol):
    """Protocol for deployment controllers."""
    name: str

    def run(self, context: DeploymentContext) -> DeploymentResult: ...

    def resume(self, context: DeploymentContext, resume_input: Optional[Dict[str, Any]] = None) -> DeploymentResult: ...
