from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class StageState:
    status: str = "pending"
    updated_at: str = ""
    result_path: Optional[str] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskState:
    task_id: str
    status: str = "created"
    current_stage: str = "created"
    attempt: int = 1
    stages: Dict[str, StageState] = field(default_factory=dict)
    agent_session_id: Optional[str] = None
    last_safe_stage: Optional[str] = None
    report_path: Optional[str] = None
