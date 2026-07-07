from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentAction:
    type: str
    reason: str = ""
    confidence: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    requires: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentObservation:
    task_id: str
    stage: str
    repo_dir: str = ""
    file_tree: List[str] = field(default_factory=list)
    selected_files: Dict[str, str] = field(default_factory=dict)
    deterministic_result: Dict[str, Any] = field(default_factory=dict)
    previous_results: Dict[str, Any] = field(default_factory=dict)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    selected_skills: List[Dict[str, Any]] = field(default_factory=list)
    runtime_policy: Dict[str, Any] = field(default_factory=dict)
    allowed_action_types: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    stage: str
    status: str = "skipped"
    summary: str = ""
    confidence: float = 0.0
    actions: List[AgentAction] = field(default_factory=list)
    plan_delta: Dict[str, Any] = field(default_factory=dict)
    risks: List[str] = field(default_factory=list)
    rationale: str = ""
    raw_text: str = ""
    provider: str = ""
    model: str = ""
    trace_path: str = ""
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    verify_hint: Dict[str, Any] = field(default_factory=dict)
