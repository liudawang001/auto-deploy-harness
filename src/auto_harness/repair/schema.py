from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RepairAction:
    type: str
    reason: str
    requires: Dict[str, bool] = field(default_factory=dict)
    payload: Dict = field(default_factory=dict)


@dataclass
class RepairPlan:
    root_cause: str
    confidence: float
    actions: List[RepairAction] = field(default_factory=list)
    rollback: Dict = field(default_factory=dict)
    rerun_from: str = ""
    verification_required: bool = True
    status: str = "proposed"
