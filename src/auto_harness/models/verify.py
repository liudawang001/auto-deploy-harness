from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VerifyResult:
    status: str
    trace_id: str
    service: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    diagnosis: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    next_action: str = "report"

