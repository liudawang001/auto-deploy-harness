from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    stage: str
    status: str
    summary: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    error: Optional[str] = None

