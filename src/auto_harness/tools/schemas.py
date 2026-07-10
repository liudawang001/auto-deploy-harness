from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ToolSchema:
    name: str
    risk_level: str = "low"
    side_effects: List[str] = field(default_factory=list)
    requires_policy: bool = False
    allowed_modes: List[str] = field(default_factory=lambda: ["off", "planner", "gated_actor"])
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    success_signal: str = ""
    category: str = "read_only"  # read_only | state_delta | execution
