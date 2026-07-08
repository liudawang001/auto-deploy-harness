from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentGoal:
    task_id: str
    objective: str
    success_condition: str = "final verify evidence contains current trace id"


@dataclass
class AgentRuntimeStep:
    step_id: int
    goal: Dict
    observation: Dict = field(default_factory=dict)
    belief_state_before: Dict = field(default_factory=dict)
    llm_decision: Dict = field(default_factory=dict)
    policy_result: Dict = field(default_factory=dict)
    tool_call: Dict = field(default_factory=dict)
    tool_result: Dict = field(default_factory=dict)
    belief_state_after: Dict = field(default_factory=dict)
    critique: Dict = field(default_factory=dict)
    next_step: str = "continue"
    termination_reason: str = ""
