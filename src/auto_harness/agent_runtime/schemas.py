"""Schemas for the LLM-driven verify agent.

These data structures define the contract between LLM output, policy gates,
tool execution, and artifact persistence.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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


# ------------------------------------------------------------------
# Phase 2: LLM-driven verify agent schemas
# ------------------------------------------------------------------

# Verify tools that the LLM can choose from
VERIFY_TOOLS = ("probe_http", "discover_gradio_api", "discover_openapi_schema", "probe_browser_dom")


@dataclass
class ToolCall:
    """A parsed tool call from LLM output."""
    name: str
    input: Dict = field(default_factory=dict)
    idempotency_key: str = ""


@dataclass
class AgentDecision:
    """Parsed LLM decision for verify agent."""
    status: str = "invalid"  # ok | no_action | invalid
    hypothesis: str = ""
    confidence: float = 0.0
    tool_call: Optional[ToolCall] = None
    expected_observation: str = ""
    fallback_tool_call: Optional[ToolCall] = None
    stop_reason: Optional[str] = None
    raw_response: str = ""


@dataclass
class PolicyDecision:
    """Result of tool policy validation."""
    allowed: bool = False
    reason: str = ""
    risk: str = "high"
    normalized_input: Optional[Dict] = None


@dataclass
class ToolResult:
    """Result of tool execution.

    Distinguishes between executed (real command/network/download) and
    applied (state delta written). metadata_only cannot be counted as
    repair_verified or self_healing.
    """
    status: str = "error"  # passed|failed|uncertain|error|rejected
    tool_name: str = ""
    category: str = "read_only"  # read_only|state_delta|side_effect|evidence
    policy_allowed: bool = False
    executed: bool = False
    applied: bool = False
    metadata_only: bool = False
    evidence: Dict = field(default_factory=dict)
    evidence_path: Optional[str] = None
    strong_verify_pass: bool = False
    error: Optional[str] = None
    started_at: str = ""
    ended_at: str = ""


@dataclass
class AgentVerifyResult:
    """Result of the act_verify loop."""
    triggered: bool = False
    final_status: str = "uncertain"  # passed|uncertain
    llm_helped: bool = False
    step_count: int = 0
    accepted_tool_count: int = 0
    rejected_tool_count: int = 0
    strong_verify_pass: bool = False
    evidence_paths: List[str] = field(default_factory=list)
    stop_reason: str = ""
    mode: str = ""  # planner|gated_actor


def parse_agent_decision(raw_response: str, allowed_tools: List[str] = None) -> AgentDecision:
    """Parse LLM raw response into AgentDecision.

    Strict validation:
    - Must be valid JSON
    - Must have status in (ok, no_action)
    - If status=ok, must have tool_call with name and tool_call.name in allowed_tools
    - confidence must be a number
    """
    import json
    from auto_harness.providers.json_utils import parse_json_object

    if not raw_response or not raw_response.strip():
        return AgentDecision(status="invalid", raw_response=raw_response or "", stop_reason="empty_response")

    try:
        parsed = parse_json_object(raw_response)
    except (ValueError, TypeError):
        return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="invalid_json")
    if not parsed or not isinstance(parsed, dict):
        return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="invalid_json")

    status = str(parsed.get("status", "")).lower()
    if status not in ("ok", "no_action"):
        return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="invalid_status")

    confidence = parsed.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            confidence = 0.0
    else:
        confidence = 0.0

    hypothesis = str(parsed.get("hypothesis", ""))
    expected_observation = str(parsed.get("expected_observation", ""))
    stop_reason = parsed.get("stop_reason")
    fallback_raw = parsed.get("fallback_tool_call")

    # Parse tool_call
    tool_call = None
    tool_call_raw = parsed.get("tool_call")

    if status == "ok":
        if not isinstance(tool_call_raw, dict) or not tool_call_raw.get("name"):
            return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="missing_tool_call_name")
        tool_name = str(tool_call_raw["name"])
        if allowed_tools and tool_name not in allowed_tools:
            return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="unknown_tool", hypothesis=hypothesis, confidence=confidence)
        tool_input = tool_call_raw.get("input")
        if tool_input is not None and not isinstance(tool_input, dict):
            return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="invalid_tool_input", hypothesis=hypothesis, confidence=confidence)
        tool_call = ToolCall(name=tool_name, input=tool_input if isinstance(tool_input, dict) else {})
    elif status == "no_action":
        # no_action is valid, tool_call should be null or empty dict (no name)
        if tool_call_raw is not None and isinstance(tool_call_raw, dict) and tool_call_raw.get("name"):
            return AgentDecision(status="invalid", raw_response=raw_response, stop_reason="no_action_with_tool_call")
        if not stop_reason:
            stop_reason = "no_safe_tool"

    # Parse fallback
    fallback_tool_call = None
    if isinstance(fallback_raw, dict) and fallback_raw.get("name"):
        fallback_tool_call = ToolCall(name=str(fallback_raw["name"]), input=fallback_raw.get("input") if isinstance(fallback_raw.get("input"), dict) else {})

    return AgentDecision(
        status=status,
        hypothesis=hypothesis,
        confidence=confidence,
        tool_call=tool_call,
        expected_observation=expected_observation,
        fallback_tool_call=fallback_tool_call,
        stop_reason=stop_reason,
        raw_response=raw_response,
    )
