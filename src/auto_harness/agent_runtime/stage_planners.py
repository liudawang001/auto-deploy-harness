"""Stage-specific planners for LLM Decision Gates.

Each planner builds an observation prompt for its stage and parses
the LLM response into a GateDecision. Planners do NOT execute tools.
"""
import json
from typing import Dict, List, Optional

from auto_harness.agent_runtime.stage_schemas import (
    GateDecision,
    RUNNER_TOOLS,
    ENV_TOOLS,
    MODEL_TOOLS,
    REPAIR_TOOLS,
    PLAN_TOOLS,
    PIPELINE_STAGES,
)
from auto_harness.providers.json_utils import parse_json_object


# ------------------------------------------------------------------
# Shared prompt building
# ------------------------------------------------------------------

DECISION_SYSTEM_PROMPT = """You are a deployment decision agent for stage: {stage}.
Choose exactly one next tool call from allowed_tools.
Do not return prose outside JSON.
Do not mark success yourself.
The runtime verifier decides success from tool_result.

You must respond with a JSON object matching this schema:
{{
  "status": "ok",
  "hypothesis": "string describing what you expect",
  "confidence": 0.0,
  "tool_call": {{
    "name": "string - must be one of allowed_tools",
    "input": {{}}
  }},
  "expected_observation": "string"
}}

If no safe action can improve the outcome, respond with:
{{
  "status": "no_action",
  "hypothesis": "explanation",
  "confidence": 0.0,
  "tool_call": null,
  "expected_observation": "",
  "stop_reason": "no_safe_action"
}}
"""


def _json_block(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def parse_gate_decision(raw_response: str, allowed_tools: List[str] = None, stage: str = "") -> GateDecision:
    """Parse LLM raw response into a GateDecision.

    Strict validation:
    - Must be valid JSON
    - Must have status in (ok, no_action)
    - If status=ok, must have tool_call with name in allowed_tools
    - confidence must be a number
    """
    if not raw_response or not raw_response.strip():
        return GateDecision(stage=stage, status="invalid", raw_response=raw_response or "", stop_reason="empty_response")

    try:
        parsed = parse_json_object(raw_response)
    except (ValueError, TypeError):
        return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="invalid_json")

    if not parsed or not isinstance(parsed, dict):
        return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="invalid_json")

    status = str(parsed.get("status", "")).lower()
    if status not in ("ok", "no_action"):
        return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="invalid_status")

    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else 0.0
    except (ValueError, TypeError):
        confidence = 0.0

    hypothesis = str(parsed.get("hypothesis", ""))
    expected_observation = str(parsed.get("expected_observation", ""))
    stop_reason = parsed.get("stop_reason")

    tool_call = None
    tool_call_raw = parsed.get("tool_call")

    if status == "ok":
        if not isinstance(tool_call_raw, dict) or not tool_call_raw.get("name"):
            return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="missing_tool_call_name")
        tool_name = str(tool_call_raw["name"])
        if allowed_tools and tool_name not in allowed_tools:
            return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="unknown_tool", hypothesis=hypothesis, confidence=confidence)
        tool_input = tool_call_raw.get("input")
        if tool_input is not None and not isinstance(tool_input, dict):
            return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="invalid_tool_input", hypothesis=hypothesis, confidence=confidence)
        tool_call = {"name": tool_name, "input": tool_input if isinstance(tool_input, dict) else {}}
    elif status == "no_action":
        if tool_call_raw is not None and isinstance(tool_call_raw, dict) and tool_call_raw.get("name"):
            return GateDecision(stage=stage, status="invalid", raw_response=raw_response, stop_reason="no_action_with_tool_call")
        if not stop_reason:
            stop_reason = "no_safe_action"

    return GateDecision(
        stage=stage,
        status=status,
        hypothesis=hypothesis,
        confidence=confidence,
        tool_call=tool_call,
        expected_observation=expected_observation,
        stop_reason=stop_reason,
        raw_response=raw_response,
    )


# ------------------------------------------------------------------
# Runner Planner
# ------------------------------------------------------------------

RUNNER_OBSERVATION_TEMPLATE = """## Current State
- Stage: runner
- Goal: Select the safest and most likely successful run candidate.

## Run Candidates
{run_candidates}

## Selected Files
{selected_files}

## Constraints
{constraints}

## Allowed Tools
{allowed_tools}"""


class RunnerPlanner:
    """LLM planner for the runner decision gate."""

    def plan(self, observation: Dict, provider=None, allowed_tools: List[str] = None) -> GateDecision:
        """Call LLM provider and return a parsed GateDecision for runner stage."""
        if provider is None:
            return GateDecision(stage="runner", status="no_action", stop_reason="no_provider", raw_response="")

        allowed = allowed_tools or list(RUNNER_TOOLS)
        prompt = RUNNER_OBSERVATION_TEMPLATE.format(
            run_candidates=_json_block(observation.get("run_candidates", [])),
            selected_files=_json_block(observation.get("selected_files", {})),
            constraints="\n".join("- %s" % c for c in observation.get("constraints", [])),
            allowed_tools=", ".join(allowed),
        )

        try:
            from auto_harness.providers import Message
            result = provider.complete([
                Message(role="system", content=DECISION_SYSTEM_PROMPT.format(stage="runner")),
                Message(role="user", content=prompt),
            ])
            raw_response = result.text if result and hasattr(result, "text") else ""
        except Exception:
            return GateDecision(stage="runner", status="invalid", stop_reason="provider_error", raw_response="")

        return parse_gate_decision(raw_response, allowed_tools=allowed, stage="runner")


# ------------------------------------------------------------------
# Environment Planner
# ------------------------------------------------------------------

ENV_OBSERVATION_TEMPLATE = """## Current State
- Stage: env_solve
- Goal: Diagnose and resolve dependency conflicts.

## Failed Stage
{failed_stage}

## Requirements
{requirements}

## Install Log Tail
{install_log_tail}

## Deterministic Constraints
{deterministic_constraints}

## Risk Reasons
{risk_reasons}

## Constraints
{constraints}

## Allowed Tools
{allowed_tools}"""


class EnvPlanner:
    """LLM planner for the env_solve decision gate."""

    def plan(self, observation: Dict, provider=None, allowed_tools: List[str] = None) -> GateDecision:
        if provider is None:
            return GateDecision(stage="env_solve", status="no_action", stop_reason="no_provider", raw_response="")

        allowed = allowed_tools or list(ENV_TOOLS)
        prompt = ENV_OBSERVATION_TEMPLATE.format(
            failed_stage=_json_block(observation.get("failed_stage", "")),
            requirements=_json_block(observation.get("requirements", [])),
            install_log_tail=str(observation.get("install_log_tail", ""))[:4000],
            deterministic_constraints=_json_block(observation.get("deterministic_constraints", [])),
            risk_reasons=_json_block(observation.get("risk_reasons", [])),
            constraints="\n".join("- %s" % c for c in observation.get("constraints", [])),
            allowed_tools=", ".join(allowed),
        )

        try:
            from auto_harness.providers import Message
            result = provider.complete([
                Message(role="system", content=DECISION_SYSTEM_PROMPT.format(stage="env_solve")),
                Message(role="user", content=prompt),
            ])
            raw_response = result.text if result and hasattr(result, "text") else ""
        except Exception:
            return GateDecision(stage="env_solve", status="invalid", stop_reason="provider_error", raw_response="")

        return parse_gate_decision(raw_response, allowed_tools=allowed, stage="env_solve")


# ------------------------------------------------------------------
# Model Planner
# ------------------------------------------------------------------

MODEL_OBSERVATION_TEMPLATE = """## Current State
- Stage: model_prepare
- Goal: Resolve model asset ambiguity and select download strategy.

## Model Mentions
{model_mentions}

## Detected Assets
{detected_assets}

## Cache Candidates
{cache_candidates}

## Git LFS
{git_lfs}

## Constraints
{constraints}

## Allowed Tools
{allowed_tools}"""


class ModelPlanner:
    """LLM planner for the model_prepare decision gate."""

    def plan(self, observation: Dict, provider=None, allowed_tools: List[str] = None) -> GateDecision:
        if provider is None:
            return GateDecision(stage="model_prepare", status="no_action", stop_reason="no_provider", raw_response="")

        allowed = allowed_tools or list(MODEL_TOOLS)
        prompt = MODEL_OBSERVATION_TEMPLATE.format(
            model_mentions=_json_block(observation.get("model_mentions", [])),
            detected_assets=_json_block(observation.get("detected_assets", [])),
            cache_candidates=_json_block(observation.get("cache_candidates", [])),
            git_lfs=_json_block(observation.get("git_lfs", {})),
            constraints="\n".join("- %s" % c for c in observation.get("constraints", [])),
            allowed_tools=", ".join(allowed),
        )

        try:
            from auto_harness.providers import Message
            result = provider.complete([
                Message(role="system", content=DECISION_SYSTEM_PROMPT.format(stage="model_prepare")),
                Message(role="user", content=prompt),
            ])
            raw_response = result.text if result and hasattr(result, "text") else ""
        except Exception:
            return GateDecision(stage="model_prepare", status="invalid", stop_reason="provider_error", raw_response="")

        return parse_gate_decision(raw_response, allowed_tools=allowed, stage="model_prepare")


# ------------------------------------------------------------------
# Repair Planner (for repair actuator gate)
# ------------------------------------------------------------------

REPAIR_OBSERVATION_TEMPLATE = """## Current State
- Stage: repair
- Goal: Propose and execute a repair action that resolves the failure.

## Failure
{failure}

## Diagnosis
{diagnosis}

## Previous Repair Attempts
{previous_repairs}

## Constraints
{constraints}

## Allowed Tools
{allowed_tools}"""


class RepairActuatorPlanner:
    """LLM planner for the repair actuator gate."""

    def plan(self, observation: Dict, provider=None, allowed_tools: List[str] = None) -> GateDecision:
        if provider is None:
            return GateDecision(stage="repair", status="no_action", stop_reason="no_provider", raw_response="")

        allowed = allowed_tools or list(REPAIR_TOOLS)
        prompt = REPAIR_OBSERVATION_TEMPLATE.format(
            failure=_json_block(observation.get("failure", {})),
            diagnosis=_json_block(observation.get("diagnosis", {})),
            previous_repairs=_json_block(observation.get("previous_repairs", [])),
            constraints="\n".join("- %s" % c for c in observation.get("constraints", [])),
            allowed_tools=", ".join(allowed),
        )

        try:
            from auto_harness.providers import Message
            result = provider.complete([
                Message(role="system", content=DECISION_SYSTEM_PROMPT.format(stage="repair")),
                Message(role="user", content=prompt),
            ])
            raw_response = result.text if result and hasattr(result, "text") else ""
        except Exception:
            return GateDecision(stage="repair", status="invalid", stop_reason="provider_error", raw_response="")

        return parse_gate_decision(raw_response, allowed_tools=allowed, stage="repair")


# ------------------------------------------------------------------
# Plan Planner (Cross-stage Planning / Revision)
# ------------------------------------------------------------------

PLAN_OBSERVATION_TEMPLATE = """## Current State
- Stage: plan
- Goal: Generate deployment strategy and stage-specific hints.

## Analysis Summary
{analysis_summary}

## Frameworks
{frameworks}

## Previous Results
{previous_results}

## Uncertainties
{uncertainties}

## Constraints
{constraints}

## Allowed Tools
{allowed_tools}"""


class PlanPlanner:
    """LLM planner for the cross-stage planning gate.

    Plan gate does NOT execute tools. It generates strategy hints
    that are stored and used by subsequent stages.
    """

    def plan(self, observation: Dict, provider=None, allowed_tools: List[str] = None) -> GateDecision:
        if provider is None:
            return GateDecision(stage="plan", status="no_action", stop_reason="no_provider", raw_response="")

        allowed = allowed_tools or list(PLAN_TOOLS)
        prompt = PLAN_OBSERVATION_TEMPLATE.format(
            analysis_summary=_json_block(observation.get("analysis_summary", {})),
            frameworks=", ".join(observation.get("frameworks", [])),
            previous_results=_json_block(observation.get("previous_results", {})),
            uncertainties=_json_block(observation.get("uncertainties", [])),
            constraints="\n".join("- %s" % c for c in observation.get("constraints", [])),
            allowed_tools=", ".join(allowed),
        )

        try:
            from auto_harness.providers import Message
            result = provider.complete([
                Message(role="system", content=DECISION_SYSTEM_PROMPT.format(stage="plan")),
                Message(role="user", content=prompt),
            ])
            raw_response = result.text if result and hasattr(result, "text") else ""
        except Exception:
            return GateDecision(stage="plan", status="invalid", stop_reason="provider_error", raw_response="")

        return parse_gate_decision(raw_response, allowed_tools=allowed, stage="plan")
