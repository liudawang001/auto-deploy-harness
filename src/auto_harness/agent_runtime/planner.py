"""LLM planner for the verify agent.

Calls LLM provider with a structured prompt and returns a parsed AgentDecision.
The prompt forces JSON output with a tool_call schema.
"""
from typing import Dict, List, Optional

from auto_harness.agent_runtime.schemas import AgentDecision, VERIFY_TOOLS, parse_agent_decision
from auto_harness.utils.time import utc_now_iso


SYSTEM_PROMPT = """You are a deployment verification agent.
Choose exactly one next tool call from allowed_tools.
Do not return prose outside JSON.
Do not mark success yourself.
The runtime verifier decides success from tool_result.

You must respond with a JSON object matching this schema:
{
  "status": "ok",
  "hypothesis": "string describing what you expect to find",
  "confidence": 0.0,
  "tool_call": {
    "name": "string - must be one of allowed_tools",
    "input": {}
  },
  "expected_observation": "string",
  "fallback_tool_call": null
}

If no safe local probe can prove trace handling, respond with:
{
  "status": "no_action",
  "hypothesis": "explanation",
  "confidence": 0.0,
  "tool_call": null,
  "expected_observation": "",
  "fallback_tool_call": null,
  "stop_reason": "no_safe_tool"
}
"""

OBSERVATION_TEMPLATE = """## Current State
- Stage: verify
- Goal: Find an evidence-producing local probe that proves the current service handles the current trace_id.

## Service
{service}

## Failed Checks
{failed_checks}

## Evidence Summary
{evidence_summary}

## Selected Files
{selected_files}

## Allowed Tools
{allowed_tools}

## Constraints
{constraints}"""


class VerifyPlanner:
    """LLM planner for the verify agent. Produces AgentDecision from observation."""

    def __init__(self, provider=None) -> None:
        self.provider = provider

    def plan_verify(self, observation: Dict, allowed_tools: List[str] = None) -> AgentDecision:
        """Call LLM provider and return a parsed AgentDecision.

        If provider is None or fails, returns an invalid AgentDecision.
        """
        if self.provider is None:
            return AgentDecision(status="no_action", stop_reason="no_provider", raw_response="")

        allowed = allowed_tools or list(VERIFY_TOOLS)
        prompt = self._build_prompt(observation, allowed)

        try:
            from auto_harness.providers import Message
            result = self.provider.complete([
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=prompt),
            ])
            raw_response = result.text if result and hasattr(result, "text") else ""
        except Exception:
            return AgentDecision(status="invalid", stop_reason="provider_error", raw_response="")

        decision = parse_agent_decision(raw_response, allowed_tools=allowed)
        return decision

    def _build_prompt(self, observation: Dict, allowed_tools: List[str]) -> str:
        service = observation.get("service", {})
        failed_checks = observation.get("failed_checks", [])
        evidence_summary = observation.get("evidence_summary", {})
        selected_files = observation.get("selected_files", {})
        constraints = observation.get("constraints", [])

        # Truncate selected files to avoid blowing up the prompt
        truncated_files = {}
        for name, content in selected_files.items():
            if isinstance(content, str) and len(content) > 3000:
                truncated_files[name] = content[:3000] + "\n... (truncated)"
            else:
                truncated_files[name] = content

        return OBSERVATION_TEMPLATE.format(
            service=_json_block(service),
            failed_checks=_json_block(failed_checks),
            evidence_summary=_json_block(evidence_summary),
            selected_files=_json_block(truncated_files),
            allowed_tools=", ".join(allowed_tools),
            constraints="\n".join("- %s" % c for c in constraints),
        )


def _json_block(obj) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)
