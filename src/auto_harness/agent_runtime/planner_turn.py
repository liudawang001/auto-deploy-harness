"""Versioned JSON protocol for plan-first repository observation turns."""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from auto_harness.providers.json_utils import parse_json_object


PLANNER_TURN_SCHEMA = {
    "type": "object",
    "required": ["protocol_version", "kind"],
    "properties": {
        "protocol_version": {"type": "integer", "enum": [1]},
        "kind": {"type": "string", "enum": ["observe", "final"]},
        "reason": {"type": "string"},
        "requests": {"type": "array"},
        "plan": {"type": "object"},
    },
}


class PlannerTurnValidationError(ValueError):
    """Raised when a provider response violates the planner-turn protocol."""


@dataclass
class ObservationRequest:
    request_id: str
    tool: str
    input: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"request_id": self.request_id, "tool": self.tool, "input": self.input}


@dataclass
class PlannerTurn:
    kind: str
    protocol_version: int = 1
    reason: str = ""
    requests: List[ObservationRequest] = field(default_factory=list)
    plan: Dict = field(default_factory=dict)
    raw_response: str = ""


class PlannerTurnParser:
    DEFAULT_ALLOWED_TOOLS = frozenset({
        "inspect_repo_tree",
        "search_repo",
        "read_selected_files",
        "parse_dependency_files",
    })

    def __init__(
        self,
        max_requests: int = 4,
        allowed_tools: Optional[Iterable[str]] = None,
    ):
        self.max_requests = max(1, int(max_requests))
        self.allowed_tools = frozenset(allowed_tools or self.DEFAULT_ALLOWED_TOOLS)

    def parse(self, raw_text: str) -> PlannerTurn:
        try:
            data = parse_json_object(raw_text)
        except Exception as exc:
            raise PlannerTurnValidationError(
                "invalid planner turn JSON: %s" % str(exc)
            ) from exc
        if not isinstance(data, dict):
            raise PlannerTurnValidationError("planner turn must be an object")

        # Backward-compatible final plan: old providers and deterministic test
        # fixtures may return a DeploymentPlan directly.
        if "kind" not in data and "status" in data:
            return PlannerTurn(kind="final", plan=data, raw_response=raw_text)

        # JSON-mode providers occasionally omit this metadata-only field even
        # when the response otherwise matches the current schema.  Missing
        # means the sole supported version; explicit unknown versions remain
        # fail-closed.
        if data.get("protocol_version", 1) != 1:
            raise PlannerTurnValidationError("unsupported planner turn protocol_version")
        kind = str(data.get("kind", ""))
        if kind not in {"observe", "final"}:
            raise PlannerTurnValidationError("planner turn kind must be observe or final")
        if kind == "final":
            plan = data.get("plan")
            if not isinstance(plan, dict) or not plan:
                raise PlannerTurnValidationError(
                    "final planner turn must contain a plan object"
                )
            return PlannerTurn(kind="final", plan=plan, raw_response=raw_text)

        raw_requests = data.get("requests")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise PlannerTurnValidationError(
                "observe planner turn must contain requests"
            )
        if len(raw_requests) > self.max_requests:
            raise PlannerTurnValidationError("planner turn exceeds request limit")
        requests = []
        seen = set()
        explicit_ids = {
            str(item.get("request_id", ""))
            for item in raw_requests
            if isinstance(item, dict) and str(item.get("request_id", ""))
        }
        for index, item in enumerate(raw_requests):
            if not isinstance(item, dict):
                raise PlannerTurnValidationError(
                    "requests[%d] must be an object" % index
                )
            request_id = str(item.get("request_id", ""))
            if not request_id:
                # request_id is only an internal correlation key.  Some
                # otherwise valid JSON-mode providers omit it, so derive a
                # stable value instead of aborting the whole deployment plan.
                suffix = index + 1
                request_id = "auto_request_%d" % suffix
                while request_id in explicit_ids or request_id in seen:
                    suffix += 1
                    request_id = "auto_request_%d" % suffix
            tool = str(item.get("tool", ""))
            tool_input = item.get("input", {})
            if request_id in seen:
                raise PlannerTurnValidationError(
                    "request_id must be unique"
                )
            if not tool or not isinstance(tool_input, dict):
                raise PlannerTurnValidationError(
                    "observation request requires tool and object input"
                )
            if tool not in self.allowed_tools:
                raise PlannerTurnValidationError(
                    "observation tool is not allowed in plan/replan: %s" % tool
                )
            seen.add(request_id)
            requests.append(ObservationRequest(request_id, tool, tool_input))
        return PlannerTurn(
            kind="observe",
            reason=str(data.get("reason", ""))[:1000],
            requests=requests,
            raw_response=raw_text,
        )
