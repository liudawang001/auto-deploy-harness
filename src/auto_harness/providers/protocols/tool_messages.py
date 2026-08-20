"""Safe conversion between normalized results and provider tool messages."""

import json
from typing import Any, Dict

from auto_harness.providers.base import LLMResult, Message
from auto_harness.providers.protocols.schemas import (
    NormalizedToolResult,
    canonical_json_hash,
)
from auto_harness.agent.safety import AgentInputSanitizer


_SENSITIVE_KEY_PARTS = (
    "api_key", "apikey", "authorization", "credential", "password",
    "private_key", "secret", "token",
)


def redact_tool_payload(value: Any) -> Any:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            result[str(key)] = (
                "[REDACTED]"
                if any(part in lowered for part in _SENSITIVE_KEY_PARTS)
                else redact_tool_payload(item)
            )
        return result
    if isinstance(value, list):
        return [redact_tool_payload(item) for item in value]
    if isinstance(value, str):
        return AgentInputSanitizer().scan_text(value)["text"]
    return value


def assistant_tool_call_message(result: LLMResult) -> Message:
    """Preserve the assistant call envelope required by provider protocols."""
    return Message(
        role="assistant",
        content=str(result.text or ""),
        reasoning_content=str(result.reasoning_content or ""),
        tool_calls=list(result.tool_calls or []),
    )


def tool_result_message(
    result: NormalizedToolResult,
    *,
    max_chars: int = 12000,
) -> Message:
    """Create a bounded, redacted tool message without leaking local metadata."""
    payload = redact_tool_payload(result.to_dict())
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > max_chars:
        payload["result"] = {
            "truncated": True,
            "original_result_hash": canonical_json_hash(payload.get("result", {})),
        }
        payload["truncated"] = True
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > max_chars:
        # Retain the execution contract and hash even under an unusually small
        # configured budget. Do not include a potentially sensitive preview.
        payload = {
            "call_id": result.call_id,
            "operation_id": result.operation_id,
            "tool_name": result.tool_name,
            "status": result.status,
            "executed": result.executed,
            "applied": result.applied,
            "truncated": True,
            "result_hash": result.result_hash,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return Message(role="tool", content=encoded[:max_chars], tool_call_id=result.call_id)


__all__ = [
    "assistant_tool_call_message",
    "redact_tool_payload",
    "tool_result_message",
]
