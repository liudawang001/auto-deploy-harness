"""Normalize JSON Action and provider-native tool calls."""

import json
import re
from typing import Any, Dict, Mapping, Optional

from auto_harness.providers.protocols.schemas import (
    NormalizedToolCall,
    canonical_json_hash,
)


TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


class ToolCallProtocolError(ValueError):
    """Provider response does not satisfy the tool-call transport contract."""


class ToolCallConflictError(ToolCallProtocolError):
    """A call id was reused with different semantic arguments."""


def _parse_arguments(value: Any) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ToolCallProtocolError("tool arguments must be a JSON object or string")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ToolCallProtocolError("tool arguments are not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ToolCallProtocolError("tool arguments JSON must decode to an object")
    return parsed


def _generated_call_id(
    protocol: str,
    provider_name: str,
    provider_model: str,
    turn_index: int,
    call_index: int,
    tool_name: str,
    arguments_hash: str,
) -> str:
    digest = canonical_json_hash({
        "protocol": protocol,
        "provider_name": provider_name,
        "provider_model": provider_model,
        "turn_index": turn_index,
        "call_index": call_index,
        "tool_name": tool_name,
        "arguments_hash": arguments_hash,
    })
    return "call_" + digest.split(":", 1)[1][:24]


def _validate_conflict(
    call: NormalizedToolCall,
    seen_call_hashes: Optional[Mapping[str, str]],
) -> None:
    if not seen_call_hashes:
        return
    previous = seen_call_hashes.get(call.call_id)
    if previous and previous != call.arguments_hash:
        raise ToolCallConflictError("tool call id reused with different arguments")


def normalize_provider_tool_call(
    raw_call: Dict[str, Any],
    *,
    provider_name: str = "",
    provider_model: str = "",
    turn_index: int = 0,
    call_index: int = 0,
    seen_call_hashes: Optional[Mapping[str, str]] = None,
) -> NormalizedToolCall:
    """Normalize OpenAI/DeepSeek or Anthropic-shaped native calls."""
    if not isinstance(raw_call, dict):
        raise ToolCallProtocolError("tool call must be an object")
    function = raw_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = _parse_arguments(function.get("arguments"))
    else:
        name = raw_call.get("name")
        arguments = _parse_arguments(raw_call.get("input", raw_call.get("arguments")))
    name = str(name or "")
    if not TOOL_NAME.fullmatch(name):
        raise ToolCallProtocolError("invalid tool name")
    arguments_hash = canonical_json_hash(arguments)
    call_id = str(raw_call.get("id") or raw_call.get("call_id") or "")
    if not call_id:
        call_id = _generated_call_id(
            "native_tools", provider_name, provider_model, turn_index,
            call_index, name, arguments_hash,
        )
    call = NormalizedToolCall(
        call_id=call_id,
        tool_name=name,
        arguments=arguments,
        arguments_hash=arguments_hash,
        provider_protocol="native_tools",
        provider_name=str(provider_name),
        provider_model=str(provider_model),
        turn_index=int(turn_index),
        call_index=int(call_index),
        raw_call_hash=canonical_json_hash(raw_call),
    )
    _validate_conflict(call, seen_call_hashes)
    return call


def normalize_json_action_call(
    raw_call: Any,
    *,
    provider_name: str = "",
    provider_model: str = "",
    turn_index: int = 0,
    call_index: int = 0,
    seen_call_hashes: Optional[Mapping[str, str]] = None,
) -> NormalizedToolCall:
    """Normalize the current JSON Action ToolCall/dict contract."""
    if hasattr(raw_call, "name") and hasattr(raw_call, "input"):
        payload = {
            "name": getattr(raw_call, "name"),
            "input": getattr(raw_call, "input") or {},
            "id": getattr(raw_call, "call_id", "")
            or getattr(raw_call, "idempotency_key", ""),
        }
    elif isinstance(raw_call, dict):
        payload = {
            "name": raw_call.get("name"),
            "input": raw_call.get("input", {}),
            "id": raw_call.get("call_id") or raw_call.get("idempotency_key"),
        }
    else:
        raise ToolCallProtocolError("JSON Action tool call must be an object")
    name = str(payload.get("name") or "")
    if not TOOL_NAME.fullmatch(name):
        raise ToolCallProtocolError("invalid tool name")
    arguments = _parse_arguments(payload.get("input"))
    arguments_hash = canonical_json_hash(arguments)
    call_id = str(payload.get("id") or "") or _generated_call_id(
        "json_action", provider_name, provider_model, turn_index,
        call_index, name, arguments_hash,
    )
    safe_raw = {"name": name, "input": arguments}
    call = NormalizedToolCall(
        call_id=call_id,
        tool_name=name,
        arguments=arguments,
        arguments_hash=arguments_hash,
        provider_protocol="json_action",
        provider_name=str(provider_name),
        provider_model=str(provider_model),
        turn_index=int(turn_index),
        call_index=int(call_index),
        raw_call_hash=canonical_json_hash(safe_raw),
    )
    _validate_conflict(call, seen_call_hashes)
    return call
