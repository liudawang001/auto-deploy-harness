import json
from typing import Any, Dict, Iterable, Optional

from auto_harness.context.models import NormalizedUsage, ProviderCapabilities
from auto_harness.providers.base import Message


class ConservativeTokenEstimator:
    name = "utf8_bytes_upper_bound"

    def estimate_text(self, text: str) -> int:
        return len(str(text or "").encode("utf-8"))

    def estimate_request(
        self,
        messages: Iterable[Message],
        capabilities: ProviderCapabilities,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> int:
        message_list = list(messages)
        total = capabilities.request_overhead_tokens
        for message in message_list:
            total += capabilities.message_overhead_tokens
            total += self.estimate_text(getattr(message, "role", ""))
            total += self.estimate_text(getattr(message, "content", ""))
        if tools:
            total += self.estimate_text(
                json.dumps(list(tools), ensure_ascii=False, sort_keys=True)
            )
        if output_schema:
            total += self.estimate_text(
                json.dumps(output_schema, ensure_ascii=False, sort_keys=True)
            )
        return total


def normalize_usage(
    usage: Optional[Dict[str, Any]],
    estimated_input_tokens: int,
) -> NormalizedUsage:
    payload = usage if isinstance(usage, dict) else {}
    input_tokens = _first_int(
        payload,
        "input_tokens",
        "prompt_tokens",
        "inputTokenCount",
        "promptTokenCount",
    )
    output_tokens = _first_int(
        payload,
        "output_tokens",
        "completion_tokens",
        "outputTokenCount",
        "completionTokenCount",
    )
    total_tokens = _first_int(payload, "total_tokens", "totalTokenCount")
    source = "provider_reported" if input_tokens is not None else "estimated"
    if input_tokens is None:
        input_tokens = estimated_input_tokens
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    # Extract DeepSeek cache token fields if present
    cache_hit_tokens = None
    cache_miss_tokens = None
    if "prompt_cache_hit_tokens" in payload:
        cache_hit_tokens = _safe_int(payload.get("prompt_cache_hit_tokens"))
    if "prompt_cache_miss_tokens" in payload:
        cache_miss_tokens = _safe_int(payload.get("prompt_cache_miss_tokens"))
    # Also check for generic cache fields
    if cache_hit_tokens is None:
        cache_hit_tokens = _safe_int(payload.get("cache_hit_tokens"))
    if cache_miss_tokens is None:
        cache_miss_tokens = _safe_int(payload.get("cache_read_input_tokens"))

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        source=source,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
    )


def _first_int(payload: Dict[str, Any], *keys: str):
    for key in keys:
        value = payload.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _safe_int(value) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
