"""Unified LLM runtime settings parser.

Resolves effective (provider_name, model, context_window_tokens,
max_output_tokens) from the priority chain:

    CLI runtime overrides > Provider env vars > Harness env vars >
    provider_configs > HarnessConfig fields > code defaults.

This module NEVER reads or returns API keys. It is intentionally
importable before any provider instance is created so the CLI and
Context Governance layers can agree on the same effective values.
"""

from typing import Any, Dict, Iterable, Optional

from auto_harness.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Whitelist for runtime override keys
# ---------------------------------------------------------------------------
_ALLOWED_RUNTIME_OVERRIDE_KEYS = frozenset({
    "model",
    "context_window_tokens",
    "max_output_tokens",
})

_FORBIDDEN_RUNTIME_OVERRIDE_KEYS = frozenset({
    "api_key",
    "authorization",
    "token",
    "password",
    "secret",
})


def normalize_provider_name(name: str) -> str:
    """Normalize a provider name, consistent with ProviderRegistry."""
    return ProviderRegistry.normalize_name(name)


def get_runtime_overrides(config: Any, provider_name: str) -> Dict[str, Any]:
    """Return a shallow copy of the runtime overrides for *provider_name*.

    Returns an empty dict when no overrides exist.  The returned dict is
    safe to mutate without affecting the config.
    """
    overrides = _get_llm_runtime_overrides(config)
    normalized = normalize_provider_name(provider_name)
    provider_overrides = overrides.get(normalized, {})
    if not isinstance(provider_overrides, dict):
        return {}
    result = {}
    for key in _ALLOWED_RUNTIME_OVERRIDE_KEYS:
        if key in provider_overrides:
            result[key] = provider_overrides[key]
    return result


def set_runtime_overrides(
    config: Any,
    provider_names: Iterable[str],
    *,
    model: Optional[str] = None,
    context_window_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> None:
    """Validate and apply runtime overrides IN-PROCESS.

    Does NOT modify ``os.environ``.  Overrides are stored on
    ``config.llm_runtime_overrides``, which is never persisted to JSON.
    """
    overrides = _get_llm_runtime_overrides(config)

    # Validate values
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
    for name, value in (
        ("context_window_tokens", context_window_tokens),
        ("max_output_tokens", max_output_tokens),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be a positive integer" % name)
            parsed = int(value)
            if parsed <= 0:
                raise ValueError("%s must be a positive integer" % name)

    for raw_name in provider_names:
        normalized = normalize_provider_name(raw_name)
        entry = overrides.setdefault(normalized, {})
        if model is not None:
            entry["model"] = str(model).strip()
        if context_window_tokens is not None:
            entry["context_window_tokens"] = int(context_window_tokens)
        if max_output_tokens is not None:
            entry["max_output_tokens"] = int(max_output_tokens)


def has_explicit_model_override(config: Any, provider_name: str) -> bool:
    """Check whether any explicit model override exists for *provider_name*.

    Returns True when the model has been set via:
      - CLI runtime override (--model)
      - Provider-specific env var (<PROVIDER>_MODEL)
      - Generic harness env var (AUTO_HARNESS_LLM_MODEL)
    """
    import os

    runtime = get_runtime_overrides(config, provider_name)
    if runtime.get("model"):
        return True

    prefix = _env_prefix(provider_name)
    if os.environ.get("%s_MODEL" % prefix):
        return True
    if os.environ.get("AUTO_HARNESS_LLM_MODEL"):
        return True

    return False


def validate_runtime_overrides_payload(data: Dict[str, Any]) -> None:
    """Reject any secret-bearing keys in a runtime overrides payload.

    Raises ValueError when forbidden keys are present.
    """
    if not isinstance(data, dict):
        raise ValueError("runtime overrides payload must be an object")
    forbidden = _FORBIDDEN_RUNTIME_OVERRIDE_KEYS.intersection(
        str(k).lower() for k in data
    )
    if forbidden:
        raise ValueError(
            "runtime overrides must not contain secret values: %s"
            % ", ".join(sorted(forbidden))
        )
    unknown = set(data) - _ALLOWED_RUNTIME_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            "unknown runtime override keys: %s; allowed: %s"
            % (", ".join(sorted(unknown)), ", ".join(sorted(_ALLOWED_RUNTIME_OVERRIDE_KEYS)))
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_llm_runtime_overrides(config: Any) -> Dict[str, Dict[str, Any]]:
    """Return the llm_runtime_overrides dict from config."""
    if isinstance(config, dict):
        return config.get("llm_runtime_overrides", {})
    return getattr(config, "llm_runtime_overrides", {})


def _env_prefix(provider_name: str) -> str:
    """Map a provider name to its environment variable prefix."""
    normalized = str(provider_name or "").strip().lower().replace("-", "_")
    known = {
        "openai_compatible": "AUTO_HARNESS_LLM",
        "openai": "OPENAI",
        "deepseek": "DEEPSEEK",
        "qwen": "DASHSCOPE",
        "dashscope": "DASHSCOPE",
        "volcengine": "VOLCENGINE",
        "zhipu": "ZHIPU",
        "vllm": "VLLM",
        "ollama": "OLLAMA",
    }
    return known.get(normalized, normalized.upper())
