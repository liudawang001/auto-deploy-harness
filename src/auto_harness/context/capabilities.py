import os
from typing import Any

from auto_harness.context.models import ProviderCapabilities


def resolve_provider_capabilities(provider: Any, config: Any = None) -> ProviderCapabilities:
    provider_name = provider.__class__.__name__ if provider is not None else ""
    model = str(
        getattr(provider, "model", "")
        or getattr(provider, "model_name", "")
        or ""
    )
    provider_window = _positive_int(getattr(provider, "context_window_tokens", None))
    configured_window = _positive_int(_config_get(config, "agent_context_window_tokens"))
    max_output = _positive_int(getattr(provider, "max_tokens", None))
    configured_output = _positive_int(
        _config_get(config, "agent_context_reserved_output_tokens")
    )

    source = "provider"
    if provider_window is None and provider_name in {
        "MockLLMProvider",
        "MemoryEvolutionMockProvider",
    }:
        provider_window = 65536
        source = "registry"
    if provider_window is None and provider_name == "XunfeiSparkProvider":
        env_window = _positive_int(os.environ.get("XUNFEI_CONTEXT_WINDOW_TOKENS"))
        if env_window:
            provider_window = env_window
            source = "explicit_config"
    if configured_window is not None:
        provider_window = min(provider_window, configured_window) if provider_window else configured_window
        source = "explicit_config"
    if provider_window is None:
        provider_window = _positive_int(
            _config_get(config, "agent_context_unknown_model_fallback_tokens")
        ) or 8192
        source = "fallback"

    if max_output is None:
        max_output = configured_output or 2048
    elif configured_output is not None:
        max_output = min(max_output, configured_output)

    return ProviderCapabilities(
        provider_name=provider_name,
        model=model,
        context_window_tokens=provider_window,
        max_output_tokens=max_output,
        tokenizer_id=None,
        supports_tool_calling=callable(getattr(provider, "complete_with_tools", None)),
        usage_format="provider_native",
        source=source,
    )


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _config_get(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)
