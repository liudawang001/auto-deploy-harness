import os
from typing import Any

from auto_harness.context.models import ProviderCapabilities


# Known DeepSeek model capabilities (from official docs)
_DEEPSEEK_CAPABILITIES = {
    "deepseek-v4-flash": {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "api_supports_tool_calling": True,
        "supports_json_mode": True,
        "supports_thinking": True,
        "api_supports_streaming": True,
    },
    "deepseek-v4-pro": {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 384_000,
        "api_supports_tool_calling": True,
        "supports_json_mode": True,
        "supports_thinking": True,
        "api_supports_streaming": True,
    },
}


def resolve_provider_capabilities(provider: Any, config: Any = None) -> ProviderCapabilities:
    provider_class = provider.__class__.__name__ if provider is not None else ""
    provider_name = str(
        getattr(provider, "provider_name", "") or provider_class
    )
    model = str(
        getattr(provider, "model", "")
        or getattr(provider, "model_name", "")
        or ""
    )

    # Check for DeepSeek known capabilities first
    model_lower = model.strip().lower()
    if model_lower in _DEEPSEEK_CAPABILITIES:
        caps = _DEEPSEEK_CAPABILITIES[model_lower]
        configured_provider_window = _positive_int(
            getattr(provider, "context_window_tokens", None)
        )
        provider_window = (
            min(caps["context_window_tokens"], configured_provider_window)
            if configured_provider_window is not None
            else caps["context_window_tokens"]
        )
        configured_provider_output = _positive_int(
            getattr(provider, "max_tokens", None)
        )
        max_output = (
            min(caps["max_output_tokens"], configured_provider_output)
            if configured_provider_output is not None
            else caps["max_output_tokens"]
        )
        supports_tool_calling = bool(
            caps["api_supports_tool_calling"]
            and callable(getattr(provider, "complete_with_tools", None))
            and bool(getattr(provider, "native_tool_calling", False))
        )
        source = (
            "provider_config"
            if configured_provider_window is not None
            and configured_provider_window < caps["context_window_tokens"]
            else "deepseek_model_registry"
        )
    else:
        provider_window = _positive_int(getattr(provider, "context_window_tokens", None))
        max_output = _positive_int(getattr(provider, "max_tokens", None))
        supports_tool_calling = callable(getattr(provider, "complete_with_tools", None))
        source = "provider"

    configured_window = _positive_int(_config_get(config, "agent_context_window_tokens"))
    configured_output = _positive_int(
        _config_get(config, "agent_context_reserved_output_tokens")
    )

    # Mock providers get a known window
    if provider_window is None and provider_class in {
        "MockLLMProvider",
        "MemoryEvolutionMockProvider",
    }:
        provider_window = 65536
        source = "registry"

    # Xunfei provider
    if provider_window is None and provider_class == "XunfeiSparkProvider":
        env_window = _positive_int(os.environ.get("XUNFEI_CONTEXT_WINDOW_TOKENS"))
        if env_window:
            provider_window = env_window
            source = "explicit_config"

    # Apply project operational budget — DEFAULT does NOT raise to 1M
    if configured_window is not None:
        provider_window = (
            min(provider_window, configured_window)
            if provider_window
            else configured_window
        )
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
        supports_tool_calling=supports_tool_calling,
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
