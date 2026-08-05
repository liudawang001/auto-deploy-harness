"""Registry and factory for pluggable LLM providers."""
import re
from typing import Any, Callable, Dict, Tuple


ProviderFactory = Callable[[Any, str, str], Any]
_VALID_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class ProviderRegistry:
    """Map stable provider names to factories.

    A factory receives ``(config, purpose, provider_name)``. ``purpose`` is one
    of ``agent``, ``plan_first``, ``memory_evolution``, ``llm_test`` or
    ``live_smoke`` and allows a provider family to select a purpose-specific
    implementation without leaking that choice into business code.
    """

    def __init__(self, include_builtins: bool = True) -> None:
        self._factories: Dict[str, ProviderFactory] = {}
        if include_builtins:
            self._register_builtins()

    def register(
        self,
        name: str,
        factory: ProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        normalized = self.normalize_name(name)
        if not callable(factory):
            raise TypeError("provider factory must be callable")
        if normalized in self._factories and not replace:
            raise ValueError("provider already registered: %s" % normalized)
        self._factories[normalized] = factory

    def create(
        self,
        name: str,
        *,
        config: Any = None,
        purpose: str = "agent",
    ):
        normalized = self.normalize_name(name)
        factory = self._factories.get(normalized)
        if factory is None and _has_provider_config(config, normalized):
            from auto_harness.providers.openai_compatible import (
                OpenAICompatibleProvider,
            )

            factory = (
                lambda current_config, current_purpose, provider_name:
                OpenAICompatibleProvider(
                    provider_name=provider_name,
                    config=current_config,
                )
            )
        if factory is None:
            raise ValueError(
                "unknown provider '%s'; available providers: %s"
                % (normalized, ", ".join(self.names()))
            )
        provider = factory(config, str(purpose or "agent"), normalized)
        if not callable(getattr(provider, "complete", None)):
            raise TypeError(
                "provider factory '%s' returned an object without complete()"
                % normalized
            )
        return provider

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._factories))

    def contains(self, name: str) -> bool:
        try:
            normalized = self.normalize_name(name)
        except ValueError:
            return False
        return normalized in self._factories

    @staticmethod
    def normalize_name(name: str) -> str:
        normalized = str(name or "").strip().lower().replace("-", "_")
        if not normalized or not _VALID_PROVIDER_NAME.fullmatch(normalized):
            raise ValueError("invalid provider name: %r" % name)
        return normalized

    def _register_builtins(self) -> None:
        from auto_harness.providers.memory_evolution_mock import (
            MemoryEvolutionMockProvider,
        )
        from auto_harness.providers.mock import MockLLMProvider
        from auto_harness.providers.openai_compatible import (
            OpenAICompatibleProvider,
        )
        from auto_harness.providers.xunfei import XunfeiSparkProvider

        self.register(
            "mock",
            lambda config, purpose, name: (
                MemoryEvolutionMockProvider()
                if purpose == "memory_evolution"
                else MockLLMProvider()
            ),
        )
        self.register(
            "xunfei",
            lambda config, purpose, name: XunfeiSparkProvider(config=config),
        )

        # DeepSeek uses dedicated provider with purpose-driven model selection
        self.register(
            "deepseek",
            lambda config, purpose, name: _create_deepseek_provider(
                config, purpose, name
            ),
        )

        # Other OpenAI-compatible providers use generic adapter
        for name in (
            "openai_compatible",
            "openai",
            "qwen",
            "dashscope",
            "volcengine",
            "zhipu",
            "vllm",
            "ollama",
        ):
            self.register(
                name,
                lambda config, purpose, registered_name: OpenAICompatibleProvider(
                    provider_name=registered_name,
                    config=config,
                ),
            )


def _create_deepseek_provider(config, purpose, provider_name):
    """Create a DeepSeekProvider with purpose-specific configuration.

    This factory function is extracted so it can be patched in tests
    without affecting other provider factories in the lambda closure.
    """
    from auto_harness.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(
        provider_name=provider_name,
        config=config,
        purpose=purpose,
    )


DEFAULT_PROVIDER_REGISTRY = ProviderRegistry()


def create_provider(
    name: str,
    *,
    config: Any = None,
    purpose: str = "agent",
):
    return DEFAULT_PROVIDER_REGISTRY.create(
        name,
        config=config,
        purpose=purpose,
    )


def provider_names() -> Tuple[str, ...]:
    return DEFAULT_PROVIDER_REGISTRY.names()


def _has_provider_config(config: Any, name: str) -> bool:
    if isinstance(config, dict):
        provider_configs = config.get("provider_configs") or {}
    else:
        provider_configs = getattr(config, "provider_configs", {}) or {}
    if not isinstance(provider_configs, dict):
        return False
    if isinstance(provider_configs.get(name), dict):
        return True
    for configured_name, settings in provider_configs.items():
        try:
            normalized = ProviderRegistry.normalize_name(configured_name)
        except ValueError:
            continue
        if normalized == name and isinstance(settings, dict):
            return True
    return False
