"""Secure terminal configuration for an ephemeral custom provider session."""
import getpass
from dataclasses import dataclass
from typing import Any, Callable, Dict
from urllib.parse import urlparse

from auto_harness.providers.openai_compatible import OpenAICompatibleProvider
from auto_harness.providers.registry import ProviderRegistry


@dataclass(frozen=True)
class InteractiveProviderSession:
    provider_name: str
    api_base: str
    model: str
    context_window_tokens: int
    max_tokens: int
    requires_api_key: bool

    def safe_summary(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "api_base": self.api_base,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "max_tokens": self.max_tokens,
            "api_key": "provided_in_memory"
            if self.requires_api_key
            else "not_required",
            "persisted": False,
        }


class InteractiveProviderConfigurator:
    """Prompt for non-secret settings and read the API key without echo."""

    def __init__(
        self,
        registry: ProviderRegistry,
        input_fn: Callable[[str], str] = input,
        secret_fn: Callable[[str], str] = getpass.getpass,
    ) -> None:
        self.registry = registry
        self.input_fn = input_fn
        self.secret_fn = secret_fn

    def configure(
        self,
        *,
        config: Any = None,
        default_name: str = "custom",
    ) -> InteractiveProviderSession:
        name = self.registry.normalize_name(
            self._prompt("自定义厂商名称", default_name)
        )
        api_base = self._prompt("API 调用地址（到 /v1）", "")
        self._validate_api_base(api_base)
        api_key = str(
            self.secret_fn(
                "API Key（输入时不回显；本地免鉴权接口可直接回车）: "
            )
            or ""
        )
        require_api_key = bool(api_key)
        model = self._prompt("模型名称", "")
        if not model:
            raise ValueError("模型名称不能为空")
        context_window = self._positive_int_prompt(
            "Context Window Token 数",
            32768,
        )
        max_tokens = self._positive_int_prompt(
            "最大输出 Token 数",
            4096,
        )

        settings = {
            "api_base": api_base,
            "model": model,
            "context_window_tokens": context_window,
            "max_tokens": max_tokens,
            "require_api_key": require_api_key,
        }
        self.registry.register(
            name,
            lambda current_config, purpose, provider_name: (
                OpenAICompatibleProvider(
                    provider_name=provider_name,
                    config=current_config,
                    settings_override=settings,
                    api_key=api_key,
                )
            ),
            replace=True,
        )
        return InteractiveProviderSession(
            provider_name=name,
            api_base=api_base,
            model=model,
            context_window_tokens=context_window,
            max_tokens=max_tokens,
            requires_api_key=require_api_key,
        )

    def _prompt(self, label: str, default: str) -> str:
        suffix = " [%s]" % default if default else ""
        value = str(self.input_fn("%s%s: " % (label, suffix)) or "").strip()
        return value or default

    def _positive_int_prompt(self, label: str, default: int) -> int:
        value = self._prompt(label, str(default))
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s必须是正整数" % label) from exc
        if parsed <= 0:
            raise ValueError("%s必须是正整数" % label)
        return parsed

    @staticmethod
    def _validate_api_base(value: str) -> None:
        parsed = urlparse(str(value or ""))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API 调用地址必须是有效的 http/https URL")
