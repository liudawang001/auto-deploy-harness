"""Provider for OpenAI-compatible chat completion APIs.

The adapter intentionally depends only on the Python standard library. Secrets
are read from environment variables and are never accepted as plain-text
configuration values.
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from auto_harness.providers.base import LLMResult, Message


_ENV_PREFIXES = {
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


class OpenAICompatibleProvider:
    """Call a configurable OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        provider_name: str = "openai_compatible",
        config: Any = None,
        urlopen=None,
        settings_override: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider_name = _normalize_name(provider_name)
        self.config = config
        self.settings = _provider_settings(config, self.provider_name)
        if settings_override:
            self.settings.update(dict(settings_override))
        prefix = _ENV_PREFIXES.get(
            self.provider_name,
            self.provider_name.upper().replace("-", "_").replace(".", "_"),
        )
        self.api_base = _first_nonempty(
            os.environ.get("%s_API_BASE" % prefix),
            os.environ.get("%s_BASE_URL" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_API_BASE"),
            self.settings.get("api_base"),
        ).rstrip("/")
        self.api_url = _first_nonempty(
            os.environ.get("%s_API_URL" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_API_URL"),
            self.settings.get("api_url"),
        )
        self.model = _first_nonempty(
            os.environ.get("%s_MODEL" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_MODEL"),
            self.settings.get("model"),
        )
        key_env = _first_nonempty(
            self.settings.get("api_key_env"),
            "%s_API_KEY" % prefix,
        )
        self.api_key_env = key_env
        self.api_key = (
            str(api_key)
            if api_key is not None
            else _first_nonempty(
                os.environ.get(key_env),
                os.environ.get("AUTO_HARNESS_LLM_API_KEY"),
            )
        )
        self.timeout_seconds = _positive_int(
            os.environ.get("%s_TIMEOUT_SECONDS" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_TIMEOUT_SECONDS"),
            self.settings.get("timeout_seconds"),
            default=60,
        )
        self.max_tokens = _positive_int(
            os.environ.get("%s_MAX_TOKENS" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_MAX_TOKENS"),
            self.settings.get("max_tokens"),
            default=4096,
        )
        self.context_window_tokens = _positive_int(
            os.environ.get("%s_CONTEXT_WINDOW_TOKENS" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_CONTEXT_WINDOW_TOKENS"),
            self.settings.get("context_window_tokens"),
            _config_get(config, "agent_context_window_tokens"),
            default=0,
        )
        self.organization = _first_nonempty(
            os.environ.get("%s_ORGANIZATION" % prefix),
            self.settings.get("organization"),
        )
        self.urlopen = urlopen or urllib.request.urlopen

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: int = None,
    ) -> LLMResult:
        missing = self.missing_configuration()
        if missing:
            raise RuntimeError(
                "%s provider is not configured: missing %s"
                % (self.provider_name, ", ".join(missing))
            )
        output_limit = self.max_tokens
        if max_output_tokens:
            output_limit = min(output_limit, int(max_output_tokens))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role
                    if message.role in {"system", "user", "assistant"}
                    else "user",
                    "content": message.content,
                }
                for message in messages
            ],
            "temperature": temperature,
            "max_tokens": output_limit,
        }
        request = urllib.request.Request(
            self._resolve_url(),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        if self.api_key:
            request.add_header("Authorization", "Bearer %s" % self.api_key)
        if self.organization:
            request.add_header("OpenAI-Organization", self.organization)

        started = time.time()
        try:
            with self.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                "%s HTTP error %s: %s"
                % (self.provider_name, exc.code, detail)
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "%s network error: %s" % (self.provider_name, str(exc.reason))
            ) from exc

        latency_ms = int((time.time() - started) * 1000)
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError:
            return LLMResult(
                text=raw_text,
                raw={"raw_text": raw_text[:10000]},
                latency_ms=latency_ms,
            )
        return LLMResult(
            text=self._extract_text(raw),
            raw=raw,
            usage=raw.get("usage") if isinstance(raw, dict) else None,
            latency_ms=latency_ms,
            protocol="json_action",
            tool_calls=self._extract_tool_calls(raw),
        )

    def missing_configuration(self) -> List[str]:
        missing = []
        if not (self.api_url or self.api_base):
            missing.append("api_url or api_base")
        if not self.model:
            missing.append("model")
        if bool(self.settings.get("require_api_key", True)) and not self.api_key:
            missing.append(self.api_key_env or "API key")
        if not self.context_window_tokens:
            missing.append("context_window_tokens")
        return missing

    def _resolve_url(self) -> str:
        if self.api_url:
            return self.api_url
        if self.api_base.endswith("/chat/completions"):
            return self.api_base
        return self.api_base + "/chat/completions"

    @staticmethod
    def _extract_text(raw: Any) -> str:
        if not isinstance(raw, dict):
            return str(raw)
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("text") is not None:
                                parts.append(str(item["text"]))
                        if parts:
                            return "\n".join(parts)
                if choice.get("text") is not None:
                    return str(choice["text"])
        for key in ("content", "text", "answer"):
            if raw.get(key) is not None:
                return str(raw[key])
        return json.dumps(raw, ensure_ascii=False)

    @staticmethod
    def _extract_tool_calls(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return []
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        return calls if isinstance(calls, list) else []


def _provider_settings(config: Any, name: str) -> Dict[str, Any]:
    if isinstance(config, dict):
        provider_configs = config.get("provider_configs") or {}
    else:
        provider_configs = getattr(config, "provider_configs", {}) or {}
    if not isinstance(provider_configs, dict):
        return {}
    settings = provider_configs.get(name)
    if not isinstance(settings, dict):
        for configured_name, configured_settings in provider_configs.items():
            try:
                normalized_name = _normalize_name(configured_name)
            except ValueError:
                continue
            if normalized_name == name:
                settings = configured_settings
                break
    if not isinstance(settings, dict):
        settings = provider_configs.get("openai_compatible")
    return dict(settings) if isinstance(settings, dict) else {}


def _config_get(config: Any, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _normalize_name(value: str) -> str:
    return str(value or "openai_compatible").strip().lower().replace("-", "_")


def _first_nonempty(*values) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _positive_int(*values, default: int) -> int:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return default
