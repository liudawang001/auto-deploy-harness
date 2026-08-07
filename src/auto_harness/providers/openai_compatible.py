"""Provider for OpenAI-compatible chat completion APIs.

The adapter intentionally depends only on the Python standard library. Secrets
are read from environment variables and are never accepted as plain-text
configuration values.

Refactored to expose reusable transport methods that vendor-specific
providers (e.g. DeepSeekProvider) can override individually.
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
            _runtime_get(config, self.provider_name, "model"),
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
            _runtime_get(config, self.provider_name, "max_output_tokens"),
            os.environ.get("%s_MAX_TOKENS" % prefix),
            os.environ.get("AUTO_HARNESS_LLM_MAX_TOKENS"),
            self.settings.get("max_tokens"),
            default=4096,
        )
        self.context_window_tokens = _positive_int(
            _runtime_get(config, self.provider_name, "context_window_tokens"),
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

    # ------------------------------------------------------------------
    # Reusable transport methods (overridable by subclasses)
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the JSON payload for a chat completion request."""
        output_limit = self.max_tokens
        if max_output_tokens:
            output_limit = min(output_limit, int(max_output_tokens))
        return {
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

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for the request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _perform_request(
        self,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send the request and return parsed JSON or raise an error.

        Returns:
            Parsed response dict with an internal ``_latency_ms`` field.

        Raises:
            urllib.error.HTTPError: on HTTP errors.
            urllib.error.URLError: on network errors.

        Malformed JSON is returned as ``{"_raw_text": ...}`` so a
        vendor-specific parser can classify it without losing the safe preview.
        """
        if headers is None:
            headers = self._build_headers()
        url = self._resolve_url()
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)

        started = time.time()
        try:
            request_timeout = (
                max(0.1, float(timeout_seconds))
                if timeout_seconds is not None
                else self.timeout_seconds
            )
            with self.urlopen(request, timeout=request_timeout) as response:
                raw_text = response.read().decode("utf-8")
        finally:
            pass

        latency_ms = int((time.time() - started) * 1000)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            raw = {"_raw_text": raw_text[:10000]}
        else:
            raw = (
                parsed
                if isinstance(parsed, dict)
                else {"_invalid_response_shape": type(parsed).__name__}
            )
        raw["_latency_ms"] = latency_ms
        return raw

    def _parse_response(
        self,
        raw: Dict[str, Any],
        messages: Optional[List[Message]] = None,
    ) -> LLMResult:
        """Parse a raw API response into an LLMResult."""
        latency_ms = raw.pop("_latency_ms", 0)
        return LLMResult(
            text=self._extract_text(raw),
            raw=raw,
            usage=raw.get("usage") if isinstance(raw, dict) else None,
            latency_ms=latency_ms,
            protocol="json_action",
            tool_calls=self._extract_tool_calls(raw),
        )

    def _classify_http_error(
        self,
        exc: urllib.error.HTTPError,
    ) -> Dict[str, Any]:
        """Classify an HTTPError into structured error info.

        Returns a dict with keys: category, retryable, safe_detail, request_id.
        """
        status = int(exc.code)
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass

        from auto_harness.providers.errors import ErrorCategory, sanitize_error_body

        safe = sanitize_error_body(detail)
        request_id = ""

        if status == 400:
            # Check for context overflow in the detail
            if any(marker in detail.lower() for marker in (
                "context length", "context_length", "too many tokens",
                "prompt is too long", "request too large",
            )):
                return {
                    "category": ErrorCategory.CONTEXT_OVERFLOW,
                    "retryable": False,
                    "safe_detail": safe,
                    "request_id": request_id,
                }
            return {
                "category": ErrorCategory.INVALID_REQUEST,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
            }
        elif status == 401:
            return {
                "category": ErrorCategory.AUTHENTICATION_FAILED,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
            }
        elif status == 402:
            return {
                "category": ErrorCategory.INSUFFICIENT_BALANCE,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
            }
        elif status == 422:
            return {
                "category": ErrorCategory.INVALID_PARAMETER,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
            }
        elif status == 429:
            return {
                "category": ErrorCategory.RATE_LIMITED,
                "retryable": True,
                "safe_detail": safe,
                "request_id": request_id,
            }
        elif status == 503:
            return {
                "category": ErrorCategory.SERVER_OVERLOADED,
                "retryable": True,
                "safe_detail": safe,
                "request_id": request_id,
            }
        else:
            return {
                "category": ErrorCategory.SERVER_ERROR,
                "retryable": status >= 500,
                "safe_detail": safe,
                "request_id": request_id,
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        payload = self._build_payload(messages, temperature, max_output_tokens)
        headers = self._build_headers()

        try:
            raw = self._perform_request(payload, headers)
        except urllib.error.HTTPError as exc:
            classified = self._classify_http_error(exc)
            from auto_harness.providers.errors import ProviderError
            raise ProviderError(
                "%s HTTP error %s" % (self.provider_name, exc.code),
                provider_name=self.provider_name,
                status_code=int(exc.code),
                category=classified["category"],
                request_id=classified.get("request_id", ""),
                safe_detail=classified.get("safe_detail", ""),
            ) from exc
        except urllib.error.URLError as exc:
            from auto_harness.providers.errors import ProviderError, ErrorCategory
            raise ProviderError(
                "%s network error: %s" % (self.provider_name, str(exc.reason)),
                provider_name=self.provider_name,
                category=ErrorCategory.NETWORK_ERROR,
                safe_detail=str(exc.reason)[:500],
            ) from exc

        return self._parse_response(raw, messages)

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


# ------------------------------------------------------------------
# Configuration helpers (public for reuse by subclasses)
# ------------------------------------------------------------------

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


def _runtime_get(config: Any, provider_name: str, key: str):
    """Read a single non-sensitive runtime override for *provider_name*.

    Returns None when no override exists or the config does not support
    runtime overrides.
    """
    if config is None:
        return None
    overrides = getattr(config, "llm_runtime_overrides", None)
    if not isinstance(overrides, dict):
        return None
    normalized = _normalize_name(provider_name)
    provider_overrides = overrides.get(normalized)
    if not isinstance(provider_overrides, dict):
        return None
    value = provider_overrides.get(key)
    if value is None or value == "":
        return None
    return value
