"""DeepSeek V4 Provider with purpose-driven model selection, Thinking Mode,
JSON Output, structured error handling, and bounded retry.

DeepSeek V4 models:
  - deepseek-v4-flash
  - deepseek-v4-pro

Retired model names (rejected at configuration time):
  - deepseek-chat
  - deepseek-reasoner

Architecture:
  ProviderRegistry -> DeepSeekProvider
                          -> OpenAICompatibleProvider (HTTP transport)
                          -> DeepSeek request builder (override _build_payload)
                          -> DeepSeek response parser (override _parse_response)
                          -> DeepSeek error classifier (override _classify_http_error)
                          -> DeepSeek retry loop (override complete)
"""

import hashlib
import json
import os
import random
import time
import urllib.error
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from auto_harness.providers.base import (
    LLMResult,
    Message,
    ProviderRequestContext,
)
from auto_harness.providers.errors import (
    ErrorCategory,
    ProviderError,
    configuration_error,
    empty_content_error,
    invalid_response_error,
)
from auto_harness.providers.openai_compatible import (
    OpenAICompatibleProvider,
)

# ---------------------------------------------------------------------------
# Retired model names — rejected BEFORE any network request
# ---------------------------------------------------------------------------
_RETIRED_DEEPSEEK_MODELS = frozenset({
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-chat-v1",
    "deepseek-coder",
    "deepseek-coder-v1",
})

# Known current V4 models
_V4_MODELS = frozenset({
    "deepseek-v4-flash",
    "deepseek-v4-pro",
})

# Purpose → model defaults (all default to V4 Pro)
_DEFAULT_PURPOSE_MODELS = {
    "plan_first": "deepseek-v4-pro",
    "agent": "deepseek-v4-pro",
    "memory_evolution": "deepseek-v4-pro",
    "llm_test": "deepseek-v4-pro",
    "live_smoke": "deepseek-v4-pro",
}

# Purpose → thinking defaults
_DEFAULT_PURPOSE_THINKING = {
    "plan_first": "enabled",
    "agent": "disabled",
    "memory_evolution": "disabled",
    "llm_test": "disabled",
    "live_smoke": "disabled",
}

# Purpose → reasoning_effort defaults
_DEFAULT_PURPOSE_REASONING_EFFORT = {
    "plan_first": "high",
}

# Purpose → json_mode defaults
_DEFAULT_PURPOSE_JSON_MODE = {
    "plan_first": True,
    "agent": True,
    "memory_evolution": True,
    "llm_test": False,
    "live_smoke": True,
}
_VALID_PURPOSES = frozenset(_DEFAULT_PURPOSE_MODELS)


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek-specific provider with V4 model support and Thinking Mode.

    Extends OpenAICompatibleProvider with:
    - Purpose-driven model, thinking, and JSON mode selection
    - Explicit thinking/reasoning_effort parameters
    - JSON Output mode with response_format
    - Empty content detection with bounded retry
    - Retired model rejection at configuration time
    - Exponential backoff retry for transient errors
    - Reasoning content privacy (never written to disk)
    - Structured error classification
    """

    def __init__(
        self,
        provider_name: str = "deepseek",
        config: Any = None,
        urlopen=None,
        settings_override: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        purpose: str = "agent",
    ) -> None:
        self._purpose = str(purpose or "agent")
        if self._purpose not in _VALID_PURPOSES:
            raise configuration_error(
                provider_name,
                "unsupported purpose '%s'" % self._purpose,
            )
        super().__init__(
            provider_name=provider_name,
            config=config,
            urlopen=urlopen,
            settings_override=settings_override,
            api_key=api_key,
        )
        # Resolve purpose-specific settings
        self.purpose = self._purpose
        self._validate_feature_flag_types()
        self._validate_all_purpose_settings()
        self._resolve_purpose_config()

        # Retry configuration
        self.max_retries = _strict_non_negative_int(
            os.environ.get(
                "DEEPSEEK_MAX_RETRIES",
                self.settings.get("max_retries", 2),
            ),
            provider_name=self.provider_name,
            setting_name="max_retries",
        )
        self.retry_base_seconds = _strict_positive_float(
            self.settings.get("retry_base_seconds", 1.0),
            provider_name=self.provider_name,
            setting_name="retry_base_seconds",
        )
        self.retry_max_seconds = _strict_positive_float(
            self.settings.get("retry_max_seconds", 8.0),
            provider_name=self.provider_name,
            setting_name="retry_max_seconds",
        )
        if self.retry_max_seconds < self.retry_base_seconds:
            raise configuration_error(
                self.provider_name,
                "retry_max_seconds must be greater than or equal to "
                "retry_base_seconds",
            )

        # Feature flags
        self.native_tool_calling = self.settings.get(
            "native_tool_calling", False
        )
        self.allow_beta = self.settings.get("allow_beta", False)
        self.allow_custom_endpoint = self.settings.get(
            "allow_custom_endpoint", False
        )

        # Validate configuration eagerly
        self._validate_models()
        self._validate_token_limits()
        self._validate_thinking()
        self._validate_json_mode()
        self._validate_api_base()
        self._validate_native_tool_calling()

    # ------------------------------------------------------------------
    # Purpose-driven configuration
    # ------------------------------------------------------------------

    def _resolve_purpose_config(self) -> None:
        """Resolve model, thinking, and JSON mode from purpose-specific config.

        Model priority (highest first):
          runtime_overrides.model > DEEPSEEK_MODEL > AUTO_HARNESS_LLM_MODEL
          > provider_configs.deepseek.models[purpose]
          > provider_configs.deepseek.model
          > _DEFAULT_PURPOSE_MODELS[purpose]
        """
        from auto_harness.providers.settings import has_explicit_model_override

        # Model: respect runtime/env override; otherwise purpose → single → default
        explicit = has_explicit_model_override(self.config, self.provider_name)
        if not explicit:
            models_config = self.settings.get("models", {})
            if isinstance(models_config, dict) and self.purpose in models_config:
                purpose_model = models_config[self.purpose]
                if purpose_model:
                    self.model = str(purpose_model)
            elif not self.model:
                self.model = _DEFAULT_PURPOSE_MODELS.get(
                    self.purpose, "deepseek-v4-pro"
                )

        # Thinking
        thinking_config = self.settings.get("thinking", {})
        if isinstance(thinking_config, dict) and self.purpose in thinking_config:
            self.thinking_mode = str(thinking_config[self.purpose])
        else:
            self.thinking_mode = _DEFAULT_PURPOSE_THINKING.get(
                self.purpose, "disabled"
            )

        # Reasoning effort
        reasoning_config = self.settings.get("reasoning_effort", {})
        if isinstance(reasoning_config, dict) and self.purpose in reasoning_config:
            self.reasoning_effort = str(reasoning_config[self.purpose])
        else:
            self.reasoning_effort = _DEFAULT_PURPOSE_REASONING_EFFORT.get(
                self.purpose, ""
            )

        # JSON mode
        json_mode_config = self.settings.get("json_mode", {})
        if isinstance(json_mode_config, dict) and self.purpose in json_mode_config:
            self.json_mode = json_mode_config[self.purpose]
        else:
            self.json_mode = _DEFAULT_PURPOSE_JSON_MODE.get(
                self.purpose, True
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_feature_flag_types(self) -> None:
        for name in (
            "native_tool_calling",
            "allow_beta",
            "allow_custom_endpoint",
            "allow_unknown_model",
        ):
            if name in self.settings and not isinstance(self.settings[name], bool):
                raise configuration_error(
                    self.provider_name,
                    "%s must be boolean" % name,
                )
        for name in ("models", "thinking", "reasoning_effort", "json_mode"):
            if name in self.settings and not isinstance(self.settings[name], dict):
                raise configuration_error(
                    self.provider_name,
                    "%s must be an object keyed by purpose" % name,
                )

    def _validate_all_purpose_settings(self) -> None:
        models = self.settings.get("models", {})
        thinking = self.settings.get("thinking", {})
        reasoning = self.settings.get("reasoning_effort", {})
        json_modes = self.settings.get("json_mode", {})
        for mapping_name, mapping in (
            ("models", models),
            ("thinking", thinking),
            ("reasoning_effort", reasoning),
            ("json_mode", json_modes),
        ):
            for purpose in mapping:
                if purpose not in _VALID_PURPOSES:
                    raise configuration_error(
                        self.provider_name,
                        "%s has unsupported purpose '%s'"
                        % (mapping_name, purpose),
                    )
        for purpose, model in models.items():
            if not isinstance(model, str) or not model.strip():
                raise configuration_error(
                    self.provider_name,
                    "models.%s must be a non-empty string" % purpose,
                )
            normalized = model.strip().lower()
            if normalized in _RETIRED_DEEPSEEK_MODELS:
                raise configuration_error(
                    self.provider_name,
                    "models.%s uses retired model '%s'" % (purpose, model),
                )
            if (
                normalized not in _V4_MODELS
                and not self.settings.get("allow_unknown_model", False)
            ):
                raise configuration_error(
                    self.provider_name,
                    "models.%s uses unknown model '%s'" % (purpose, model),
                )
            if normalized not in _V4_MODELS:
                self._require_explicit_unknown_model_capacity()
        for purpose, value in thinking.items():
            if value not in ("enabled", "disabled"):
                raise configuration_error(
                    self.provider_name,
                    "thinking.%s must be 'enabled' or 'disabled'" % purpose,
                )
        for purpose, value in reasoning.items():
            if value not in ("high", "max"):
                raise configuration_error(
                    self.provider_name,
                    "reasoning_effort.%s must be 'high' or 'max'" % purpose,
                )
        for purpose, value in json_modes.items():
            if not isinstance(value, bool):
                raise configuration_error(
                    self.provider_name,
                    "json_mode.%s must be boolean" % purpose,
                )

    def _validate_models(self) -> None:
        """Reject retired model names and validate model format."""
        if not self.model:
            return
        model_lower = self.model.strip().lower()
        if model_lower in _RETIRED_DEEPSEEK_MODELS:
            raise configuration_error(
                self.provider_name,
                "model '%s' is retired; use deepseek-v4-flash or deepseek-v4-pro"
                % self.model,
            )
        if model_lower in _V4_MODELS:
            return
        # Unknown model — require explicit allow_unknown_model
        if not bool(self.settings.get("allow_unknown_model", False)):
            raise configuration_error(
                self.provider_name,
                "unknown model '%s'; set allow_unknown_model=true in config or use "
                "deepseek-v4-flash / deepseek-v4-pro" % self.model,
            )
        self._require_explicit_unknown_model_capacity()

    def _validate_token_limits(self) -> None:
        """Reject impossible or model-incompatible token budgets eagerly."""
        context_window = int(self.context_window_tokens or 0)
        max_output = int(self.max_tokens or 0)
        if context_window <= 0 or max_output <= 0:
            return
        if max_output >= context_window:
            raise configuration_error(
                self.provider_name,
                "max_output_tokens must be smaller than context_window_tokens",
            )
        if self.model.strip().lower() in _V4_MODELS:
            if context_window > 1_000_000:
                raise configuration_error(
                    self.provider_name,
                    "context_window_tokens exceeds DeepSeek V4 limit of 1000000",
                )
            if max_output > 384_000:
                raise configuration_error(
                    self.provider_name,
                    "max_output_tokens exceeds DeepSeek V4 limit of 384000",
                )

    def _require_explicit_unknown_model_capacity(self) -> None:
        explicit_context = bool(
            self.settings.get("context_window_tokens")
            or os.environ.get("DEEPSEEK_CONTEXT_WINDOW_TOKENS")
        )
        explicit_output = bool(
            self.settings.get("max_tokens")
            or os.environ.get("DEEPSEEK_MAX_TOKENS")
        )
        if not (explicit_context and explicit_output):
            raise configuration_error(
                self.provider_name,
                "unknown models require explicit context_window_tokens and "
                "max_tokens",
            )

    def _validate_thinking(self) -> None:
        """Validate thinking mode is 'enabled' or 'disabled'."""
        if self.thinking_mode not in ("enabled", "disabled"):
            raise configuration_error(
                self.provider_name,
                "thinking must be 'enabled' or 'disabled', got '%s'"
                % self.thinking_mode,
            )
        if self.thinking_mode == "enabled" and self.reasoning_effort:
            if self.reasoning_effort not in ("high", "max"):
                raise configuration_error(
                    self.provider_name,
                    "reasoning_effort must be 'high' or 'max', got '%s'"
                    % self.reasoning_effort,
                )

    def _validate_json_mode(self) -> None:
        """Validate json_mode is boolean."""
        if not isinstance(self.json_mode, bool):
            raise configuration_error(
                self.provider_name,
                "json_mode must be boolean, got %s" % type(self.json_mode).__name__,
            )

    def _validate_api_base(self) -> None:
        """Validate the effective endpoint, including environment overrides."""
        endpoint = self.api_url or self.api_base
        if not endpoint:
            return
        parsed = urlparse(endpoint)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise configuration_error(
                self.provider_name,
                "DeepSeek endpoint must be an absolute HTTPS URL",
            )
        if (
            (parsed.hostname or "").lower() != "api.deepseek.com"
            and not self.allow_custom_endpoint
        ):
            raise configuration_error(
                self.provider_name,
                "custom DeepSeek endpoint requires allow_custom_endpoint=true",
            )
        if "/beta" in endpoint and not self.allow_beta:
            raise configuration_error(
                self.provider_name,
                "api_base contains /beta but allow_beta is false; "
                "set allow_beta=true to use beta endpoints",
            )

    def _validate_native_tool_calling(self) -> None:
        """Fail closed until the real multi-turn tools protocol exists."""
        if self.native_tool_calling:
            raise configuration_error(
                self.provider_name,
                "native_tool_calling is not implemented; keep it false and use "
                "the json_action protocol",
            )

    # ------------------------------------------------------------------
    # Request building (override)
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build DeepSeek-specific payload with Thinking and JSON Output.

        Rules:
        - thinking=enabled → no temperature, set reasoning_effort
        - thinking=disabled → temperature allowed
        - json_mode=true → add response_format + system prompt must contain 'json'
        """
        output_limit = self.max_tokens
        if max_output_tokens:
            output_limit = min(output_limit, int(max_output_tokens))

        # Build messages list
        payload_messages = []
        for message in messages:
            if message.role == "tool" or message.tool_calls or message.tool_call_id:
                raise ProviderError(
                    "native tool-call messages require complete_with_tools(), which "
                    "is not implemented",
                    provider_name=self.provider_name,
                    category=ErrorCategory.INVALID_REQUEST,
                )
            msg = {
                "role": message.role
                if message.role in {"system", "user", "assistant"}
                else "user",
                "content": message.content,
            }
            payload_messages.append(msg)

        if self.json_mode and not any(
            "json" in str(item.get("content", "")).lower()
            for item in payload_messages
        ):
            payload_messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Return exactly one non-empty valid JSON object.",
                },
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": output_limit,
        }

        # Thinking mode — DeepSeek-specific
        payload["thinking"] = {"type": self.thinking_mode}
        if self.thinking_mode == "enabled":
            # DeepSeek requires: no temperature when thinking is enabled
            # temperature is simply NOT sent
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        else:
            payload["temperature"] = temperature

        # JSON Output mode
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        return payload

    # ------------------------------------------------------------------
    # Response parsing (override)
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: Dict[str, Any],
        messages: Optional[List[Message]] = None,
    ) -> LLMResult:
        """Parse DeepSeek response including reasoning_content and finish_reason.

        Returns LLMResult with:
        - text: message.content
        - reasoning_content: message.reasoning_content (NOT persisted to disk)
        - finish_reason: from choices[0].finish_reason
        - request_id: from response id
        - provider_name/provider_model: for audit trail
        """
        latency_ms = raw.pop("_latency_ms", 0)
        raw_dict = raw if isinstance(raw, dict) else {}

        # Extract text
        text = self._extract_text(raw_dict)

        # Extract reasoning_content (memory only, never written to disk artifacts)
        reasoning_content = ""
        choices = raw_dict.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    reasoning_content = str(
                        message.get("reasoning_content", "")
                    )

        # Extract finish_reason
        finish_reason = ""
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                finish_reason = str(choice.get("finish_reason", ""))

        # Extract request_id
        request_id = str(raw_dict.get("id", ""))

        # Extract usage
        usage = raw_dict.get("usage") if isinstance(raw_dict, dict) else None

        # Extract tool_calls (for P3 native tool calling)
        tool_calls = self._extract_tool_calls(raw_dict)

        # Create context with reasoning hash (NOT the full reasoning text)
        reasoning_context = {
            "thinking_enabled": self.thinking_mode == "enabled",
            "reasoning_effort": self.reasoning_effort,
            "json_mode": self.json_mode,
            "reasoning_present": False,
            "reasoning_chars": 0,
            "reasoning_sha256": "",
        }
        if reasoning_content:
            reasoning_context.update({
                "reasoning_present": True,
                "reasoning_chars": len(reasoning_content),
                "reasoning_sha256": hashlib.sha256(
                    reasoning_content.encode("utf-8")
                ).hexdigest(),
            })

        # Keep raw provider data useful for diagnostics without retaining the
        # private chain-of-thought a second time in a generally serialisable dict.
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    message.pop("reasoning_content", None)

        return LLMResult(
            text=text,
            raw=raw_dict,
            usage=usage,
            latency_ms=latency_ms,
            protocol="json_action",
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            request_id=request_id,
            provider_name=self.provider_name,
            provider_model=self.model,
            retry_count=0,
            context=reasoning_context,
        )

    @staticmethod
    def _shorter_json_retry_messages(messages: List[Message]) -> List[Message]:
        """Build one smaller retry prompt while preserving trusted instructions."""
        result: List[Message] = []
        for message in messages:
            if message.role == "system":
                result.append(message)
                continue
            content = str(message.content or "")
            target = min(6000, max(256, len(content) // 2))
            shortened = content[:target]
            if len(shortened) < len(content):
                shortened += "\n[Input compacted after an empty model response.]"
            result.append(
                Message(
                    role=message.role,
                    content=shortened,
                    tool_calls=list(message.tool_calls),
                    tool_call_id=message.tool_call_id,
                )
            )
        result.append(
            Message(
                role="user",
                content="Retry once. Return exactly one non-empty valid JSON object.",
            )
        )
        return result

    def _request_timeout(
        self,
        request_context: Optional[ProviderRequestContext],
    ) -> float:
        remaining = self._remaining_seconds(request_context)
        if remaining is None:
            return float(self.timeout_seconds)
        if remaining < 0.1:
            self._ensure_deadline(request_context)
            raise ProviderError(
                "DeepSeek call deadline has insufficient network budget",
                provider_name=self.provider_name,
                category=ErrorCategory.DEADLINE_EXCEEDED,
            )
        return max(0.1, min(float(self.timeout_seconds), remaining))

    def _ensure_deadline(
        self,
        request_context: Optional[ProviderRequestContext],
    ) -> None:
        remaining = self._remaining_seconds(request_context)
        if remaining is not None and remaining <= 0:
            raise ProviderError(
                "DeepSeek call deadline exceeded",
                provider_name=self.provider_name,
                category=ErrorCategory.DEADLINE_EXCEEDED,
            )

    @staticmethod
    def _remaining_seconds(
        request_context: Optional[ProviderRequestContext],
    ) -> Optional[float]:
        deadline = getattr(request_context, "deadline_at", "") if request_context else ""
        if not deadline:
            return None
        try:
            parsed = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return (parsed - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            raise configuration_error("deepseek", "invalid request deadline")

    # ------------------------------------------------------------------
    # Public API with retry
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: int = None,
        request_context: Optional[ProviderRequestContext] = None,
    ) -> LLMResult:
        """Complete with DeepSeek-specific retry and error handling.

        Retry strategy:
        - 429/500/503 + network errors → bounded exponential backoff
        - 400 (context_overflow) → no retry (handled by Context Executor)
        - 401/402/422 → no retry
        - empty content → one retry with shorter prompt
        """
        missing = self.missing_configuration()
        if missing:
            raise configuration_error(
                self.provider_name,
                "missing configuration: %s" % ", ".join(missing),
            )

        active_messages = list(messages)
        transport_attempt = 0
        transient_retries = 0
        empty_retry_used = False
        empty_retry_metadata: Dict[str, Any] = {}

        while True:
            self._ensure_deadline(request_context)
            try:
                result = self._complete_once(
                    active_messages,
                    temperature,
                    max_output_tokens,
                    request_context=request_context,
                )
                result.retry_count = transport_attempt
                if empty_retry_metadata:
                    result.context["empty_content_retry"] = empty_retry_metadata
                return result
            except ProviderError as exc:
                exc.retry_count = transport_attempt

                # DeepSeek JSON Output can occasionally return empty content.
                # This retry is independent from transient transport retries and
                # happens exactly once, even when max_retries is zero.
                if (
                    exc.category == ErrorCategory.EMPTY_CONTENT
                    and not empty_retry_used
                ):
                    empty_retry_used = True
                    empty_retry_metadata = {
                        "attempted": True,
                        "first_request_id": exc.request_id,
                        "first_finish_reason": getattr(exc, "finish_reason", ""),
                        "first_usage": getattr(exc, "usage", None),
                    }
                    active_messages = self._shorter_json_retry_messages(messages)
                    transport_attempt += 1
                    continue
                if exc.category == ErrorCategory.EMPTY_CONTENT:
                    exc.context = {
                        "empty_content_retry": empty_retry_metadata,
                    }

                # Never retry these categories
                if not exc.retryable:
                    raise

                # Context overflow → let Context Executor handle it
                if exc.category == ErrorCategory.CONTEXT_OVERFLOW:
                    raise

                # Last attempt → give up
                if transient_retries >= self.max_retries:
                    raise

                # Wait with exponential backoff + jitter
                self._retry_delay(
                    transient_retries,
                    exc,
                    request_context=request_context,
                )
                transient_retries += 1
                transport_attempt += 1
                continue
            except Exception as exc:
                # Wrap unexpected errors
                wrapped_error = ProviderError(
                    "%s unexpected provider error" % self.provider_name,
                    provider_name=self.provider_name,
                    category=ErrorCategory.PROVIDER_UNAVAILABLE,
                    safe_detail=str(exc)[:500],
                    retry_count=transport_attempt,
                )
                if transient_retries >= self.max_retries:
                    raise wrapped_error from exc
                self._retry_delay(
                    transient_retries,
                    wrapped_error,
                    request_context=request_context,
                )
                transient_retries += 1
                transport_attempt += 1
                continue

    def _complete_once(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: int = None,
        *,
        request_context: Optional[ProviderRequestContext] = None,
    ) -> LLMResult:
        """Single completion attempt with empty-content detection."""
        payload = self._build_payload(messages, temperature, max_output_tokens)
        headers = self._build_headers()

        try:
            raw = self._perform_request(
                payload,
                headers,
                timeout_seconds=self._request_timeout(request_context),
            )
        except urllib.error.HTTPError as exc:
            classified = self._classify_http_error(exc)
            raise ProviderError(
                "%s HTTP error %s" % (self.provider_name, exc.code),
                provider_name=self.provider_name,
                status_code=int(exc.code),
                error_code=classified.get("error_code", ""),
                category=classified["category"],
                request_id=classified.get("request_id", ""),
                safe_detail=classified.get("safe_detail", ""),
                retry_after_seconds=classified.get("retry_after_seconds"),
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                "%s network error: %s" % (self.provider_name, str(exc.reason)),
                provider_name=self.provider_name,
                category=(
                    ErrorCategory.NETWORK_TIMEOUT
                    if "timeout" in str(exc.reason).lower()
                    else ErrorCategory.NETWORK_ERROR
                ),
                safe_detail=str(exc.reason)[:500],
            ) from exc
        except TimeoutError as exc:
            raise ProviderError(
                "%s network timeout" % self.provider_name,
                provider_name=self.provider_name,
                category=ErrorCategory.NETWORK_TIMEOUT,
                safe_detail=str(exc)[:500],
            ) from exc

        if isinstance(raw, dict) and (
            "_raw_text" in raw or "_invalid_response_shape" in raw
        ):
            raise invalid_response_error(
                self.provider_name,
                "response body is not a valid JSON object",
            )

        result = self._parse_response(raw, messages)

        # Empty content detection with one bounded retry
        if not result.text.strip():
            last_error = empty_content_error(
                self.provider_name,
                request_id=result.request_id,
            )
            if result.tool_calls:
                raise invalid_response_error(
                    self.provider_name,
                    "received native tool_calls while json_action is active",
                    request_id=result.request_id,
                )
            last_error.finish_reason = result.finish_reason
            last_error.usage = result.usage
            raise last_error

        # Non-JSON content rejection when json_mode is enabled
        if self.json_mode and result.text.strip():
            stripped = result.text.strip()
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                raise invalid_response_error(
                    self.provider_name,
                    "response is not valid JSON: %s" % stripped[:200],
                    request_id=result.request_id,
                )
            if not isinstance(parsed, dict):
                raise invalid_response_error(
                    self.provider_name,
                    "JSON Action response must be an object",
                    request_id=result.request_id,
                )

        return result

    def _retry_delay(
        self,
        attempt: int,
        exc: Optional[ProviderError] = None,
        *,
        request_context: Optional[ProviderRequestContext] = None,
    ) -> None:
        """Calculate and apply retry delay with exponential backoff + jitter.

        Prefers the parsed Retry-After response header when available.
        """
        # Check for Retry-After
        retry_after = getattr(exc, "retry_after_seconds", None) if exc else None

        if retry_after is not None:
            delay = min(float(retry_after), self.retry_max_seconds)
            jitter = 0.0
        else:
            delay = min(
                self.retry_max_seconds,
                self.retry_base_seconds * (2 ** attempt),
            )
            # Jitter only computed backoff. Never retry earlier than a
            # server-provided Retry-After value because of negative jitter.
            jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay = max(0.1, delay + jitter)
        remaining = self._remaining_seconds(request_context)
        if remaining is not None and delay >= remaining:
            raise ProviderError(
                "DeepSeek call deadline exceeded before retry",
                provider_name=self.provider_name,
                category=ErrorCategory.DEADLINE_EXCEEDED,
                retry_count=getattr(exc, "retry_count", 0),
            )
        time.sleep(delay)

    # ------------------------------------------------------------------
    # Error classification (override)
    # ------------------------------------------------------------------

    def _classify_http_error(
        self,
        exc: urllib.error.HTTPError,
    ) -> Dict[str, Any]:
        """Classify HTTP errors with DeepSeek-specific error code extraction."""
        status = int(exc.code)
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass

        from auto_harness.providers.errors import sanitize_error_body

        safe = sanitize_error_body(detail)
        retry_after_seconds = _parse_retry_after(
            exc.headers.get("Retry-After") if exc.headers else None
        )

        # Try to extract DeepSeek error code from response body
        error_code = ""
        request_id = ""
        try:
            body = json.loads(detail) if detail else {}
            if isinstance(body, dict):
                error_obj = body.get("error", {})
                if isinstance(error_obj, dict):
                    error_code = str(error_obj.get("code", ""))
                    request_id = str(body.get("request_id", ""))
        except (json.JSONDecodeError, ValueError):
            pass

        if status == 400:
            if any(marker in detail.lower() for marker in (
                "context length", "context_length", "too many tokens",
                "prompt is too long", "request too large",
                "maximum context length",
            )):
                return {
                    "category": ErrorCategory.CONTEXT_OVERFLOW,
                    "retryable": False,
                    "safe_detail": safe,
                    "request_id": request_id,
                    "error_code": error_code,
                    "retry_after_seconds": retry_after_seconds,
                }
            return {
                "category": ErrorCategory.INVALID_REQUEST,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }
        elif status == 401:
            return {
                "category": ErrorCategory.AUTHENTICATION_FAILED,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }
        elif status == 402:
            return {
                "category": ErrorCategory.INSUFFICIENT_BALANCE,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }
        elif status == 422:
            return {
                "category": ErrorCategory.INVALID_PARAMETER,
                "retryable": False,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }
        elif status == 429:
            return {
                "category": ErrorCategory.RATE_LIMITED,
                "retryable": True,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }
        elif status == 503:
            return {
                "category": ErrorCategory.SERVER_OVERLOADED,
                "retryable": True,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }
        else:
            return {
                "category": ErrorCategory.SERVER_ERROR,
                "retryable": status >= 500,
                "safe_detail": safe,
                "request_id": request_id,
                "error_code": error_code,
                "retry_after_seconds": retry_after_seconds,
            }

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> Dict[str, Any]:
        """DeepSeek-specific capability profile."""
        return {
            "provider_name": self.provider_name,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_tokens,
            "supports_tool_calling": False,
            "supports_json_mode": True,
            "supports_thinking": True,
            "supports_streaming": False,
            "source": (
                "provider_config"
                if self.context_window_tokens < 1_000_000
                else "deepseek_model_registry"
            ),
        }


def _strict_non_negative_int(
    value: Any,
    *,
    provider_name: str,
    setting_name: str,
) -> int:
    if isinstance(value, bool):
        raise configuration_error(
            provider_name, "%s must be an integer" % setting_name
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise configuration_error(
            provider_name, "%s must be an integer" % setting_name
        )
    if str(value).strip() not in {str(parsed), "%s.0" % parsed} or parsed < 0:
        raise configuration_error(
            provider_name,
            "%s must be a non-negative integer" % setting_name,
        )
    return parsed


def _strict_positive_float(
    value: Any,
    *,
    provider_name: str,
    setting_name: str,
) -> float:
    if isinstance(value, bool):
        raise configuration_error(
            provider_name, "%s must be positive" % setting_name
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise configuration_error(
            provider_name, "%s must be positive" % setting_name
        )
    if parsed <= 0:
        raise configuration_error(
            provider_name, "%s must be positive" % setting_name
        )
    return parsed


def _parse_retry_after(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (parsed - datetime.now(timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return None
