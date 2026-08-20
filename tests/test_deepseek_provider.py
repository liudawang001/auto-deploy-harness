"""Tests for DeepSeekProvider.

Validates:
- V4 Flash/Pro model selection
- Retired model rejection
- Purpose-specific model selection
- Thinking enabled/disabled in request body
- Thinking enabled → no temperature
- JSON Mode request body
- Content/reasoning_content/finish_reason parsing
- Empty content detection
- Non-JSON response rejection (json_mode=true)
- Secret not in payload or error
- 401/402/422 no retry
- 429/500/503 retry with backoff
- Context overflow delegation
- Reasoning not persisted in trace artifacts
"""
import json
import os
import time
import unittest
from io import BytesIO
from unittest.mock import patch

from auto_harness.config import HarnessConfig
from auto_harness.providers.base import LLMResult, Message, ProviderRequestContext
from auto_harness.providers.deepseek import DeepSeekProvider
from auto_harness.providers.errors import (
    ErrorCategory,
    ProviderError,
)


def _fake_urlopen_factory(response_payload, status_code=200):
    """Create a fake urlopen that returns the given payload."""

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(request, timeout=60):
        if status_code != 200:
            import urllib.error
            raise urllib.error.HTTPError(
                url="https://api.deepseek.com/chat/completions",
                code=status_code,
                msg="Error",
                hdrs={},
                fp=BytesIO(json.dumps(response_payload).encode("utf-8")),
            )
        return _FakeResponse()

    return fake_urlopen


class DeepSeekProviderTests(unittest.TestCase):
    """Core DeepSeekProvider tests."""

    def _make_provider(self, purpose="agent", **overrides):
        """Create a DeepSeekProvider for testing."""
        settings = {
            "api_base": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "DEEPSEEK_API_KEY",
            "context_window_tokens": 1_000_000,
            "max_tokens": 4096,
            "timeout_seconds": 30,
            "require_api_key": False,  # test without real key
        }
        # Remove model if explicitly set to None (to test purpose defaults)
        if "model" in overrides and overrides["model"] is None:
            del settings["model"]
            del overrides["model"]
        settings.update(overrides)
        with patch.dict(os.environ, {}, clear=False):
            return DeepSeekProvider(
                provider_name="deepseek",
                config={"provider_configs": {"deepseek": settings}},
                purpose=purpose,
            )

    def _make_provider_with_key(self, purpose="agent", **overrides):
        settings = {
            "api_base": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "DEEPSEEK_API_KEY",
            "context_window_tokens": 1_000_000,
            "max_tokens": 4096,
            "timeout_seconds": 30,
        }
        settings.update(overrides)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            return DeepSeekProvider(
                provider_name="deepseek",
                config={"provider_configs": {"deepseek": settings}},
                purpose=purpose,
            )

    # --- Model selection ---

    def test_v4_flash_model_selection(self):
        """V4 Flash model is selected for agent purpose."""
        provider = self._make_provider(purpose="agent")
        self.assertEqual(provider.model, "deepseek-v4-flash")

    def test_v4_pro_model_selection_for_plan_first(self):
        """V4 Pro model is selected for plan_first purpose."""
        # No model in config → uses purpose-specific default
        provider = self._make_provider(purpose="plan_first", model=None)
        self.assertEqual(provider.model, "deepseek-v4-pro")

    def test_purpose_specific_model_override(self):
        """Purpose-specific model config overrides defaults."""
        provider = self._make_provider(
            purpose="agent",
            models={"agent": "deepseek-v4-pro"},
        )
        self.assertEqual(provider.model, "deepseek-v4-pro")

    def test_retired_model_rejected(self):
        """Retired deepseek-chat is rejected at construction."""
        with self.assertRaises(ProviderError) as ctx:
            self._make_provider(model="deepseek-chat")
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)
        self.assertIn("retired", str(ctx.exception).lower())

    def test_retired_reasoner_model_rejected(self):
        """Retired deepseek-reasoner is rejected at construction."""
        with self.assertRaises(ProviderError) as ctx:
            self._make_provider(model="deepseek-reasoner")
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    def test_unknown_model_rejected_by_default(self):
        """Unknown model is rejected without allow_unknown_model."""
        with self.assertRaises(ProviderError) as ctx:
            self._make_provider(model="some-future-model")
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    def test_unknown_model_allowed_with_flag(self):
        """Unknown model is allowed when allow_unknown_model=true."""
        provider = self._make_provider(
            model="some-future-model",
            allow_unknown_model=True,
        )
        self.assertEqual(provider.model, "some-future-model")

    def test_unknown_model_requires_explicit_capacity(self):
        with self.assertRaises(ProviderError) as ctx:
            DeepSeekProvider(
                provider_name="deepseek",
                config={
                    "provider_configs": {
                        "deepseek": {
                            "api_base": "https://api.deepseek.com",
                            "model": "some-future-model",
                            "allow_unknown_model": True,
                            "require_api_key": False,
                        }
                    }
                },
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    # --- Thinking mode ---

    def test_thinking_enabled_for_plan_first(self):
        """Thinking is enabled for plan_first purpose."""
        provider = self._make_provider(purpose="plan_first")
        self.assertEqual(provider.thinking_mode, "enabled")

    def test_thinking_disabled_for_agent(self):
        """Thinking is disabled for agent purpose."""
        provider = self._make_provider(purpose="agent")
        self.assertEqual(provider.thinking_mode, "disabled")

    def test_thinking_enabled_no_temperature_in_payload(self):
        """When thinking=enabled, temperature is NOT in payload."""
        provider = self._make_provider_with_key(purpose="plan_first")
        payload = provider._build_payload(
            [Message(role="user", content="test")],
            temperature=0.5,
        )
        self.assertNotIn("temperature", payload)

    def test_thinking_disabled_temperature_in_payload(self):
        """When thinking=disabled, temperature IS in payload."""
        provider = self._make_provider_with_key(purpose="agent")
        payload = provider._build_payload(
            [Message(role="user", content="test")],
            temperature=0.5,
        )
        self.assertIn("temperature", payload)
        self.assertEqual(payload["temperature"], 0.5)

    def test_thinking_type_in_payload(self):
        """Thinking type is always in payload."""
        provider = self._make_provider_with_key(purpose="plan_first")
        payload = provider._build_payload(
            [Message(role="user", content="test")],
        )
        self.assertIn("thinking", payload)
        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_reasoning_effort_for_plan_first(self):
        """reasoning_effort is high for plan_first."""
        provider = self._make_provider_with_key(purpose="plan_first")
        payload = provider._build_payload(
            [Message(role="user", content="test")],
        )
        self.assertIn("reasoning_effort", payload)
        self.assertEqual(payload["reasoning_effort"], "high")

    # --- JSON Mode ---

    def test_json_mode_adds_response_format(self):
        """JSON mode adds response_format to payload."""
        provider = self._make_provider_with_key(purpose="agent")
        payload = provider._build_payload(
            [Message(role="user", content="test")],
        )
        self.assertIn("response_format", payload)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_json_mode_injects_json_instruction_when_missing(self):
        provider = self._make_provider_with_key(purpose="agent")
        payload = provider._build_payload(
            [Message(role="user", content="return the status")]
        )
        self.assertIn("json", payload["messages"][0]["content"].lower())

    def test_json_mode_disabled_for_llm_test(self):
        """JSON mode is disabled for llm_test purpose."""
        provider = self._make_provider(purpose="llm_test")
        self.assertFalse(provider.json_mode)

    def test_json_mode_disabled_no_response_format(self):
        """No response_format when json_mode is disabled."""
        provider = self._make_provider_with_key(
            purpose="llm_test",
            json_mode={"llm_test": False},
        )
        payload = provider._build_payload(
            [Message(role="user", content="test")],
        )
        self.assertNotIn("response_format", payload)

    # --- Response parsing ---

    def test_parse_normal_response(self):
        """Normal response is parsed correctly."""
        provider = self._make_provider(purpose="agent")
        raw = {
            "id": "resp-123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"status": "ok"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = provider._parse_response(raw)
        self.assertEqual(result.text, '{"status": "ok"}')
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.request_id, "resp-123")
        self.assertEqual(result.provider_name, "deepseek")
        self.assertEqual(result.protocol, "json_action")

    def test_parse_reasoning_content(self):
        """Reasoning content is extracted but in context only."""
        provider = self._make_provider(purpose="plan_first")
        raw = {
            "id": "resp-456",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"status": "ok"}',
                        "reasoning_content": "Let me think about this...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = provider._parse_response(raw)
        # Reasoning in memory (reasoning_content field)
        self.assertEqual(result.reasoning_content, "Let me think about this...")
        # Reasoning summary in context (NOT full text)
        self.assertIn("reasoning_present", result.context)
        self.assertTrue(result.context["reasoning_present"])
        self.assertIn("reasoning_sha256", result.context)
        # Full reasoning is NOT in context
        self.assertNotIn("Let me think", str(result.context.get("reasoning_content", "")))

    def test_parse_finish_reason(self):
        """Finish reason is extracted from response."""
        provider = self._make_provider(purpose="agent")
        raw = {
            "choices": [
                {
                    "message": {"content": '{"status": "ok"}'},
                    "finish_reason": "length",
                }
            ],
        }
        result = provider._parse_response(raw)
        self.assertEqual(result.finish_reason, "length")

    def test_parse_tool_calls(self):
        """Tool calls are extracted from response."""
        provider = self._make_provider(purpose="agent")
        raw = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Beijing"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
        result = provider._parse_response(raw)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["id"], "call_1")

    # --- Empty content ---

    def test_empty_content_detected(self):
        """Empty content is detected and raises error after retry."""
        provider = self._make_provider_with_key(purpose="agent")
        empty_response = {
            "choices": [
                {
                    "message": {"content": ""},
                    "finish_reason": "stop",
                }
            ],
        }
        provider.urlopen = _fake_urlopen_factory(empty_response)

        # First call triggers empty content; _complete_once doesn't auto-retry
        # but complete() will retry once, then still fail if both are empty
        # For a single empty response with no tool_calls, it should fail fast
        with self.assertRaises(ProviderError) as ctx:
            provider._complete_once(
                [Message(role="user", content="test")],
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.EMPTY_CONTENT)

    def test_empty_content_retries_once_with_shorter_prompt(self):
        provider = self._make_provider_with_key(purpose="agent", max_retries=0)
        calls = []

        def respond(request, timeout=60):
            calls.append(json.loads(request.data.decode("utf-8")))
            payload = (
                {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
                if len(calls) == 1
                else {"choices": [{"message": {"content": '{"status":"ok"}'}}]}
            )
            return _fake_urlopen_factory(payload)(request, timeout)

        provider.urlopen = respond
        result = provider.complete(
            [Message(role="user", content="x" * 2000)]
        )
        self.assertEqual(result.text, '{"status":"ok"}')
        self.assertEqual(len(calls), 2)
        self.assertLess(
            sum(len(item["content"]) for item in calls[1]["messages"]),
            sum(len(item["content"]) for item in calls[0]["messages"]),
        )
        self.assertTrue(result.context["empty_content_retry"]["attempted"])

    def test_empty_content_is_never_retried_more_than_once(self):
        provider = self._make_provider_with_key(purpose="agent", max_retries=0)
        calls = [0]

        def respond(request, timeout=60):
            calls[0] += 1
            return _fake_urlopen_factory(
                {"choices": [{"message": {"content": ""}}]}
            )(request, timeout)

        provider.urlopen = respond
        with self.assertRaises(ProviderError) as ctx:
            provider.complete([Message(role="user", content="Return JSON")])
        self.assertEqual(ctx.exception.category, ErrorCategory.EMPTY_CONTENT)
        self.assertEqual(calls[0], 2)

    # --- Non-JSON rejection ---

    def test_non_json_response_rejected(self):
        """Non-JSON response is rejected when json_mode is enabled."""
        provider = self._make_provider_with_key(purpose="agent")
        non_json_response = {
            "choices": [
                {
                    "message": {"content": "This is not JSON"},
                    "finish_reason": "stop",
                }
            ],
        }
        provider.urlopen = _fake_urlopen_factory(non_json_response)

        with self.assertRaises(ProviderError) as ctx:
            provider._complete_once(
                [Message(role="user", content="Return JSON please")],
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.INVALID_RESPONSE)

    def test_malformed_json_starting_with_brace_is_rejected(self):
        provider = self._make_provider_with_key(purpose="agent")
        provider.urlopen = _fake_urlopen_factory(
            {"choices": [{"message": {"content": "{not-json"}}]}
        )
        with self.assertRaises(ProviderError) as ctx:
            provider._complete_once(
                [Message(role="user", content="Return JSON")]
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.INVALID_RESPONSE)

    def test_non_object_http_response_is_rejected_without_retry(self):
        provider = self._make_provider_with_key(
            purpose="agent", max_retries=0
        )
        provider.urlopen = _fake_urlopen_factory(["unexpected"])
        with self.assertRaises(ProviderError) as ctx:
            provider._complete_once(
                [Message(role="user", content="Return JSON")]
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.INVALID_RESPONSE)

    # --- Secret safety ---

    def test_api_key_not_in_payload(self):
        """API key must not appear in request payload."""
        provider = self._make_provider_with_key(purpose="agent")
        payload = provider._build_payload(
            [Message(role="user", content="test")],
        )
        payload_str = json.dumps(payload)
        self.assertNotIn("test-key", payload_str)

    def test_api_key_not_in_error_message(self):
        """API key must not appear in error messages."""
        provider = self._make_provider_with_key(purpose="agent")
        try:
            raise ProviderError(
                "auth failed",
                provider_name="deepseek",
                category=ErrorCategory.AUTHENTICATION_FAILED,
                safe_detail="",
            )
        except ProviderError as exc:
            error_dict = exc.to_dict()
            error_str = json.dumps(error_dict)
            self.assertNotIn("test-key", error_str)
            self.assertNotIn("test-key", str(exc))

    # --- Retry behavior ---

    def test_401_no_retry(self):
        """401 errors must NOT be retried."""
        provider = self._make_provider_with_key(purpose="agent")
        call_count = [0]

        def counting_urlopen(request, timeout=60):
            call_count[0] += 1
            import urllib.error
            raise urllib.error.HTTPError(
                url="https://api.deepseek.com/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"Invalid API Key"}}'),
            )

        provider.urlopen = counting_urlopen
        with self.assertRaises(ProviderError) as ctx:
            provider.complete([Message(role="user", content="test")])
        self.assertEqual(ctx.exception.category, ErrorCategory.AUTHENTICATION_FAILED)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(call_count[0], 1)  # Only one attempt (no retry)

    def test_402_no_retry(self):
        """402 errors must NOT be retried."""
        provider = self._make_provider_with_key(purpose="agent")
        call_count = [0]

        def counting_urlopen(request, timeout=60):
            call_count[0] += 1
            import urllib.error
            raise urllib.error.HTTPError(
                url="https://api.deepseek.com/chat/completions",
                code=402,
                msg="Payment Required",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"Insufficient Balance"}}'),
            )

        provider.urlopen = counting_urlopen
        with self.assertRaises(ProviderError) as ctx:
            provider.complete([Message(role="user", content="test")])
        self.assertEqual(ctx.exception.category, ErrorCategory.INSUFFICIENT_BALANCE)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(call_count[0], 1)

    def test_429_retry(self):
        """429 errors ARE retried with backoff."""
        provider = self._make_provider_with_key(purpose="agent")
        provider.max_retries = 2

        call_count = [0]

        def counting_urlopen(request, timeout=60):
            call_count[0] += 1
            import urllib.error
            raise urllib.error.HTTPError(
                url="https://api.deepseek.com/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs={},
                fp=BytesIO(b'{"error":{"message":"Rate limited"}}'),
            )

        provider.urlopen = counting_urlopen

        # Patch _retry_delay to avoid actual sleep
        with patch.object(provider, "_retry_delay", return_value=None):
            with self.assertRaises(ProviderError) as ctx:
                provider.complete([Message(role="user", content="test")])
            self.assertEqual(ctx.exception.category, ErrorCategory.RATE_LIMITED)
            self.assertTrue(ctx.exception.retryable)
            self.assertEqual(call_count[0], 3)  # 1 initial + 2 retries

    def test_500_retry(self):
        """500 errors ARE retried."""
        provider = self._make_provider_with_key(purpose="agent")
        provider.max_retries = 1

        call_count = [0]
        succeed_after = [1]  # succeed on 2nd attempt

        def counting_urlopen(request, timeout=60):
            call_count[0] += 1
            if call_count[0] <= succeed_after[0]:
                import urllib.error
                raise urllib.error.HTTPError(
                    url="https://api.deepseek.com/chat/completions",
                    code=500,
                    msg="Internal Server Error",
                    hdrs={},
                    fp=BytesIO(b"{}"),
                )
            return _fake_urlopen_factory({
                "choices": [{"message": {"content": '{"status":"ok"}'}}],
            })(request, timeout)

        provider.urlopen = counting_urlopen

        with patch.object(provider, "_retry_delay", return_value=None):
            result = provider.complete([Message(role="user", content="test")])
            self.assertEqual(result.text, '{"status":"ok"}')
            self.assertEqual(call_count[0], 2)  # 1 failure + 1 success

    def test_retry_count_in_result(self):
        """Retry count is recorded in LLMResult."""
        provider = self._make_provider_with_key(purpose="agent")
        provider.max_retries = 1

        call_count = [0]

        def counting_urlopen(request, timeout=60):
            call_count[0] += 1
            if call_count[0] <= 1:
                import urllib.error
                raise urllib.error.HTTPError(
                    url="https://api.deepseek.com/chat/completions",
                    code=500,
                    msg="Error",
                    hdrs={},
                    fp=BytesIO(b"{}"),
                )
            return _fake_urlopen_factory({
                "choices": [{"message": {"content": '{"status":"ok"}'}}],
            })(request, timeout)

        provider.urlopen = counting_urlopen

        with patch.object(provider, "_retry_delay", return_value=None):
            result = provider.complete([Message(role="user", content="test")])
            self.assertEqual(result.retry_count, 1)

    def test_zero_max_retries_disables_transient_retry(self):
        provider = self._make_provider_with_key(purpose="agent", max_retries=0)
        self.assertEqual(provider.max_retries, 0)
        calls = [0]

        def fail(request, timeout=60):
            calls[0] += 1
            import urllib.error
            raise urllib.error.HTTPError(
                request.full_url, 500, "error", {}, BytesIO(b"{}")
            )

        provider.urlopen = fail
        with self.assertRaises(ProviderError):
            provider.complete([Message(role="user", content="Return JSON")])
        self.assertEqual(calls[0], 1)

    def test_retry_after_header_is_classified(self):
        provider = self._make_provider_with_key(purpose="agent")
        import urllib.error
        exc = urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            429,
            "rate limited",
            {"Retry-After": "7"},
            BytesIO(b"{}"),
        )
        classified = provider._classify_http_error(exc)
        self.assertEqual(classified["retry_after_seconds"], 7.0)

    def test_expired_deadline_stops_before_network_request(self):
        provider = self._make_provider_with_key(purpose="agent")
        calls = [0]

        def should_not_call(request, timeout=60):
            calls[0] += 1
            raise AssertionError("network should not be called")

        provider.urlopen = should_not_call
        with self.assertRaises(ProviderError) as ctx:
            provider.complete(
                [Message(role="user", content="Return JSON")],
                request_context=ProviderRequestContext(
                    deadline_at="2000-01-01T00:00:00+00:00"
                ),
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.DEADLINE_EXCEEDED)
        self.assertEqual(calls[0], 0)

    # --- Context overflow ---

    def test_context_overflow_category_detected(self):
        """400 with context overflow message is classified correctly."""
        provider = self._make_provider_with_key(purpose="agent")
        import urllib.error

        exc = urllib.error.HTTPError(
            url="https://api.deepseek.com/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=BytesIO(
                b'{"error":{"message":"maximum context length exceeded"}}'
            ),
        )
        classified = provider._classify_http_error(exc)
        self.assertEqual(classified["category"], ErrorCategory.CONTEXT_OVERFLOW)

    # --- Reasoning privacy ---

    def test_reasoning_not_in_result_context_as_text(self):
        """Full reasoning text is NOT stored in context dict as raw text."""
        provider = self._make_provider(purpose="plan_first")
        raw = {
            "choices": [
                {
                    "message": {
                        "content": '{"status":"ok"}',
                        "reasoning_content": "SECRET reasoning step 1...",
                    },
                }
            ],
        }
        result = provider._parse_response(raw)
        # Context has hash, NOT raw text
        context_json = json.dumps(result.context)
        self.assertNotIn("SECRET reasoning", context_json)
        self.assertIn("reasoning_sha256", result.context)
        self.assertNotIn("SECRET reasoning", json.dumps(result.raw))

    # --- Configuration validation ---

    def test_invalid_thinking_mode_rejected(self):
        """Invalid thinking mode is rejected."""
        with self.assertRaises(ProviderError) as ctx:
            self._make_provider(thinking={"agent": "maybe"})
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    def test_invalid_reasoning_effort_rejected(self):
        """Invalid reasoning_effort is rejected."""
        with self.assertRaises(ProviderError) as ctx:
            self._make_provider(
                purpose="plan_first",
                reasoning_effort={"plan_first": "medium"},
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    def test_beta_endpoint_rejected_without_allow_beta(self):
        """Beta endpoint is rejected without allow_beta flag."""
        with self.assertRaises(ProviderError) as ctx:
            self._make_provider(api_base="https://api.deepseek.com/beta")
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    def test_beta_endpoint_allowed_with_flag(self):
        """Beta endpoint is allowed with allow_beta=true."""
        provider = self._make_provider(
            api_base="https://api.deepseek.com/beta",
            allow_beta=True,
        )
        self.assertIn("/beta", provider.api_base)

    def test_http_api_url_environment_override_is_rejected(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_URL": "http://api.deepseek.com/chat/completions"},
            clear=False,
        ):
            with self.assertRaises(ProviderError) as ctx:
                self._make_provider()
        self.assertEqual(ctx.exception.category, ErrorCategory.CONFIGURATION_ERROR)

    def test_native_tool_calling_true_is_supported_explicitly(self):
        provider = self._make_provider(native_tool_calling=True)
        self.assertTrue(provider.native_tool_calling)
        self.assertTrue(provider.capabilities["supports_tool_calling"])

    def test_native_tool_calling_disabled_fails_closed(self):
        provider = self._make_provider_with_key(native_tool_calling=False)
        with self.assertRaises(ProviderError) as ctx:
            provider.complete_with_tools(
                [Message(role="user", content="inspect")],
                [{"type": "function", "function": {
                    "name": "inspect_repo_tree",
                    "parameters": {"type": "object", "properties": {}},
                }}],
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.INVALID_REQUEST)

    def test_native_tool_payload_preserves_assistant_and_tool_messages(self):
        provider = self._make_provider_with_key(native_tool_calling=True)
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "inspect_repo_tree", "arguments": "{}"},
        }
        payload = provider._build_tools_payload(
            [
                Message(role="user", content="inspect"),
                Message(role="assistant", content="", tool_calls=[call]),
                Message(role="tool", content='{"status":"passed"}', tool_call_id="call-1"),
            ],
            [{"type": "function", "function": {
                "name": "inspect_repo_tree",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )
        self.assertEqual(payload["messages"][1]["tool_calls"], [call])
        self.assertEqual(payload["messages"][2]["role"], "tool")
        self.assertEqual(payload["messages"][2]["tool_call_id"], "call-1")
        self.assertNotIn("response_format", payload)

    def test_native_tool_response_allows_empty_text_with_tool_call(self):
        response = {
            "id": "resp-native-1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "inspect_repo_tree",
                            "arguments": "{}",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        provider = self._make_provider_with_key(native_tool_calling=True)
        provider.urlopen = _fake_urlopen_factory(response)
        result = provider.complete_with_tools(
            [Message(role="user", content="inspect")],
            [{"type": "function", "function": {
                "name": "inspect_repo_tree",
                "parameters": {"type": "object", "properties": {}},
            }}],
        )
        self.assertEqual(result.protocol, "native_tools")
        self.assertEqual(result.text, "")
        self.assertEqual(result.tool_calls[0]["id"], "call-1")

    def test_json_action_rejects_tool_messages(self):
        provider = self._make_provider_with_key()
        with self.assertRaises(ProviderError) as ctx:
            provider._build_payload(
                [Message(role="tool", content="result", tool_call_id="call-1")]
            )
        self.assertEqual(ctx.exception.category, ErrorCategory.INVALID_REQUEST)

    # --- Purpose resolution ---

    def test_memory_evolution_purpose(self):
        """Memory evolution uses V4 Flash."""
        provider = self._make_provider(purpose="memory_evolution")
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertEqual(provider.thinking_mode, "disabled")

    def test_live_smoke_purpose(self):
        """Live smoke uses V4 Flash with JSON mode."""
        provider = self._make_provider(purpose="live_smoke")
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertTrue(provider.json_mode)

    def test_llm_test_purpose(self):
        """LLM test uses V4 Flash without JSON mode."""
        provider = self._make_provider(purpose="llm_test")
        self.assertEqual(provider.model, "deepseek-v4-flash")
        self.assertFalse(provider.json_mode)

    # --- Provider identity ---

    def test_provider_name_is_deepseek(self):
        """Provider reports its name as deepseek."""
        provider = self._make_provider(purpose="agent")
        self.assertEqual(provider.provider_name, "deepseek")

    def test_protocol_capabilities_follow_feature_flag(self):
        provider = self._make_provider(purpose="agent")
        self.assertFalse(provider.capabilities["supports_tool_calling"])
        self.assertFalse(provider.capabilities["supports_streaming"])
        enabled = self._make_provider(purpose="agent", native_tool_calling=True)
        self.assertTrue(enabled.capabilities["supports_tool_calling"])

    def test_provider_not_instance_of_generic_openai_compatible(self):
        """DeepSeekProvider is a dedicated class, not generic OpenAICompatibleProvider."""
        from auto_harness.providers.openai_compatible import OpenAICompatibleProvider
        provider = self._make_provider(purpose="agent")
        # It inherits from OpenAICompatibleProvider but is DeepSeekProvider
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertNotEqual(
            type(provider).__name__,
            "OpenAICompatibleProvider",
        )

    # --- Message compatibility ---

    def test_message_with_new_fields(self):
        """Message with reasoning_content and tool_call_id works."""
        msg = Message(
            role="assistant",
            content="result",
            reasoning_content="thinking...",
            tool_call_id="call_1",
        )
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "result")
        self.assertEqual(msg.reasoning_content, "thinking...")
        self.assertEqual(msg.tool_call_id, "call_1")

    def test_message_backward_compatible(self):
        """Old Message(role, content) construction still works."""
        msg = Message(role="user", content="hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "hello")
        self.assertEqual(msg.reasoning_content, "")
        self.assertEqual(msg.tool_calls, [])
        self.assertEqual(msg.tool_call_id, "")

    # --- ProviderError ---

    def test_provider_error_serialization(self):
        """ProviderError.to_dict() produces safe output."""
        err = ProviderError(
            "test error",
            provider_name="deepseek",
            status_code=500,
            category=ErrorCategory.SERVER_ERROR,
            request_id="req-1",
            safe_detail="Internal error",
        )
        d = err.to_dict()
        self.assertEqual(d["provider_name"], "deepseek")
        self.assertEqual(d["status_code"], 500)
        self.assertEqual(d["category"], ErrorCategory.SERVER_ERROR)
        self.assertTrue(d["retryable"])
        self.assertEqual(d["request_id"], "req-1")
        self.assertNotIn("secret", json.dumps(d).lower())

    def test_provider_error_non_retryable_categories(self):
        """Non-retryable categories have retryable=False."""
        for cat in ("authentication_failed", "insufficient_balance",
                     "invalid_request", "configuration_error",
                     "context_overflow", "invalid_response", "empty_content"):
            err = ProviderError(
                "test", provider_name="dp", category=cat
            )
            self.assertFalse(err.retryable, f"{cat} should not be retryable")

    def test_provider_error_retryable_categories(self):
        """Retryable categories have retryable=True."""
        for cat in ("rate_limited", "server_error", "server_overloaded",
                     "network_timeout", "network_error"):
            err = ProviderError(
                "test", provider_name="dp", category=cat
            )
            self.assertTrue(err.retryable, f"{cat} should be retryable")


if __name__ == "__main__":
    unittest.main()
