"""Tests for DeepSeek context and capability integration.

Validates:
- resolve_provider_capabilities for DeepSeek models
- NormalizedUsage includes cache fields
- Context governance with ProviderError
- is_context_overflow_error recognizes ProviderError category
"""
import unittest

from auto_harness.config import HarnessConfig
from auto_harness.context.capabilities import resolve_provider_capabilities
from auto_harness.context.executor import is_context_overflow_error
from auto_harness.context.tokens import normalize_usage
from auto_harness.providers.errors import (
    ErrorCategory,
    ProviderError,
    context_overflow_error,
)


class CapabilityResolutionTests(unittest.TestCase):
    """Tests for DeepSeek capability resolution."""

    def test_v4_flash_capabilities_from_registry(self):
        """V4 Flash capabilities come from the known registry."""
        provider = _FakeDeepSeekProvider(
            provider_name="deepseek",
            model="deepseek-v4-flash",
            context_window_tokens=1_000_000,
            max_tokens=4096,
        )
        caps = resolve_provider_capabilities(provider)
        self.assertEqual(caps.provider_name, "deepseek")
        self.assertEqual(caps.model, "deepseek-v4-flash")
        self.assertEqual(caps.source, "deepseek_model_registry")
        self.assertFalse(caps.supports_tool_calling)

    def test_v4_pro_capabilities_from_registry(self):
        """V4 Pro capabilities come from the known registry."""
        provider = _FakeDeepSeekProvider(
            provider_name="deepseek",
            model="deepseek-v4-pro",
            context_window_tokens=1_000_000,
            max_tokens=4096,
        )
        caps = resolve_provider_capabilities(provider)
        self.assertEqual(caps.model, "deepseek-v4-pro")
        self.assertEqual(caps.source, "deepseek_model_registry")

    def test_project_budget_caps_provider_capability(self):
        """Project budget caps provider's 1M window."""
        provider = _FakeDeepSeekProvider(
            provider_name="deepseek",
            model="deepseek-v4-flash",
            context_window_tokens=1_000_000,
            max_tokens=4096,
        )
        config = HarnessConfig(agent_context_window_tokens=65536)
        caps = resolve_provider_capabilities(provider, config)
        self.assertEqual(caps.context_window_tokens, 65536)
        self.assertEqual(caps.source, "explicit_config")

    def test_no_auto_raise_to_1m(self):
        """Project budget caps DeepSeek's 1M window."""
        provider = _FakeDeepSeekProvider(
            provider_name="deepseek",
            model="deepseek-v4-flash",
            context_window_tokens=1_000_000,
            max_tokens=4096,
        )
        # With explicit project budget of 64K, caps to 64K
        config = HarnessConfig(agent_context_window_tokens=65536)
        caps = resolve_provider_capabilities(provider, config)
        # Should be min(provider=1M, configured=64K) = 64K
        self.assertEqual(caps.context_window_tokens, 65536)
        self.assertEqual(caps.source, "explicit_config")

    def test_provider_level_context_cap_is_not_raised_to_model_maximum(self):
        provider = _FakeDeepSeekProvider(
            provider_name="deepseek",
            model="deepseek-v4-flash",
            context_window_tokens=65536,
            max_tokens=4096,
        )
        caps = resolve_provider_capabilities(provider)
        self.assertEqual(caps.context_window_tokens, 65536)
        self.assertEqual(caps.source, "provider_config")


class NormalizedUsageTests(unittest.TestCase):
    """Tests for usage normalization including DeepSeek cache fields."""

    def test_cache_hit_tokens_extracted(self):
        """DeepSeek cache_hit_tokens are extracted."""
        usage = normalize_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 60,
                "prompt_cache_miss_tokens": 40,
            },
            estimated_input_tokens=120,
        )
        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.cache_hit_tokens, 60)
        self.assertEqual(usage.cache_miss_tokens, 40)
        self.assertEqual(usage.source, "provider_reported")

    def test_cache_fields_none_when_missing(self):
        """Cache fields are None when not in usage payload."""
        usage = normalize_usage(
            {"prompt_tokens": 100, "completion_tokens": 50},
            estimated_input_tokens=120,
        )
        self.assertIsNone(usage.cache_hit_tokens)
        self.assertIsNone(usage.cache_miss_tokens)

    def test_estimated_source_when_no_provider_usage(self):
        """Source is 'estimated' when provider doesn't report usage."""
        usage = normalize_usage(None, estimated_input_tokens=200)
        self.assertEqual(usage.source, "estimated")
        self.assertEqual(usage.input_tokens, 200)


class ContextOverflowErrorTests(unittest.TestCase):
    """Tests for context overflow detection with ProviderError."""

    def test_provider_error_context_overflow_detected(self):
        """ProviderError with context_overflow category is detected."""
        err = context_overflow_error("deepseek", detail="too many tokens")
        self.assertTrue(is_context_overflow_error(err))

    def test_provider_error_auth_not_context_overflow(self):
        """ProviderError with auth category is NOT context overflow."""
        from auto_harness.providers.errors import authentication_error
        err = authentication_error("deepseek")
        self.assertFalse(is_context_overflow_error(err))

    def test_generic_exception_with_context_keywords(self):
        """Generic exceptions with context keywords still detected."""
        err = RuntimeError("context length exceeded: 100000 > 65536")
        self.assertTrue(is_context_overflow_error(err))

    def test_generic_exception_without_context_keywords(self):
        """Generic exceptions without context keywords not detected."""
        err = RuntimeError("network timeout")
        self.assertFalse(is_context_overflow_error(err))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _FakeDeepSeekProvider:
    """Minimal fake for capability resolution tests."""
    def __init__(self, provider_name, model, context_window_tokens, max_tokens):
        self.provider_name = provider_name
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.max_tokens = max_tokens


if __name__ == "__main__":
    unittest.main()
