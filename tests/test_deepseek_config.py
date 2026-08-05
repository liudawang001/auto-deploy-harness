"""Tests for DeepSeek configuration validation.

Validates:
- API base must be https
- /beta endpoint requires allow_beta
- Retired model names rejected (single + per-purpose)
- Thinking/enabled-disabled validation
- Reasoning effort validation
- JSON mode boolean validation
- Retry config validation
- Provider registry creates DeepSeekProvider not OpenAICompatibleProvider
- Other providers unaffected
- Plan-first uses purpose=plan_first
"""
import os
import unittest
from unittest.mock import patch

from auto_harness.config import HarnessConfig, _validate_deepseek_config
from auto_harness.providers import (
    DeepSeekProvider,
    OpenAICompatibleProvider,
    ProviderRegistry,
)


class DeepSeekConfigValidationTests(unittest.TestCase):
    """Tests for DeepSeek-specific configuration validation."""

    # --- API base ---

    def test_http_api_base_rejected(self):
        """Non-HTTPS API base is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "http://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                    }
                }
            )
        self.assertIn("https", str(ctx.exception))

    def test_https_api_base_accepted(self):
        """HTTPS API base is accepted."""
        config = HarnessConfig(
            provider_configs={
                "deepseek": {
                    "api_base": "https://api.deepseek.com",
                    "model": "deepseek-v4-flash",
                }
            }
        )
        self.assertIn("deepseek", config.provider_configs)

    # --- Beta endpoint ---

    def test_beta_endpoint_rejected_without_flag(self):
        """Beta endpoint requires allow_beta=true."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com/beta",
                        "model": "deepseek-v4-flash",
                    }
                }
            )
        self.assertIn("beta", str(ctx.exception).lower())

    def test_beta_endpoint_accepted_with_flag(self):
        """Beta endpoint accepted with allow_beta=true."""
        config = HarnessConfig(
            provider_configs={
                "deepseek": {
                    "api_base": "https://api.deepseek.com/beta",
                    "model": "deepseek-v4-flash",
                    "allow_beta": True,
                }
            }
        )
        self.assertTrue(
            config.provider_configs["deepseek"]["allow_beta"]
        )

    # --- Retired models ---

    def test_retired_model_in_single_config_rejected(self):
        """Retired deepseek-chat in model field is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-chat",
                    }
                }
            )
        self.assertIn("retired", str(ctx.exception).lower())

    def test_retired_model_in_purpose_config_rejected(self):
        """Retired model in purpose-specific models is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "models": {
                            "agent": "deepseek-reasoner",
                        },
                    }
                }
            )
        self.assertIn("retired", str(ctx.exception).lower())

    def test_v4_model_accepted(self):
        """V4 models are accepted in config."""
        config = HarnessConfig(
            provider_configs={
                "deepseek": {
                    "api_base": "https://api.deepseek.com",
                    "models": {
                        "agent": "deepseek-v4-flash",
                        "plan_first": "deepseek-v4-pro",
                    },
                }
            }
        )
        models = config.provider_configs["deepseek"]["models"]
        self.assertEqual(models["agent"], "deepseek-v4-flash")
        self.assertEqual(models["plan_first"], "deepseek-v4-pro")

    # --- Thinking validation ---

    def test_invalid_thinking_value_rejected(self):
        """Invalid thinking value is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "thinking": {"agent": "maybe"},
                    }
                }
            )
        self.assertIn("thinking", str(ctx.exception).lower())

    # --- Reasoning effort ---

    def test_invalid_reasoning_effort_rejected(self):
        """Invalid reasoning_effort is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "reasoning_effort": {"plan_first": "low"},
                    }
                }
            )
        self.assertIn("reasoning_effort", str(ctx.exception).lower())

    # --- JSON mode ---

    def test_json_mode_non_bool_rejected(self):
        """Non-boolean json_mode is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "json_mode": {"agent": "yes"},
                    }
                }
            )
        self.assertIn("json_mode", str(ctx.exception).lower())

    # --- Retry config ---

    def test_negative_max_retries_rejected(self):
        """Negative max_retries is rejected."""
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "max_retries": -1,
                    }
                }
            )
        self.assertIn("max_retries", str(ctx.exception).lower())

    def test_native_tool_calling_true_rejected_until_implemented(self):
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "native_tool_calling": True,
                    }
                }
            )
        self.assertIn("not implemented", str(ctx.exception).lower())

    def test_http_api_url_rejected(self):
        with self.assertRaises(ValueError):
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_url": "http://api.deepseek.com/chat/completions",
                        "model": "deepseek-v4-flash",
                    }
                }
            )

    def test_custom_https_endpoint_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://proxy.example.com/v1",
                        "model": "deepseek-v4-flash",
                    }
                }
            )
        config = HarnessConfig(
            provider_configs={
                "deepseek": {
                    "api_base": "https://proxy.example.com/v1",
                    "model": "deepseek-v4-flash",
                    "allow_custom_endpoint": True,
                }
            }
        )
        self.assertTrue(
            config.provider_configs["deepseek"]["allow_custom_endpoint"]
        )

    def test_provider_timeout_cannot_exceed_agent_deadline(self):
        with self.assertRaises(ValueError) as ctx:
            HarnessConfig(
                agent_decision_timeout_seconds=30,
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "timeout_seconds": 60,
                    }
                },
            )
        self.assertIn("agent_decision_timeout_seconds", str(ctx.exception))

    # --- Non-deepseek providers unaffected ---

    def test_non_deepseek_provider_skips_validation(self):
        """Non-deepseek config is not validated by DeepSeek rules."""
        # Should not raise
        config = HarnessConfig(
            provider_configs={
                "qwen": {
                    "api_base": "http://localhost:8080/v1",
                    "model": "qwen-model",
                }
            }
        )
        self.assertIn("qwen", config.provider_configs)

    def test_validate_deepseek_config_skips_non_deepseek(self):
        """_validate_deepseek_config returns early for non-deepseek providers."""
        # Should not raise
        _validate_deepseek_config("qwen", {"api_base": "http://localhost:8080"})


class ProviderRegistryDeepSeekTests(unittest.TestCase):
    """Registry integration tests for DeepSeekProvider."""

    def test_deepseek_creates_deepseek_provider(self):
        """deepseek name creates DeepSeekProvider, not OpenAICompatibleProvider."""
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key"},
            clear=False,
        ):
            config = HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "context_window_tokens": 1_000_000,
                    }
                }
            )
            provider = ProviderRegistry().create(
                "deepseek",
                config=config,
                purpose="agent",
            )
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertNotEqual(
            type(provider).__name__,
            "OpenAICompatibleProvider",
        )

    def test_plan_first_uses_purpose_plan_first(self):
        """Provider registry passes purpose=plan_first correctly."""
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key"},
            clear=False,
        ):
            config = HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "context_window_tokens": 1_000_000,
                    }
                }
            )
            provider = ProviderRegistry().create(
                "deepseek",
                config=config,
                purpose="plan_first",
            )
        self.assertEqual(provider.purpose, "plan_first")

    def test_other_providers_still_openai_compatible(self):
        """Other vendors still use OpenAICompatibleProvider."""
        config = HarnessConfig(
            provider_configs={
                "qwen": {
                    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-model",
                }
            }
        )
        with patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "test-key"},
            clear=False,
        ):
            provider = ProviderRegistry().create(
                "qwen",
                config=config,
                purpose="agent",
            )
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertNotIsInstance(provider, DeepSeekProvider)

    def test_memory_evolution_provider_name(self):
        """memory_evolution purpose creates correct provider."""
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key"},
            clear=False,
        ):
            config = HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "context_window_tokens": 1_000_000,
                    }
                }
            )
            provider = ProviderRegistry().create(
                "deepseek",
                config=config,
                purpose="memory_evolution",
            )
        self.assertEqual(provider.purpose, "memory_evolution")


if __name__ == "__main__":
    unittest.main()
