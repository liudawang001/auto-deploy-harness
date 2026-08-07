"""Tests for LLM runtime settings, configuration priority, and CLI behavior.

These tests verify:
- Fresh defaults are DeepSeek V4 Pro
- CLI > env > config priority
- Runtime overrides are not persisted to JSON
- Mixed provider + uniform override rejection
- Flash model only affects current config
- Context and output synced to Governance
- Queue snapshot does not save API keys
- Missing key fails before deploy()
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from unittest.mock import MagicMock, patch

from auto_harness.config import HarnessConfig, validate_runtime_overrides
from auto_harness.providers.deepseek import DeepSeekProvider
from auto_harness.providers.errors import ProviderError
from auto_harness.providers.settings import (
    get_runtime_overrides,
    has_explicit_model_override,
    normalize_provider_name,
    set_runtime_overrides,
    validate_runtime_overrides_payload,
)


# ---------------------------------------------------------------------------
# Fresh default tests
# ---------------------------------------------------------------------------

class TestFreshDefaults:
    """Verify HarnessConfig() produces DeepSeek V4 Pro defaults."""

    def test_default_provider_is_deepseek(self):
        config = HarnessConfig()
        assert config.agent_provider == "deepseek"

    def test_default_plan_first_provider_is_deepseek(self):
        config = HarnessConfig()
        assert config.agent_plan_first_provider == "deepseek"

    def test_default_memory_evolution_provider_is_deepseek(self):
        config = HarnessConfig()
        assert config.memory_evolution_provider == "deepseek"

    def test_default_context_window_tokens(self):
        config = HarnessConfig()
        assert config.agent_context_window_tokens == 262144

    def test_default_reserved_output_tokens(self):
        config = HarnessConfig()
        assert config.agent_context_reserved_output_tokens == 16384

    def test_llm_runtime_overrides_is_empty_dict(self):
        config = HarnessConfig()
        assert config.llm_runtime_overrides == {}
        assert isinstance(config.llm_runtime_overrides, dict)

    def test_default_provider_configs_contains_deepseek(self):
        config = HarnessConfig()
        deepseek = config.provider_configs["deepseek"]
        assert deepseek["api_base"] == "https://api.deepseek.com"
        assert deepseek["api_key_env"] == "DEEPSEEK_API_KEY"
        assert deepseek["model"] == "deepseek-v4-pro"
        assert deepseek["context_window_tokens"] == 262144
        assert deepseek["max_tokens"] == 16384
        assert "api_key" not in deepseek


# ---------------------------------------------------------------------------
# Runtime override tests
# ---------------------------------------------------------------------------

class TestRuntimeOverrides:
    """Verify the non-persistent runtime override mechanism."""

    def test_set_and_get_runtime_overrides(self):
        config = HarnessConfig()
        set_runtime_overrides(
            config,
            ["deepseek"],
            model="deepseek-v4-flash",
        )
        overrides = get_runtime_overrides(config, "deepseek")
        assert overrides["model"] == "deepseek-v4-flash"

    def test_runtime_overrides_not_in_json_roundtrip(self):
        config = HarnessConfig()
        set_runtime_overrides(
            config,
            ["deepseek"],
            model="deepseek-v4-flash",
            context_window_tokens=524288,
            max_output_tokens=32768,
        )
        # ``init=False`` prevents HarnessConfig.load() from accepting this
        # process-local field from JSON. The dedicated load test below verifies
        # an injected JSON value is ignored.
        field_info = HarnessConfig.__dataclass_fields__["llm_runtime_overrides"]
        assert field_info.init is False

    def test_validate_runtime_overrides_rejects_secret_keys(self):
        import pytest
        with pytest.raises(ValueError, match="secret"):
            validate_runtime_overrides({"api_key": "sk-secret"})
        with pytest.raises(ValueError, match="secret"):
            validate_runtime_overrides({"token": "abc"})
        with pytest.raises(ValueError, match="secret"):
            validate_runtime_overrides({"password": "pwd"})

    def test_validate_runtime_overrides_rejects_unknown_keys(self):
        import pytest
        with pytest.raises(ValueError, match="unknown"):
            validate_runtime_overrides({"unknown_key": "value"})

    def test_validate_runtime_overrides_payload_function(self):
        import pytest
        validate_runtime_overrides_payload({"model": "deepseek-v4-flash"})  # ok
        with pytest.raises(ValueError):
            validate_runtime_overrides_payload({"api_key": "sk-xxx"})

    def test_set_runtime_overrides_validates_positive_int(self):
        import pytest
        config = HarnessConfig()
        with pytest.raises(ValueError, match="positive integer"):
            set_runtime_overrides(config, ["deepseek"], context_window_tokens=-1)
        with pytest.raises(ValueError, match="positive integer"):
            set_runtime_overrides(config, ["deepseek"], max_output_tokens=0)


# ---------------------------------------------------------------------------
# Priority tests (CLI > env > config)
# ---------------------------------------------------------------------------

class TestPriority:
    """Verify config resolution priority: CLI > env > config."""

    def test_has_explicit_model_override_detects_runtime(self):
        config = HarnessConfig()
        set_runtime_overrides(config, ["deepseek"], model="deepseek-v4-flash")
        assert has_explicit_model_override(config, "deepseek") is True

    def test_has_explicit_model_override_detects_env(self):
        config = HarnessConfig()
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "deepseek-v4-flash"}):
            assert has_explicit_model_override(config, "deepseek") is True

    def test_has_explicit_model_override_detects_generic_env(self):
        config = HarnessConfig()
        with patch.dict(os.environ, {"AUTO_HARNESS_LLM_MODEL": "some-model"}):
            assert has_explicit_model_override(config, "deepseek") is True

    def test_has_explicit_model_override_false_without_any(self):
        config = HarnessConfig()
        assert has_explicit_model_override(config, "deepseek") is False


# ---------------------------------------------------------------------------
# Config load tests
# ---------------------------------------------------------------------------

class TestConfigLoad:
    """Verify HarnessConfig.load() behaviour."""

    def test_load_does_not_accept_runtime_overrides_from_json(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "test_config.json"
            config_path.write_text(json.dumps({
                "agent_provider": "mock",
                "llm_runtime_overrides": {
                    "deepseek": {"model": "injected"},
                },
            }))
            config = HarnessConfig.load(str(config_path))
            # llm_runtime_overrides must stay empty (init=False field)
            assert config.llm_runtime_overrides == {}

    def test_load_respects_provider_configs_from_json(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "test_config.json"
            config_path.write_text(json.dumps({
                "agent_provider": "openai",
                "provider_configs": {
                    "openai": {
                        "api_base": "https://api.openai.com",
                        "api_key_env": "OPENAI_API_KEY",
                        "model": "gpt-4",
                        "context_window_tokens": 128000,
                        "max_tokens": 4096,
                    }
                },
            }))
            config = HarnessConfig.load(str(config_path))
            assert config.agent_provider == "openai"
            assert "openai" in config.provider_configs


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Verify runtime override whitelist validation."""

    def test_validate_runtime_overrides_accepts_valid_keys(self):
        validate_runtime_overrides({"model": "v4-flash"})
        validate_runtime_overrides({"context_window_tokens": 262144})
        validate_runtime_overrides({"max_output_tokens": 16384})
        validate_runtime_overrides({
            "model": "v4-flash",
            "context_window_tokens": 100000,
            "max_output_tokens": 32000,
        })

    def test_validate_runtime_overrides_not_dict(self):
        import pytest
        with pytest.raises(ValueError, match="object"):
            validate_runtime_overrides([])

    def test_provider_name_normalization(self):
        assert normalize_provider_name("deepseek") == "deepseek"
        assert normalize_provider_name("DeepSeek") == "deepseek"
        assert normalize_provider_name("DEEPSEEK") == "deepseek"
        assert normalize_provider_name("deep-seek") == "deep_seek"


# ---------------------------------------------------------------------------
# Provider settings tests
# ---------------------------------------------------------------------------

class TestProviderSettings:
    """Verify the settings module provides correct interfaces."""

    def test_get_runtime_overrides_empty_when_none_set(self):
        config = HarnessConfig()
        assert get_runtime_overrides(config, "deepseek") == {}

    def test_get_runtime_overrides_only_returns_allowed_keys(self):
        config = HarnessConfig()
        # Manually inject forbidden key into overrides dict
        config.llm_runtime_overrides["deepseek"] = {
            "model": "v4-flash",
            "api_key": "should-not-appear",
        }
        result = get_runtime_overrides(config, "deepseek")
        assert "model" in result
        assert "api_key" not in result

    def test_set_runtime_overrides_multiple_providers(self):
        config = HarnessConfig()
        set_runtime_overrides(
            config,
            ["deepseek", "openai"],
            model="test-model",
        )
        assert get_runtime_overrides(config, "deepseek")["model"] == "test-model"
        assert get_runtime_overrides(config, "openai")["model"] == "test-model"


# ---------------------------------------------------------------------------
# Queue snapshot tests
# ---------------------------------------------------------------------------

class TestQueueSnapshot:
    """Verify queue persists only non-sensitive LLM snapshots."""

    def test_sanitize_rejects_secret_keys(self):
        from auto_harness.queue import _sanitize_llm_snapshot
        import pytest
        with pytest.raises(ValueError, match="secret"):
            _sanitize_llm_snapshot({"api_key": "sk-secret"})

    def test_sanitize_accepts_allowed_keys(self):
        from auto_harness.queue import _sanitize_llm_snapshot
        snapshot = _sanitize_llm_snapshot({
            "agent_provider": "deepseek",
            "model": "deepseek-v4-flash",
            "context_window_tokens": 262144,
            "max_output_tokens": 16384,
        })
        assert snapshot is not None
        assert snapshot["model"] == "deepseek-v4-flash"
        assert "agent_provider" in snapshot

    def test_sanitize_returns_none_for_empty(self):
        from auto_harness.queue import _sanitize_llm_snapshot
        assert _sanitize_llm_snapshot(None) is None
        assert _sanitize_llm_snapshot({}) is None

    def test_apply_restores_providers_overrides_and_governance_budgets(self):
        from auto_harness.queue import _apply_queue_llm_snapshot

        config = HarnessConfig(
            agent_provider="mock",
            agent_plan_first_provider="mock",
            agent_context_window_tokens=8192,
            agent_context_reserved_output_tokens=1024,
        )
        _apply_queue_llm_snapshot(config, {
            "agent_provider": "deepseek",
            "plan_first_provider": "deepseek",
            "model": "deepseek-v4-flash",
            "context_window_tokens": 524288,
            "max_output_tokens": 32768,
        })

        assert config.agent_provider == "deepseek"
        assert config.agent_plan_first_provider == "deepseek"
        overrides = get_runtime_overrides(config, "deepseek")
        assert overrides == {
            "model": "deepseek-v4-flash",
            "context_window_tokens": 524288,
            "max_output_tokens": 32768,
        }
        assert config.agent_context_window_tokens == 524288
        assert config.agent_context_reserved_output_tokens == 32768

    def test_provider_precheck_always_checks_agent_and_plan_first(self):
        from auto_harness.queue import _validate_job_providers

        config = HarnessConfig(
            agent_provider="deepseek",
            agent_plan_first=False,
            agent_plan_first_provider="deepseek",
        )
        registry = MagicMock()
        provider = MagicMock()
        provider.missing_configuration.return_value = []
        registry.create.return_value = provider
        runner = MagicMock(provider_registry=registry)

        _validate_job_providers(runner, config, {})

        purposes = [call.kwargs["purpose"] for call in registry.create.call_args_list]
        assert purposes == ["agent", "plan_first"]


# ---------------------------------------------------------------------------
# DeepSeek token limit tests
# ---------------------------------------------------------------------------

class TestDeepSeekTokenLimits:
    """Reject impossible budgets before the first API request."""

    @staticmethod
    def _provider(context_window_tokens, max_tokens):
        return DeepSeekProvider(
            config={
                "provider_configs": {
                    "deepseek": {
                        "api_base": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "require_api_key": False,
                        "model": "deepseek-v4-pro",
                        "context_window_tokens": context_window_tokens,
                        "max_tokens": max_tokens,
                    }
                }
            },
            purpose="agent",
        )

    def test_output_must_be_smaller_than_context_window(self):
        import pytest

        with pytest.raises(ProviderError, match="smaller than"):
            self._provider(16384, 16384)

    def test_v4_context_window_cap_is_enforced(self):
        import pytest

        with pytest.raises(ProviderError, match="1000000"):
            self._provider(1_000_001, 16384)

    def test_v4_output_cap_is_enforced(self):
        import pytest

        with pytest.raises(ProviderError, match="384000"):
            self._provider(1_000_000, 384_001)


# ---------------------------------------------------------------------------
# Memory evolution: only --propose creates Provider
# ---------------------------------------------------------------------------

class TestMemoryEvolutionProviderCreation:
    """Verify only --propose requires API Key validation."""

    def test_memory_evolve_no_propose_no_provider_check(self):
        """_providers_for_command returns empty list for memory-evolve without --propose."""
        config = HarnessConfig()
        # Simulate args for memory-evolve without --propose
        args = mock.Mock()
        args.command = "memory-evolve"
        args.propose = False
        args.provider = None

        from auto_harness.cli import _providers_for_command
        providers = _providers_for_command(config, args)
        assert providers == []

    def test_memory_evolve_propose_returns_provider(self):
        """_providers_for_command returns provider for memory-evolve --propose."""
        config = HarnessConfig()
        args = mock.Mock()
        args.command = "memory-evolve"
        args.propose = True
        args.provider = None

        from auto_harness.cli import _providers_for_command
        providers = _providers_for_command(config, args)
        assert len(providers) == 1
        assert providers[0][0] == "deepseek"
        assert providers[0][1] == "memory_evolution"


# ---------------------------------------------------------------------------
# Context Governance sync tests
# ---------------------------------------------------------------------------

class TestContextGovernanceSync:
    """Verify Context Governance is synced with Provider config."""

    def test_agent_context_window_default_matches_provider_deepseek(self):
        config = HarnessConfig.load()
        deepseek_cfg = config.provider_configs.get("deepseek", {})
        provider_ctx = deepseek_cfg.get("context_window_tokens")
        assert config.agent_context_window_tokens == provider_ctx == 262144

    def test_agent_reserved_output_default_matches_provider_deepseek(self):
        config = HarnessConfig.load()
        deepseek_cfg = config.provider_configs.get("deepseek", {})
        provider_max = deepseek_cfg.get("max_tokens")
        assert config.agent_context_reserved_output_tokens == provider_max == 16384

    def test_cli_syncs_two_deploy_purposes_using_same_provider(self):
        from auto_harness.cli import main

        config = HarnessConfig(
            agent_provider="deepseek",
            agent_plan_first_provider="deepseek",
        )
        provider = MagicMock(
            context_window_tokens=524288,
            max_tokens=32768,
        )
        provider.missing_configuration.return_value = []
        runner = MagicMock()
        runner.deploy.return_value = "task_123"

        with patch("auto_harness.cli.HarnessConfig.load", return_value=config), patch(
            "auto_harness.cli.DEFAULT_PROVIDER_REGISTRY.create",
            return_value=provider,
        ), patch("auto_harness.cli.TaskRunner", return_value=runner):
            exit_code = main([
                "deploy",
                "--repo",
                "https://example.com/repo.git",
                "--context-window-tokens",
                "524288",
                "--max-output-tokens",
                "32768",
            ])

        assert exit_code == 0
        assert config.agent_context_window_tokens == 524288
        assert config.agent_context_reserved_output_tokens == 32768


# ---------------------------------------------------------------------------
# API key safety tests
# ---------------------------------------------------------------------------

class TestApiKeySafety:
    """Verify API keys are never in config, overrides, or logs."""

    def test_default_config_has_no_api_key(self):
        config = HarnessConfig.load()
        deepseek_cfg = config.provider_configs.get("deepseek", {})
        assert "api_key" not in deepseek_cfg
        assert deepseek_cfg.get("api_key_env") == "DEEPSEEK_API_KEY"

    def test_runtime_overrides_reject_api_key(self):
        import pytest
        config = HarnessConfig()
        with pytest.raises(ValueError):
            validate_runtime_overrides({"api_key": "sk-xxx"})

    def test_config_validation_rejects_api_key_in_provider_configs(self):
        import pytest
        with pytest.raises(ValueError):
            HarnessConfig(
                provider_configs={
                    "deepseek": {
                        "api_key": "sk-should-not-be-here",
                        "api_base": "https://api.deepseek.com",
                    }
                }
            )
