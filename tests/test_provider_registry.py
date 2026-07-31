import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_harness.cli import build_parser
from auto_harness.config import HarnessConfig
from auto_harness.orchestrator import TaskRunner
from auto_harness.providers import (
    InteractiveProviderConfigurator,
    LLMResult,
    MemoryEvolutionMockProvider,
    Message,
    MockLLMProvider,
    OpenAICompatibleProvider,
    ProviderRegistry,
    provider_names,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _CustomProvider:
    provider_name = "custom"
    context_window_tokens = 8192
    max_tokens = 1024

    def complete(self, messages, temperature=0.2, max_output_tokens=None):
        return LLMResult(text='{"status":"ok"}')


class ProviderRegistryTests(unittest.TestCase):
    def test_builtin_registry_exposes_vendor_aliases(self):
        names = set(provider_names())
        self.assertTrue(
            {
                "mock",
                "xunfei",
                "openai_compatible",
                "openai",
                "deepseek",
                "qwen",
                "vllm",
                "ollama",
            }.issubset(names)
        )

    def test_registry_selects_purpose_specific_mock(self):
        registry = ProviderRegistry()
        self.assertIsInstance(
            registry.create("mock", purpose="agent"),
            MockLLMProvider,
        )
        self.assertIsInstance(
            registry.create("mock", purpose="memory_evolution"),
            MemoryEvolutionMockProvider,
        )

    def test_registry_supports_custom_provider_without_business_changes(self):
        registry = ProviderRegistry(include_builtins=False)
        calls = []

        def factory(config, purpose, provider_name):
            calls.append((config, purpose, provider_name))
            return _CustomProvider()

        registry.register("custom", factory)
        provider = registry.create(
            "custom",
            config={"flag": True},
            purpose="plan_first",
        )
        self.assertIsInstance(provider, _CustomProvider)
        self.assertEqual(
            calls,
            [({"flag": True}, "plan_first", "custom")],
        )
        with self.assertRaises(ValueError):
            registry.register("custom", factory)

    def test_unknown_provider_fails_with_available_names(self):
        with self.assertRaisesRegex(ValueError, "available providers"):
            ProviderRegistry().create("not_registered")

    def test_registry_accepts_arbitrary_configured_vendor_name(self):
        config = HarnessConfig(
            provider_configs={
                "my-vendor": {
                    "api_base": "https://vendor.invalid/v1",
                    "model": "vendor-model",
                    "api_key_env": "MY_VENDOR_API_KEY",
                    "context_window_tokens": 16384,
                }
            }
        )
        with patch.dict(
            os.environ,
            {"MY_VENDOR_API_KEY": "session-secret"},
            clear=False,
        ):
            provider = ProviderRegistry().create(
                "my-vendor",
                config=config,
            )
        self.assertIsInstance(provider, OpenAICompatibleProvider)
        self.assertEqual(provider.provider_name, "my_vendor")
        self.assertEqual(provider.model, "vendor-model")

    def test_openai_compatible_request_and_response(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _FakeResponse(
                {
                    "choices": [
                        {"message": {"content": '{"status":"ok"}'}}
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                    },
                }
            )

        config = HarnessConfig(
            provider_configs={
                "deepseek": {
                    "api_base": "https://provider.invalid/v1",
                    "model": "test-model",
                    "api_key_env": "TEST_DEEPSEEK_KEY",
                    "context_window_tokens": 32768,
                    "max_tokens": 2048,
                    "timeout_seconds": 15,
                }
            }
        )
        with patch.dict(
            os.environ,
            {"TEST_DEEPSEEK_KEY": "secret-value"},
            clear=False,
        ):
            provider = OpenAICompatibleProvider(
                "deepseek",
                config=config,
                urlopen=fake_urlopen,
            )
            result = provider.complete(
                [
                    Message(role="system", content="guard"),
                    Message(role="user", content="return json"),
                ],
                max_output_tokens=512,
            )

        self.assertEqual(result.text, '{"status":"ok"}')
        self.assertEqual(
            captured["url"],
            "https://provider.invalid/v1/chat/completions",
        )
        self.assertEqual(captured["body"]["model"], "test-model")
        self.assertEqual(captured["body"]["max_tokens"], 512)
        self.assertEqual(captured["timeout"], 15)
        self.assertEqual(
            captured["headers"]["Authorization"],
            "Bearer secret-value",
        )
        self.assertNotIn("secret-value", json.dumps(captured["body"]))
        self.assertEqual(result.usage["total_tokens"], 15)

    def test_openai_compatible_supports_keyless_local_endpoint(self):
        config = HarnessConfig(
            provider_configs={
                "vllm": {
                    "api_base": "http://127.0.0.1:8000/v1",
                    "model": "local-model",
                    "require_api_key": False,
                    "context_window_tokens": 8192,
                }
            }
        )
        provider = OpenAICompatibleProvider("vllm", config=config)
        self.assertEqual(provider.missing_configuration(), [])

    def test_config_rejects_plaintext_provider_secret(self):
        with self.assertRaisesRegex(ValueError, "must not contain secret"):
            HarnessConfig(
                provider_configs={
                    "openai": {
                        "api_key": "must-not-be-in-config",
                    }
                }
            )

    def test_cli_provider_choices_come_from_registry(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "deploy",
                "--repo",
                ".",
                "--agent-provider",
                "deepseek",
                "--agent-plan-first-provider",
                "qwen",
            ]
        )
        self.assertEqual(args.agent_provider, "deepseek")
        self.assertEqual(args.agent_plan_first_provider, "qwen")

    def test_cli_accepts_arbitrary_provider_name(self):
        args = build_parser().parse_args(
            [
                "llm-test",
                "--provider",
                "my_vendor",
            ]
        )
        self.assertEqual(args.provider, "my_vendor")

    def test_interactive_provider_keeps_secret_in_memory_only(self):
        answers = iter(
            [
                "my-vendor",
                "https://vendor.invalid/v1",
                "vendor-model",
                "16384",
                "2048",
            ]
        )
        registry = ProviderRegistry()
        configurator = InteractiveProviderConfigurator(
            registry,
            input_fn=lambda prompt: next(answers),
            secret_fn=lambda prompt: "interactive-secret",
        )
        config = HarnessConfig()
        environment_before = dict(os.environ)
        session = configurator.configure(config=config)
        provider = registry.create(
            "my_vendor",
            config=config,
            purpose="agent",
        )

        self.assertEqual(session.provider_name, "my_vendor")
        self.assertEqual(provider.api_key, "interactive-secret")
        self.assertNotIn(
            "interactive-secret",
            json.dumps(session.safe_summary()),
        )
        self.assertNotIn(
            "interactive-secret",
            json.dumps(config.provider_configs),
        )
        self.assertEqual(dict(os.environ), environment_before)

    def test_interactive_provider_allows_keyless_local_endpoint(self):
        answers = iter(
            [
                "local_llm",
                "http://127.0.0.1:8000/v1",
                "local-model",
                "8192",
                "1024",
            ]
        )
        registry = ProviderRegistry()
        with patch.dict(
            os.environ,
            {"AUTO_HARNESS_LLM_API_KEY": "must-not-be-used"},
            clear=False,
        ):
            session = InteractiveProviderConfigurator(
                registry,
                input_fn=lambda prompt: next(answers),
                secret_fn=lambda prompt: "",
            ).configure(config=HarnessConfig())
            provider = registry.create("local_llm")

        self.assertFalse(session.requires_api_key)
        self.assertEqual(provider.api_key, "")
        self.assertEqual(provider.missing_configuration(), [])

    def test_task_runner_uses_injected_registry(self):
        registry = ProviderRegistry(include_builtins=False)
        registry.register(
            "custom",
            lambda config, purpose, provider_name: _CustomProvider(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "models"),
                agent_provider="custom",
                agent_plan_first_provider="custom",
            )
            runner = TaskRunner(config, provider_registry=registry)
            self.assertIsInstance(runner._agent_provider(), _CustomProvider)
            self.assertIsInstance(
                runner._create_plan_first_provider(),
                _CustomProvider,
            )


if __name__ == "__main__":
    unittest.main()
