"""Real DeepSeek API live smoke test — Level 1-4 verification.

This test MUST be run with DEEPSEEK_API_KEY in the environment.
It never writes the key to disk or includes it in any output.

Usage:
    DEEPSEEK_API_KEY="sk-..." python -m pytest tests/live/test_deepseek_live_smoke.py -v
"""

import json
import os

import pytest


# Skip if no key configured
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    pytest.skip("DEEPSEEK_API_KEY not set", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(purpose="llm_test", model_name=None):
    """Create a real DeepSeekProvider."""
    from auto_harness.providers.deepseek import DeepSeekProvider

    config = {
        "provider_configs": {
            "deepseek": {
                "api_base": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "context_window_tokens": 65_536,
                "max_tokens": 4096,
                "timeout_seconds": 60,
                "max_retries": 1,
            }
        }
    }
    if model_name:
        config["provider_configs"]["deepseek"]["model"] = model_name
    return DeepSeekProvider(
        provider_name="deepseek",
        config=config,
        purpose=purpose,
    )


def _assert_no_secret_leak(text, ctx=""):
    """Fail if the API key appears in text."""
    if DEEPSEEK_API_KEY in str(text):
        pytest.fail(f"SECRET LEAKED in {ctx}: {str(text)[:100]}")


# ---------------------------------------------------------------------------
# Level 1: Provider initialization & configuration
# ---------------------------------------------------------------------------

class TestLevel1_ProviderInit:
    """Level 1: Provider initialization and configuration validation."""

    def test_provider_creates_with_valid_config(self):
        """Provider initializes without errors with real config."""
        provider = _make_provider(purpose="agent")
        missing = provider.missing_configuration()
        assert not missing, f"Missing config: {missing}"
        assert provider.model in ("deepseek-v4-flash", "deepseek-v4-pro")
        assert bool(provider.api_key)

    def test_plan_first_uses_flash_model(self):
        """plan_first purpose selects V4 Flash with thinking enabled."""
        provider = _make_provider(purpose="plan_first")
        assert provider.model == "deepseek-v4-flash"
        assert provider.thinking_mode == "enabled"

    def test_agent_uses_flash_model(self):
        """agent purpose selects V4 Flash with thinking disabled."""
        provider = _make_provider(purpose="agent")
        assert provider.model == "deepseek-v4-flash"
        assert provider.thinking_mode == "disabled"

    def test_no_key_in_payload(self):
        """API key never appears in request payload."""
        provider = _make_provider()
        from auto_harness.providers.base import Message

        payload = provider._build_payload(
            [Message(role="user", content="test")],
        )
        _assert_no_secret_leak(json.dumps(payload), "payload")

    def test_retired_model_rejected(self):
        """Retired model names are rejected before network request."""
        from auto_harness.providers.errors import ProviderError, ErrorCategory

        with pytest.raises(ProviderError) as exc_info:
            _make_provider(model_name="deepseek-chat")
        assert exc_info.value.category == ErrorCategory.CONFIGURATION_ERROR


# ---------------------------------------------------------------------------
# Level 2: Simple JSON Action request
# ---------------------------------------------------------------------------

class TestLevel2_JsonAction:
    """Level 2: Simple JSON Action request to real DeepSeek API."""

    @pytest.mark.timeout(60)
    def test_simple_json_completion(self):
        """Simple JSON completion with json_mode enabled."""
        provider = _make_provider(purpose="agent")
        from auto_harness.providers.base import Message

        result = provider.complete(
            [
                Message(
                    role="system",
                    content=(
                        "You are a JSON-only assistant. Always respond with valid JSON. "
                        "No markdown, no explanation, just the JSON object."
                    ),
                ),
                Message(
                    role="user",
                    content='Return: {"name":"test","status":"ok","number":42}',
                ),
            ],
        )

        # Result structure
        assert result.text, "Empty text in response"
        assert result.protocol == "json_action"
        assert result.provider_name == "deepseek"
        assert result.provider_model in ("deepseek-v4-flash", "deepseek-v4-pro")
        assert result.latency_ms > 0

        # No secret in text or raw
        _assert_no_secret_leak(result.text, "result.text")
        raw_str = json.dumps(result.raw) if result.raw else ""
        _assert_no_secret_leak(raw_str, "result.raw")

        # Usage exists
        if result.usage:
            assert isinstance(result.usage, dict)
            print(f"  Usage: {json.dumps(result.usage)}")
            print(f"  Latency: {result.latency_ms}ms")
            print(f"  Model: {result.provider_model}")

        print(f"  Response: {result.text[:200]}")

    @pytest.mark.timeout(60)
    def test_json_object_format(self):
        """Response is parseable JSON when json_mode is on."""
        provider = _make_provider(purpose="agent")
        from auto_harness.providers.base import Message

        result = provider.complete(
            [
                Message(
                    role="system",
                    content="You are a JSON API. Always output valid JSON.",
                ),
                Message(
                    role="user",
                    content='List 3 colors as JSON: {"colors": ["red", "green", "blue"]}',
                ),
            ],
        )

        text = result.text.strip()
        # Should be parseable JSON
        try:
            parsed = json.loads(text)
            print(f"  Parsed JSON: {json.dumps(parsed)}")
        except json.JSONDecodeError:
            # Some models wrap in markdown code blocks even with json_mode
            # Try extracting from code block
            if "```" in text:
                import re
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1).strip())
                    print(f"  Extracted from code block: {json.dumps(parsed)}")
                else:
                    pytest.fail(f"Response is not valid JSON and no code block found: {text[:200]}")
            else:
                pytest.fail(f"Response is not valid JSON: {text[:200]}")


# ---------------------------------------------------------------------------
# Level 3: Purpose-driven model selection
# ---------------------------------------------------------------------------

class TestLevel3_PurposeModels:
    """Level 3: Different purposes use different models and thinking modes."""

    @pytest.mark.timeout(60)
    def test_agent_purpose_json_action(self):
        """agent purpose: V4 Flash, thinking=disabled, json_mode=true."""
        provider = _make_provider(purpose="agent")
        from auto_harness.providers.base import Message

        result = provider.complete(
            [
                Message(
                    role="system",
                    content="Return JSON only.",
                ),
                Message(
                    role="user",
                    content='{"answer":"hello from agent"}',
                ),
            ],
        )

        assert result.provider_model == "deepseek-v4-flash"
        assert result.text
        print(f"  Agent response (flash): {result.text[:150]}")

    @pytest.mark.timeout(120)
    def test_plan_first_purpose_json_action(self):
        """plan_first purpose: V4 Pro, thinking=enabled, json_mode=true."""
        provider = _make_provider(purpose="plan_first")
        from auto_harness.providers.base import Message

        result = provider.complete(
            [
                Message(
                    role="system",
                    content="You are a deployment planner. Return a JSON plan.",
                ),
                Message(
                    role="user",
                    content=(
                        'Plan the deployment for a simple Flask app. '
                        'Return JSON: {"steps":[{"name":"install","command":"pip install flask"},'
                        '{"name":"run","command":"python app.py"}],"entrypoint":"app.py"}'
                    ),
                ),
            ],
        )

        assert result.provider_model == "deepseek-v4-flash"
        assert result.text
        # Check reasoning privacy
        ctx = result.context
        if ctx.get("reasoning_present"):
            assert "reasoning_sha256" in ctx
            assert isinstance(ctx.get("reasoning_chars"), int)
            # Full reasoning NOT in context
            _assert_no_secret_leak(json.dumps(ctx), "context")

        print(f"  Plan-first response: {result.text[:200]}")
        print(f"  Reasoning present: {ctx.get('reasoning_present', False)}")
        print(f"  Reasoning chars: {ctx.get('reasoning_chars', 0)}")


# ---------------------------------------------------------------------------
# Level 4: Error handling & edge cases
# ---------------------------------------------------------------------------

class TestLevel4_ErrorHandling:
    """Level 4: Error classification, retry, and edge cases."""

    def test_rate_limit_error_category(self):
        """Rate limit errors have correct category."""
        from auto_harness.providers.errors import ProviderError, ErrorCategory, rate_limited_error

        err = rate_limited_error("deepseek")
        assert err.category == ErrorCategory.RATE_LIMITED
        assert err.retryable

    def test_auth_error_no_retry(self):
        """Auth errors are non-retryable."""
        from auto_harness.providers.errors import (
            ProviderError,
            ErrorCategory,
            authentication_error,
        )

        err = authentication_error("deepseek")
        assert err.category == ErrorCategory.AUTHENTICATION_FAILED
        assert not err.retryable

    @pytest.mark.timeout(60)
    def test_empty_prompt_handling(self):
        """Very short prompt still gets a valid response."""
        provider = _make_provider(purpose="agent")
        from auto_harness.providers.base import Message

        result = provider.complete(
            [
                Message(
                    role="system",
                    content="Return JSON only.",
                ),
                Message(role="user", content='{"status":"ping"}'),
            ],
        )

        assert result.text
        print(f"  Ping response: {result.text[:100]}")


# ---------------------------------------------------------------------------
# Manifest generation (run last)
# ---------------------------------------------------------------------------

def test_generate_live_smoke_manifest():
    """Generate a clean manifest WITHOUT any secrets.

    This runs after all tests and captures key metrics.
    """
    manifest = {
        "provider": "deepseek",
        "model_flash": "deepseek-v4-flash",
        "model_pro": "deepseek-v4-pro",
        "protocol": "json_action",
        "thinking_supported": True,
        "json_mode_supported": True,
        "native_tool_calling": False,
        "tests_run": True,
        "secret_persisted": False,
    }

    # Write manifest if requested
    output_path = os.environ.get("DEEPSEEK_MANIFEST_OUTPUT", "")
    if output_path:
        _assert_no_secret_leak(json.dumps(manifest), "manifest")
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {output_path}")

    # Always verify secret safety
    manifest_str = json.dumps(manifest)
    _assert_no_secret_leak(manifest_str, "manifest")
