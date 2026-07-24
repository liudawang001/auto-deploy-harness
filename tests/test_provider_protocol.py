"""Tests for Provider Protocol clarification.

Validates:
- Current providers report json_action protocol
- Agent trace contains provider_protocol
- LLMResult has protocol and tool_calls fields
- ToolCallingLLMProvider protocol exists for future native adapters
"""
import pytest

from auto_harness.providers.base import LLMResult, Message, ToolCallingLLMProvider
from auto_harness.providers.mock import MockLLMProvider


class TestProviderProtocol:
    """Test that providers correctly report their protocol."""

    def test_mock_provider_reports_json_action(self):
        """MockLLMProvider must report json_action protocol."""
        provider = MockLLMProvider()
        messages = [Message(role="user", content="test")]
        result = provider.complete(messages)
        assert result.protocol == "json_action"

    def test_xunfei_text_response_reports_json_action(self):
        """XunfeiSparkProvider must report json_action for text responses."""
        # Use a mock urlopen that returns a text response
        from io import BytesIO
        from auto_harness.providers.xunfei import XunfeiSparkProvider

        response_data = {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        import json

        class MockResponse:
            def read(self):
                return json.dumps(response_data).encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout=60):
            return MockResponse()

        provider = XunfeiSparkProvider(urlopen=mock_urlopen)
        # Set required env-like attrs
        provider.api_url = "http://test.example.com/v1/messages"
        provider.model = "test-model"

        result = provider.complete([Message(role="user", content="test")])
        assert result.protocol == "json_action"

    def test_llm_result_default_protocol_is_json_action(self):
        """Default LLMResult protocol must be json_action."""
        result = LLMResult(text="test")
        assert result.protocol == "json_action"

    def test_llm_result_has_tool_calls_field(self):
        """LLMResult must have tool_calls field defaulting to empty list."""
        result = LLMResult(text="test")
        assert result.tool_calls == []

    def test_tool_calling_provider_protocol_exists(self):
        """ToolCallingLLMProvider protocol must exist for future native adapters."""
        # Just verify the protocol class is importable
        assert ToolCallingLLMProvider is not None

    def test_json_action_still_passes_schema_and_policy(self):
        """json_action protocol results must still work with schema and policy validation."""
        from auto_harness.agent_runtime.policy import ToolPolicy
        from auto_harness.agent_runtime.schemas import ToolCall, parse_agent_decision

        # Parse a json_action decision
        raw = '{"status": "ok", "tool_call": {"name": "probe_http", "input": {"url": "http://127.0.0.1/", "trace_template": "t_{{trace_id}}"}}, "hypothesis": "test", "confidence": 0.9}'
        decision = parse_agent_decision(raw, allowed_tools=["probe_http"])
        assert decision.status == "ok"
        assert decision.tool_call is not None

        # Policy should still validate
        policy = ToolPolicy()
        result = policy.validate(
            tool_call=decision.tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        assert result.allowed

    def test_agent_trace_contains_provider_protocol(self):
        """Agent trace results must contain provider_protocol field."""
        from auto_harness.agent_runtime.state import AgentVerifyState

        state = AgentVerifyState(trace_id="test-trace")
        result = state.to_result(
            final_status="uncertain",
            stop_reason="max_steps",
            mode="gated_actor",
            llm_helped=False,
        )
        assert "provider_protocol" in result
        assert result["provider_protocol"] == "json_action"
