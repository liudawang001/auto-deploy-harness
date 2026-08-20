import json

import pytest

from auto_harness.providers.base import Message
from auto_harness.providers.openai_compatible import OpenAICompatibleProvider


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _provider(native=True):
    captured = {}
    response = {
        "id": "resp-1",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
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
    }

    def opener(request, timeout=60):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response(response)

    provider = OpenAICompatibleProvider(
        "openai",
        config={"provider_configs": {"openai": {
            "api_base": "https://example.invalid/v1",
            "model": "model-1",
            "context_window_tokens": 8192,
            "require_api_key": False,
            "native_tool_calling": native,
        }}},
        urlopen=opener,
    )
    return provider, captured


def _tools():
    return [{"type": "function", "function": {
        "name": "inspect_repo_tree",
        "parameters": {"type": "object", "properties": {}},
    }}]


def test_openai_compatible_native_request_and_response_contract():
    provider, captured = _provider()
    result = provider.complete_with_tools(
        [Message(role="user", content="inspect")], _tools(),
    )
    assert result.protocol == "native_tools"
    assert result.tool_calls[0]["id"] == "call-1"
    assert captured["payload"]["tools"] == _tools()
    assert captured["payload"]["tool_choice"] == "auto"


def test_openai_compatible_native_messages_keep_call_ids():
    provider, _ = _provider()
    call = {
        "id": "call-1", "type": "function",
        "function": {"name": "inspect_repo_tree", "arguments": "{}"},
    }
    messages = provider._serialize_native_messages([
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="{}", tool_call_id="call-1"),
    ])
    assert messages[0]["tool_calls"] == [call]
    assert messages[1]["tool_call_id"] == "call-1"


def test_openai_compatible_native_disabled_does_not_fallback():
    provider, captured = _provider(native=False)
    with pytest.raises(RuntimeError, match="disabled"):
        provider.complete_with_tools(
            [Message(role="user", content="inspect")], _tools(),
        )
    assert captured == {}


def test_custom_compatible_endpoint_requires_explicit_capability_flag():
    provider = OpenAICompatibleProvider(
        "vllm",
        config={"provider_configs": {"vllm": {
            "api_base": "https://example.invalid/v1",
            "model": "model-1",
            "context_window_tokens": 8192,
            "require_api_key": False,
            "supports_native_tool_calling": True,
        }}},
    )
    assert provider.native_tool_calling is True
    assert provider.supports_native_tool_calling is True
