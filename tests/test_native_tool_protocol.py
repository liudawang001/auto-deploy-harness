from dataclasses import dataclass

import pytest

from auto_harness.agent_runtime.schemas import ToolCall
from auto_harness.agent_runtime.schemas import parse_agent_decision
from auto_harness.providers.protocols import (
    ProviderProtocolError,
    ToolCallConflictError,
    canonical_json_hash,
    normalize_json_action_call,
    normalize_provider_tool_call,
    select_provider_protocol,
    tool_operation_id,
)


def test_canonical_arguments_hash_ignores_key_order():
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})


def test_normalize_json_action_tool_call_without_raw_provider_payload():
    normalized = normalize_json_action_call(
        ToolCall(name="probe_http", input={"url": "http://127.0.0.1"}),
        provider_name="mock",
    )
    assert normalized.provider_protocol == "json_action"
    assert normalized.tool_name == "probe_http"
    assert normalized.call_id.startswith("call_")
    assert "raw" not in normalized.to_dict()


def test_json_action_parser_populates_protocol_identity():
    decision = parse_agent_decision(
        '{"status":"ok","tool_call":{"name":"probe_http","input":{"url":"http://127.0.0.1"}}}',
        allowed_tools=["probe_http"],
    )
    assert decision.tool_call.provider_protocol == "json_action"
    assert decision.tool_call.call_id.startswith("call_")
    assert decision.tool_call.arguments_hash.startswith("sha256:")
    assert decision.tool_call.idempotency_key.startswith("sha256:")


def test_normalize_openai_native_call_parses_arguments_and_preserves_id():
    normalized = normalize_provider_tool_call({
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "inspect_repo_tree",
            "arguments": '{"max_depth": 2, "path": "."}',
        },
    }, provider_name="deepseek", provider_model="model", turn_index=1)
    assert normalized.call_id == "call-1"
    assert normalized.arguments == {"max_depth": 2, "path": "."}
    assert normalized.provider_protocol == "native_tools"


def test_normalize_anthropic_shaped_call():
    normalized = normalize_provider_tool_call({
        "id": "toolu-1",
        "name": "read_selected_files",
        "input": {"files": [{"path": "README.md"}]},
    })
    assert normalized.tool_name == "read_selected_files"
    assert normalized.arguments["files"][0]["path"] == "README.md"


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        normalize_provider_tool_call({
            "id": "call-1",
            "function": {"name": "inspect_repo_tree", "arguments": "{"},
        })


def test_same_call_id_with_changed_arguments_is_conflict():
    first = normalize_provider_tool_call({
        "id": "call-1",
        "function": {"name": "inspect_repo_tree", "arguments": '{"max_depth": 1}'},
    })
    with pytest.raises(ToolCallConflictError):
        normalize_provider_tool_call({
            "id": "call-1",
            "function": {"name": "inspect_repo_tree", "arguments": '{"max_depth": 2}'},
        }, seen_call_hashes={"call-1": first.arguments_hash})


def test_operation_identity_ignores_provider_call_id():
    first = tool_operation_id(
        task_id="t", stage="plan", tool_name="inspect_repo_tree",
        arguments={"path": "."}, repository_fingerprint="repo",
    )
    second = tool_operation_id(
        task_id="t", stage="plan", tool_name="inspect_repo_tree",
        arguments={"path": "."}, repository_fingerprint="repo",
    )
    assert first == second


@dataclass
class Caps:
    supports_tool_calling: bool


class NativeProvider:
    native_tool_calling = True

    def complete_with_tools(self, *args, **kwargs):
        raise AssertionError("not called by selection")


class JsonOnlyProvider:
    pass


def test_protocol_default_stays_json_action():
    selection = select_provider_protocol("", NativeProvider(), Caps(True))
    assert selection.selected_protocol == "json_action"


def test_explicit_native_is_fail_closed_when_unsupported():
    with pytest.raises(ProviderProtocolError, match="no complete_with_tools"):
        select_provider_protocol("native_tools", JsonOnlyProvider(), Caps(False))


def test_auto_selects_only_implemented_and_enabled_native_protocol():
    assert select_provider_protocol("auto", NativeProvider(), Caps(True)).selected_protocol == "native_tools"
    assert select_provider_protocol("auto", NativeProvider(), Caps(False)).selected_protocol == "json_action"
