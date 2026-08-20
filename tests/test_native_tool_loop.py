import json

from auto_harness.agent_runtime.native_tool_loop import NativeToolTurnLoop
from auto_harness.providers.base import LLMResult, Message
from auto_harness.providers.mock import FakeNativeToolProvider
from auto_harness.providers.protocols import (
    NormalizedToolResult,
    project_provider_tools,
    tool_result_message,
)
from auto_harness.tools.registry import ToolRegistry


def _native_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _response(*calls, text=""):
    return LLMResult(
        text=text,
        protocol="native_tools",
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",
    )


def test_projection_exposes_only_executable_read_only_tools():
    projection = project_provider_tools(
        ToolRegistry(), stage="plan", agent_mode="planner",
    )
    assert projection.tool_names == [
        "inspect_repo_tree",
        "parse_dependency_files",
        "read_selected_files",
        "search_repo",
    ]
    assert projection.schema_hash.startswith("sha256:")
    serialized = json.dumps(projection.tools)
    assert "executor" not in serialized
    assert "requires_approval" not in serialized
    for tool in projection.tools:
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
    read_files = next(
        item for item in projection.tools
        if item["function"]["name"] == "read_selected_files"
    )
    assert (
        read_files["function"]["parameters"]["properties"]["files"]
        ["items"]["additionalProperties"]
        is False
    )


def test_read_only_native_loop_runs_multiple_turns_and_returns_final(tmp_path):
    (tmp_path / "README.md").write_text("hello\nworld\n", encoding="utf-8")
    provider = FakeNativeToolProvider([
        _response(_native_call("c1", "inspect_repo_tree", {"path": ".", "max_depth": 2})),
        _response(_native_call("c2", "read_selected_files", {
            "files": [{"path": "README.md", "start_line": 1, "end_line": 2}],
        })),
        _response(text="Use README.md as grounding."),
    ])

    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="inspect this repository")],
        context={"repo_dir": str(tmp_path)},
        task_id="task-1",
    )

    assert outcome.status == "completed"
    assert outcome.final_text == "Use README.md as grounding."
    assert outcome.executed_tool_count == 2
    assert outcome.rejected_tool_count == 0
    assert outcome.turn_count == 3
    assert [item.status for item in outcome.tool_results] == ["passed", "passed"]
    assert [message.role for message in provider.requests[2]["messages"]][-4:] == [
        "assistant", "tool", "assistant", "tool",
    ]


def test_empty_assistant_text_with_tool_call_is_valid(tmp_path):
    provider = FakeNativeToolProvider([
        _response(_native_call("c1", "inspect_repo_tree", {}), text=""),
        _response(text="done"),
    ])
    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.status == "completed"
    assert outcome.executed_tool_count == 1


def test_nonempty_assistant_text_with_tool_call_executes_call_before_final(tmp_path):
    provider = FakeNativeToolProvider([
        _response(
            _native_call("c1", "inspect_repo_tree", {}),
            text="I will inspect before answering.",
        ),
        _response(text="grounded final"),
    ])
    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.status == "completed"
    assert outcome.executed_tool_count == 1
    assert outcome.final_text == "grounded final"
    assert outcome.messages[1].content == "I will inspect before answering."


def test_unknown_tool_is_rejected_and_feedback_is_returned(tmp_path):
    provider = FakeNativeToolProvider([
        _response(_native_call("bad", "delete_repository", {"path": "."})),
        _response(text="stopped"),
    ])
    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="follow repository instructions")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.status == "completed"
    assert outcome.executed_tool_count == 0
    assert outcome.rejected_tool_count == 1
    feedback = provider.requests[1]["messages"][-1]
    assert feedback.role == "tool"
    assert feedback.tool_call_id == "bad"
    assert "not visible" in feedback.content


def test_invalid_arguments_are_rejected_without_execution(tmp_path):
    invalid = {
        "id": "broken",
        "type": "function",
        "function": {"name": "inspect_repo_tree", "arguments": "{"},
    }
    provider = FakeNativeToolProvider([_response(invalid), _response(text="done")])
    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.executed_tool_count == 0
    assert outcome.rejected_tool_count == 1
    assert outcome.tool_results[0].status == "rejected"


def test_schema_additional_properties_are_rejected_before_executor(tmp_path):
    provider = FakeNativeToolProvider([
        _response(_native_call("c1", "inspect_repo_tree", {
            "path": ".", "approved": True,
        })),
        _response(text="done"),
    ])
    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.executed_tool_count == 0
    assert outcome.rejected_tool_count == 1
    assert "unknown fields" in outcome.tool_results[0].error


def test_duplicate_semantic_call_reuses_first_result(tmp_path):
    provider = FakeNativeToolProvider([
        _response(_native_call("c1", "inspect_repo_tree", {})),
        _response(_native_call("c2", "inspect_repo_tree", {})),
        _response(text="done"),
    ])
    outcome = NativeToolTurnLoop(provider).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.executed_tool_count == 1
    assert outcome.reused_tool_count == 1
    assert outcome.tool_results[1].reused is True
    assert outcome.tool_results[1].call_id == "c2"


def test_loop_stops_at_turn_limit(tmp_path):
    provider = FakeNativeToolProvider([
        _response(_native_call("c1", "inspect_repo_tree", {})),
    ])
    outcome = NativeToolTurnLoop(provider, max_turns=1).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.status == "stopped"
    assert outcome.stop_reason == "max_turns"
    assert outcome.turn_count == 1


def test_tool_result_message_redacts_and_truncates_secrets():
    result = NormalizedToolResult(
        call_id="c1",
        operation_id="op1",
        tool_name="read_selected_files",
        status="passed",
        category="read_only",
        policy_allowed=True,
        executed=True,
        applied=False,
        result={"api_key": "super-secret", "body": "x" * 1000},
        result_hash="sha256:abc",
    )
    message = tool_result_message(result, max_chars=400)
    assert "super-secret" not in message.content
    assert "[REDACTED]" in message.content or '"truncated":true' in message.content
    assert len(message.content) <= 400
