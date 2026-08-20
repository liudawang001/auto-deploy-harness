import json

import pytest

from auto_harness.agent_runtime.native_tool_loop import NativeToolTurnLoop
from auto_harness.agent_runtime.schemas import ToolResult
from auto_harness.providers.base import LLMResult, Message
from auto_harness.providers.mock import FakeNativeToolProvider
from auto_harness.providers.protocols import (
    normalize_provider_tool_call,
    project_provider_tools,
    tool_operation_id,
)
from auto_harness.recovery.journal import OperationJournal
from auto_harness.recovery.tool_call_ledger import ToolCallLedger
from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.repository_executor import RepositoryToolExecutor
from auto_harness.tools.schemas import ToolSchema


class CrashAt:
    def __init__(self, point):
        self.point = point
        self.raised = False

    def __call__(self, point, operation_id):
        if point == self.point and not self.raised:
            self.raised = True
            raise RuntimeError("crash:%s" % point)


class CountingRepositoryExecutor:
    def __init__(self, registry):
        self.delegate = RepositoryToolExecutor(registry=registry)
        self.count = 0

    def execute(self, call, context):
        self.count += 1
        return self.delegate.execute(call, context)


class CountingSideEffectExecutor:
    def __init__(self):
        self.count = 0

    def execute(self, call, context):
        self.count += 1
        return ToolResult(
            status="passed",
            tool_name=call.name,
            category="side_effect",
            policy_allowed=True,
            executed=True,
            applied=True,
            evidence={"effect_id": "effect-1"},
        )


def _call(call_id, name="inspect_repo_tree", arguments=None):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}, sort_keys=True),
        },
    }


def _tool_response(call):
    return LLMResult(
        text="", protocol="native_tools", tool_calls=[call],
        finish_reason="tool_calls", request_id="request-tools",
    )


def _final_response():
    return LLMResult(
        text="done", protocol="native_tools", tool_calls=[],
        finish_reason="stop", request_id="request-final",
    )


@pytest.mark.parametrize("fault_point", [
    "after_provider_before_call_ledger",
    "after_call_ledger_before_policy",
    "after_tool_result_before_provider_feedback",
    "after_provider_feedback_before_checkpoint",
])
def test_read_only_fault_windows_resume_without_duplicate_execution(tmp_path, fault_point):
    repo_dir = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("fixture", encoding="utf-8")
    registry = ToolRegistry()
    executor = CountingRepositoryExecutor(registry)
    crashing_provider = FakeNativeToolProvider([
        _tool_response(_call("call-before-crash")),
    ])
    crashing_loop = NativeToolTurnLoop(
        crashing_provider,
        registry=registry,
        executor=executor,
        run_dir=run_dir,
        fault_injector=CrashAt(fault_point),
    )
    with pytest.raises(RuntimeError, match="crash"):
        crashing_loop.run(
            [Message(role="user", content="inspect")],
            context={"repo_dir": str(repo_dir)},
            task_id="task-recovery",
            repository_fingerprint="repo-v1",
        )

    resumed_provider = FakeNativeToolProvider([
        _tool_response(_call("call-after-restart")),
        _final_response(),
    ])
    outcome = NativeToolTurnLoop(
        resumed_provider,
        registry=registry,
        executor=executor,
        run_dir=run_dir,
    ).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(repo_dir)},
        task_id="task-recovery",
        repository_fingerprint="repo-v1",
    )

    assert outcome.status == "completed"
    assert executor.count == 1
    expected_reuse = fault_point in {
        "after_tool_result_before_provider_feedback",
        "after_provider_feedback_before_checkpoint",
    }
    assert bool(outcome.reused_tool_count) is expected_reuse


@pytest.mark.parametrize("fault_point,expected_execution,expected_status", [
    ("after_journal_begin_before_side_effect", 0, "unknown"),
    ("after_side_effect_before_journal_commit", 1, "unknown"),
    ("after_journal_commit_before_tool_result", 1, "committed"),
])
def test_side_effect_fault_windows_require_reconcile_or_reuse_commit(
    tmp_path, fault_point, expected_execution, expected_status,
):
    run_dir = tmp_path / "run"
    registry = ToolRegistry()
    registry.tools["test_side_effect"] = ToolSchema(
        name="test_side_effect",
        input_schema={
            "type": "object",
            "required": ["target"],
            "properties": {"target": {"type": "string"}},
        },
        category="side_effect",
        implemented=True,
        executor="test",
        stages=["plan"],
    )
    executor = CountingSideEffectExecutor()
    call_args = {"target": "fixture"}
    provider = FakeNativeToolProvider([
        _tool_response(_call("side-before", "test_side_effect", call_args)),
    ])
    with pytest.raises(RuntimeError, match="crash"):
        NativeToolTurnLoop(
            provider,
            registry=registry,
            executor=executor,
            allowed_categories=("side_effect",),
            run_dir=run_dir,
            fault_injector=CrashAt(fault_point),
        ).run(
            [Message(role="user", content="act")],
            context={},
            task_id="task-side",
        )

    resumed = FakeNativeToolProvider([
        _tool_response(_call("side-after", "test_side_effect", call_args)),
        _final_response(),
    ])
    outcome = NativeToolTurnLoop(
        resumed,
        registry=registry,
        executor=executor,
        allowed_categories=("side_effect",),
        run_dir=run_dir,
    ).run(
        [Message(role="user", content="act")],
        context={},
        task_id="task-side",
    )
    operation_id = tool_operation_id(
        task_id="task-side", stage="plan", tool_name="test_side_effect",
        arguments=call_args,
    )
    journal_record = OperationJournal(run_dir).load(operation_id)
    assert executor.count == expected_execution
    assert journal_record["status"] == expected_status
    assert outcome.status == "completed"
    if expected_status == "committed":
        assert outcome.reused_tool_count == 1
        assert outcome.tool_results[0].status == "passed"
    else:
        assert outcome.tool_results[0].status == "rejected"
        assert "reconciliation" in outcome.tool_results[0].error


def test_ledger_detects_cross_process_call_id_conflict(tmp_path):
    ledger = ToolCallLedger(tmp_path)
    first = normalize_provider_tool_call(_call("same", arguments={"max_depth": 1}))
    projection = project_provider_tools(ToolRegistry(), stage="plan", agent_mode="planner")
    ledger.record_call(
        first,
        task_id="task",
        operation_id="tool-operation-one",
        tool_schema_hash=projection.schema_hash,
    )
    second = normalize_provider_tool_call(_call("same", arguments={"max_depth": 2}))
    with pytest.raises(ValueError, match="different arguments"):
        ledger.record_call(
            second,
            task_id="task",
            operation_id="tool-operation-two",
            tool_schema_hash=projection.schema_hash,
        )


def test_ledger_rebuilds_deterministic_protocol_messages(tmp_path):
    repo_dir = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo_dir.mkdir()
    provider = FakeNativeToolProvider([
        _tool_response(_call("rebuild-call")), _final_response(),
    ])
    NativeToolTurnLoop(provider, run_dir=run_dir).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(repo_dir)},
        task_id="task-rebuild",
    )
    messages = ToolCallLedger(run_dir).rebuild_exchange("rebuild-call")
    assert [message.role for message in messages] == ["assistant", "tool"]
    assert messages[0].tool_calls[0]["id"] == "rebuild-call"
    assert messages[1].tool_call_id == "rebuild-call"
    assert json.loads(messages[1].content)["status"] == "passed"


def test_pre_policy_ledger_snapshot_redacts_provider_secret_fields(tmp_path):
    repo_dir = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo_dir.mkdir()
    provider = FakeNativeToolProvider([
        _tool_response(_call("secret-call", arguments={
            "path": ".", "api_key": "sk-1234567890abcdef",
        })),
        _final_response(),
    ])
    outcome = NativeToolTurnLoop(provider, run_dir=run_dir).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(repo_dir)},
        task_id="secret-ledger",
    )
    assert outcome.tool_results[0].status == "rejected"
    text = (
        run_dir / "agent_tool_calls" / "calls" / "secret-call.json"
    ).read_text(encoding="utf-8")
    assert "sk-1234567890abcdef" not in text
    assert "REDACTED" in text
