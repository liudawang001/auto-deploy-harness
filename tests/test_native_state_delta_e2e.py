import json

from auto_harness.agent_runtime.contribution import AgentContributionAnalyzer
from auto_harness.agent_runtime.native_tool_loop import NativeToolTurnLoop
from auto_harness.providers.base import LLMResult, Message
from auto_harness.providers.mock import FakeNativeToolProvider


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def _response(*calls, text=""):
    return LLMResult(
        text=text,
        protocol="native_tools",
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",
    )


def _deterministic_runner_status(state):
    """Small deterministic stage consumer for the reproducible fixture."""
    selected = (state.get("run_candidates") or [{}])[0]
    return "passed" if selected.get("id") == "working" else "uncertain"


def test_native_runner_selection_changes_consumed_state_and_proves_contribution(tmp_path):
    state = {
        "run_candidates": [
            {"id": "broken", "cmd": ["python", "missing.py"]},
            {"id": "working", "cmd": ["python", "app.py"]},
        ],
    }
    baseline_status = _deterministic_runner_status(state)
    provider = FakeNativeToolProvider([
        _response(_call("runner-call", "select_runner_candidate", {
            "candidate_id": "working",
            "reason": "app.py is the discovered entrypoint",
        })),
        _response(text="runner selection complete"),
    ])
    outcome = NativeToolTurnLoop(
        provider,
        stage="runner",
        agent_mode="gated_actor",
        allowed_categories=("state_delta",),
        run_dir=tmp_path,
    ).run(
        [Message(role="user", content="select the grounded runner")],
        context={"state": state},
        task_id="state-delta-fixture",
    )
    agent_status = _deterministic_runner_status(state)

    assert baseline_status == "uncertain"
    assert agent_status == "passed"
    assert outcome.status == "completed"
    assert outcome.tool_results[0].applied is True
    evidence = outcome.tool_results[0].result["evidence"]
    assert evidence["before_state_hash"] != evidence["after_state_hash"]
    assert "selected_runner_candidate_id" in evidence["changed_fields"]
    assert state["run_candidates"][0]["id"] == "working"

    contribution = AgentContributionAnalyzer().evaluate_llm_required(
        tmp_path,
        baseline_status=baseline_status,
        agent_status=agent_status,
        results={"analyze": {"data": {"agent_decision": {}}}, "verify": {"status": "passed"}},
    )
    assert contribution["llm_required"] is True
    native_gate = contribution["evidence"]["gate_results"][0]
    assert native_gate["tool_name"] == "select_runner_candidate"
    assert native_gate["before_state_hash"] != native_gate["after_state_hash"]


def test_native_stage_hint_is_bounded_internal_state_only(tmp_path):
    state = {"stage_hints": {}}
    provider = FakeNativeToolProvider([
        _response(_call("hint-call", "set_stage_hint", {
            "stage": "verify",
            "hints": {"method": "GET", "path": "/health"},
        })),
        _response(text="hint recorded"),
    ])
    outcome = NativeToolTurnLoop(
        provider,
        stage="plan",
        agent_mode="gated_actor",
        allowed_categories=("state_delta",),
        run_dir=tmp_path,
    ).run(
        [Message(role="user", content="set a verify hint")],
        context={"state": state},
        task_id="hint-fixture",
    )
    assert outcome.tool_results[0].applied is True
    assert state["stage_hints"]["verify"] == {"method": "GET", "path": "/health"}
    assert "status" not in state
    assert "final_status" not in state


def test_state_delta_cannot_write_final_stage_status(tmp_path):
    state = {"stage_hints": {}}
    provider = FakeNativeToolProvider([
        _response(_call("bad-hint", "set_stage_hint", {
            "stage": "verify",
            "hints": {"status": "passed"},
        })),
        _response(text="done"),
    ])
    outcome = NativeToolTurnLoop(
        provider,
        stage="plan",
        agent_mode="gated_actor",
        allowed_categories=("state_delta",),
        run_dir=tmp_path,
    ).run(
        [Message(role="user", content="do not bypass verification")],
        context={"state": state},
        task_id="status-bypass",
    )
    assert outcome.tool_results[0].status == "rejected"
    assert outcome.tool_results[0].applied is False
    assert state == {"stage_hints": {}}


def test_no_accepted_native_delta_never_reports_llm_helped(tmp_path):
    report = AgentContributionAnalyzer().analyze(
        tmp_path,
        results={"analyze": {"data": {"agent_decision": {}}}, "verify": {"status": "passed"}},
    )
    assert report["accepted_action_count"] == 0
    assert report["llm_helped"] is False
