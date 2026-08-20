import json

from auto_harness.agent_runtime.runtime import AgentRuntime
from auto_harness.agent_runtime.native_tool_loop import NativeToolTurnLoop
from auto_harness.config import HarnessConfig
from auto_harness.modules.reporter import ReportGenerator
from auto_harness.providers.base import LLMResult, Message
from auto_harness.providers.mock import FakeNativeToolProvider
from auto_harness.readiness import CapabilityMatrix


def _call(call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "inspect_repo_tree", "arguments": "{}"},
    }


def test_default_config_keeps_json_action_and_native_disabled():
    config = HarnessConfig()
    assert config.provider_protocol == "json_action"
    assert config.native_tool_calling["enabled"] is False
    assert config.native_tool_calling["parallel_calls"] is False
    assert config.native_tool_calling["allow_side_effect_tools"] is False


def test_runtime_auto_selection_writes_reason_and_native_summary(tmp_path):
    repo = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repo.mkdir()
    provider = FakeNativeToolProvider([
        LLMResult(text="", protocol="native_tools", tool_calls=[_call()], request_id="r1"),
        LLMResult(text="done", protocol="native_tools", request_id="r2", finish_reason="stop"),
    ])
    config = HarnessConfig(
        provider_protocol="auto",
        native_tool_calling={"enabled": True},
    )
    outcome = AgentRuntime().run_protocol_session(
        provider=provider,
        messages=[Message(role="user", content="inspect")],
        context={"repo_dir": str(repo)},
        run_dir=run_dir,
        task_id="runtime-native",
        stage="plan",
        config=config,
    )
    assert outcome.status == "completed"
    selection = json.loads((run_dir / "reports" / "provider_protocol.json").read_text())
    assert selection["selected_protocol"] == "native_tools"
    assert selection["reason"] == "auto_selected_native_tools"
    summary = json.loads((run_dir / "reports" / "native_tool_calling_summary.json").read_text())
    assert summary["native_tool_turn_count"] == 2
    assert summary["native_tool_call_accepted_count"] == 1
    assert summary["tool_schema_estimated_tokens"] > 0
    assert summary["provider_request_ids"] == ["r1", "r2"]

    ReportGenerator().generate(
        run_dir,
        task={"project": {"name": "fixture", "repo_url": ""}},
        results={},
    )
    report = (run_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "## Native Tool Calling" in report
    assert "- Turns: `2`" in report


def test_native_context_budget_includes_tool_schema_and_fails_before_request(tmp_path):
    provider = FakeNativeToolProvider([
        LLMResult(text="unused", protocol="native_tools"),
    ])
    provider.context_window_tokens = 200
    provider.max_tokens = 50
    outcome = NativeToolTurnLoop(
        provider,
        config={"agent_context_safety_margin_tokens": 20},
    ).run(
        [Message(role="user", content="inspect")],
        context={"repo_dir": str(tmp_path)},
    )
    assert outcome.stop_reason == "context_budget_exceeded"
    assert outcome.tool_schema_estimated_tokens > outcome.max_input_tokens
    assert provider.requests == []


def test_readiness_never_promotes_fake_evidence_to_live(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "native_tool_calling_summary.json").write_text(json.dumps({
        "contract_tests_passed": True,
        "fake_tests_passed": True,
        "recovery_tests_passed": True,
        "state_delta_tests_passed": True,
    }), encoding="utf-8")
    evidence = tmp_path / "docs" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "native-tool-live-smoke-manifest.json").write_text(json.dumps({
        "provider_protocol": "native_tools",
        "provider_name": "fake_native",
        "network_transport": "live",
        "tool_call_count": 2,
        "final_status": "passed",
    }), encoding="utf-8")
    readiness = CapabilityMatrix(tmp_path)._native_tool_readiness()
    assert readiness["details"]["readiness_level"] == "state_delta_verified"
    assert readiness["details"]["live_read_only_verified"] is False
    assert readiness["details"]["v0_3_release_gate_passed"] is False
