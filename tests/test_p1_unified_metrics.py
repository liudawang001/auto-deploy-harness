import json

from auto_harness.agent.metrics import AgentMetricsCollector
from auto_harness.models.base import read_json, write_json
from auto_harness.observability.metrics import UnifiedMetricsCollector
from auto_harness.recovery import FaultInjector, InjectedFault, OperationJournal


def _build_artifacts(run_dir):
    (run_dir / "logs" / "agent_calls").mkdir(parents=True)
    (run_dir / "repairs").mkdir()
    (run_dir / "reports").mkdir()

    write_json(run_dir / "logs" / "agent_calls" / "repair.json", {
        "stage": "repair",
        "provider": "mock",
        "model": "deterministic-fixture",
        "parsed_decision": {"actions": [{"type": "install_package"}]},
        "policy_result": {
            "accepted_actions": [{"type": "install_package"}],
            "rejected_actions": [{"type": "run_shell"}],
        },
        "latency_ms": 12,
    })
    write_json(run_dir / "repairs" / "repair_apply_result.json", {
        "status": "applied",
        "action_results": [{
            "action_type": "install_package",
            "executed": True,
            "exit_code": 0,
        }],
    })
    write_json(run_dir / "repairs" / "repair_loop_state.json", {
        "history": [{"stage": "env_deploy", "status": "verified"}],
    })
    write_json(run_dir / "reports" / "pipeline_results.json", {
        "verify": {"status": "passed", "data": {}},
    })
    write_json(run_dir / "reports" / "skill_effects.json", {
        "effects": [{
            "skill_name": "dependency-repair",
            "stage": "repair",
            "accepted_by_policy": True,
            "field_changed": True,
        }],
    })

    operation = {
        "operation_id": "p1-operation",
        "idempotency_key": "p1-operation",
        "task_id": run_dir.name,
        "run_dir": str(run_dir),
        "stage": "env_deploy",
        "action": "install_dependencies",
        "resource_type": "dependency_install",
        "normalized_input_hash": "fixture",
    }
    journal = OperationJournal(run_dir)
    journal.begin(operation)
    journal.transition("p1-operation", "unknown")
    journal.transition(
        "p1-operation",
        "committed",
        reconcile_result={"decision": "reuse", "reason": "already_installed"},
    )

    try:
        FaultInjector(
            ["env_deploy:after_commit_before_checkpoint"]
        ).raise_if_configured(
            run_dir=run_dir,
            task_id=run_dir.name,
            stage="env_deploy",
            window="after_commit_before_checkpoint",
            operation_id="p1-operation",
        )
    except InjectedFault:
        pass


def test_unified_metrics_are_derived_from_real_artifacts(tmp_path):
    run_dir = tmp_path / "p1-metrics"
    _build_artifacts(run_dir)

    report = UnifiedMetricsCollector().collect(run_dir)
    counters = report["summary"]["counters"]

    assert counters == {
        "llm_calls": 1,
        "policy_accepted": 1,
        "policy_rejected": 1,
        "repair_actions_executed": 1,
        "repair_attempts": 1,
        "recovery_operations": 1,
        "duplicate_execution_prevented": 1,
        "faults_injected": 1,
        "verify_passes": 1,
        "verify_failures": 0,
        "skill_influences": 1,
        "skill_harms": 0,
    }
    assert report["summary"]["rates"]["policy_accept_rate"] == 0.5
    assert report["summary"]["rates"]["skill_harm_rate"] == 0.0
    assert "operations/p1-operation.json" in report["provenance"]
    assert "reports/pipeline_results.json" in report["provenance"]

    event_lines = (
        run_dir / "reports" / "agent_metric_events.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in event_lines]
    assert len(events) == report["event_count"]
    assert all(event["source_artifact"] for event in events)
    assert len({event["event_id"] for event in events}) == len(events)


def test_collection_is_idempotent_and_agent_metrics_links_summary(tmp_path):
    run_dir = tmp_path / "p1-metrics"
    _build_artifacts(run_dir)

    first = UnifiedMetricsCollector().collect(run_dir)
    second = UnifiedMetricsCollector().collect(run_dir)
    first_ids = {
        json.loads(line)["event_id"]
        for line in (
            run_dir / "reports" / "agent_metric_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    }
    assert first["event_count"] == second["event_count"] == len(first_ids)

    agent_report = AgentMetricsCollector().collect(
        run_dir,
        output_path=run_dir / "reports" / "agent_metrics.json",
    )
    assert (
        agent_report["unified_metrics"]["counters"]["duplicate_execution_prevented"]
        == 1
    )
    persisted = read_json(run_dir / "reports" / "agent_metrics.json")
    assert persisted["unified_metrics_path"].endswith("reports/unified_metrics.json")
