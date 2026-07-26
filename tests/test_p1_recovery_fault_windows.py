import subprocess
import sys
from types import SimpleNamespace

import pytest

from auto_harness.agent_runtime.stage_executor import StageExecutionResult
from auto_harness.graph.nodes import make_recovery_gate_node, make_stage_node
from auto_harness.recovery import FaultInjector, InjectedFault, OperationJournal
from auto_harness.recovery.graph_adapter import GraphRecoveryAdapter


def _state(tmp_path):
    return {
        "task_id": "p1-recovery",
        "run_dir": str(tmp_path),
        "repo_dir": str(tmp_path / "workspace" / "repo"),
        "runtime_policy": {},
        "compiled_analysis": {"install_plan": [["pip", "install", "flask"]]},
        "stage_results": {},
        "dry_run": False,
        "skill_contexts": {},
    }


class _CountingExecutor:
    def __init__(self):
        self.calls = 0

    def execute_stage(self, **_kwargs):
        self.calls += 1
        return StageExecutionResult(
            stage="env_deploy",
            before_status="pending",
            after_status="passed",
            result={"status": "passed", "data": {"calls": self.calls}},
            changed=True,
        )


class _ReuseReconciler:
    def reconcile(self, _operation):
        return {
            "decision": "reuse",
            "reason": "external_resource_observed",
            "observed_state": {"exists": True},
        }


def _deps(adapter, executor, injector):
    return SimpleNamespace(
        recovery_adapter=adapter,
        stage_executor=executor,
        fault_injector=injector,
        merge_analysis=lambda deterministic, compiled: dict(compiled),
    )


def _gate_state(state, deps):
    update = make_recovery_gate_node("env_deploy", deps)(state)
    return {**state, **update}


def test_before_side_effect_fault_leaves_running_and_never_calls_executor(tmp_path):
    executor = _CountingExecutor()
    deps = _deps(
        GraphRecoveryAdapter(),
        executor,
        FaultInjector(["env_deploy:before_side_effect"]),
    )
    state = _gate_state(_state(tmp_path), deps)

    with pytest.raises(InjectedFault) as raised:
        make_stage_node("env_deploy", deps)(state)

    assert raised.value.point == "env_deploy:before_side_effect"
    assert executor.calls == 0
    record = OperationJournal(tmp_path).load(state["pending_operation_id"])
    assert record["status"] == "running"
    assert record["idempotency_key"] == record["operation_id"]


def test_after_effect_before_commit_resume_reuses_durable_result(tmp_path):
    executor = _CountingExecutor()
    first_deps = _deps(
        GraphRecoveryAdapter(reconcilers={"dependency_install": _ReuseReconciler()}),
        executor,
        FaultInjector(["env_deploy:after_side_effect_before_commit"]),
    )
    state = _gate_state(_state(tmp_path), first_deps)

    with pytest.raises(InjectedFault):
        make_stage_node("env_deploy", first_deps)(state)

    assert executor.calls == 1
    journal = OperationJournal(tmp_path)
    assert journal.load(state["pending_operation_id"])["status"] == "running"
    assert (
        tmp_path / "operations" / ("%s_result.json" % state["pending_operation_id"])
    ).exists()

    resumed_deps = _deps(
        GraphRecoveryAdapter(reconcilers={"dependency_install": _ReuseReconciler()}),
        executor,
        FaultInjector(["env_deploy:after_side_effect_before_commit"]),
    )
    resumed = _gate_state(_state(tmp_path), resumed_deps)
    assert resumed["recovery_decision"] == "reuse"
    assert resumed["recovery_skip_stage"] is True
    assert resumed["stage_results"]["env_deploy"]["data"]["calls"] == 1

    result = make_stage_node("env_deploy", resumed_deps)(resumed)
    assert result["node_history"][0]["status"] == "skipped_recovery_reuse"
    assert executor.calls == 1


def test_after_commit_before_checkpoint_resume_hydrates_without_duplicate(tmp_path):
    executor = _CountingExecutor()
    first_deps = _deps(
        GraphRecoveryAdapter(),
        executor,
        FaultInjector(["env_deploy:after_commit_before_checkpoint"]),
    )
    state = _gate_state(_state(tmp_path), first_deps)

    with pytest.raises(InjectedFault):
        make_stage_node("env_deploy", first_deps)(state)

    assert executor.calls == 1
    assert OperationJournal(tmp_path).load(state["pending_operation_id"])["status"] == "committed"

    resumed_deps = _deps(
        GraphRecoveryAdapter(),
        executor,
        FaultInjector(["env_deploy:after_commit_before_checkpoint"]),
    )
    resumed = _gate_state(_state(tmp_path), resumed_deps)
    assert resumed["recovery_decision"] == "reuse"
    assert resumed["recovery_skip_stage"] is True
    make_stage_node("env_deploy", resumed_deps)(resumed)
    assert executor.calls == 1


def test_fault_marker_is_persistent_across_injector_instances(tmp_path):
    kwargs = {
        "run_dir": tmp_path,
        "task_id": "p1-recovery",
        "stage": "env_deploy",
        "window": "before_side_effect",
        "operation_id": "stable-operation",
    }
    with pytest.raises(InjectedFault):
        FaultInjector(["env_deploy:before_side_effect"]).raise_if_configured(**kwargs)

    injected = FaultInjector(
        ["env_deploy:before_side_effect"]
    ).raise_if_configured(**kwargs)
    assert injected is False


def test_cross_process_resume_after_effect_does_not_duplicate(tmp_path):
    script = r"""
import sys
import os
from pathlib import Path
from auto_harness.recovery import FaultInjector, InjectedFault
from auto_harness.recovery.graph_adapter import GraphRecoveryAdapter

run_dir = Path(sys.argv[1])
state = {
    "task_id": "p1-subprocess",
    "run_dir": str(run_dir),
    "repo_dir": str(run_dir / "workspace" / "repo"),
    "runtime_policy": {},
    "compiled_analysis": {"install_plan": [["pip", "install", "flask"]]},
}
adapter = GraphRecoveryAdapter()
decision = adapter.prepare_or_reconcile(state, "env_deploy")
adapter.persist_result(
    state,
    "env_deploy",
    {"status": "passed", "data": {"side_effect_calls": 1}},
)
try:
    FaultInjector(
        ["env_deploy:after_side_effect_before_commit"]
    ).raise_if_configured(
        run_dir=run_dir,
        task_id=state["task_id"],
        stage="env_deploy",
        window="after_side_effect_before_commit",
        operation_id=decision.operation["operation_id"],
    )
except InjectedFault:
    os._exit(73)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 73, completed.stderr

    class Reuse:
        def reconcile(self, _operation):
            return {"decision": "reuse", "reason": "resource_observed"}

    state = {
        "task_id": "p1-subprocess",
        "run_dir": str(tmp_path),
        "repo_dir": str(tmp_path / "workspace" / "repo"),
        "runtime_policy": {},
        "compiled_analysis": {"install_plan": [["pip", "install", "flask"]]},
    }
    resumed = GraphRecoveryAdapter(
        reconcilers={"dependency_install": Reuse()}
    ).prepare_or_reconcile(state, "env_deploy")
    assert resumed.decision == "reuse"
    assert resumed.hydrated_stage_result["data"]["side_effect_calls"] == 1
