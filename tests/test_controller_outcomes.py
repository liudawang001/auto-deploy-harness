import json

from auto_harness.controllers.base import DeploymentResult
from auto_harness.controllers.outcomes import controller_exit_code
from auto_harness.controllers.state_sync import sync_controller_state
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.state.store import StateStore


def _create_task(tmp_path):
    store = StateStore(tmp_path / "runs")
    spec = TaskSpec(
        task_id="task1",
        project=ProjectSpec(name="demo", repo_url="local"),
        runtime=RuntimePolicy(workspace_root=str(tmp_path / "workspace")),
        created_at="2026-01-01T00:00:00Z",
        controller="langgraph",
    )
    run_dir = store.create_task(spec)
    return store, run_dir


def test_controller_exit_code_contract():
    assert controller_exit_code("completed") == 0
    assert controller_exit_code("completed_dry_run") == 0
    assert controller_exit_code("stopped") == 1
    assert controller_exit_code("uncertain") == 1
    assert controller_exit_code("failed") == 3


def test_sync_completed_dry_run_updates_user_visible_state(tmp_path):
    store, run_dir = _create_task(tmp_path)
    pipeline = {
        "analyze": {"status": "passed"},
        "verify": {"status": "uncertain", "error": None},
    }
    reports = run_dir / "reports"
    (reports / "pipeline_results.json").write_text(
        json.dumps(pipeline), encoding="utf-8"
    )
    result = DeploymentResult(
        task_id="task1",
        status="completed_dry_run",
        stop_reason="dry_run_completed",
        controller="langgraph",
        verify_status="uncertain",
    )

    sync_controller_state(store, "task1", result)

    state = store.load_state("task1")
    assert state.status == "completed_dry_run"
    assert state.current_stage == "report"
    assert state.stages["analyze"].status == "passed"
    assert state.stages["verify"].status == "uncertain"
    assert state.stages["report"].status == "passed"


def test_sync_stopped_does_not_claim_completion(tmp_path):
    store, _ = _create_task(tmp_path)
    result = DeploymentResult(
        task_id="task1",
        status="stopped",
        stop_reason="policy_rejected",
        controller="langgraph",
    )
    sync_controller_state(store, "task1", result)
    state = store.load_state("task1")
    assert state.status == "stopped"
    assert state.current_stage == "controller"
    assert state.stages["report"].status == "pending"
