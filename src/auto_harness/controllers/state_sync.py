"""Synchronize graph/controller progress into the user-visible TaskStore."""
from pathlib import Path
from typing import Any

from auto_harness.controllers.outcomes import SUCCESS_STATUSES
from auto_harness.models.base import read_json, write_json
from auto_harness.models.state import StageState
from auto_harness.utils.time import utc_now_iso


class StateConsistencyError(RuntimeError):
    """Raised when persisted controller and task terminal states disagree."""


def mark_controller_running(store, task_id: str) -> None:
    state = store.load_state(task_id)
    state.status = "running"
    state.current_stage = "controller"
    store.save_state(state)
    store.events(task_id).append(
        "controller", "controller_started", {"status": "running"}
    )


def sync_controller_state(store, task_id: str, result: Any) -> None:
    """Reconcile final graph artifacts and terminal controller truth."""
    run_dir = Path(store.run_dir(task_id))
    pipeline_path = run_dir / "reports" / "pipeline_results.json"
    if pipeline_path.exists():
        try:
            pipeline = read_json(pipeline_path)
        except (OSError, ValueError):
            pipeline = {}
        if isinstance(pipeline, dict):
            state = store.load_state(task_id)
            for stage, payload in pipeline.items():
                if stage not in state.stages or not isinstance(payload, dict):
                    continue
                raw_status = str(payload.get("status", "uncertain"))
                stage_status = "passed" if raw_status == "pass" else raw_status
                state.current_stage = stage
                state.stages[stage] = StageState(
                    status=stage_status,
                    updated_at=utc_now_iso(),
                    result_path=str(pipeline_path),
                    error=payload.get("error"),
                    progress=(
                        payload.get("progress", {})
                        if isinstance(payload.get("progress", {}), dict)
                        else {}
                    ),
                )
                if stage_status == "passed":
                    state.last_safe_stage = stage
            store.save_state(state)

    state = store.load_state(task_id)
    result_status = str(getattr(result, "status", "failed") or "failed")
    if result_status == "interrupted":
        state.status = "waiting_for_approval"
    elif result_status == "blocked":
        state.status = "stopped"
    else:
        state.status = result_status
    if result_status in SUCCESS_STATUSES:
        state.current_stage = "report"
        state.stages["report"] = StageState(
            status="passed",
            updated_at=utc_now_iso(),
            result_path=str(run_dir / "reports" / "report.md"),
        )
        state.last_safe_stage = "report"
        report_md = run_dir / "reports" / "report.md"
        state.report_path = str(report_md if report_md.exists() else pipeline_path)
    elif state.current_stage in {"created", "controller"}:
        state.current_stage = "controller"
    store.save_state(state)
    store.events(task_id).append(
        "controller",
        "controller_terminal",
        {
            "status": state.status,
            "controller_status": result_status,
            "stop_reason": str(getattr(result, "stop_reason", "")),
            "verify_status": str(getattr(result, "verify_status", "")),
        },
    )


def assert_terminal_consistency(store, task_id: str, result: Any) -> None:
    """Enforce controller/state/report invariants after result persistence."""
    run_dir = Path(store.run_dir(task_id))
    state = store.load_state(task_id)
    status = str(getattr(result, "status", ""))
    verify_status = str(getattr(result, "verify_status", ""))
    errors = []
    if status == "completed":
        if state.status != "completed":
            errors.append("completed controller result requires completed state")
        if verify_status not in {"pass", "passed"}:
            errors.append("completed controller result requires passed verification")
    elif status == "completed_dry_run":
        if state.status != "completed_dry_run":
            errors.append("completed_dry_run result requires matching state")
    elif status == "stopped" and state.status != "stopped":
        errors.append("stopped controller result requires stopped state")
    elif status == "failed" and state.status != "failed":
        errors.append("failed controller result requires failed state")
    elif status == "interrupted" and state.status != "waiting_for_approval":
        errors.append("interrupted result requires waiting_for_approval state")

    if status in SUCCESS_STATUSES:
        if not state.report_path or not Path(state.report_path).exists():
            errors.append("successful terminal state requires an existing report")

    if errors:
        error_path = run_dir / "reports" / "state_consistency_error.json"
        write_json(error_path, {
            "task_id": task_id,
            "controller_status": status,
            "state_status": state.status,
            "verify_status": verify_status,
            "errors": errors,
        })
        raise StateConsistencyError("; ".join(errors))
