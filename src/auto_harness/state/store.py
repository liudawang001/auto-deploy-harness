import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.models.base import read_json, to_plain, write_json
from auto_harness.models.state import StageState, TaskState
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.state.events import EventLog
from auto_harness.utils.files import ensure_dir
from auto_harness.utils.time import utc_now_iso


class StateStore:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = ensure_dir(runs_dir)

    def run_dir(self, task_id: str) -> Path:
        return self.runs_dir / task_id

    def create_task(self, spec: TaskSpec) -> Path:
        run_dir = ensure_dir(self.run_dir(spec.task_id))
        ensure_dir(run_dir / "workspace" / "repo")
        ensure_dir(run_dir / "workspace" / "verify_workspace")
        ensure_dir(run_dir / "logs" / "agent_calls")
        ensure_dir(run_dir / "evidence")
        ensure_dir(run_dir / "reports")
        write_json(run_dir / "task.json", spec)
        state = TaskState(
            task_id=spec.task_id,
            stages={
                "analyze": StageState(),
                "resource_plan": StageState(),
                "env_solve": StageState(),
                "env_deploy": StageState(),
                "model_prepare": StageState(),
                "runner": StageState(),
                "verify": StageState(),
                "report": StageState(),
            },
        )
        self.save_state(state)
        self.events(spec.task_id).append("task", "created", {"task_id": spec.task_id})
        return run_dir

    def load_task(self, task_id: str) -> TaskSpec:
        data = read_json(self.run_dir(task_id) / "task.json")
        project = ProjectSpec(**data["project"])
        runtime = RuntimePolicy(**data["runtime"])
        return TaskSpec(
            task_id=data["task_id"],
            project=project,
            runtime=runtime,
            created_at=data["created_at"],
            source_report=data.get("source_report"),
        )

    def load_state(self, task_id: str) -> TaskState:
        data = read_json(self.run_dir(task_id) / "state.json")
        stages = {}
        for name, value in data.get("stages", {}).items():
            stages[name] = StageState(**value)
        return TaskState(
            task_id=data["task_id"],
            status=data.get("status", "created"),
            current_stage=data.get("current_stage", "created"),
            attempt=data.get("attempt", 1),
            stages=stages,
            agent_session_id=data.get("agent_session_id"),
            last_safe_stage=data.get("last_safe_stage"),
            report_path=data.get("report_path"),
        )

    def save_state(self, state: TaskState) -> None:
        write_json(self.run_dir(state.task_id) / "state.json", state)

    def update_stage(
        self,
        task_id: str,
        stage: str,
        status: str,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self.load_state(task_id)
        state.current_stage = stage
        state.status = "running" if status == "running" else state.status
        if status in ("passed", "failed", "uncertain"):
            state.last_safe_stage = stage if status == "passed" else state.last_safe_stage
        state.stages[stage] = StageState(
            status=status,
            updated_at=utc_now_iso(),
            result_path=result_path,
            error=error,
            progress=progress or {},
        )
        if stage == "report" and result_path:
            state.report_path = result_path
            state.status = "completed"
        self.save_state(state)
        self.events(task_id).append(stage, "stage_update", {"status": status, "result_path": result_path, "error": error, "progress": progress or {}})

    def save_result(self, task_id: str, stage: str, data: Any) -> Path:
        path = self.run_dir(task_id) / "reports" / ("%s_result.json" % stage)
        write_json(path, data)
        return path

    def events(self, task_id: str) -> EventLog:
        return EventLog(self.run_dir(task_id) / "events.jsonl")

    def task_summary(self, task_id: str) -> Dict[str, Any]:
        state = self.load_state(task_id)
        return to_plain(state)
