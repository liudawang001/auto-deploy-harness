"""Deterministic, auditable fault injection for recovery validation."""
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from auto_harness.utils.atomic import FileLock, atomic_write_text
from auto_harness.utils.time import utc_now_iso


FAULT_WINDOWS = frozenset({
    "before_side_effect",
    "after_side_effect_before_commit",
    "after_commit_before_checkpoint",
    "after_provider_before_call_ledger",
    "after_call_ledger_before_policy",
    "after_journal_begin_before_side_effect",
    "after_side_effect_before_journal_commit",
    "after_journal_commit_before_tool_result",
    "after_tool_result_before_provider_feedback",
    "after_provider_feedback_before_checkpoint",
})


class InjectedFault(RuntimeError):
    """Raised when a configured recovery fault point is reached."""

    def __init__(self, point: str, operation_id: str) -> None:
        self.point = point
        self.operation_id = operation_id
        super().__init__("injected recovery fault at %s" % point)


class FaultInjector:
    """Raises each configured fault point at most once per run.

    The marker is persisted before the exception is raised. A restarted
    process therefore observes the same configuration without repeatedly
    crashing at the same point.
    """

    def __init__(self, points: Iterable[str] = ()) -> None:
        normalized = set()
        for raw in points or ():
            point = str(raw).strip()
            if not point:
                continue
            self._validate_point(point)
            normalized.add(point)
        self.points = frozenset(normalized)

    @staticmethod
    def point(stage: str, window: str) -> str:
        if window not in FAULT_WINDOWS:
            raise ValueError("unsupported fault window: %s" % window)
        return "%s:%s" % (stage, window)

    @staticmethod
    def _validate_point(point: str) -> None:
        parts = point.split(":", 1)
        if len(parts) != 2 or not parts[0] or parts[1] not in FAULT_WINDOWS:
            raise ValueError("invalid recovery fault point: %s" % point)

    def raise_if_configured(
        self,
        run_dir: Path,
        task_id: str,
        stage: str,
        window: str,
        operation_id: str,
    ) -> bool:
        """Persist an audit marker and raise once when the point is enabled."""
        point = self.point(stage, window)
        if point not in self.points:
            return False
        if not operation_id:
            raise ValueError("operation_id is required for fault injection")

        root = Path(run_dir) / "operations" / "fault_injections"
        root.mkdir(parents=True, exist_ok=True)
        marker_name = hashlib.sha256(
            ("%s:%s" % (operation_id, point)).encode("utf-8")
        ).hexdigest()[:24]
        marker_path = root / ("%s.json" % marker_name)

        with FileLock(marker_path):
            if marker_path.exists():
                return False
            event = {
                "schema_version": 1,
                "task_id": str(task_id),
                "operation_id": str(operation_id),
                "point": point,
                "injected_at": utc_now_iso(),
            }
            atomic_write_text(
                marker_path,
                json.dumps(event, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            )
            self._append_event(root.parent / "fault_injections.jsonl", event)

        raise InjectedFault(point, operation_id)

    @staticmethod
    def _append_event(path: Path, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n"
        with FileLock(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
