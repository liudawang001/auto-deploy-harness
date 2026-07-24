"""Operation Journal: persistent record of side-effect operations.

Each operation gets a JSON snapshot file and events are appended to
a JSONL audit log. Uses FileLock for concurrent access safety and
atomic_write_text for crash-safe snapshot updates.

Storage layout:
  runs/<task-id>/operations/events.jsonl
  runs/<task-id>/operations/<operation-id>.json
"""
import json
import os
from pathlib import Path

from auto_harness.models.base import read_json
from auto_harness.utils.atomic import FileLock, atomic_write_text
from auto_harness.utils.time import utc_now_iso


# Allowed state transitions for operation records
# Note: planned -> committed/failed is kept for legacy migration of old
# artifacts created before begin() existed. New side-effect operations
# go through begin() (planned/none -> running atomically) and should
# never transition planned -> committed/failed directly.
ALLOWED_TRANSITIONS = {
    "planned": {"running", "committed", "failed", "conflict", "manual"},
    "running": {"committed", "failed", "unknown"},
    "unknown": {"committed", "retryable", "failed", "conflict", "manual"},
    # retryable still needs re-observation of external facts
    "retryable": {"running", "committed", "failed", "conflict", "manual"},
    "manual": {"retryable", "failed"},
    "committed": set(),
    "failed": {"retryable"},
    "conflict": set(),
}


class OperationJournal:
    """Persistent journal of side-effect operations.

    Thread-safe via FileLock. Each operation has:
    - A snapshot file (<operation-id>.json) for fast reads
    - Events appended to events.jsonl for audit trail

    The snapshot lock and events lock are separate to avoid deadlock.
    Never hold the snapshot lock while calling a method that acquires
    the same snapshot lock again.
    """

    def __init__(self, run_dir: Path) -> None:
        self.root = Path(run_dir) / "operations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"

    def record_path(self, operation_id: str) -> Path:
        """Get the snapshot file path for an operation ID.

        Validates that the operation_id contains only alphanumeric
        characters and hyphens to prevent path traversal.
        """
        if not operation_id or not operation_id.replace("-", "").isalnum():
            raise ValueError("invalid operation_id")
        return self.root / (operation_id + ".json")

    def load(self, operation_id: str):
        """Load an operation record by ID. Returns None if not found."""
        path = self.record_path(operation_id)
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def create(self, record):
        """Create a new operation record.

        If a record with the same operation_id already exists and has
        the same normalized_input_hash, returns the existing record
        (idempotent). If the hash differs, raises ValueError (identity
        collision).

        Returns the record (new or existing).
        """
        path = self.record_path(record["operation_id"])
        with FileLock(path):
            existing = self.load(record["operation_id"])
            if existing:
                if existing.get("normalized_input_hash") != record.get("normalized_input_hash"):
                    raise ValueError("operation identity collision")
                return existing
            new_record = dict(record)
            new_record["status"] = "planned"
            new_record.setdefault("attempt", 0)
            new_record.setdefault("observed_resource", {})
            new_record.setdefault("started_at", "")
            new_record.setdefault("committed_at", "")
            new_record.setdefault("last_checked_at", "")
            new_record.setdefault("result_artifacts", [])
            new_record.setdefault("error", "")
            new_record["created_at"] = utc_now_iso()
            self._write_snapshot(path, new_record)
            self._append_event("created", new_record)
            return new_record

    def begin(self, record):
        """Atomically persist an operation as running before side effect.

        Unlike create()+transition(), this is a single write that sets
        status=running, preventing the crash window where an operation
        is planned but never entered running.

        If the record already exists:
        - running/committed/manual/conflict: return existing (no-op)
        - planned/retryable: transition to running, increment attempt
        - hash collision: raise ValueError

        If no record exists: create as running with attempt=1.
        """
        operation_id = record["operation_id"]
        path = self.record_path(operation_id)

        with FileLock(path):
            existing = self.load(operation_id)

            if existing:
                if existing.get("normalized_input_hash") != record.get(
                    "normalized_input_hash"
                ):
                    raise ValueError("operation identity collision")

                status = existing.get("status")

                if status in ("running", "committed", "manual", "conflict"):
                    return existing

                if status not in ("planned", "retryable"):
                    return existing

                updated = dict(existing)
                updated["status"] = "running"
                updated["attempt"] = int(existing.get("attempt", 0)) + 1
                updated["started_at"] = utc_now_iso()
                updated["updated_at"] = updated["started_at"]
                self._write_snapshot(path, updated)
                self._append_event("started", {
                    "operation_id": operation_id,
                    "attempt": updated["attempt"],
                    "at": updated["started_at"],
                })
                return updated

            started = dict(record)
            started["status"] = "running"
            started["attempt"] = 1
            started["created_at"] = utc_now_iso()
            started["started_at"] = started["created_at"]
            started.setdefault("observed_resource", {})
            started.setdefault("committed_at", "")
            started.setdefault("last_checked_at", "")
            started.setdefault("result_artifacts", [])
            started.setdefault("error", "")
            self._write_snapshot(path, started)
            self._append_event("started", started)
            return started

    def transition(self, operation_id: str, new_status: str, **updates):
        """Transition an operation to a new status.

        Validates the transition is allowed by ALLOWED_TRANSITIONS.
        Applies any keyword updates to the record.
        Writes the updated snapshot and appends a transition event.

        Returns the updated record.
        """
        path = self.record_path(operation_id)
        with FileLock(path):
            record = self.load(operation_id)
            if not record:
                raise KeyError("operation not found: %s" % operation_id)
            old_status = record["status"]
            allowed = ALLOWED_TRANSITIONS.get(old_status, set())
            if new_status not in allowed:
                raise ValueError(
                    "invalid transition: %s -> %s" % (old_status, new_status)
                )
            record.update(updates)
            record["status"] = new_status
            record["updated_at"] = utc_now_iso()
            self._write_snapshot(path, record)
            self._append_event("transition", {
                "operation_id": operation_id,
                "from": old_status,
                "to": new_status,
                "at": record["updated_at"],
            })
            return record

    def recover_running(self, operation_id: str):
        """Mark a running operation as unknown (crash recovery).

        If the operation is in 'running' status, transitions it to
        'unknown' so that reconciliation must happen before re-execution.
        If not running, returns the record unchanged.
        """
        record = self.load(operation_id)
        if record and record.get("status") == "running":
            return self.transition(operation_id, "unknown")
        return record

    def _write_snapshot(self, path: Path, record):
        """Write an operation snapshot file atomically."""
        text = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        atomic_write_text(path, text)

    def _append_event(self, event_type: str, payload):
        """Append an event to the JSONL audit log.

        Uses a separate lock from the snapshot to avoid deadlock.
        Flushes and fsyncs for durability.
        """
        line = json.dumps(
            {"type": event_type, "data": payload},
            ensure_ascii=False,
        ) + "\n"
        with FileLock(self.events_path):
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
