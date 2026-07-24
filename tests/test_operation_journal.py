"""Tests for OperationJournal, RecoveryService, and schemas.

Phase 1 tests: Journal lifecycle, stable IDs, state transitions,
secret exclusion, and RecoveryService orchestration.
"""
import json
import pytest
from pathlib import Path

from auto_harness.recovery.schemas import (
    canonical_json,
    compute_operation_id,
    OPERATION_STATUSES,
    RECONCILE_DECISIONS,
)
from auto_harness.recovery.journal import OperationJournal, ALLOWED_TRANSITIONS
from auto_harness.recovery.service import RecoveryService


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_record(
    task_id="test_task",
    stage="runner",
    action="start_service",
    resource_type="local_process",
    operation_id=None,
    **overrides,
):
    """Build a minimal operation record dict."""
    normalized_input = {"command": "python app.py", "port": 8501}
    resource_identity = {
        "command_hash": "abc123",
        "repo_path": "/workspace/repo",
        "expected_port": "8501",
    }
    if operation_id is None:
        operation_id = compute_operation_id(
            task_id, stage, action, normalized_input, resource_identity,
        )
    record = {
        "schema_version": 1,
        "operation_id": operation_id,
        "task_id": task_id,
        "stage": stage,
        "action": action,
        "resource_type": resource_type,
        "resource_identity": resource_identity,
        "observed_resource": {},
        "normalized_input_hash": canonical_json(normalized_input),
        "status": "planned",
        "attempt": 0,
        "started_at": "",
        "committed_at": "",
        "last_checked_at": "",
        "result_artifacts": [],
        "error": "",
    }
    record.update(overrides)
    return record


# -------------------------------------------------------------------
# Schema Tests
# -------------------------------------------------------------------

class TestCanonicalJson:
    def test_key_ordering(self):
        assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'

    def test_escaping(self):
        result = canonical_json({"key": "val\"ue"})
        assert "val" in result

    def test_nested(self):
        result = canonical_json({"a": {"c": 3, "b": 2}})
        assert result == '{"a":{"b":2,"c":3}}'

    def test_deterministic(self):
        """Same input always produces same output."""
        data = {"z": 1, "a": {"b": 2, "a": 1}, "m": [3, 2]}
        assert canonical_json(data) == canonical_json(data)

    def test_no_whitespace(self):
        result = canonical_json({"a": 1, "b": 2})
        assert " " not in result


class TestComputeOperationId:
    def test_same_inputs_same_id(self):
        """Stable ID: identical inputs always produce the same ID."""
        id1 = compute_operation_id(
            "task1", "runner", "start",
            {"cmd": "python app.py"}, {"type": "process"},
        )
        id2 = compute_operation_id(
            "task1", "runner", "start",
            {"cmd": "python app.py"}, {"type": "process"},
        )
        assert id1 == id2

    def test_changed_inputs_different_id(self):
        """Changed inputs produce a different ID."""
        id1 = compute_operation_id(
            "task1", "runner", "start",
            {"cmd": "python app.py"}, {"type": "process"},
        )
        id2 = compute_operation_id(
            "task1", "runner", "start",
            {"cmd": "python other.py"}, {"type": "process"},
        )
        assert id1 != id2

    def test_id_length(self):
        """ID is 24 hex characters (12 bytes of SHA-256)."""
        oid = compute_operation_id("t", "s", "a", {}, {})
        assert len(oid) == 24
        assert oid.isalnum()

    def test_dict_order_invariant(self):
        """Dict key ordering doesn't affect the ID."""
        id1 = compute_operation_id(
            "t", "s", "a",
            {"a": 1, "b": 2}, {"x": "y"},
        )
        id2 = compute_operation_id(
            "t", "s", "a",
            {"b": 2, "a": 1}, {"x": "y"},
        )
        assert id1 == id2

    def test_secret_not_in_hash(self):
        """Secret values must be excluded before calling compute_operation_id.
        This test verifies the hash is computed from what's passed in;
        callers must sanitize before calling."""
        id1 = compute_operation_id(
            "t", "s", "a",
            {"key": "public"}, {"type": "x"},
        )
        id2 = compute_operation_id(
            "t", "s", "a",
            {"key": "SECRET_VALUE"}, {"type": "x"},
        )
        # Different inputs → different IDs (caller must sanitize before calling)
        assert id1 != id2

    def test_different_stage_different_id(self):
        id1 = compute_operation_id("t", "runner", "start", {}, {})
        id2 = compute_operation_id("t", "model_prepare", "download", {}, {})
        assert id1 != id2

    def test_different_task_different_id(self):
        id1 = compute_operation_id("task1", "s", "a", {}, {})
        id2 = compute_operation_id("task2", "s", "a", {}, {})
        assert id1 != id2


# -------------------------------------------------------------------
# OperationJournal Tests
# -------------------------------------------------------------------

class TestOperationJournal:
    def test_create_writes_snapshot(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        created = journal.create(record)
        assert created["status"] == "planned"
        assert created["operation_id"] == record["operation_id"]
        path = journal.record_path(record["operation_id"])
        assert path.exists()

    def test_create_appends_events(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        assert journal.events_path.exists()
        lines = journal.events_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        event = json.loads(lines[0])
        assert event["type"] == "created"

    def test_create_idempotent(self, tmp_path):
        """Creating the same operation twice returns the existing record."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        first = journal.create(record)
        second = journal.create(record)
        assert first["operation_id"] == second["operation_id"]

    def test_create_collision_raises(self, tmp_path):
        """Same operation_id but different hash raises ValueError."""
        journal = OperationJournal(tmp_path)
        record1 = make_record(operation_id="abc123")
        record1["normalized_input_hash"] = "hash1"
        journal.create(record1)
        record2 = make_record(operation_id="abc123")
        record2["normalized_input_hash"] = "hash2"
        with pytest.raises(ValueError, match="identity collision"):
            journal.create(record2)

    def test_transition_valid(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        updated = journal.transition(record["operation_id"], "running")
        assert updated["status"] == "running"

    def test_transition_invalid_raises(self, tmp_path):
        """Invalid transition (committed -> running) raises ValueError."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.transition(record["operation_id"], "committed")
        with pytest.raises(ValueError, match="invalid transition"):
            journal.transition(record["operation_id"], "running")

    def test_transition_appends_event(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        lines = journal.events_path.read_text().strip().splitlines()
        events = [json.loads(line) for line in lines]
        types = [e["type"] for e in events]
        assert "created" in types
        assert "transition" in types

    def test_transition_with_updates(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        updated = journal.transition(
            record["operation_id"], "running",
            started_at="2025-01-01T00:00:00Z",
            attempt=1,
        )
        assert updated["started_at"] == "2025-01-01T00:00:00Z"
        assert updated["attempt"] == 1

    def test_load_nonexistent(self, tmp_path):
        journal = OperationJournal(tmp_path)
        assert journal.load("nonexistent") is None

    def test_recover_running_marks_unknown(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        recovered = journal.recover_running(record["operation_id"])
        assert recovered["status"] == "unknown"

    def test_recover_non_running_unchanged(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        recovered = journal.recover_running(record["operation_id"])
        assert recovered["status"] == "planned"

    def test_invalid_operation_id_rejected(self, tmp_path):
        journal = OperationJournal(tmp_path)
        with pytest.raises(ValueError, match="invalid operation_id"):
            journal.record_path("")
        with pytest.raises(ValueError, match="invalid operation_id"):
            journal.record_path("../../../etc/passwd")

    def test_committed_is_terminal(self, tmp_path):
        """Committed operations cannot transition to any other state."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.transition(record["operation_id"], "committed")
        for target in ("running", "planned", "failed"):
            with pytest.raises(ValueError, match="invalid transition"):
                journal.transition(record["operation_id"], target)

    def test_conflict_is_terminal(self, tmp_path):
        """Conflict operations cannot transition to any other state."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.transition(record["operation_id"], "unknown")
        journal.transition(record["operation_id"], "conflict")
        for target in ("running", "planned", "retryable"):
            with pytest.raises(ValueError, match="invalid transition"):
                journal.transition(record["operation_id"], target)

    def test_unknown_requires_reconcile(self, tmp_path):
        """Unknown operations can go to retryable (after reconcile), not to running directly."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.recover_running(record["operation_id"])
        # Unknown -> running is NOT allowed; must go through retryable
        with pytest.raises(ValueError, match="invalid transition"):
            journal.transition(record["operation_id"], "running")
        # Unknown -> retryable IS allowed
        updated = journal.transition(record["operation_id"], "retryable")
        assert updated["status"] == "retryable"
        # retryable -> running IS allowed
        final = journal.transition(record["operation_id"], "running")
        assert final["status"] == "running"

    def test_failed_can_retry(self, tmp_path):
        """Failed operations can transition to retryable."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.transition(record["operation_id"], "failed")
        updated = journal.transition(record["operation_id"], "retryable")
        assert updated["status"] == "retryable"

    def test_manual_can_retry_or_fail(self, tmp_path):
        """Manual operations can transition to retryable or failed."""
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.transition(record["operation_id"], "unknown")
        manual = journal.transition(record["operation_id"], "manual")
        assert manual["status"] == "manual"
        retryable = journal.transition(record["operation_id"], "retryable")
        assert retryable["status"] == "retryable"


# -------------------------------------------------------------------
# RecoveryService Tests
# -------------------------------------------------------------------

class FakeReconciler:
    """Reconciler that returns a configurable decision."""
    def __init__(self, decision="retry", reason="test", **observed):
        self.decision = decision
        self.reason = reason
        self.observed = observed
        self.calls = []

    def reconcile(self, operation):
        self.calls.append(operation)
        return {
            "decision": self.decision,
            "observed_state": self.observed,
            "reason": self.reason,
            "evidence_paths": [],
        }


class TestRecoveryService:
    def test_prepare_creates_journal_record(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        assert prepared["status"] == "planned"

    def test_reconcile_dispatches_to_reconciler(self, tmp_path):
        journal = OperationJournal(tmp_path)
        reconciler = FakeReconciler(decision="reuse", reason="already done")
        service = RecoveryService(journal, {"local_process": reconciler})
        record = make_record(resource_type="local_process")
        prepared = service.prepare(record)
        result = service.reconcile(prepared)
        assert result["decision"] == "reuse"
        assert len(reconciler.calls) == 1

    def test_reconcile_unknown_type_returns_manual(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record(resource_type="unknown_type")
        prepared = service.prepare(record)
        result = service.reconcile(prepared)
        assert result["decision"] == "manual"
        assert result["reason"] == "reconciler_not_registered"

    def test_apply_decision_reuse(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "reuse", "observed_state": {}, "reason": "done"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "committed"

    def test_apply_decision_retry(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "retry", "observed_state": {}, "reason": "not found"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "running"
        assert updated["attempt"] == 1

    def test_apply_decision_continue_increments_attempt(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "continue", "observed_state": {"offset": 4096}, "reason": "partial"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "running"
        assert updated["attempt"] == 1

    def test_apply_decision_conflict(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "conflict", "observed_state": {}, "reason": "pid reused"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "conflict"

    def test_apply_decision_manual(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "manual", "observed_state": {}, "reason": "uncertain"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "manual"

    def test_apply_decision_cleanup_then_retry(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "cleanup_then_retry", "observed_state": {}, "reason": "config changed"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "manual"  # Needs approval

    def test_apply_decision_unknown_defaults_to_manual(self, tmp_path):
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        reconcile_result = {"decision": "unknown_decision", "observed_state": {}, "reason": "???"}
        updated = service.apply_decision(prepared, reconcile_result)
        assert updated["status"] == "manual"

    def test_unknown_operation_goes_through_retryable(self, tmp_path):
        """Unknown → retryable → running (never skips)."""
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        journal.transition(record["operation_id"], "running")
        journal.recover_running(record["operation_id"])
        loaded = journal.load(record["operation_id"])
        assert loaded["status"] == "unknown"
        reconcile_result = {"decision": "continue", "observed_state": {}, "reason": "partial"}
        updated = service.apply_decision(loaded, reconcile_result)
        assert updated["status"] == "running"


# -------------------------------------------------------------------
# Committed not re-executed test
# -------------------------------------------------------------------

class TestCommittedNotReExecuted:
    def test_committed_operation_is_not_executed_twice(self, tmp_path):
        """A committed operation should not be executed again."""
        journal = OperationJournal(tmp_path)
        service = RecoveryService(journal, {})
        record = make_record()
        prepared = service.prepare(record)
        journal.transition(record["operation_id"], "running")
        journal.transition(record["operation_id"], "committed")
        # Second call: prepare returns the committed record
        second = service.prepare(record)
        assert second["status"] == "committed"


# -------------------------------------------------------------------
# begin() atomic entry tests (Task 3)
# -------------------------------------------------------------------

class TestJournalBegin:
    """begin() atomically persists an operation as running."""

    def test_begin_creates_running_atomically(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        result = journal.begin(record)
        assert result["status"] == "running"
        assert result["attempt"] == 1
        assert result["started_at"]
        # Persisted to disk
        loaded = journal.load(record["operation_id"])
        assert loaded["status"] == "running"
        assert loaded["attempt"] == 1

    def test_begin_increments_retryable_attempt(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        # Create, run, crash to unknown, then to retryable
        journal.create(record)
        journal.transition(record["operation_id"], "running")
        journal.recover_running(record["operation_id"])  # running -> unknown
        journal.transition(record["operation_id"], "retryable")  # unknown -> retryable
        loaded = journal.load(record["operation_id"])
        assert loaded["status"] == "retryable"
        # begin should increment attempt and set running
        result = journal.begin(record)
        assert result["status"] == "running"
        assert result["attempt"] == 1

    def test_begin_increments_planned_attempt(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        loaded = journal.load(record["operation_id"])
        assert loaded["status"] == "planned"
        assert loaded["attempt"] == 0
        result = journal.begin(record)
        assert result["status"] == "running"
        assert result["attempt"] == 1

    def test_begin_returns_existing_running_without_increment(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        first = journal.begin(record)
        assert first["attempt"] == 1
        # Calling begin again on already-running should be a no-op
        second = journal.begin(record)
        assert second["status"] == "running"
        assert second["attempt"] == 1

    def test_begin_returns_committed_without_change(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.begin(record)
        journal.transition(record["operation_id"], "committed")
        result = journal.begin(record)
        assert result["status"] == "committed"

    def test_begin_rejects_hash_collision(self, tmp_path):
        journal = OperationJournal(tmp_path)
        record = make_record()
        journal.create(record)
        # Same operation_id, different hash
        colliding = make_record(operation_id=record["operation_id"])
        colliding["normalized_input_hash"] = "different_hash_value"
        with pytest.raises(ValueError, match="operation identity collision"):
            journal.begin(colliding)
