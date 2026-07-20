"""Tests for execute checkpoint resume with capability map.

Phase 6 tests: capability map replaces global boolean block,
committed operations not re-executed, unknown operations reconcile first,
conflict stops safely, supported types resume, unsupported types stay blocked.
"""
import json
import pytest
from pathlib import Path

from auto_harness.controllers.langgraph import (
    LangGraphController,
    SIDE_EFFECT_STAGES,
    STAGE_TO_CAPABILITY,
    can_resume_stage,
    build_initial_state,
)
from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.recovery.journal import OperationJournal
from auto_harness.recovery.service import RecoveryService
from auto_harness.recovery.schemas import compute_operation_id, canonical_json
from auto_harness.recovery.download import DownloadReconciler
from auto_harness.recovery.process import ProcessReconciler, ProcessProbe
from auto_harness.recovery.docker import DockerReconciler
from auto_harness.models.base import read_json, write_json


# -------------------------------------------------------------------
# can_resume_stage Tests
# -------------------------------------------------------------------

class TestCanResumeStage:
    def test_non_side_effect_always_allowed(self):
        """Non-side-effect stages can always resume."""
        assert can_resume_stage("analyze", {}, dry_run=False) is True
        assert can_resume_stage("resource_plan", {}, dry_run=False) is True
        assert can_resume_stage("env_solve", {}, dry_run=False) is True
        assert can_resume_stage("verify", {}, dry_run=False) is True

    def test_side_effect_blocked_without_capability(self):
        """Side-effect stages blocked when capability is False."""
        caps = {"download": False, "local_process": False, "docker_service": False, "dependency_install": False}
        assert can_resume_stage("model_prepare", caps, dry_run=False) is False
        assert can_resume_stage("runner", caps, dry_run=False) is False
        assert can_resume_stage("env_deploy", caps, dry_run=False) is False

    def test_side_effect_allowed_with_capability(self):
        """Side-effect stages allowed when corresponding capability is True."""
        caps = {"download": True, "local_process": False, "docker_service": False, "dependency_install": False}
        assert can_resume_stage("model_prepare", caps, dry_run=False) is True
        assert can_resume_stage("runner", caps, dry_run=False) is False

    def test_dry_run_always_allowed(self):
        """All stages allowed in dry_run mode."""
        caps = {"download": False, "local_process": False, "docker_service": False, "dependency_install": False}
        assert can_resume_stage("model_prepare", caps, dry_run=True) is True
        assert can_resume_stage("runner", caps, dry_run=True) is True
        assert can_resume_stage("env_deploy", caps, dry_run=True) is True

    def test_runner_with_docker_capability(self):
        """Runner stage allowed with docker_service capability."""
        caps = {"download": False, "local_process": False, "docker_service": True, "dependency_install": False}
        assert can_resume_stage("runner", caps, dry_run=False) is True

    def test_runner_with_local_process_capability(self):
        """Runner stage allowed with local_process capability."""
        caps = {"download": False, "local_process": True, "docker_service": False, "dependency_install": False}
        assert can_resume_stage("runner", caps, dry_run=False) is True

    def test_env_deploy_with_dependency_capability(self):
        """env_deploy stage allowed with dependency_install capability."""
        caps = {"download": False, "local_process": False, "docker_service": False, "dependency_install": True}
        assert can_resume_stage("env_deploy", caps, dry_run=False) is True


# -------------------------------------------------------------------
# Integration: Journal + Reconciler + Capability Map
# -------------------------------------------------------------------

class TestExecuteCheckpointResume:
    def test_committed_not_reexecuted(self, tmp_path):
        """Committed operation should not be re-executed."""
        journal = OperationJournal(tmp_path)
        normalized_input = {"command": "python app.py"}
        resource_identity = {"command_hash": "abc", "repo_path": "/repo", "expected_port": "8501"}
        operation_id = compute_operation_id(
            "task1", "runner", "start", normalized_input, resource_identity,
        )
        record = {
            "operation_id": operation_id,
            "task_id": "task1",
            "stage": "runner",
            "action": "start",
            "resource_type": "local_process",
            "resource_identity": resource_identity,
            "normalized_input_hash": canonical_json(normalized_input),
        }
        # Create, run, and commit
        created = journal.create(record)
        journal.transition(operation_id, "running")
        journal.transition(operation_id, "committed", committed_at="2025-01-01T00:00:00Z")

        # Second call: prepare returns the committed record
        second = journal.create(record)
        assert second["status"] == "committed"

    def test_unknown_reconciles_first(self, tmp_path):
        """Unknown operation must reconcile before re-execution."""
        journal = OperationJournal(tmp_path)
        normalized_input = {"command": "python app.py"}
        resource_identity = {"command_hash": "abc", "repo_path": "/repo", "expected_port": "8501"}
        operation_id = compute_operation_id(
            "task1", "runner", "start", normalized_input, resource_identity,
        )
        record = {
            "operation_id": operation_id,
            "task_id": "task1",
            "stage": "runner",
            "action": "start",
            "resource_type": "local_process",
            "resource_identity": resource_identity,
            "normalized_input_hash": canonical_json(normalized_input),
        }
        journal.create(record)
        journal.transition(operation_id, "running")
        # Simulate crash: recover running marks unknown
        journal.recover_running(operation_id)
        loaded = journal.load(operation_id)
        assert loaded["status"] == "unknown"
        # Cannot go directly to running; must go through reconcile → retryable → running
        with pytest.raises(ValueError, match="invalid transition"):
            journal.transition(operation_id, "running")
        # Correct path: unknown → retryable → running
        journal.transition(operation_id, "retryable")
        journal.transition(operation_id, "running")
        loaded = journal.load(operation_id)
        assert loaded["status"] == "running"

    def test_conflict_stops_safely(self, tmp_path):
        """Conflict operations are terminal — no auto-cleanup."""
        journal = OperationJournal(tmp_path)
        normalized_input = {"source": "huggingface", "repo_id": "org/model"}
        resource_identity = {"target_path": "/cache/model.bin", "expected_size": "1024"}
        operation_id = compute_operation_id(
            "task1", "model_prepare", "download", normalized_input, resource_identity,
        )
        record = {
            "operation_id": operation_id,
            "task_id": "task1",
            "stage": "model_prepare",
            "action": "download",
            "resource_type": "model_download",
            "resource_identity": resource_identity,
            "normalized_input_hash": canonical_json(normalized_input),
        }
        journal.create(record)
        journal.transition(operation_id, "running")
        journal.transition(operation_id, "unknown")
        # Conflict is terminal
        journal.transition(operation_id, "conflict")
        loaded = journal.load(operation_id)
        assert loaded["status"] == "conflict"
        # Cannot transition out of conflict
        with pytest.raises(ValueError, match="invalid transition"):
            journal.transition(operation_id, "running")

    def test_capability_map_in_initial_state(self, tmp_path):
        """Initial state includes recovery_capabilities map."""
        from tests.test_langgraph_controller import make_context
        ctx = make_context(tmp_path)
        state = build_initial_state(ctx, max_replans=2)
        assert "recovery_capabilities" in state
        caps = state["recovery_capabilities"]
        assert caps["download"] is False
        assert caps["local_process"] is False
        assert caps["docker_service"] is False
        assert caps["dependency_install"] is False

    def test_reconciler_with_journal_integration(self, tmp_path):
        """Full flow: journal create → reconcile → apply_decision for download."""
        journal = OperationJournal(tmp_path)
        # Create a target file to reconcile
        target = tmp_path / "cache" / "model.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00" * 1024)
        normalized_input = {"source": "huggingface", "repo_id": "org/model"}
        resource_identity = {
            "target_path": str(target),
            "expected_size": "1024",
            "sha256": "",
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "main",
            "relative_path": "model.bin",
            "etag": "",
        }
        operation_id = compute_operation_id(
            "task1", "model_prepare", "download", normalized_input, resource_identity,
        )
        record = {
            "operation_id": operation_id,
            "task_id": "task1",
            "stage": "model_prepare",
            "action": "download",
            "resource_type": "model_download",
            "resource_identity": resource_identity,
            "normalized_input_hash": canonical_json(normalized_input),
        }
        reconciler = DownloadReconciler()
        service = RecoveryService(journal, {"model_download": reconciler})
        # Prepare
        prepared = service.prepare(record)
        assert prepared["status"] == "planned"
        # Reconcile — file exists and size matches
        result = service.reconcile(prepared)
        assert result["decision"] == "reuse"
        # Apply — marks committed
        updated = service.apply_decision(prepared, result)
        assert updated["status"] == "committed"

    def test_events_jsonl_accumulates(self, tmp_path):
        """Multiple operations produce multiple events in events.jsonl."""
        journal = OperationJournal(tmp_path)
        for i in range(3):
            normalized_input = {"index": i}
            resource_identity = {"type": "test"}
            operation_id = compute_operation_id(
                "task1", "stage", "action_%d" % i, normalized_input, resource_identity,
            )
            record = {
                "operation_id": operation_id,
                "task_id": "task1",
                "stage": "stage",
                "action": "action_%d" % i,
                "resource_type": "test",
                "resource_identity": resource_identity,
                "normalized_input_hash": canonical_json(normalized_input),
            }
            journal.create(record)
        lines = journal.events_path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            event = json.loads(line)
            assert event["type"] == "created"

    def test_operation_journal_cross_process(self, tmp_path):
        """Journal state persists and can be read from a subprocess."""
        import subprocess
        import sys
        journal = OperationJournal(tmp_path)
        normalized_input = {"key": "value"}
        resource_identity = {"type": "test"}
        operation_id = compute_operation_id(
            "task1", "stage", "action", normalized_input, resource_identity,
        )
        record = {
            "operation_id": operation_id,
            "task_id": "task1",
            "stage": "stage",
            "action": "action",
            "resource_type": "test",
            "resource_identity": resource_identity,
            "normalized_input_hash": canonical_json(normalized_input),
        }
        journal.create(record)
        journal.transition(operation_id, "running")

        # Verify snapshot file readable from subprocess
        snapshot_path = journal.record_path(operation_id)
        assert snapshot_path.exists()
        result = subprocess.run(
            [sys.executable, "-c",
             "import json; d = json.load(open('%s')); print(d['status'])" % snapshot_path],
            capture_output=True, text=True, timeout=10,
        )
        assert "running" in result.stdout


# -------------------------------------------------------------------
# Capability Map Upgrade Tests
# -------------------------------------------------------------------

class TestCapabilityMapUpgrade:
    def test_all_capabilities_disabled_by_default(self, tmp_path):
        """By default, all recovery capabilities are disabled."""
        from tests.test_langgraph_controller import make_context
        ctx = make_context(tmp_path)
        state = build_initial_state(ctx, max_replans=2)
        caps = state["recovery_capabilities"]
        for key in ("download", "local_process", "docker_service", "dependency_install"):
            assert caps[key] is False

    def test_supported_type_enables_resume(self):
        """When a capability is True, the corresponding stage can resume."""
        caps = {"download": True, "local_process": True, "docker_service": True, "dependency_install": False}
        assert can_resume_stage("model_prepare", caps, dry_run=False) is True
        assert can_resume_stage("runner", caps, dry_run=False) is True
        assert can_resume_stage("env_deploy", caps, dry_run=False) is False

    def test_partial_capabilities(self):
        """Only enabled capabilities allow resume; others stay blocked."""
        caps = {"download": True, "local_process": False, "docker_service": False, "dependency_install": False}
        assert can_resume_stage("model_prepare", caps, dry_run=False) is True
        assert can_resume_stage("runner", caps, dry_run=False) is False
        assert can_resume_stage("env_deploy", caps, dry_run=False) is False
