"""Tests for ProcessReconciler and ProcessProbe.

Phase 3 tests: process reconciliation, PID reuse detection,
command hash verification, port readiness, and cwd matching.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.recovery.process import (
    ProcessProbe,
    ProcessReconciler,
    sha256_text,
    normalize_command,
)
from auto_harness.recovery.download import reconcile_result
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_process_operation(
    pid=12345,
    command_hash=None,
    repo_path="/workspace/repo",
    expected_port=8501,
    process_start_time="Mon Jan  1 00:00:00 2025",
    actual_command="python app.py",
):
    """Build a local process operation dict for testing."""
    if command_hash is None:
        command_hash = sha256_text(normalize_command(actual_command))
    identity = {
        "command_hash": command_hash,
        "repo_path": repo_path,
        "expected_port": str(expected_port),
    }
    observed_resource = {
        "pid": str(pid),
        "process_start_time": process_start_time,
        "command": actual_command,
        "cwd": repo_path,
    }
    normalized_input = {"command": actual_command, "port": expected_port}
    operation_id = compute_operation_id(
        "test_task", "runner", "start_service",
        normalized_input, identity,
    )
    return {
        "operation_id": operation_id,
        "task_id": "test_task",
        "stage": "runner",
        "action": "start_service",
        "resource_type": "local_process",
        "resource_identity": identity,
        "observed_resource": observed_resource,
        "normalized_input_hash": canonical_json(normalized_input),
        "status": "running",
    }


class FakeProcessProbe:
    """Configurable process probe for testing."""
    def __init__(self, observations=None):
        self.observations = observations or {}

    def observe(self, pid):
        return self.observations.get(pid, {"exists": False})


def fake_port_probe(host, port):
    """Port probe that always returns True."""
    return True


def fake_port_probe_closed(host, port):
    """Port probe that always returns False."""
    return False


# -------------------------------------------------------------------
# Utility Tests
# -------------------------------------------------------------------

class TestSha256Text:
    def test_deterministic(self):
        assert sha256_text("hello") == sha256_text("hello")

    def test_different_inputs(self):
        assert sha256_text("hello") != sha256_text("world")

    def test_hex_length(self):
        assert len(sha256_text("test")) == 64


class TestNormalizeCommand:
    def test_collapse_whitespace(self):
        assert normalize_command("python   app.py") == "python app.py"

    def test_preserve_arguments(self):
        assert normalize_command("python app.py --port 8501") == "python app.py --port 8501"

    def test_no_change_needed(self):
        assert normalize_command("python app.py") == "python app.py"


# -------------------------------------------------------------------
# ProcessProbe Tests (lightweight, using real ps)
# -------------------------------------------------------------------

class TestProcessProbe:
    def test_nonexistent_pid(self):
        probe = ProcessProbe()
        result = probe.observe(99999999)
        assert result["exists"] is False

    def test_invalid_pid(self):
        probe = ProcessProbe()
        result = probe.observe(0)
        assert result["exists"] is False

    def test_negative_pid(self):
        probe = ProcessProbe()
        result = probe.observe(-1)
        assert result["exists"] is False


# -------------------------------------------------------------------
# ProcessReconciler Tests
# -------------------------------------------------------------------

class TestProcessReconciler:
    def test_retry_when_process_gone(self):
        """PID doesn't exist → retry."""
        probe = FakeProcessProbe({12345: {"exists": False}})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(pid=12345)
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
        assert "no longer exists" in result["reason"]

    def test_manual_when_identity_incomplete(self):
        """PID exists but identity incomplete → manual."""
        probe = FakeProcessProbe({12345: {"exists": True, "identity_complete": False}})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(pid=12345)
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"
        assert "identity" in result["reason"]

    def test_conflict_on_pid_reuse(self):
        """PID start time doesn't match → conflict (PID reuse)."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "DIFFERENT_START_TIME",
            "command": "python app.py",
            "cwd": "/workspace/repo",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "pid was reused" in result["reason"]

    def test_conflict_on_command_change(self):
        """Command hash doesn't match → conflict."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python OTHER.py",  # Different command
            "cwd": "/workspace/repo",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",  # Original command
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "command changed" in result["reason"]

    def test_manual_when_missing_repo_path(self):
        """Missing expected cwd → manual."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python app.py",
            "cwd": "/workspace/repo",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",
        )
        op["resource_identity"]["repo_path"] = ""  # Missing
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"
        assert "cwd is missing" in result["reason"]

    def test_conflict_on_cwd_change(self):
        """Cwd doesn't match → conflict."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python app.py",
            "cwd": "/DIFFERENT/path",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",
            repo_path="/workspace/repo",
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "cwd changed" in result["reason"]

    def test_manual_when_cwd_unverifiable(self):
        """Cwd can't be verified → manual."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python app.py",
            "cwd": "",  # Can't verify
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",
            repo_path="/workspace/repo",
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"
        assert "cwd cannot be verified" in result["reason"]

    def test_manual_when_port_not_ready(self):
        """Process matches but port not ready → manual."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python app.py",
            "cwd": "/workspace/repo",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe_closed)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",
            repo_path="/workspace/repo",
            expected_port=8501,
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"
        assert "port is not ready" in result["reason"]

    def test_reuse_when_all_match(self):
        """All identity fields match and port ready → reuse."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python app.py",
            "cwd": "/workspace/repo",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",
            repo_path="/workspace/repo",
            expected_port=8501,
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"
        assert "still running" in result["reason"]

    def test_reuse_without_port_check(self):
        """No expected_port → skip port check, reuse if all else matches."""
        probe = FakeProcessProbe({12345: {
            "exists": True,
            "identity_complete": True,
            "start_time": "Mon Jan  1 00:00:00 2025",
            "command": "python app.py",
            "cwd": "/workspace/repo",
        }})
        reconciler = ProcessReconciler(probe, fake_port_probe_closed)
        op = make_process_operation(
            pid=12345,
            process_start_time="Mon Jan  1 00:00:00 2025",
            actual_command="python app.py",
            repo_path="/workspace/repo",
            expected_port=0,  # No port check
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"

    def test_resource_type(self):
        assert ProcessReconciler.resource_type == "local_process"

    def test_no_recorded_pid(self):
        """No recorded PID → retry."""
        probe = FakeProcessProbe()
        reconciler = ProcessReconciler(probe, fake_port_probe)
        op = make_process_operation(pid=0)
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
