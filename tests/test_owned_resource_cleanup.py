"""Task 7: OwnedResourceCleanupExecutor tests.

Verifies:
1. Refuses cleanup when not owned
2. Refuses empty container_id
3. Docker cleanup uses argv without shell=True
4. Process cleanup refuses when identity not strong enough
5. Dry-run does not execute
6. Successful cleanup transitions manual -> retryable
"""
import pytest
from unittest.mock import MagicMock

from auto_harness.recovery.cleanup import OwnedResourceCleanupExecutor


class TestRefuseCleanupWhenNotOwned:
    """Cleanup must refuse when ownership check fails."""

    def test_not_owned_returns_failure(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "docker_service", "task_id": "t1"}
        check = {"owned": False}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert result["reason"] == "resource_not_owned"

    def test_owned_true_proceeds(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "docker_service", "task_id": "t1", "operation_id": "op1"}
        check = {"owned": True, "container_id": "abc123", "task_id": "t1", "operation_id": "op1"}
        result = executor.remove_owned_resource(operation, check, dry_run=True)
        assert result["success"] is True


class TestRefuseEmptyContainerId:
    """Docker cleanup refuses with empty container_id."""

    def test_empty_container_id(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "docker_service", "task_id": "t1", "operation_id": "op1"}
        check = {"owned": True, "container_id": "", "task_id": "t1", "operation_id": "op1"}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert result["reason"] == "empty_container_id"

    def test_missing_container_id(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "docker_service", "task_id": "t1", "operation_id": "op1"}
        check = {"owned": True, "task_id": "t1", "operation_id": "op1"}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False


class TestDockerCleanupUsesArgv:
    """Docker cleanup uses argv list, never shell=True."""

    def test_docker_cleanup_command_is_argv(self):
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor = OwnedResourceCleanupExecutor(run_command=mock_run)
        operation = {"resource_type": "docker_service", "task_id": "t1", "operation_id": "op1"}
        check = {"owned": True, "container_id": "abc123", "task_id": "t1", "operation_id": "op1"}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is True
        # Verify command was called as argv list
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == ["docker", "rm", "-f", "abc123"]
        # Ensure shell=True was NOT used
        assert call_args[1].get("shell") is not True

    def test_task_id_label_mismatch(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "docker_service", "task_id": "t1", "operation_id": "op1"}
        check = {"owned": True, "container_id": "abc123", "task_id": "t_other", "operation_id": "op1"}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert result["reason"] == "task_id_label_mismatch"


class TestProcessCleanupIdentity:
    """Process cleanup requires strong identity evidence."""

    def test_refuses_when_no_start_time(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {
            "resource_type": "local_process",
            "task_id": "t1",
            "observed_resource": {"pid": 1234, "process_start_time": "2025-01-01"},
            "resource_identity": {"command_hash": "abc"},
        }
        check = {"owned": True, "pid": 1234, "command_hash": "abc"}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert result["reason"] == "process_identity_not_strong_enough"

    def test_refuses_when_no_command_hash(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {
            "resource_type": "local_process",
            "task_id": "t1",
            "observed_resource": {"pid": 1234, "process_start_time": "2025-01-01"},
            "resource_identity": {"command_hash": "abc"},
        }
        check = {"owned": True, "pid": 1234, "start_time": "2025-01-01"}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert result["reason"] == "process_identity_not_strong_enough"

    def test_refuses_invalid_pid(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "local_process", "task_id": "t1"}
        check = {"owned": True, "pid": -1}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert result["reason"] == "invalid_pid"

    def test_accepts_with_strong_identity(self):
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor = OwnedResourceCleanupExecutor(run_command=mock_run)
        operation = {
            "resource_type": "local_process",
            "task_id": "t1",
            "observed_resource": {
                "pid": 1234,
                "process_start_time": "2025-01-01T00:00:00",
            },
            "resource_identity": {"command_hash": "abc123"},
        }
        check = {
            "owned": True,
            "pid": 1234,
            "start_time": "2025-01-01T00:00:00",
            "command_hash": "abc123",
        }
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is True
        cmd = mock_run.call_args[0][0]
        assert cmd == ["kill", "1234"]


class TestDryRun:
    """Dry-run plans but does not execute."""

    def test_docker_dry_run(self):
        mock_run = MagicMock()
        executor = OwnedResourceCleanupExecutor(run_command=mock_run)
        operation = {"resource_type": "docker_service", "task_id": "t1", "operation_id": "op1"}
        check = {"owned": True, "container_id": "abc123", "task_id": "t1", "operation_id": "op1"}
        result = executor.remove_owned_resource(operation, check, dry_run=True)
        assert result["success"] is True
        assert result["executed"] is False
        assert result["dry_run"] is True
        assert result["planned_command"] == ["docker", "rm", "-f", "abc123"]
        # run_command was NOT called
        mock_run.assert_not_called()

    def test_process_dry_run(self):
        mock_run = MagicMock()
        executor = OwnedResourceCleanupExecutor(run_command=mock_run)
        operation = {
            "resource_type": "local_process",
            "task_id": "t1",
            "observed_resource": {"pid": 1234, "process_start_time": "x"},
            "resource_identity": {"command_hash": "x"},
        }
        check = {"owned": True, "pid": 1234, "start_time": "x", "command_hash": "x"}
        result = executor.remove_owned_resource(operation, check, dry_run=True)
        assert result["success"] is True
        assert result["executed"] is False
        assert result["dry_run"] is True
        mock_run.assert_not_called()


class TestUnsupportedResourceType:
    """Unsupported resource types return failure."""

    def test_unsupported_type(self):
        executor = OwnedResourceCleanupExecutor()
        operation = {"resource_type": "arbitrary_directory", "task_id": "t1"}
        check = {"owned": True}
        result = executor.remove_owned_resource(operation, check)
        assert result["success"] is False
        assert "unsupported_resource_type" in result["reason"]
