"""OwnedResourceCleanupExecutor: real cleanup for docker_service and local_process.

Only removes resources that pass ownership verification.
Never uses shell=True. Never pkill by process name.
First version supports docker_service and local_process only.
"""
import subprocess
from typing import Any, Callable, Dict, Optional


class OwnedResourceCleanupExecutor:
    """Cleanup executor that verifies ownership before removing resources.

    Supports:
    - docker_service: removes container by ID with ownership label check
    - local_process: refuses cleanup if process identity cannot be verified
    """

    def __init__(self, run_command=None):
        self.run_command = run_command or subprocess.run

    def remove_owned_resource(
        self,
        operation: Dict[str, Any],
        check: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Remove an owned resource after ownership verification.

        Args:
            operation: Operation record with resource_type, task_id, etc.
            check: Ownership check result with 'owned' flag and details.
            dry_run: If True, plan but don't execute.

        Returns:
            Dict with success, executed, dry_run, and planned_command or error.
        """
        if not check.get("owned"):
            return {
                "success": False,
                "reason": "resource_not_owned",
            }

        resource_type = operation.get("resource_type", "")

        if resource_type == "docker_service":
            return self._cleanup_docker(operation, check, dry_run=dry_run)

        if resource_type == "local_process":
            return self._cleanup_process(operation, check, dry_run=dry_run)

        return {
            "success": False,
            "reason": "unsupported_resource_type:%s" % resource_type,
        }

    def _cleanup_docker(
        self,
        operation: Dict[str, Any],
        check: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Remove a Docker container after verifying ownership.

        Conditions:
        - check.owned == True
        - container_id is non-empty
        - task_id label matches operation.task_id
        """
        container_id = check.get("container_id", "")
        if not container_id:
            return {
                "success": False,
                "reason": "empty_container_id",
            }

        # Both ownership labels must be present and match the journal record.
        check_task_id = check.get("task_id", "")
        operation_task_id = operation.get("task_id", "")
        if not check_task_id or not operation_task_id:
            return {
                "success": False,
                "reason": "task_id_label_missing",
            }
        if check_task_id != operation_task_id:
            return {
                "success": False,
                "reason": "task_id_label_mismatch",
            }
        check_operation_id = check.get("operation_id", "")
        operation_id = operation.get("operation_id", "")
        if not check_operation_id or not operation_id:
            return {
                "success": False,
                "reason": "operation_id_label_missing",
            }
        if check_operation_id != operation_id:
            return {
                "success": False,
                "reason": "operation_id_label_mismatch",
            }

        cmd = ["docker", "rm", "-f", container_id]

        if dry_run:
            return {
                "success": True,
                "executed": False,
                "dry_run": True,
                "planned_command": cmd,
            }

        try:
            result = self.run_command(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "executed": True,
                "exit_code": result.returncode,
                "command": cmd,
                "stdout": result.stdout[:500],
                "stderr": result.stderr[:500],
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "cleanup_command_failed",
                "error": str(exc)[:500],
            }

    def _cleanup_process(
        self,
        operation: Dict[str, Any],
        check: Dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Remove a local process after verifying identity.

        First version: refuses cleanup if process identity cannot be
        strongly verified (start_time + command hash match).

        We do NOT pkill by process name.
        """
        pid = check.get("pid", 0)
        if not isinstance(pid, int) or pid <= 0:
            return {
                "success": False,
                "reason": "invalid_pid",
            }

        recorded = operation.get("observed_resource", {})
        identity = operation.get("resource_identity", {})
        expected_pid = int(recorded.get("pid") or 0)
        expected_start_time = recorded.get("process_start_time", "")
        expected_command_hash = identity.get("command_hash", "")
        actual_start_time = check.get("start_time", "")
        actual_command_hash = check.get("command_hash", "")

        if not (
            expected_pid > 0
            and expected_start_time
            and expected_command_hash
            and actual_start_time
            and actual_command_hash
        ):
            return {
                "success": False,
                "reason": "process_identity_not_strong_enough",
            }
        if pid != expected_pid:
            return {
                "success": False,
                "reason": "process_pid_mismatch",
            }
        if actual_start_time != expected_start_time:
            return {
                "success": False,
                "reason": "process_start_time_mismatch",
            }
        if actual_command_hash != expected_command_hash:
            return {
                "success": False,
                "reason": "process_command_hash_mismatch",
            }

        if dry_run:
            return {
                "success": True,
                "executed": False,
                "dry_run": True,
                "planned_command": ["kill", str(pid)],
            }

        try:
            result = self.run_command(
                ["kill", str(pid)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "success": result.returncode == 0,
                "executed": True,
                "exit_code": result.returncode,
                "command": ["kill", str(pid)],
            }
        except Exception as exc:
            return {
                "success": False,
                "reason": "kill_command_failed",
                "error": str(exc)[:500],
            }
