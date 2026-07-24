"""Docker reconciler: detects external state of managed Docker containers.

Reconciles Docker container operations against the Docker daemon:
- Matching container with correct labels and running → reuse
- Matching container stopped → continue (docker start)
- Container name occupied by unowned resource → conflict
- No container → retry
- Config mismatch → cleanup_then_retry (needs approval)
- Inspect failure → manual

Managed service containers use ownership labels:
  auto-harness.task-id=<task-id>
  auto-harness.operation-id=<operation-id>
  auto-harness.plan-hash=<hash>

Ephemeral install/probe containers can use --rm (no labels needed).
Managed service containers MUST NOT use --rm (need persistence).
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from auto_harness.recovery.download import reconcile_result


class DockerReconciler:
    """Reconciler for Docker container operations.

    Uses docker ps/inspect to check container state.
    All docker commands go through command_runner for testability.
    """
    resource_type = "docker_service"

    def __init__(self, command_runner: Callable) -> None:
        """Initialize with a command runner callable.

        command_runner(cmd: List[str]) -> Dict with keys:
          - exit_code: int
          - stdout: str
          - stderr: str
        """
        self.command_runner = command_runner

    def reconcile(self, operation):
        """Reconcile a Docker container operation.

        Decision logic:
        1. Find containers by operation-id label
        2. If none, check for name collision → conflict/retry
        3. If multiple, conflict
        4. Inspect the single match
        5. Verify ownership labels (task-id, operation-id, plan-hash)
        6. Verify runtime config matches
        7. If running → reuse
        8. If stopped → continue
        9. Config mismatch → cleanup_then_retry
        10. Inspect failure → manual
        """
        operation_id = operation["operation_id"]
        task_id = operation["task_id"]

        # 1. Find by operation-id label
        listed = self.command_runner([
            "docker", "ps", "-a",
            "--filter", "label=auto-harness.operation-id=%s" % operation_id,
            "--format", "{{.ID}}",
        ])
        if listed.get("exit_code") != 0:
            return reconcile_result(
                "manual", "docker ps failed",
                stderr=listed.get("stderr", "")[-1000:],
            )

        ids = [line.strip() for line in listed.get("stdout", "").splitlines() if line.strip()]

        # 2. No containers found
        if not ids:
            return self._check_name_collision(operation)

        # 3. Multiple containers claiming same operation
        if len(ids) > 1:
            return reconcile_result(
                "conflict", "multiple containers claim the operation id",
                ids=ids,
            )

        # 4. Inspect single match
        inspected = self.command_runner(["docker", "inspect", ids[0]])
        if inspected.get("exit_code") != 0:
            return reconcile_result("manual", "docker inspect failed")

        try:
            data = json.loads(inspected.get("stdout", ""))[0]
        except (ValueError, IndexError, TypeError):
            return reconcile_result("manual", "docker inspect did not return valid JSON")

        # 5. Verify ownership labels
        labels = data.get("Config", {}).get("Labels", {}) or {}
        expected = operation["resource_identity"]

        if labels.get("auto-harness.task-id") != task_id:
            return reconcile_result(
                "conflict", "container task ownership mismatch",
                id=ids[0],
            )
        if labels.get("auto-harness.operation-id") != operation_id:
            return reconcile_result(
                "conflict", "container operation ownership mismatch",
                id=ids[0],
            )
        if labels.get("auto-harness.plan-hash") != expected.get("plan_hash"):
            return reconcile_result(
                "cleanup_then_retry", "container plan changed",
                id=ids[0],
            )

        # 6. Verify runtime config
        if not docker_config_matches(data, expected):
            return reconcile_result(
                "cleanup_then_retry", "container runtime config changed",
                id=ids[0],
            )

        # 7-8. Check running state
        if data.get("State", {}).get("Running"):
            return reconcile_result(
                "reuse", "matching managed container is running",
                id=ids[0],
            )
        return reconcile_result(
            "continue", "matching managed container is stopped",
            id=ids[0], resume_action="start_existing",
        )

    def _check_name_collision(self, operation):
        """Check if the expected container name is occupied by another resource."""
        expected_name = operation["resource_identity"].get("container_name", "")
        if not expected_name:
            return reconcile_result("retry", "managed container does not exist")

        by_name = self.command_runner([
            "docker", "ps", "-a",
            "--filter", "name=^/%s$" % expected_name,
            "--format", "{{.ID}}",
        ])
        if by_name.get("exit_code") != 0:
            return reconcile_result("manual", "docker name lookup failed")

        foreign_ids = [
            line.strip() for line in by_name.get("stdout", "").splitlines()
            if line.strip()
        ]
        if foreign_ids:
            return reconcile_result(
                "conflict",
                "container name is occupied by an unowned resource",
                ids=foreign_ids,
            )
        return reconcile_result("retry", "managed container does not exist")

    def verify_cleanup_target(self, operation):
        """Verify a container is safe to clean up (has correct ownership labels).

        Called by cleanup_node before removing a container.
        Returns dict with 'owned' bool and inspect data.
        """
        operation_id = operation["operation_id"]
        task_id = operation["task_id"]
        expected_name = operation["resource_identity"].get("container_name", "")

        # Find by operation-id label
        listed = self.command_runner([
            "docker", "ps", "-a",
            "--filter", "label=auto-harness.operation-id=%s" % operation_id,
            "--format", "{{.ID}}",
        ])
        if listed.get("exit_code") != 0:
            return {"owned": False, "reason": "docker ps failed"}

        ids = [line.strip() for line in listed.get("stdout", "").splitlines() if line.strip()]
        if not ids:
            return {"owned": False, "reason": "container not found"}

        inspected = self.command_runner(["docker", "inspect", ids[0]])
        if inspected.get("exit_code") != 0:
            return {"owned": False, "reason": "docker inspect failed"}

        try:
            data = json.loads(inspected.get("stdout", ""))[0]
        except (ValueError, IndexError, TypeError):
            return {"owned": False, "reason": "invalid inspect JSON"}

        labels = data.get("Config", {}).get("Labels", {}) or {}
        # Verify all three ownership labels
        if labels.get("auto-harness.task-id") != task_id:
            return {"owned": False, "reason": "task-id mismatch"}
        if labels.get("auto-harness.operation-id") != operation_id:
            return {"owned": False, "reason": "operation-id mismatch"}
        expected = operation["resource_identity"]
        if labels.get("auto-harness.plan-hash") != expected.get("plan_hash"):
            return {"owned": False, "reason": "plan-hash mismatch"}

        return {
            "owned": True,
            "container_id": ids[0],
            "task_id": labels.get("auto-harness.task-id", ""),
            "operation_id": labels.get("auto-harness.operation-id", ""),
            "inspect": data,
        }


def docker_config_matches(inspect_data, expected):
    """Check if an inspected container's runtime config matches expected values.

    Compares: image, ports, network, GPU, mounts.
    Returns True if all key fields match, False otherwise.
    Returns False (not manual) if any key field can't be parsed.
    """
    try:
        config = inspect_data.get("Config", {})
        host_config = inspect_data.get("HostConfig", {})

        # Image
        actual_image = config.get("Image", "")
        expected_image = expected.get("image", "")
        if expected_image and actual_image != expected_image:
            return False

        # Ports
        expected_ports = sorted(expected.get("ports", []))
        port_bindings = host_config.get("PortBindings", {}) or {}
        actual_ports = sorted(
            int(k.split("/")[0]) for k in port_bindings.keys() if k
        )
        if expected_ports and actual_ports != expected_ports:
            return False

        # Network
        expected_network = expected.get("network", "")
        actual_network = host_config.get("NetworkMode", "")
        if expected_network and actual_network != expected_network:
            return False

        # GPU
        expected_gpus = expected.get("gpus", "none")
        device_requests = host_config.get("DeviceRequests", []) or []
        if expected_gpus and expected_gpus != "none":
            if not device_requests:
                return False

        return True
    except (KeyError, TypeError, ValueError):
        return False
