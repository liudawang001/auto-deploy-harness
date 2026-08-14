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
        if "runtime_plan_hash" in expected or "model_hash" in expected:
            config_ok = docker_model_runtime_config_matches(data, expected)
        else:
            config_ok = docker_config_matches(data, expected)
        if not config_ok:
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


def _size_bytes(value) -> int:
    """Parse a size string ('8g', '512m', '1g') or int into bytes."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if not text:
        return 0
    units = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    try:
        if text[-1] in units:
            return int(float(text[:-1]) * units[text[-1]])
        return int(float(text))
    except (ValueError, IndexError):
        return 0


def docker_model_runtime_config_matches(inspect_data, expected) -> bool:
    """Check a managed vLLM container's runtime config against expected values.

    Compares image + digest, model identity label, the exact read-only
    ``/models/current`` mount source, GPU device index, shm/memory/CPU/PID
    limits, read-only rootfs and user. A check is enforced only when the
    corresponding expected key is present.
    """
    try:
        config = inspect_data.get("Config", {})
        host_config = inspect_data.get("HostConfig", {})
        labels = config.get("Labels", {}) or {}

        # Image and digest
        actual_image = str(config.get("Image", ""))
        expected_image = str(expected.get("image", ""))
        if expected_image and actual_image != expected_image:
            return False
        expected_digest = str(expected.get("image_digest", ""))
        if expected_digest and expected_digest not in actual_image:
            return False

        # Model identity label
        expected_model_hash = str(expected.get("model_hash", ""))
        if expected_model_hash and labels.get("auto-harness.model-hash") != expected_model_hash:
            return False

        # Exact read-only model mount (both sides resolved so symlinked roots
        # such as /tmp -> /private/tmp compare consistently).
        expected_model_path = str(expected.get("model_host_path", ""))
        if expected_model_path:
            mounts = inspect_data.get("Mounts", []) or []
            wanted = str(Path(expected_model_path).resolve())
            found = any(
                str(m.get("Destination", "")) == "/models/current"
                and str(Path(str(m.get("Source", ""))).resolve()) == wanted
                and str(m.get("Mode", "")).lower().startswith("ro")
                for m in mounts
                if isinstance(m, dict)
            )
            if not found:
                return False

        # GPU device index
        expected_gpus = expected.get("gpu_indexes", [])
        if expected_gpus:
            device_requests = host_config.get("DeviceRequests", []) or []
            device_ids = []
            for request in device_requests:
                if isinstance(request, dict):
                    device_ids.extend(str(i) for i in (request.get("DeviceIDs") or []))
            if not any(str(i) in device_ids for i in expected_gpus):
                return False

        # shm size
        expected_shm = str(expected.get("shm_size", ""))
        if expected_shm:
            if _size_bytes(host_config.get("ShmSize", 0)) != _size_bytes(expected_shm):
                return False

        # memory
        expected_memory = str(expected.get("memory", ""))
        if expected_memory:
            if int(host_config.get("Memory", 0) or 0) != _size_bytes(expected_memory):
                return False

        # CPU (NanoCpus = cpus * 1e9)
        expected_cpus = expected.get("cpus")
        if expected_cpus:
            wanted_nano = int(float(expected_cpus) * 1e9)
            if int(host_config.get("NanoCpus", 0) or 0) != wanted_nano:
                return False

        # PID limit
        expected_pids = expected.get("pids_limit")
        if expected_pids:
            if int(host_config.get("PidsLimit", 0) or 0) != int(expected_pids):
                return False

        # read-only rootfs
        if expected.get("read_only_rootfs") is not None:
            if bool(host_config.get("ReadonlyRootfs", False)) != bool(expected["read_only_rootfs"]):
                return False

        # user
        expected_user = str(expected.get("user", ""))
        if expected_user and str(host_config.get("User", "")) != expected_user:
            return False

        return True
    except (KeyError, TypeError, ValueError):
        return False
