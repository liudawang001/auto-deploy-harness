"""Tests for DockerReconciler and managed container lifecycle.

Phase 4 tests: Docker container reconciliation, ownership labels,
config matching, name collision, and cleanup verification.
"""
import json
import pytest
from unittest.mock import MagicMock

from auto_harness.recovery.docker import (
    DockerReconciler,
    docker_config_matches,
)
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_docker_operation(
    container_name="auto-harness-abc123-8501",
    plan_hash="hash123",
    image="python:3.10-slim",
    ports=None,
    network="bridge",
    gpus="none",
    task_id="test_task",
    operation_id=None,
):
    """Build a Docker container operation dict for testing."""
    if ports is None:
        ports = [8501]
    identity = {
        "container_name": container_name,
        "plan_hash": plan_hash,
        "image": image,
        "ports": ports,
        "network": network,
        "gpus": gpus,
    }
    normalized_input = {"image": image, "ports": ports}
    if operation_id is None:
        operation_id = compute_operation_id(
            task_id, "runner", "start_service",
            normalized_input, identity,
        )
    return {
        "operation_id": operation_id,
        "task_id": task_id,
        "stage": "runner",
        "action": "start_service",
        "resource_type": "docker_service",
        "resource_identity": identity,
        "observed_resource": {},
        "normalized_input_hash": canonical_json(normalized_input),
        "status": "running",
    }


class FakeCommandRunner:
    """Command runner that returns pre-configured results."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        key = tuple(cmd[:3])
        if key in self.responses:
            return self.responses[key]
        # Try full command match
        for pattern, response in self.responses.items():
            if isinstance(pattern, tuple) and len(cmd) >= len(pattern):
                if tuple(cmd[:len(pattern)]) == pattern:
                    return response
        return {"exit_code": 1, "stdout": "", "stderr": "not configured"}


def make_inspect_response(
    container_id="abc123",
    running=True,
    labels=None,
    image="python:3.10-slim",
    ports=None,
    network="bridge",
    gpus="none",
):
    """Build a docker inspect response."""
    if labels is None:
        labels = {
            "auto-harness.task-id": "test_task",
            "auto-harness.operation-id": "op123",
            "auto-harness.plan-hash": "hash123",
        }
    if ports is None:
        ports = [8501]
    port_bindings = {}
    for p in ports:
        port_bindings["%d/tcp" % p] = [{"HostIp": "127.0.0.1", "HostPort": str(p)}]
    device_requests = []
    if gpus and gpus != "none":
        device_requests = [{"Driver": "nvidia", "Count": -1}]
    data = [{
        "Id": container_id,
        "Config": {
            "Image": image,
            "Labels": labels,
        },
        "State": {
            "Running": running,
            "Status": "running" if running else "exited",
        },
        "HostConfig": {
            "PortBindings": port_bindings,
            "NetworkMode": network,
            "DeviceRequests": device_requests,
        },
    }]
    return {
        "exit_code": 0,
        "stdout": json.dumps(data),
        "stderr": "",
    }


# -------------------------------------------------------------------
# DockerReconciler Tests
# -------------------------------------------------------------------

class TestDockerReconciler:
    def test_reuse_when_matching_running(self):
        """Matching container with correct labels and running → reuse."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                running=True,
                labels={
                    "auto-harness.task-id": "test_task",
                    "auto-harness.operation-id": "op123",
                    "auto-harness.plan-hash": "hash123",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"
        assert "running" in result["reason"]

    def test_continue_when_matching_stopped(self):
        """Matching container but stopped → continue."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                running=False,
                labels={
                    "auto-harness.task-id": "test_task",
                    "auto-harness.operation-id": "op123",
                    "auto-harness.plan-hash": "hash123",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "continue"
        assert result["observed_state"]["resume_action"] == "start_existing"

    def test_conflict_when_task_id_mismatch(self):
        """Container has wrong task-id label → conflict."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                labels={
                    "auto-harness.task-id": "DIFFERENT_TASK",
                    "auto-harness.operation-id": "op123",
                    "auto-harness.plan-hash": "hash123",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "task ownership" in result["reason"]

    def test_conflict_when_operation_id_mismatch(self):
        """Container has wrong operation-id label → conflict."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                labels={
                    "auto-harness.task-id": "test_task",
                    "auto-harness.operation-id": "DIFFERENT_OP",
                    "auto-harness.plan-hash": "hash123",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"

    def test_cleanup_then_retry_when_plan_changed(self):
        """Container plan-hash changed → cleanup_then_retry."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                labels={
                    "auto-harness.task-id": "test_task",
                    "auto-harness.operation-id": "op123",
                    "auto-harness.plan-hash": "DIFFERENT_HASH",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "cleanup_then_retry"

    def test_retry_when_no_container(self):
        """No container found by label → retry (after checking name)."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "", "stderr": "",
            },
            # Name collision check also returns empty
        })
        # Override name collision check to return empty
        runner.responses[("docker", "ps", "-a")] = {
            "exit_code": 0, "stdout": "", "stderr": "",
        }
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"

    def test_conflict_when_name_occupied(self):
        """Container name occupied by unowned resource → conflict."""
        op = make_docker_operation(operation_id="op123")
        # First ps (by label) returns nothing, second (by name) finds something
        call_count = [0]
        def dynamic_runner(cmd):
            call_count[0] += 1
            if "--filter" in cmd and "label=" in " ".join(cmd):
                return {"exit_code": 0, "stdout": "", "stderr": ""}
            if "--filter" in cmd and "name=" in " ".join(cmd):
                return {"exit_code": 0, "stdout": "foreign_id\n", "stderr": ""}
            return {"exit_code": 1, "stdout": "", "stderr": ""}
        reconciler = DockerReconciler(dynamic_runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "unowned resource" in result["reason"]

    def test_manual_when_docker_ps_fails(self):
        """docker ps command fails → manual."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 1, "stdout": "", "stderr": "error",
            },
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"

    def test_manual_when_inspect_fails(self):
        """docker inspect fails → manual."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): {
                "exit_code": 1, "stdout": "", "stderr": "error",
            },
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"

    def test_manual_when_inspect_invalid_json(self):
        """docker inspect returns invalid JSON → manual."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): {
                "exit_code": 0, "stdout": "not json", "stderr": "",
            },
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"

    def test_conflict_when_multiple_containers(self):
        """Multiple containers with same operation-id label → conflict."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\ndef456\n", "stderr": "",
            },
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "multiple" in result["reason"]

    def test_resource_type(self):
        assert DockerReconciler.resource_type == "docker_service"


# -------------------------------------------------------------------
# Docker Config Matches Tests
# -------------------------------------------------------------------

class TestDockerConfigMatches:
    def test_matching_config(self):
        inspect_data = {
            "Config": {"Image": "python:3.10-slim"},
            "HostConfig": {
                "PortBindings": {"8501/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8501"}]},
                "NetworkMode": "bridge",
                "DeviceRequests": [],
            },
        }
        expected = {
            "image": "python:3.10-slim",
            "ports": [8501],
            "network": "bridge",
            "gpus": "none",
        }
        assert docker_config_matches(inspect_data, expected) is True

    def test_image_mismatch(self):
        inspect_data = {
            "Config": {"Image": "python:3.9-slim"},
            "HostConfig": {"PortBindings": {}, "NetworkMode": "bridge", "DeviceRequests": []},
        }
        expected = {"image": "python:3.10-slim", "ports": [], "network": "bridge", "gpus": "none"}
        assert docker_config_matches(inspect_data, expected) is False

    def test_port_mismatch(self):
        inspect_data = {
            "Config": {"Image": "python:3.10-slim"},
            "HostConfig": {
                "PortBindings": {"9000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "9000"}]},
                "NetworkMode": "bridge",
                "DeviceRequests": [],
            },
        }
        expected = {"image": "python:3.10-slim", "ports": [8501], "network": "bridge", "gpus": "none"}
        assert docker_config_matches(inspect_data, expected) is False

    def test_gpu_expected_but_missing(self):
        inspect_data = {
            "Config": {"Image": "python:3.10-slim"},
            "HostConfig": {"PortBindings": {}, "NetworkMode": "bridge", "DeviceRequests": []},
        }
        expected = {"image": "python:3.10-slim", "ports": [], "network": "bridge", "gpus": "all"}
        assert docker_config_matches(inspect_data, expected) is False

    def test_empty_expected_matches(self):
        inspect_data = {
            "Config": {"Image": "python:3.10-slim"},
            "HostConfig": {"PortBindings": {}, "NetworkMode": "bridge", "DeviceRequests": []},
        }
        expected = {}  # No constraints
        assert docker_config_matches(inspect_data, expected) is True


# -------------------------------------------------------------------
# Cleanup Verification Tests
# -------------------------------------------------------------------

class TestVerifyCleanupTarget:
    def test_owned_container(self):
        """Container with all three ownership labels → owned=True."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                labels={
                    "auto-harness.task-id": "test_task",
                    "auto-harness.operation-id": "op123",
                    "auto-harness.plan-hash": "hash123",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.verify_cleanup_target(op)
        assert result["owned"] is True
        assert result["container_id"] == "abc123"

    def test_unowned_container(self):
        """Container with wrong task-id → owned=False."""
        op = make_docker_operation(operation_id="op123")
        runner = FakeCommandRunner({
            ("docker", "ps", "-a"): {
                "exit_code": 0, "stdout": "abc123\n", "stderr": "",
            },
            ("docker", "inspect"): make_inspect_response(
                labels={
                    "auto-harness.task-id": "DIFFERENT",
                    "auto-harness.operation-id": "op123",
                    "auto-harness.plan-hash": "hash123",
                },
            ),
        })
        reconciler = DockerReconciler(runner)
        result = reconciler.verify_cleanup_target(op)
        assert result["owned"] is False
