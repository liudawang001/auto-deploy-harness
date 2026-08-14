"""Startup Readiness Gate (Document B Phase B5).

Waits for a managed vLLM container to reach a *verified* ready state. A running
container, an open port, or a log keyword is only an intermediate state — ready
requires every condition to hold simultaneously:

1. the managed container is still running,
2. container labels + runtime plan hash still match,
3. ``/v1/models`` returns 2xx and valid JSON,
4. the model list contains the served model name,
5. the current time is within the startup deadline.

The loop uses ``time.monotonic`` and bounded polling; every dependency
(command runner, urlopen, clock, sleeper) is injectable for offline tests.
"""
import json
import time
import urllib.request
from typing import Callable, Optional

from auto_harness.model_runtime.schemas import ModelRuntimeStartupEvidence
from auto_harness.utils.time import utc_now_iso

MODEL_STARTUP_STATES = (
    "created",
    "starting",
    "loading_weights",
    "ready",
    "failed",
    "timed_out",
    "container_exited",
)

# (category, marker) pairs evaluated in order against the container log tail.
_FAILURE_MARKERS = (
    ("cuda_out_of_memory", "cuda out of memory"),
    ("driver_cuda_mismatch", "cuda driver version"),
    ("driver_cuda_mismatch", "driver is insufficient"),
    ("unsupported_architecture", "unsupported architecture"),
    ("invalid_model_path", "no such file or directory"),
    ("missing_model_file", "does not appear to have a file"),
    ("missing_model_file", "no model files found"),
    ("checksum_or_corruption", "checksum"),
    ("checksum_or_corruption", "corrupt safetensors"),
    ("port_conflict", "address already in use"),
    ("permission_denied", "permission denied"),
    ("container_runtime_failure", "oci runtime"),
    ("unexpected_remote_download", "hf_hub_download"),
    ("unexpected_remote_download", "from_pretrained"),
    ("unexpected_remote_download", "huggingface.co"),
)


def classify_model_startup_failure(text: str) -> str:
    """Classify a vLLM startup log tail into a failure category ('' if none)."""
    lowered = (text or "").lower()
    for category, marker in _FAILURE_MARKERS:
        if marker in lowered:
            return category
    return ""


class ModelRuntimeReadiness:
    """Deterministic readiness waiter for a managed vLLM container."""

    def __init__(
        self,
        command_runner: Optional[Callable] = None,
        urlopen: Optional[Callable] = None,
        clock: Optional[Callable] = None,
        sleeper: Optional[Callable] = None,
    ) -> None:
        self.command_runner = command_runner or self._subprocess_runner()
        self.urlopen = urlopen or urllib.request.urlopen
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep

    def wait(
        self,
        *,
        runtime_plan,
        task_id: str,
        operation_id: str,
        container_id: str,
        container_created_at: str = "",
        labels=None,
        gpu_before=None,
        poll_interval: float = 1.0,
        progress_callback: Optional[Callable] = None,
        max_rounds: Optional[int] = None,
    ) -> ModelRuntimeStartupEvidence:
        labels = dict(labels or {})
        start_mono = self._clock()
        deadline = start_mono + float(runtime_plan.startup_timeout_seconds)
        started_at = utc_now_iso()
        rounds = 0
        last_log_tail = ""

        while True:
            rounds += 1
            now = self._clock()

            if max_rounds is not None and rounds > max_rounds:
                return self._evidence(
                    status="timed_out",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((now - start_mono) * 1000),
                    failure_reason="readiness rounds exhausted",
                    log_tail=last_log_tail,
                )

            # 1. Container must still be running.
            running, inspect_data, inspect_error = self._container_inspect(container_id)
            if inspect_error:
                return self._evidence(
                    status="failed",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((now - start_mono) * 1000),
                    failure_reason="container_runtime_failure: %s" % inspect_error[-200:],
                    log_tail=last_log_tail,
                )
            if not running:
                category = classify_model_startup_failure(last_log_tail) or "container_runtime_failure"
                return self._evidence(
                    status="container_exited",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((now - start_mono) * 1000),
                    failure_reason="container exited before ready (category=%s)" % category,
                    log_tail=last_log_tail,
                )

            # 2. Labels + plan hash still match.
            if not self._labels_match(inspect_data, labels):
                return self._evidence(
                    status="failed",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((now - start_mono) * 1000),
                    failure_reason="container labels or plan hash mismatch",
                    log_tail=last_log_tail,
                )

            # 3. /v1/models returns 2xx + valid JSON containing the served model.
            status, models = self._probe_models(runtime_plan)
            if status == 200 and self._served_model_present(models, runtime_plan.served_model_name):
                ready_mono = self._clock()
                return self._evidence(
                    status="ready",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((ready_mono - start_mono) * 1000),
                    models_endpoint_status=200,
                    models_endpoint_match=True,
                )

            # 4. Classify the refreshed log tail for a deterministic failure.
            last_log_tail = self._log_tail(container_id)
            category = classify_model_startup_failure(last_log_tail)
            if category:
                return self._evidence(
                    status="failed",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((now - start_mono) * 1000),
                    failure_reason="startup failure: %s" % category,
                    log_tail=last_log_tail,
                    models_endpoint_status=status or 0,
                )

            # 5. Deadline.
            if now >= deadline:
                return self._evidence(
                    status="timed_out",
                    runtime_plan=runtime_plan,
                    task_id=task_id,
                    operation_id=operation_id,
                    container_id=container_id,
                    container_created_at=container_created_at,
                    labels=labels,
                    gpu_before=gpu_before,
                    started_at=started_at,
                    latency_ms=int((now - start_mono) * 1000),
                    failure_reason="startup deadline exceeded",
                    log_tail=last_log_tail,
                    models_endpoint_status=status or 0,
                )

            if progress_callback:
                progress_callback({
                    "state": "loading_weights",
                    "elapsed_ms": int((now - start_mono) * 1000),
                    "round": rounds,
                    "last_log_category": category or "none",
                })
            self._sleeper(poll_interval)

    # -- probes / helpers ----------------------------------------------

    @staticmethod
    def _subprocess_runner():
        import subprocess

        def _run(cmd):
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"exit_code": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}

        return _run

    def _container_inspect(self, container_id):
        result = self.command_runner(["docker", "inspect", container_id])
        if result.get("exit_code") != 0:
            return False, None, result.get("stderr", "")
        try:
            data = json.loads(result.get("stdout") or "")[0]
        except (ValueError, IndexError, TypeError):
            return False, None, "invalid docker inspect json"
        return bool(data.get("State", {}).get("Running")), data, ""

    def _log_tail(self, container_id, lines: int = 50) -> str:
        result = self.command_runner(["docker", "logs", "--tail", str(lines), container_id])
        if result.get("exit_code") != 0:
            return ""
        return (result.get("stdout", "") + result.get("stderr", ""))[-8000:]

    def _probe_models(self, runtime_plan):
        url = "http://%s:%s%s" % (
            runtime_plan.expected_host,
            runtime_plan.expected_port,
            runtime_plan.health_path,
        )
        try:
            req = urllib.request.Request(url, method="GET")
            with self.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                status = getattr(resp, "status", None) or getattr(resp, "code", None)
            models = None
            if status == 200:
                try:
                    models = json.loads(body)
                except ValueError:
                    models = None
            return status, models
        except Exception:
            return None, None

    @staticmethod
    def _served_model_present(models, served_name) -> bool:
        if not isinstance(models, dict):
            return False
        data = models.get("data")
        if not isinstance(data, list):
            return False
        return any(isinstance(item, dict) and item.get("id") == served_name for item in data)

    @staticmethod
    def _labels_match(inspect_data, labels) -> bool:
        actual = inspect_data.get("Config", {}).get("Labels", {}) or {}
        for key, value in labels.items():
            if value and actual.get(key) != value:
                return False
        return True

    @staticmethod
    def _evidence(
        *,
        status: str,
        runtime_plan,
        task_id: str,
        operation_id: str,
        container_id: str,
        container_created_at: str,
        labels,
        gpu_before,
        started_at: str,
        latency_ms: int,
        failure_reason: str = "",
        log_tail: str = "",
        models_endpoint_status: int = 0,
        models_endpoint_match: bool = False,
    ) -> ModelRuntimeStartupEvidence:
        from auto_harness.model_runtime.schemas import hash_payload

        log_tail_hash = hash_payload(log_tail or "") if log_tail else ""
        return ModelRuntimeStartupEvidence(
            status=status,
            task_id=task_id,
            operation_id=operation_id,
            runtime_plan_hash=runtime_plan.plan_hash,
            container_id=container_id,
            container_name=runtime_plan.container_name,
            container_created_at=container_created_at,
            container_labels=dict(labels),
            image_digest=runtime_plan.image_digest,
            model_identity=runtime_plan.model_identity,
            served_model_name=runtime_plan.served_model_name,
            started_at=started_at,
            ready_at=utc_now_iso() if status == "ready" else "",
            startup_latency_ms=int(latency_ms),
            models_endpoint_status=int(models_endpoint_status),
            models_endpoint_match=bool(models_endpoint_match),
            gpu_before=dict(gpu_before or {}),
            gpu_ready=dict(gpu_before or {}),
            log_tail_hash=log_tail_hash,
            evidence_paths=[],
            failure_reason=failure_reason,
        )
