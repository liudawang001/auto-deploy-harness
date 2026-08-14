"""Phase B5 tests: startup readiness gate.

Covers delayed ready (no premature verify), wrong/old-port model rejection,
container exit -> immediate failure, timeout with saved diagnostics, and the
failure classifier.
"""
import json

import pytest

from auto_harness.model_runtime.readiness import (
    ModelRuntimeReadiness,
    classify_model_startup_failure,
)
from auto_harness.model_runtime.schemas import InferenceRuntimePlan

DIGEST = "sha256:" + "d" * 64
MODEL_ID = "huggingface:org/model@" + "c" * 40


def _plan(**overrides):
    data = dict(
        runtime="vllm",
        deployment_mode="managed_vllm",
        image="vllm/vllm-openai:v0.6.1@" + DIGEST,
        image_digest=DIGEST,
        model_identity=MODEL_ID,
        resolved_model_hash="sha256:model",
        model_host_path="/tmp/model_cache/key",
        model_container_path="/models/current",
        served_model_name="org/model",
        command=["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/current"],
        expected_host="127.0.0.1",
        expected_port=8000,
        startup_timeout_seconds=900,
        request_timeout_seconds=120,
        health_path="/v1/models",
        container_name="auto-harness-abc-vllm",
        gpu_indexes=[0],
        security_profile="model_runtime_v1",
    )
    data.update(overrides)
    plan = InferenceRuntimePlan(**data)
    plan.plan_hash = plan.compute_plan_hash()
    return plan


def _labels(plan, task_id="task-1", operation_id="op-1"):
    return {
        "auto-harness.task-id": task_id,
        "auto-harness.operation-id": operation_id,
        "auto-harness.plan-hash": plan.plan_hash,
        "auto-harness.model-hash": plan.resolved_model_hash,
    }


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self.code = status
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _inspect(running=True, labels=None):
    return {
        "exit_code": 0,
        "stdout": json.dumps([{
            "State": {"Running": running, "Status": "running" if running else "exited"},
            "Config": {"Labels": labels or {}},
        }]),
        "stderr": "",
    }


def _logs(text=""):
    return {"exit_code": 0, "stdout": text, "stderr": ""}


def _readiness(plan, inspect, logs, urlopen, clock=None, sleeper=None, max_rounds=None):
    def runner(cmd):
        if cmd[1] == "inspect":
            return inspect() if callable(inspect) else inspect
        if cmd[1] == "logs":
            return logs() if callable(logs) else logs
        return {"exit_code": 1, "stdout": "", "stderr": "unexpected"}

    return ModelRuntimeReadiness(
        command_runner=runner,
        urlopen=urlopen,
        clock=clock,
        sleeper=sleeper,
    ).wait(
        runtime_plan=plan,
        task_id="task-1",
        operation_id="op-1",
        container_id="cid",
        labels=_labels(plan),
        max_rounds=max_rounds,
    )


def _models_urlopen(ready, served="org/model"):
    def open(req, timeout=5):
        if not ready:
            raise OSError("connection refused")
        return FakeResponse(200, json.dumps({"data": [{"id": served}]}))

    return open


def test_ready_when_models_endpoint_serves_model():
    plan = _plan()
    readiness = _readiness(
        plan,
        inspect=_inspect(running=True, labels=_labels(plan)),
        logs=_logs(""),
        urlopen=_models_urlopen(True),
    )
    assert readiness.status == "ready"
    assert readiness.models_endpoint_match is True
    assert readiness.models_endpoint_status == 200


def test_delayed_ready_not_before_loading_complete():
    plan = _plan(startup_timeout_seconds=900)
    clock = FakeClock(0.0)

    def sleeper(interval):
        clock.value += 1.0

    def urlopen(req, timeout=5):
        # vLLM only serves /v1/models after 3 minutes of weight loading.
        if clock.value < 180.0:
            raise OSError("connection refused")
        return FakeResponse(200, json.dumps({"data": [{"id": "org/model"}]}))

    readiness = ModelRuntimeReadiness(
        command_runner=lambda cmd: _inspect(running=True, labels=_labels(plan)) if cmd[1] == "inspect" else _logs(""),
        urlopen=urlopen,
        clock=clock,
        sleeper=sleeper,
    ).wait(runtime_plan=plan, task_id="task-1", operation_id="op-1", container_id="cid", labels=_labels(plan))

    assert readiness.status == "ready"
    assert readiness.startup_latency_ms >= 180000


def test_old_port_listener_cannot_pass():
    plan = _plan()
    # A stale process answers /v1/models with a different model id.
    readiness = _readiness(
        plan,
        inspect=_inspect(running=True, labels=_labels(plan)),
        logs=_logs(""),
        urlopen=_models_urlopen(True, served="stale/model"),
        sleeper=lambda interval: None,
        max_rounds=3,
    )
    # It never becomes ready within a bounded number of rounds.
    assert readiness.status == "timed_out"


def test_wrong_model_not_ready():
    plan = _plan()
    readiness = _readiness(
        plan,
        inspect=_inspect(running=True, labels=_labels(plan)),
        logs=_logs(""),
        urlopen=_models_urlopen(True, served="org/other-model"),
        sleeper=lambda interval: None,
        max_rounds=3,
    )
    assert readiness.status == "timed_out"


def test_container_exit_immediate_failure():
    plan = _plan()
    readiness = _readiness(
        plan,
        inspect=_inspect(running=False, labels=_labels(plan)),
        logs=_logs(""),
        urlopen=_models_urlopen(False),
    )
    assert readiness.status == "container_exited"
    assert readiness.failure_reason


def test_timeout_saves_diagnostics():
    plan = _plan(startup_timeout_seconds=900)
    clock = FakeClock(0.0)

    def sleeper(interval):
        clock.value += 1000.0  # jump past the deadline immediately

    readiness = ModelRuntimeReadiness(
        command_runner=lambda cmd: _inspect(running=True, labels=_labels(plan)) if cmd[1] == "inspect" else _logs("Loading safetensors checkpoint"),
        urlopen=_models_urlopen(False),
        clock=clock,
        sleeper=sleeper,
    ).wait(runtime_plan=plan, task_id="task-1", operation_id="op-1", container_id="cid", labels=_labels(plan))

    assert readiness.status == "timed_out"
    assert readiness.log_tail_hash


def test_label_mismatch_fails():
    plan = _plan()
    bad_labels = dict(_labels(plan))
    bad_labels["auto-harness.plan-hash"] = "sha256:wrong"
    readiness = _readiness(
        plan,
        inspect=_inspect(running=True, labels=bad_labels),
        logs=_logs(""),
        urlopen=_models_urlopen(False),
    )
    assert readiness.status == "failed"
    assert "mismatch" in readiness.failure_reason


def test_oom_failure_classified():
    plan = _plan()
    readiness = _readiness(
        plan,
        inspect=_inspect(running=True, labels=_labels(plan)),
        logs=_logs("CUDA out of memory. Tried to allocate 8.00 GiB"),
        urlopen=_models_urlopen(False),
    )
    assert readiness.status == "failed"
    assert "cuda_out_of_memory" in readiness.failure_reason


def test_classify_model_startup_failure():
    assert classify_model_startup_failure("CUDA out of memory") == "cuda_out_of_memory"
    assert classify_model_startup_failure("Address already in use") == "port_conflict"
    assert classify_model_startup_failure("downloading from huggingface.co") == "unexpected_remote_download"
    assert classify_model_startup_failure("all systems nominal") == ""
