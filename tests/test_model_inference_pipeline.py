"""Phase B8 tests: model runtime mainline integration (offline).

Covers dry-run plan generation, full offline execute with fake Docker + fake
vLLM, feature-flag gating, resume reusing an existing container (no duplicate
start), fresh trace per verify, and historical-trace rejection.
"""
import hashlib
import json
from pathlib import Path

import pytest

from auto_harness.assets.cache import COMPLETE_MARKER_NAME, revision_cache_key
from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.controller import ModelRuntimeController
from auto_harness.model_runtime.evidence import ModelArtifactWriter, ModelInferenceGate
from auto_harness.model_runtime.schemas import (
    CacheCompleteMarker,
    InferenceResourceDecision,
    ModelFilePlan,
    ResolvedModelSpec,
)
from auto_harness.models.base import write_json

COMMIT = "a" * 40
MODEL_ID = "huggingface:org/model@" + COMMIT
DIGEST = "sha256:" + "d" * 64
IMAGE = "vllm/vllm-openai:v0.6.1@" + DIGEST


def _config(**overrides):
    data = dict(model_inference_enabled=True, model_runtime_image=IMAGE)
    data.update(overrides)
    return HarnessConfig(**data)


def _build_artifacts(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "task-1"
    cache_root = Path(tmp_path) / "model_cache"
    content = b"fake-weights"
    sha = hashlib.sha256(content).hexdigest()
    files = [{"path": "model-00001-of-00001.safetensors", "role": "weight_shard", "size_bytes": len(content), "sha256": sha, "required": True}]

    spec = ResolvedModelSpec(
        status="resolved", source="huggingface", repo_id="org/model",
        resolved_revision=COMMIT, model_identity=MODEL_ID, model_type="qwen2",
        architectures=["Qwen2ForCausalLM"], task="text-generation", dtype="float16",
        source_metadata_hash="sha256:model",
    )
    plan = ModelFilePlan(
        status="verified", model_identity=MODEL_ID, format="safetensors",
        variant="fp16", files=files, total_size_bytes=len(content),
        remaining_download_bytes=0, integrity_level="strong",
    )
    plan.plan_hash = plan.compute_plan_hash()
    decision = InferenceResourceDecision(
        status="allowed", model_identity=MODEL_ID, runtime="vllm", gpu_indexes=[0],
        required_vram_bytes=18450000000, selected_dtype="float16",
        max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    decision.decision_hash = decision.compute_decision_hash()
    marker = CacheCompleteMarker(
        status="complete", model_identity=MODEL_ID, file_plan_hash=plan.plan_hash,
        files=[{"path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]} for f in files],
    )
    marker.marker_hash = marker.compute_marker_hash()

    writer = ModelArtifactWriter(run_dir)
    writer.write_resolved_model(spec)
    writer.write_file_plan(plan)
    writer.write_resource_decision(decision)
    cache_dir = cache_root / "huggingface" / revision_cache_key("huggingface", "org/model", COMMIT, plan.plan_hash)
    cache_dir.mkdir(parents=True)
    (cache_dir / files[0]["path"]).write_bytes(content)
    write_json(cache_dir / COMPLETE_MARKER_NAME, marker.to_dict())
    return run_dir, cache_root


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


class FakeStreamResponse:
    def __init__(self, status, lines):
        self.status = status
        self.code = status
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


class FakeHTTP:
    def __init__(self, served="org/model"):
        self.served = served
        self.chat_calls = 0

    def __call__(self, req, timeout=5):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/v1/models" in url:
            return FakeResponse(200, json.dumps({"data": [{"id": self.served}]}))
        body = json.loads(req.data.decode("utf-8")) if isinstance(req.data, (bytes, bytearray)) else {}
        self.chat_calls += 1
        trace = body["messages"][1]["content"].rsplit(" ", 1)[-1]
        if body.get("stream"):
            lines = []
            lines.append("data: " + json.dumps({"choices": [{"delta": {"content": "token "}}]}))
            lines.append("")
            lines.append("data: " + json.dumps({"choices": [{"delta": {"content": trace}}]}))
            lines.append("")
            lines.append("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 18, "completion_tokens": 6}}))
            lines.append("")
            lines.append("data: [DONE]")
            lines.append("")
            return FakeStreamResponse(200, lines)
        return FakeResponse(200, json.dumps({
            "model": self.served,
            "choices": [{"message": {"content": "token %s" % trace}}],
            "usage": {"prompt_tokens": 18, "completion_tokens": 6},
        }))


class FakeDocker:
    def __init__(self, running=True):
        self.run_calls = []
        self.labels = {}
        self.running = running

    def __call__(self, cmd):
        if cmd[:2] == ["docker", "run"]:
            self.run_calls.append(cmd)
            i = 0
            while i + 1 < len(cmd):
                if cmd[i] == "--label":
                    key, value = cmd[i + 1].split("=", 1)
                    self.labels[key] = value
                i += 1
            return {"exit_code": 0, "stdout": "cid123\n", "stderr": ""}
        if cmd[:2] == ["docker", "inspect"]:
            return {
                "exit_code": 0,
                "stdout": json.dumps([{"State": {"Running": self.running}, "Config": {"Labels": self.labels}}]),
                "stderr": "",
            }
        if cmd[:2] == ["docker", "logs"]:
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        return {"exit_code": 1, "stdout": "", "stderr": "unexpected"}


class FakeReconciler:
    def __init__(self, decision="reuse", container_id="cid-existing"):
        self.decision = decision
        self.container_id = container_id
        self.calls = 0

    def reconcile(self, op):
        self.calls += 1
        return {"decision": self.decision, "reason": "test", "observed_state": {"id": self.container_id}}


def test_dry_run_generates_plan(tmp_path):
    run_dir, cache_root = _build_artifacts(tmp_path)
    controller = ModelRuntimeController()
    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=_config(), cache_root=cache_root,
        execute=False, allow_start=False,
    )
    assert phase.status == "passed"
    assert phase.plan is not None
    assert phase.startup_evidence is None
    assert phase.plan.plan_hash


def test_feature_flag_off_blocked(tmp_path):
    run_dir, cache_root = _build_artifacts(tmp_path)
    controller = ModelRuntimeController()
    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=_config(model_inference_enabled=False),
        cache_root=cache_root, execute=True, allow_start=True,
    )
    assert phase.status == "blocked"
    assert phase.policy.get("reason_code") == "model_inference_disabled"


def test_full_execute_passes(tmp_path):
    run_dir, cache_root = _build_artifacts(tmp_path)
    docker = FakeDocker()
    http = FakeHTTP()
    controller = ModelRuntimeController()
    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=_config(), cache_root=cache_root,
        execute=True, allow_start=True, command_runner=docker, urlopen=http,
    )
    assert phase.status == "passed"
    assert phase.container_id == "cid123"
    assert phase.startup_evidence.status == "ready"
    assert len(docker.run_calls) == 1

    verify = controller.verify_phase(
        run_dir=run_dir, task_id="task-1", runtime_plan=phase.plan,
        startup_evidence=phase.startup_evidence, urlopen=http,
    )
    assert verify.status == "passed"
    assert verify.data["non_stream"]["status"] == "passed"
    assert verify.data["stream"]["status"] == "passed"


def test_resume_reuses_container_no_duplicate(tmp_path):
    run_dir, cache_root = _build_artifacts(tmp_path)
    from auto_harness.model_runtime.controller import runtime_labels, stable_operation_id

    controller = ModelRuntimeController()
    # Derive the exact labels the resumed container would already carry.
    dry = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=_config(), cache_root=cache_root,
        execute=False, allow_start=False,
    )
    op_id = stable_operation_id("task-1", dry.plan.plan_hash)

    docker = FakeDocker()
    docker.labels = runtime_labels("task-1", op_id, dry.plan)
    http = FakeHTTP()

    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=_config(), cache_root=cache_root,
        execute=True, allow_start=True, command_runner=docker, urlopen=http,
        reconciler=FakeReconciler(decision="reuse", container_id="cid-existing"),
    )
    assert phase.status == "passed"
    assert phase.container_id == "cid-existing"
    assert docker.run_calls == []  # reused, not re-started


def test_fresh_trace_per_verify():
    gate = ModelInferenceGate()
    traces = {gate.generate_trace_id() for _ in range(20)}
    assert len(traces) == 20
    assert all(t.startswith("infer_") for t in traces)


def test_historical_trace_rejected(tmp_path):
    run_dir, cache_root = _build_artifacts(tmp_path)
    controller = ModelRuntimeController()
    # A verify run against a server that only echoes a stale trace must fail.
    from auto_harness.model_runtime.evidence import ModelInferenceGate
    from auto_harness.model_runtime.schemas import ModelRuntimeStartupEvidence

    plan = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=_config(), cache_root=cache_root,
        execute=False, allow_start=False,
    ).plan
    startup = ModelRuntimeStartupEvidence(status="ready", ready_at="2000-01-01T00:00:00Z", container_id="cid", served_model_name="org/model")

    def stale_http(req, timeout=5):
        return FakeResponse(200, json.dumps({
            "model": "org/model",
            "choices": [{"message": {"content": "token infer_STALE_00000000"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }))

    gate = ModelInferenceGate(urlopen=stale_http, now=lambda: "2000-01-01T01:00:00Z")
    ev = gate.verify_non_stream(runtime_plan=plan, startup_evidence=startup, task_id="t", operation_id="o")
    assert ev.status == "failed"
    assert ev.trace_found is False
