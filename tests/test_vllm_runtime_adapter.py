"""Phase B1 tests: deterministic vLLM runtime adapter.

Covers determinism, local-cache --model binding, parameter source mapping,
image-digest requirement, path escape, forbidden flags/parameter injection,
plan-hash sensitivity, and secret-free plans.
"""
import json

import pytest

from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.preparation_gate import PreparationBundle
from auto_harness.model_runtime.schemas import (
    InferenceResourceDecision,
    ResolvedModelSpec,
)
from auto_harness.model_runtime.vllm_adapter import VllmRuntimeAdapter

COMMIT = "c" * 40
MODEL_ID = "huggingface:org/model@" + COMMIT
DIGEST = "sha256:" + "d" * 64
IMAGE = "vllm/vllm-openai:v0.6.1@" + DIGEST


def _spec(**overrides):
    data = dict(
        status="resolved",
        source="huggingface",
        repo_id="org/model",
        requested_revision="main",
        resolved_revision=COMMIT,
        model_identity=MODEL_ID,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        task="text-generation",
        dtype="float16",
        source_metadata_hash="sha256:meta",
        grounding_hash="sha256:grounding",
    )
    data.update(overrides)
    return ResolvedModelSpec(**data)


def _decision(**overrides):
    data = dict(
        status="allowed",
        model_identity=MODEL_ID,
        runtime="vllm",
        gpu_indexes=[0],
        weight_bytes=15000000000,
        weight_runtime_bytes=15750000000,
        kv_cache_bytes=900000000,
        runtime_overhead_bytes=1800000000,
        required_vram_bytes=18450000000,
        usable_vram_bytes=22000000000,
        required_ram_bytes=24000000000,
        required_disk_bytes=18000000000,
        selected_dtype="float16",
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    data.update(overrides)
    return InferenceResourceDecision(**data)


def _bundle(**overrides):
    data = dict(
        status="ready",
        spec=_spec(),
        decision=_decision(),
        resolved_model_hash="sha256:meta",
        file_plan_hash="sha256:fp",
        cache_marker_hash="sha256:cm",
        resource_decision_hash="sha256:rd",
        model_host_path="/tmp/model_cache/huggingface/key",
        model_container_path="/models/current",
        gpu_indexes=[0],
    )
    data.update(overrides)
    return PreparationBundle(**data)


def _config(**overrides):
    data = dict(model_runtime_image=IMAGE)
    data.update(overrides)
    return HarnessConfig(**data)


def _build(bundle=None, config=None, **kwargs):
    return VllmRuntimeAdapter().build(
        bundle or _bundle(), config or _config(), **kwargs
    )


def test_deterministic_output():
    a = _build(task_id="task-1")
    b = _build(task_id="task-1")
    assert a.to_dict() == b.to_dict()
    assert a.plan_hash == b.plan_hash


def test_model_bound_to_local_container_path():
    plan = _build(task_id="task-1")
    args = plan.command
    model_idx = args.index("--model")
    assert args[model_idx + 1] == "/models/current"
    # The host path is the verified local cache, not a remote URL.
    assert plan.model_host_path.startswith("/tmp/model_cache/")
    assert plan.model_container_path == "/models/current"


def test_parameter_sources():
    plan = _build(task_id="task-1")
    args = plan.command
    assert args[args.index("--dtype") + 1] == "float16"
    assert args[args.index("--max-model-len") + 1] == "4096"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.9"
    assert args[args.index("--max-num-seqs") + 1] == "1"
    assert args[args.index("--tensor-parallel-size") + 1] == "1"
    assert args[args.index("--served-model-name") + 1] == "org/model"


def test_missing_image_digest_raises():
    config = _config(model_runtime_image="vllm/vllm-openai:v0.6.1")
    with pytest.raises(ValueError):
        _build(config=config)


def test_require_digest_can_be_disabled():
    config = _config(model_runtime_image="vllm/vllm-openai:v0.6.1")
    plan = _build(config=config, require_image_digest=False)
    assert plan.image_digest == ""


def test_not_ready_bundle_raises():
    with pytest.raises(ValueError):
        _build(bundle=_bundle(status="preparation_hash_mismatch"))


def test_remote_code_blocked():
    spec = _spec(requires_remote_code=True)
    with pytest.raises(ValueError):
        _build(bundle=_bundle(spec=spec))


def test_quantized_model_blocked():
    spec = _spec(quantization="awq")
    with pytest.raises(ValueError):
        _build(bundle=_bundle(spec=spec))


def test_remote_url_host_path_raises():
    with pytest.raises(ValueError):
        _build(bundle=_bundle(model_host_path="https://hf.co/org/model"))


def test_no_forbidden_flags():
    plan = _build(task_id="task-1")
    joined = " ".join(plan.command)
    for flag in (
        "--trust-remote-code",
        "--download-dir",
        "--enable-lora",
        "--lora-modules",
        "--worker-use-ray",
        "--cpu-offload-gb",
    ):
        assert flag not in joined


def test_plan_hash_changes_on_image_digest():
    a = _build(task_id="task-1")
    b = _build(task_id="task-1", config=_config(model_runtime_image="vllm/vllm-openai:v0.6.2@" + DIGEST))
    assert a.plan_hash != b.plan_hash


def test_plan_hash_changes_on_port():
    a = _build(task_id="task-1", host_port=8000)
    b = _build(task_id="task-1", host_port=8001)
    assert a.plan_hash != b.plan_hash


def test_plan_hash_changes_on_gpu_index():
    a = _build(task_id="task-1", gpu_indexes=[0])
    b = _build(task_id="task-1", gpu_indexes=[1])
    assert a.plan_hash != b.plan_hash


def test_plan_hash_changes_on_command():
    a = _build(task_id="task-1")
    d2 = _decision(max_model_len=2048)
    b = _build(task_id="task-1", bundle=_bundle(decision=d2))
    assert a.plan_hash != b.plan_hash


def test_plan_contains_no_token():
    plan = _build(task_id="task-1")
    text = json.dumps(plan.to_dict(), ensure_ascii=False)
    for token in ("hf_", "Bearer", "api_key", "token="):
        assert token not in text
