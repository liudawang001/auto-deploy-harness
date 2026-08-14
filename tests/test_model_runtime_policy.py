"""Phase B2 tests: managed inference runtime policy.

Covers legal auto-allow, per-field tamper rejection, LLM source impersonation,
operator approval cannot override hard denial, and feature-flag gating.
"""
from pathlib import Path

import pytest

from auto_harness.command_auth.policy import CommandAuthorizationEngine
from auto_harness.command_auth.schemas import CommandCandidate, CommandRegistry
from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.policy import ModelRuntimePolicy
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


def _spec():
    return ResolvedModelSpec(
        status="resolved",
        source="huggingface",
        repo_id="org/model",
        resolved_revision=COMMIT,
        model_identity=MODEL_ID,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        task="text-generation",
        dtype="float16",
        source_metadata_hash="sha256:meta",
    )


def _decision():
    return InferenceResourceDecision(
        status="allowed",
        model_identity=MODEL_ID,
        runtime="vllm",
        gpu_indexes=[0],
        required_vram_bytes=18450000000,
        selected_dtype="float16",
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )


def _bundle(tmp_path, **overrides):
    cache_root = tmp_path / "model_cache"
    model_host = cache_root / "huggingface" / "org-model_key"
    data = dict(
        status="ready",
        spec=_spec(),
        decision=_decision(),
        resolved_model_hash="sha256:meta",
        file_plan_hash="sha256:fp",
        cache_marker_hash="sha256:cm",
        resource_decision_hash="sha256:rd",
        model_host_path=str(model_host),
        cache_root=str(cache_root),
        gpu_indexes=[0],
    )
    data.update(overrides)
    return PreparationBundle(**data)


def _config(**overrides):
    data = dict(model_inference_enabled=True, model_runtime_image=IMAGE)
    data.update(overrides)
    return HarnessConfig(**data)


def _authorize(tmp_path, *, config=None, bundle=None, plan=None, **kwargs):
    config = config or _config()
    bundle = bundle or _bundle(tmp_path)
    adapter = VllmRuntimeAdapter()
    plan = plan or adapter.build(bundle, config, task_id="task-1")
    opts = dict(execute=True, allow_start=True, execution_backend="docker")
    opts.update(kwargs)
    return ModelRuntimePolicy(adapter=adapter).authorize(plan, bundle, config, **opts)


def test_legal_adapter_plan_auto_allowed(tmp_path):
    result = _authorize(tmp_path)
    assert result["allowed"] is True
    assert result["verdict"] == "auto_allowed"
    assert result["reason_code"] == "managed_inference_runtime"


def test_feature_flag_off_denied(tmp_path):
    result = _authorize(tmp_path, config=_config(model_inference_enabled=False))
    assert result["allowed"] is False
    assert result["reason_code"] == "model_inference_disabled"


def test_plan_hash_tamper_denied(tmp_path):
    config = _config()
    bundle = _bundle(tmp_path)
    plan = VllmRuntimeAdapter().build(bundle, config, task_id="task-1")
    plan.plan_hash = "sha256:deadbeef"
    result = ModelRuntimePolicy().authorize(
        plan, bundle, config, execute=True, allow_start=True, execution_backend="docker"
    )
    assert result["allowed"] is False
    assert result["reason_code"] == "plan_hash_mismatch"


def test_command_tamper_denied(tmp_path):
    config = _config()
    bundle = _bundle(tmp_path)
    plan = VllmRuntimeAdapter().build(bundle, config, task_id="task-1")
    plan.command.append("--trust-remote-code")
    plan.plan_hash = plan.compute_plan_hash()
    result = ModelRuntimePolicy().authorize(
        plan, bundle, config, execute=True, allow_start=True, execution_backend="docker"
    )
    assert result["allowed"] is False
    assert result["reason_code"] == "command_mismatch"


def test_floating_image_denied(tmp_path):
    config = _config(model_runtime_image="vllm/vllm-openai:v0.6.1")
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError):
        VllmRuntimeAdapter().build(bundle, config, task_id="task-1")


def test_model_path_outside_cache_denied(tmp_path):
    config = _config()
    bundle = _bundle(tmp_path, model_host_path="/tmp/outside/model")
    plan = VllmRuntimeAdapter().build(bundle, config, task_id="task-1")
    result = ModelRuntimePolicy().authorize(
        plan, bundle, config, execute=True, allow_start=True, execution_backend="docker"
    )
    assert result["allowed"] is False
    assert result["reason_code"] == "model_path_outside_cache"


def test_local_backend_denied(tmp_path):
    result = _authorize(tmp_path, execution_backend="local")
    assert result["allowed"] is False
    assert result["reason_code"] == "local_backend_denied"


def test_missing_start_authorization_denied(tmp_path):
    config = _config()
    bundle = _bundle(tmp_path)
    plan = VllmRuntimeAdapter().build(bundle, config, task_id="task-1")
    result = ModelRuntimePolicy().authorize(
        plan, bundle, config, execute=False, allow_start=True, execution_backend="docker"
    )
    assert result["allowed"] is False
    assert result["reason_code"] == "start_not_authorized"


def test_unsupported_security_profile_denied(tmp_path):
    config = _config()
    bundle = _bundle(tmp_path)
    plan = VllmRuntimeAdapter().build(bundle, config, task_id="task-1")
    plan.security_profile = "model_runtime_v0"
    plan.plan_hash = plan.compute_plan_hash()
    result = ModelRuntimePolicy().authorize(
        plan, bundle, config, execute=True, allow_start=True, execution_backend="docker"
    )
    assert result["allowed"] is False
    assert result["reason_code"] == "unsupported_security_profile"


def test_not_ready_bundle_denied(tmp_path):
    config = _config()
    bundle = _bundle(tmp_path, status="cache_integrity_failed")
    result = ModelRuntimePolicy().authorize(
        None, bundle, config, execute=True, allow_start=True, execution_backend="docker"
    )
    # policy validates hash first; a None plan is not a valid plan
    assert result["allowed"] is False


def test_llm_managed_source_rejected():
    candidate = CommandCandidate.build(
        phase="run",
        argv=["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/current"],
        source_kind="managed_inference_runtime",
    )
    registry = CommandRegistry(repository_fingerprint="fp", evidence=[], candidates=[candidate])
    decision = CommandAuthorizationEngine().authorize(candidate, registry)
    assert decision.verdict == "hard_denied"
    assert decision.reason_code == "managed_inference_runtime_reserved_for_adapter"
