"""Phase B0 tests: Preparation Artifact Gate.

Covers missing artifacts, hash tampering, symlink/path escape, modified cache
files, GPU free VRAM change, mixed model artifacts, secret-like fields, and
unknown schema versions.
"""
import hashlib
import json
from pathlib import Path

import pytest

from auto_harness.assets.cache import COMPLETE_MARKER_NAME, revision_cache_key
from auto_harness.model_runtime.evidence import ModelArtifactWriter
from auto_harness.model_runtime.preparation_gate import PreparationArtifactGate
from auto_harness.model_runtime.schemas import (
    CacheCompleteMarker,
    InferenceResourceDecision,
    ModelFilePlan,
    ResolvedModelSpec,
)
from auto_harness.models.base import read_json, write_json

COMMIT = "a" * 40
MODEL_ID = "huggingface:org/model@" + COMMIT


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
        parameter_count=7000000000,
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=32768,
        source_metadata_hash="sha256:metadata",
        grounding_hash="sha256:grounding",
    )
    data.update(overrides)
    return ResolvedModelSpec(**data)


def _make_files():
    content = b"fake-weights"
    sha = hashlib.sha256(content).hexdigest()
    files = [{
        "path": "model-00001-of-00001.safetensors",
        "role": "weight_shard",
        "size_bytes": len(content),
        "sha256": sha,
        "required": True,
    }]
    return content, sha, files


def _plan(files, model_identity=MODEL_ID):
    plan = ModelFilePlan(
        status="verified",
        model_identity=model_identity,
        format="safetensors",
        variant="fp16",
        files=files,
        total_size_bytes=sum(f.get("size_bytes", 0) for f in files),
        remaining_download_bytes=0,
        integrity_level="strong",
    )
    plan.plan_hash = plan.compute_plan_hash()
    return plan


def _decision(model_identity=MODEL_ID):
    decision = InferenceResourceDecision(
        status="allowed",
        model_identity=model_identity,
        runtime="vllm",
        gpu_indexes=[0],
        gpu_memory_total_bytes=25769803776,
        gpu_memory_free_bytes=24600000000,
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
    decision.decision_hash = decision.compute_decision_hash()
    return decision


def _marker(files, file_plan_hash, model_identity=MODEL_ID):
    marker = CacheCompleteMarker(
        status="complete",
        model_identity=model_identity,
        file_plan_hash=file_plan_hash,
        files=[{"path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"]} for f in files],
    )
    marker.marker_hash = marker.compute_marker_hash()
    return marker


class Setup:
    def __init__(self, tmp_path):
        self.tmp = Path(tmp_path)
        self.run_dir = self.tmp / "runs" / "task-1"
        self.cache_root = self.tmp / "model_cache"
        self.content, self.sha, self.files = _make_files()
        self.spec = _spec()
        self.plan = _plan(self.files)
        self.decision = _decision()
        self.cache_dir = self.cache_root / "huggingface" / revision_cache_key(
            "huggingface", "org/model", COMMIT, self.plan.plan_hash
        )
        self.marker = _marker(self.files, self.plan.plan_hash)

    def write_all(self):
        self.run_dir.mkdir(parents=True)
        writer = ModelArtifactWriter(self.run_dir)
        writer.write_resolved_model(self.spec)
        writer.write_file_plan(self.plan)
        writer.write_resource_decision(self.decision)
        self.cache_dir.mkdir(parents=True)
        (self.cache_dir / self.files[0]["path"]).write_bytes(self.content)
        write_json(self.cache_dir / COMPLETE_MARKER_NAME, self.marker.to_dict())

    def gate(self, host_facts=None):
        return PreparationArtifactGate(
            self.run_dir,
            self.cache_root,
            host_facts_provider=(lambda: host_facts) if host_facts is not None else None,
        )


def test_valid_artifacts_ready(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    bundle = setup.gate().validate()
    assert bundle.ok, bundle.errors
    assert bundle.model_host_path == str(setup.cache_dir)
    assert bundle.model_container_path == "/models/current"
    assert bundle.gpu_indexes == [0]
    assert bundle.file_plan_hash == setup.plan.plan_hash


def test_missing_artifact(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    (setup.run_dir / "reports" / "model" / "resolved_model.json").unlink()
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_artifact_missing"


def test_plan_hash_tamper(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    path = setup.run_dir / "reports" / "model" / "model_file_plan.json"
    data = read_json(path)
    data["plan_hash"] = "sha256:deadbeef"
    write_json(path, data)
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_hash_mismatch"


def test_decision_hash_tamper(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    path = setup.run_dir / "reports" / "model" / "resource_decision.json"
    data = read_json(path)
    data["decision_hash"] = "sha256:deadbeef"
    write_json(path, data)
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_hash_mismatch"


def test_marker_file_plan_hash_mismatch(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    setup.marker.file_plan_hash = "sha256:other"
    setup.marker.marker_hash = setup.marker.compute_marker_hash()
    write_json(setup.cache_dir / COMPLETE_MARKER_NAME, setup.marker.to_dict())
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_hash_mismatch"


def test_symlink_escape(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    # Replace the real cache dir with a symlink pointing outside the cache root.
    outside = setup.tmp / "outside"
    outside.mkdir()
    real_cache = setup.cache_dir
    # Move the real cache out and replace with a symlink.
    import shutil

    moved = outside / "model-cache"
    shutil.move(str(real_cache), str(moved))
    real_cache.symlink_to(moved, target_is_directory=True)
    bundle = setup.gate().validate()
    assert bundle.status == "cache_path_escape"


def test_required_file_missing(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    (setup.cache_dir / setup.files[0]["path"]).unlink()
    bundle = setup.gate().validate()
    assert bundle.status == "cache_file_missing"


def test_file_content_modified(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    # Same length, different content -> sha mismatch.
    path = setup.cache_dir / setup.files[0]["path"]
    path.write_bytes(b"x" * len(setup.content))
    bundle = setup.gate().validate()
    assert bundle.status == "cache_integrity_failed"


def test_gpu_index_no_longer_present(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    bundle = setup.gate(host_facts={"gpu_indexes": [], "gpu_memory_free_bytes": 24600000000}).validate()
    assert bundle.status == "resource_decision_stale"


def test_gpu_free_vram_dropped(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    bundle = setup.gate(
        host_facts={"gpu_indexes": [0], "gpu_memory_free_bytes": 1000}
    ).validate()
    assert bundle.status == "resource_no_longer_available"


def test_mixed_model_identity(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    # Rewrite the file plan for a different model identity.
    other = "huggingface:org/other@" + COMMIT
    plan = _plan(setup.files, model_identity=other)
    setup.plan = plan
    writer = ModelArtifactWriter(setup.run_dir)
    writer.write_file_plan(plan)
    bundle = setup.gate().validate()
    assert bundle.status == "model_identity_mismatch"


def test_secret_like_value(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    # Overwrite resolved_model with a secret token in the license field.
    payload = setup.spec.to_dict()
    payload["license"] = "hf_" + "a" * 30
    write_json(setup.run_dir / "reports" / "model" / "resolved_model.json", payload)
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_schema_unsupported"


def test_unknown_schema_version(tmp_path):
    setup = Setup(tmp_path)
    setup.write_all()
    payload = setup.spec.to_dict()
    payload["schema_version"] = 99
    write_json(setup.run_dir / "reports" / "model" / "resolved_model.json", payload)
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_schema_unsupported"


def test_status_not_resolved(tmp_path):
    setup = Setup(tmp_path)
    setup.spec = _spec(status="ambiguous")
    setup.write_all()
    bundle = setup.gate().validate()
    assert bundle.status == "preparation_schema_unsupported"
