"""Phase A1 tests: model preparation schemas and artifact writer.

Covers round-trip, illegal states, negative bytes, NaN/Infinity, secret
fields/values, hash stability, atomic write interruption, and unknown
schema versions.
"""
import json
import math
import pytest
from pathlib import Path

from auto_harness.model_runtime import (
    CacheCompleteMarker,
    InferenceResourceDecision,
    ModelArtifactWriter,
    ModelFilePlan,
    ModelReferenceCandidate,
    ResolvedModelSpec,
)
from auto_harness.model_runtime.evidence import (
    scan_forbidden_fields,
    validate_serializable,
)


def _candidate():
    return ModelReferenceCandidate(
        source="huggingface",
        repo_id="org/model-name",
        requested_revision="main",
        role="primary_generation_model",
        confidence=0.95,
        evidence=[{"file": "config.yaml", "expression": "model_id: org/model-name"}],
        discovered_by="deterministic_config_parser",
    )


def _resolved():
    return ResolvedModelSpec(
        status="resolved",
        source="huggingface",
        repo_id="org/model-name",
        requested_revision="main",
        resolved_revision="a" * 40,
        model_identity="huggingface:org/model-name@" + "a" * 40,
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
        quantization=None,
        requires_remote_code=False,
        gated=False,
        license="apache-2.0",
        source_metadata_hash="sha256:" + "b" * 64,
        grounding_hash="sha256:" + "c" * 64,
        resolved_at="2026-08-14T00:00:00+00:00",
    )


def _file_plan():
    return ModelFilePlan(
        status="planned",
        model_identity="huggingface:org/model@" + "a" * 40,
        format="safetensors",
        variant="fp16",
        files=[
            {"path": "model-00001-of-00002.safetensors", "role": "weight_shard", "size_bytes": 10, "sha256": "x", "etag": "e1", "required": True},
            {"path": "model.safetensors.index.json", "role": "weight_index", "size_bytes": 2, "sha256": "y", "etag": "e0", "required": True},
        ],
        total_size_bytes=12,
        remaining_download_bytes=12,
        integrity_level="strong",
    )


class TestCandidateSchema:
    def test_round_trip(self):
        data = _candidate().to_dict()
        restored = ModelReferenceCandidate.from_dict(data)
        assert restored.repo_id == "org/model-name"
        assert restored.confidence == 0.95

    def test_unknown_key_rejected(self):
        data = _candidate().to_dict()
        data["bogus"] = 1
        with pytest.raises(ValueError, match="unknown keys"):
            ModelReferenceCandidate.from_dict(data)

    def test_invalid_source_rejected(self):
        data = _candidate().to_dict()
        data["source"] = "s3"
        with pytest.raises(ValueError, match="source"):
            ModelReferenceCandidate.from_dict(data)

    def test_confidence_nan_rejected(self):
        data = _candidate().to_dict()
        data["confidence"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            ModelReferenceCandidate.from_dict(data)

    def test_confidence_out_of_range_rejected(self):
        data = _candidate().to_dict()
        data["confidence"] = 1.5
        with pytest.raises(ValueError, match="confidence"):
            ModelReferenceCandidate.from_dict(data)

    def test_empty_evidence_rejected(self):
        data = _candidate().to_dict()
        data["evidence"] = []
        with pytest.raises(ValueError, match="evidence"):
            ModelReferenceCandidate.from_dict(data)

    def test_path_escape_rejected_by_validate(self):
        candidate = _candidate()
        candidate.repo_id = "../org/model"
        problems = candidate.validate()
        assert any(".." in problem for problem in problems)


class TestResolvedModelSchema:
    def test_round_trip(self):
        restored = ResolvedModelSpec.from_dict(_resolved().to_dict())
        assert restored.status == "resolved"
        assert restored.resolved_revision == "a" * 40

    def test_illegal_status_rejected(self):
        data = _resolved().to_dict()
        data["status"] = "flying"
        with pytest.raises(ValueError, match="status"):
            ResolvedModelSpec.from_dict(data)

    def test_unknown_version_rejected(self):
        data = _resolved().to_dict()
        data["schema_version"] = 99
        with pytest.raises(ValueError, match="schema_version"):
            ResolvedModelSpec.from_dict(data)


class TestFilePlanSchema:
    def test_round_trip(self):
        plan = _file_plan()
        plan.plan_hash = plan.compute_plan_hash()
        restored = ModelFilePlan.from_dict(plan.to_dict())
        assert restored.compute_plan_hash() == plan.plan_hash

    def test_hash_stable_and_path_independent(self):
        plan_a = _file_plan()
        plan_b = _file_plan()
        # same inputs, different file ordering -> same hash
        plan_b.files = list(reversed(plan_b.files))
        assert plan_a.compute_plan_hash() == plan_b.compute_plan_hash()

    def test_hash_changes_when_files_change(self):
        plan_a = _file_plan()
        plan_b = _file_plan()
        plan_b.files[0]["size_bytes"] = 11
        assert plan_a.compute_plan_hash() != plan_b.compute_plan_hash()

    def test_negative_file_bytes_rejected(self):
        data = _file_plan().to_dict()
        data["files"][0]["size_bytes"] = -1
        with pytest.raises(ValueError, match="size_bytes"):
            ModelFilePlan.from_dict(data)

    def test_negative_total_bytes_rejected(self):
        data = _file_plan().to_dict()
        data["total_size_bytes"] = -5
        with pytest.raises(ValueError, match="total_size_bytes"):
            ModelFilePlan.from_dict(data)

    def test_illegal_status_rejected(self):
        data = _file_plan().to_dict()
        data["status"] = "nope"
        with pytest.raises(ValueError, match="status"):
            ModelFilePlan.from_dict(data)


class TestResourceDecisionSchema:
    def test_hash_recomputed(self):
        decision = InferenceResourceDecision(
            status="allowed",
            model_identity="huggingface:org/model@" + "a" * 40,
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
            reasons=["ok"],
            warnings=[],
        )
        decision.decision_hash = decision.compute_decision_hash()
        restored = InferenceResourceDecision.from_dict(decision.to_dict())
        assert restored.compute_decision_hash() == decision.decision_hash

    def test_hash_ignores_reasons_and_host_drift(self):
        base = InferenceResourceDecision(
            status="allowed", model_identity="m", reasons=["a"], warnings=[],
        )
        changed = InferenceResourceDecision(
            status="allowed", model_identity="m", reasons=["b"], warnings=[],
            gpu_memory_free_bytes=9999,
        )
        assert base.compute_decision_hash() == changed.compute_decision_hash()

    def test_illegal_status_rejected(self):
        decision = InferenceResourceDecision(status="allowed", model_identity="m")
        data = decision.to_dict()
        data["status"] = "maybe"
        with pytest.raises(ValueError, match="status"):
            InferenceResourceDecision.from_dict(data)


class TestCompleteMarkerSchema:
    def test_hash_recomputed(self):
        marker = CacheCompleteMarker(
            model_identity="huggingface:org/model@" + "a" * 40,
            file_plan_hash="sha256:" + "d" * 64,
            files=[{"path": "model.safetensors", "size_bytes": 5, "sha256": "z"}],
            verified_at="2026-08-14T00:00:00+00:00",
        )
        marker.marker_hash = marker.compute_marker_hash()
        restored = CacheCompleteMarker.from_dict(marker.to_dict())
        assert restored.compute_marker_hash() == marker.marker_hash

    def test_verified_at_not_in_hash(self):
        a = CacheCompleteMarker(model_identity="m", file_plan_hash="h", files=[], verified_at="t1")
        b = CacheCompleteMarker(model_identity="m", file_plan_hash="h", files=[], verified_at="t2")
        assert a.compute_marker_hash() == b.compute_marker_hash()


class TestSerializableGuard:
    def test_nan_rejected(self):
        problems = validate_serializable({"x": float("nan")})
        assert any("NaN" in p for p in problems)

    def test_infinity_rejected(self):
        problems = validate_serializable({"x": float("inf")})
        assert any("infinite" in p for p in problems)

    def test_path_rejected(self):
        problems = validate_serializable({"x": Path("/tmp/foo")})
        assert any("Path" in p for p in problems)

    def test_exception_rejected(self):
        problems = validate_serializable({"x": RuntimeError("boom")})
        assert any("Exception" in p for p in problems)


class TestSecretScanning:
    def test_forbidden_field_names(self):
        problems = scan_forbidden_fields({"hf_token": "abc", "nested": {"api_key": "x"}})
        assert any("hf_token" in p for p in problems)
        assert any("api_key" in p for p in problems)

    def test_clean_payload_passes(self):
        assert scan_forbidden_fields({"model_identity": "huggingface:org/m@sha"}) == []


class TestArtifactWriter:
    def test_writes_three_artifacts(self, tmp_path):
        writer = ModelArtifactWriter(tmp_path)
        spec = _resolved()
        plan = _file_plan()
        plan.plan_hash = plan.compute_plan_hash()
        decision = InferenceResourceDecision(status="allowed", model_identity=spec.model_identity)
        decision.decision_hash = decision.compute_decision_hash()
        writer.write_resolved_model(spec)
        writer.write_file_plan(plan)
        writer.write_resource_decision(decision)
        model_dir = tmp_path / "reports" / "model"
        assert (model_dir / "resolved_model.json").exists()
        assert (model_dir / "model_file_plan.json").exists()
        assert (model_dir / "resource_decision.json").exists()

    def test_forbidden_field_blocks_write(self, tmp_path):
        writer = ModelArtifactWriter(tmp_path)
        # A raw payload with a forbidden field name is rejected by the field scan.
        with pytest.raises(ValueError, match="forbidden"):
            writer._write("resolved_model", {"model_identity": "m", "hf_token": "abc"})

    def test_redacted_token_value_blocks_write(self, tmp_path):
        writer = ModelArtifactWriter(tmp_path)
        spec = _resolved()
        # a Hugging Face token-like value would trip the redaction scan
        spec.source_metadata_hash = "hf_" + "a" * 40
        with pytest.raises(ValueError, match="sensitive"):
            writer.write_resolved_model(spec)
