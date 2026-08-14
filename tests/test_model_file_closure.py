"""Phase A4 tests: deterministic Safetensors file closure."""
import json
from pathlib import Path

import pytest

from auto_harness.model_runtime.file_closure import (
    ModelFileClosure,
    is_lfs_oid,
    safe_relative_path,
)
from auto_harness.model_runtime.schemas import ResolvedModelSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "model_api"
SHA = "a" * 40


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _spec():
    return ResolvedModelSpec(
        status="resolved",
        source="huggingface",
        repo_id="org/model",
        resolved_revision=SHA,
        model_identity="huggingface:org/model@%s" % SHA,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        dtype="float16",
    )


class TestClosure:
    def test_four_shard_closure_exact(self):
        plan, errors = ModelFileClosure().build(
            _spec(),
            _load("tree_sharded_safetensors.json"),
            index_content=_load("model.safetensors.index.json"),
        )
        assert errors == []
        assert plan.status == "planned"
        assert plan.integrity_level == "strong"
        shards = sorted(f["path"] for f in plan.files if f["role"] == "weight_shard")
        assert shards == [
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ]
        # README and train.py are excluded; no extra weight files.
        paths = {f["path"] for f in plan.files}
        assert "README.md" not in paths
        assert "train.py" not in paths
        # index + config + tokenizer + optional files are included.
        roles = {f["role"] for f in plan.files}
        assert "weight_index" in roles
        assert "config" in roles
        assert "tokenizer_config" in roles
        assert "tokenizer" in roles
        assert "generation_config" in roles
        assert "special_tokens_map" in roles
        assert "added_tokens" in roles
        # total size = sum of the 11 required files
        assert plan.total_size_bytes == sum(f["size_bytes"] for f in plan.files)

    def test_hash_stable(self):
        closure = ModelFileClosure()
        a, _ = closure.build(
            _spec(), _load("tree_sharded_safetensors.json"),
            index_content=_load("model.safetensors.index.json"),
        )
        b, _ = closure.build(
            _spec(), _load("tree_sharded_safetensors.json"),
            index_content=_load("model.safetensors.index.json"),
        )
        assert a.compute_plan_hash() == b.compute_plan_hash()

    def test_single_safetensors(self):
        plan, errors = ModelFileClosure().build(
            _spec(), _load("tree_single_safetensors.json")
        )
        assert errors == []
        assert plan.status == "planned"
        shards = [f["path"] for f in plan.files if f["role"] == "weight_shard"]
        assert shards == ["model.safetensors"]

    def test_missing_shard_blocked(self):
        plan, errors = ModelFileClosure().build(
            _spec(), _load("tree_missing_shard.json"),
            index_content=_load("model.safetensors.index.json"),
        )
        assert plan.status == "blocked"
        assert any("missing" in e for e in errors)

    def test_multiple_variants_blocked(self):
        plan, errors = ModelFileClosure().build(
            _spec(), _load("tree_multiple_variants.json")
        )
        assert plan.status == "blocked"
        assert any("mixed" in e or "legacy" in e or "ambiguous" in e for e in errors)

    def test_multiple_single_safetensors_blocked(self):
        source = [
            {"path": "config.json", "size_bytes": 1, "sha256": "x"},
            {"path": "tokenizer_config.json", "size_bytes": 1, "sha256": "x"},
            {"path": "a.safetensors", "size_bytes": 1, "sha256": "x"},
            {"path": "b.safetensors", "size_bytes": 1, "sha256": "x"},
        ]
        plan, errors = ModelFileClosure().build(_spec(), source)
        assert plan.status == "blocked"

    def test_path_escape_blocked(self):
        source = [
            {"path": "../etc/passwd", "size_bytes": 1, "sha256": "x"},
        ]
        plan, errors = ModelFileClosure().build(_spec(), source)
        assert plan.status == "blocked"
        assert any("unsafe" in e for e in errors)

    def test_unsafe_index_shard_blocked(self):
        source = [
            {"path": "config.json", "size_bytes": 1, "sha256": "x"},
            {"path": "model.safetensors.index.json", "size_bytes": 1, "sha256": "x"},
        ]
        index = {"weight_map": {"w": "../outside.safetensors"}}
        plan, errors = ModelFileClosure().build(_spec(), source, index_content=index)
        assert plan.status == "blocked"

    def test_gguf_blocked(self):
        source = [
            {"path": "config.json", "size_bytes": 1, "sha256": "x"},
            {"path": "model.gguf", "size_bytes": 1, "sha256": "x"},
        ]
        plan, errors = ModelFileClosure().build(_spec(), source)
        assert plan.status == "blocked"
        assert any("GGUF" in e for e in errors)

    def test_lfs_oid_promoted_to_strong(self):
        source = [
            {"path": "config.json", "size_bytes": 1, "sha256": None, "etag": None},
            {"path": "tokenizer_config.json", "size_bytes": 1, "sha256": None, "etag": None},
            {"path": "tokenizer.json", "size_bytes": 1, "sha256": None, "etag": None},
            {"path": "model.safetensors", "size_bytes": 10, "sha256": None, "etag": "e" * 64},
        ]
        plan, errors = ModelFileClosure().build(_spec(), source)
        assert errors == []
        assert plan.integrity_level == "strong"

    def test_require_strong_blocks_bounded(self):
        source = [
            {"path": "config.json", "size_bytes": 1, "sha256": None, "etag": None},
            {"path": "tokenizer_config.json", "size_bytes": 1, "sha256": None, "etag": None},
            {"path": "tokenizer.json", "size_bytes": 1, "sha256": None, "etag": None},
            {"path": "model.safetensors", "size_bytes": 10, "sha256": None, "etag": "not-an-oid"},
        ]
        plan, errors = ModelFileClosure().build(_spec(), source, require_strong=True)
        assert plan.status == "blocked"
        assert any("strong" in e for e in errors)


class TestPathHelpers:
    def test_is_lfs_oid(self):
        assert is_lfs_oid("a" * 64)
        assert not is_lfs_oid("a" * 40)
        assert not is_lfs_oid("zz" * 32)
        assert not is_lfs_oid(None)

    def test_safe_relative_path(self):
        assert safe_relative_path("model.safetensors")
        assert safe_relative_path("sub/dir/model.safetensors")
        assert not safe_relative_path("/abs/path")
        assert not safe_relative_path("../escape")
        assert not safe_relative_path("a/../b")
        assert not safe_relative_path("a\x00b")
        assert not safe_relative_path("")
        assert not safe_relative_path(None)
