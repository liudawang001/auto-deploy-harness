"""Phase A2 tests: grounded repository model reference discovery + selection."""
import pytest
from pathlib import Path

from auto_harness.assets.detector import ModelReferenceDetector, is_valid_repo_id
from auto_harness.model_runtime.resolver import (
    ModelReferenceResolver,
    compute_grounding_hash,
)


def _write(repo_dir: Path, name: str, text: str) -> None:
    path = repo_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestDetector:
    def test_config_key_detection(self, tmp_path):
        _write(tmp_path, "config.yaml", "model_id: org/model-name\n")
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert len(candidates) == 1
        assert candidates[0].repo_id == "org/model-name"
        assert candidates[0].source == "huggingface"
        assert candidates[0].role == "primary_generation_model"
        assert candidates[0].confidence == 0.95
        assert candidates[0].evidence

    def test_hf_url_detection(self, tmp_path):
        _write(tmp_path, "README.md", "See https://huggingface.co/org/model-name for details.\n")
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert [c.repo_id for c in candidates] == ["org/model-name"]
        assert candidates[0].source == "huggingface"

    def test_modelscope_url_detection(self, tmp_path):
        _write(tmp_path, "README.md", "https://www.modelscope.cn/models/org/model-name\n")
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert [c.repo_id for c in candidates] == ["org/model-name"]
        assert candidates[0].source == "modelscope"

    def test_from_pretrained_ast(self, tmp_path):
        _write(
            tmp_path,
            "app.py",
            "from transformers import AutoModel\nmodel = AutoModel.from_pretrained('org/model-name')\n",
        )
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert any(c.repo_id == "org/model-name" for c in candidates)

    def test_from_pretrained_constant(self, tmp_path):
        _write(
            tmp_path,
            "app.py",
            "MODEL_ID = 'org/model-name'\n"
            "from transformers import AutoModel\n"
            "model = AutoModel.from_pretrained(MODEL_ID)\n",
        )
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert any(c.repo_id == "org/model-name" for c in candidates)

    def test_argparse_default(self, tmp_path):
        _write(
            tmp_path,
            "cli.py",
            "import argparse\np = argparse.ArgumentParser()\n"
            "p.add_argument('--model', default='org/model-name')\n",
        )
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert any(c.repo_id == "org/model-name" for c in candidates)

    def test_dynamic_expression_not_guessed(self, tmp_path):
        _write(
            tmp_path,
            "app.py",
            "import os\nfrom transformers import AutoModel\n"
            "model = AutoModel.from_pretrained(os.environ['MODEL'])\n",
        )
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert all(c.repo_id != "os.environ" for c in candidates)
        # The dynamic expression must not produce a concrete candidate.
        assert not any("org/" in c.repo_id for c in candidates)

    def test_accessory_role_not_primary(self, tmp_path):
        _write(tmp_path, "config.json", '{"embedding_model": "org/embed-model"}\n')
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert candidates
        assert all(c.role == "model_accessory" for c in candidates)

    def test_invalid_repo_id_not_matched(self, tmp_path):
        _write(tmp_path, "config.yaml", "model_id: ../etc/passwd\nmodel_name: /abs/path\n")
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert candidates == []

    def test_env_file_not_read(self, tmp_path):
        _write(tmp_path, ".env", "model_id: org/secret-model\n")
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert candidates == []

    def test_empty_repo(self, tmp_path):
        candidates = ModelReferenceDetector().detect(tmp_path)
        assert candidates == []


class TestIsValidRepoId:
    def test_valid(self):
        assert is_valid_repo_id("org/model-name")
        assert is_valid_repo_id("Qwen/Qwen2.5-7B-Instruct")

    def test_invalid(self):
        assert not is_valid_repo_id("../org/model")
        assert not is_valid_repo_id("/org/model")
        assert not is_valid_repo_id("org/model/extra")
        assert not is_valid_repo_id("https://huggingface.co/org/model")
        assert not is_valid_repo_id("")


class TestResolverSelection:
    def test_single_primary_resolved(self, tmp_path):
        _write(tmp_path, "config.yaml", "model_id: org/model\n")
        result = ModelReferenceResolver().resolve_reference(tmp_path)
        assert result["status"] == "resolved"
        assert result["selected"].repo_id == "org/model"

    def test_no_model_not_found(self, tmp_path):
        result = ModelReferenceResolver().resolve_reference(tmp_path)
        assert result["status"] == "not_found"

    def test_only_accessory_needs_human_input(self, tmp_path):
        _write(tmp_path, "config.json", '{"embedding_model": "org/embed"}\n')
        result = ModelReferenceResolver().resolve_reference(tmp_path)
        assert result["status"] == "needs_human_input"

    def test_multiple_primary_needs_human_input(self, tmp_path):
        _write(
            tmp_path,
            "config.yaml",
            "model_id: org/model-a\nmodel_name: org/model-b\n",
        )
        result = ModelReferenceResolver().resolve_reference(tmp_path)
        assert result["status"] == "needs_human_input"

    def test_clear_winner_resolved(self, tmp_path):
        # high-confidence config key + low-confidence README link -> clear winner
        _write(tmp_path, "config.yaml", "model_id: org/model-a\n")
        _write(tmp_path, "README.md", "check out https://huggingface.co/org/model-b\n")
        result = ModelReferenceResolver().resolve_reference(tmp_path)
        assert result["status"] == "resolved"
        assert result["selected"].repo_id == "org/model-a"

    def test_operator_override(self, tmp_path):
        _write(tmp_path, "README.md", "no model here\n")
        result = ModelReferenceResolver().resolve_reference(
            tmp_path, operator_override="huggingface:org/chosen"
        )
        assert result["status"] == "resolved"
        assert result["selected"].repo_id == "org/chosen"
        assert result["selected"].discovered_by == "operator_override"
        assert result["warnings"]

    def test_invalid_override_raises(self, tmp_path):
        with pytest.raises(ValueError):
            ModelReferenceResolver().resolve_reference(tmp_path, operator_override="../etc/passwd")


class TestGroundingHash:
    def test_grounding_hash_changes_with_evidence(self, tmp_path):
        _write(tmp_path, "config.yaml", "model_id: org/model\n")
        resolver = ModelReferenceResolver()
        first = resolver.resolve_reference(tmp_path)
        _write(tmp_path, "config.yaml", "model_id: org/model\n# extra comment\n")
        second = resolver.resolve_reference(tmp_path)
        assert first["grounding_hash"] != second["grounding_hash"]

    def test_grounding_hash_stable_for_same_evidence(self, tmp_path):
        _write(tmp_path, "config.yaml", "model_id: org/model\n")
        resolver = ModelReferenceResolver()
        a = resolver.resolve_reference(tmp_path)
        b = resolver.resolve_reference(tmp_path)
        assert a["grounding_hash"] == b["grounding_hash"]
