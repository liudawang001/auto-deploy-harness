import pytest

from auto_harness.config import HarnessConfig
from auto_harness.retrieval.service import RetrievalService
from auto_harness.retrieval.store import RetrievalStore


def _config():
    return HarnessConfig(retrieval={
        "enabled": True, "mode": "lexical", "sources": ["repository"],
    })


def test_index_transaction_rolls_back_without_partial_manifest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("stable deployment entrypoint\n", encoding="utf-8")
    store = RetrievalStore(tmp_path / "index.sqlite")
    original = RetrievalService(store, _config()).build_repository_index(repo, "fp")

    def fail(point):
        if point == "after_chunk_write_before_lexical_commit":
            raise RuntimeError("injected")

    (repo / "app.py").write_text("partial replacement must roll back\n", encoding="utf-8")
    failing = RetrievalService(
        RetrievalStore(tmp_path / "index.sqlite", fault_hook=fail), _config(),
    )
    with pytest.raises(RuntimeError, match="injected"):
        failing.build_repository_index(repo, "fp")
    recovered = store.manifest("fp")
    assert recovered.manifest_hash == original.manifest_hash
    assert "stable deployment" in store.chunks()[0].text


def test_query_refuses_missing_completed_manifest(tmp_path):
    service = RetrievalService(RetrievalStore(tmp_path / "empty.sqlite"), _config())
    with pytest.raises(ValueError, match="manifest"):
        service.retrieve(
            text="entrypoint", purpose="plan_repository",
            context={"task_id": "t", "repository_fingerprint": "fp", "stage": "plan"},
            sources=["repository"],
        )
