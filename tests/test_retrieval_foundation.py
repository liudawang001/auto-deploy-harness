import json
from pathlib import Path

import pytest

from auto_harness.config import HarnessConfig
from auto_harness.retrieval.chunkers import ChunkerRegistry
from auto_harness.retrieval.ingestion import RetrievalIngestor
from auto_harness.retrieval.lexical import BM25Retriever
from auto_harness.retrieval.policy import RetrievalPolicy
from auto_harness.retrieval.query_builder import RetrievalQueryBuilder
from auto_harness.retrieval.schemas import (
    RetrievalDocument,
    RetrievalManifest,
    RetrievalQuery,
    stable_hash,
)
from auto_harness.retrieval.store import RetrievalStore


def _document(path="app.py", content="def serve():\n    return 'ok'\n"):
    return RetrievalDocument(
        document_id="doc_1", source_type="repository",
        source_identity="repo:fp:%s" % path,
        repository_fingerprint="fp", path=path,
        content_sha256=stable_hash(content).split(":", 1)[1],
        language="python", trust_level="untrusted_repository",
    )


def test_retrieval_contracts_reject_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="unknown fields"):
        RetrievalQuery.from_dict({
            "query_id": "q", "query_text": "x", "purpose": "p",
            "unknown_authority": True,
        })
    with pytest.raises(ValueError, match="top_k"):
        RetrievalQuery(query_id="q", query_text="x", purpose="p", top_k=99)


def test_python_chunker_uses_symbols_and_falls_back_on_syntax_error():
    registry = ChunkerRegistry()
    chunks = registry.chunk(_document(), "import os\n\ndef serve():\n    return 'ok'\n")
    assert any(chunk.symbol == "serve" for chunk in chunks)
    fallback = registry.chunk(_document(), "def broken(:\n  pass\n")
    assert fallback[0].chunker_version == "line_window_v1"


def test_repository_ingestion_redacts_secrets_and_skips_denied_files(tmp_path):
    (tmp_path / "app.py").write_text("API_KEY=sk-abcdefghijklmnopqrstuvwxyz\ndef serve(): pass\n", encoding="utf-8")
    (tmp_path / ".env").write_text("PASSWORD=super-secret", encoding="utf-8")
    documents, chunks = RetrievalIngestor().ingest_repository(tmp_path, "fp")
    assert [item.path for item in documents] == ["app.py"]
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in "".join(item.text for item in chunks)
    assert "[REDACTED_SECRET]" in "".join(item.text for item in chunks)


def test_memory_ingestion_preserves_verified_quality():
    entries = [{
        "id": "m1", "stage": "verify", "category": "trace_not_observed",
        "frameworks": ["gradio"], "verified_success": True,
        "root_cause": "queue endpoint required",
        "verification_trace_id": "trace-1", "repair_action_hash": "sha256:repair",
        "repair_action_status": "executed",
    }]
    documents, chunks = RetrievalIngestor().ingest_memory_entries(entries)
    assert documents[0].source_type == "verified_memory"
    assert documents[0].trust_level == "verified_memory"
    assert chunks[0].metadata["verified"] is True


def test_runtime_and_skill_ingestion_preserve_scope_and_trust():
    ingestor = RetrievalIngestor(HarnessConfig(retrieval={"index_runtime_logs": True}))
    documents, _ = ingestor.ingest_runtime_records([
        {"stage": "runner", "message": "trace ok"},
        {"stage": "runner", "source_type": "runtime_log", "message": "raw log"},
    ], "task-1")
    assert {item.task_id for item in documents} == {"task-1"}
    assert {item.trust_level for item in documents} == {"runtime_evidence", "untrusted_runtime_log"}
    skill_documents, _ = ingestor.ingest_skill_entries([{
        "name": "gradio-verify", "version": "1", "content": "inspect queue trace",
        "stages": ["verify"],
    }])
    assert skill_documents[0].source_type == "active_skill"
    assert skill_documents[0].trust_level == "reviewed_skill"


def test_bm25_finds_exact_configuration_terms():
    registry = ChunkerRegistry()
    chunks = registry.chunk(_document("app.py"), "def serve():\n    return 'ok'\n")
    chunks += registry.chunk(_document("config.py"), "model_runtime_max_model_len = 4096\n")
    hits = BM25Retriever().search("model_runtime_max_model_len", chunks)
    assert hits[0][0].path == "config.py"


def test_sqlite_fts5_path_is_policy_filtered_when_available(tmp_path):
    store = RetrievalStore(tmp_path / "fts.sqlite")
    if not store.fts5_available:
        pytest.skip("SQLite build has no FTS5")
    allowed = _document("allowed.py")
    denied = _document("denied.py", "other")
    denied.document_id = "doc_2"
    registry = ChunkerRegistry()
    allowed_chunks = registry.chunk(allowed, "unique_deploy_symbol")
    denied_chunks = registry.chunk(denied, "unique_deploy_symbol secret")
    manifest = RetrievalManifest(repository_fingerprint="fp", document_count=2, chunk_count=2)
    store.replace([allowed, denied], allowed_chunks + denied_chunks, manifest)
    hits = store.lexical_search("unique_deploy_symbol", allowed_chunks, 10)
    assert [item[0].path for item in hits] == ["allowed.py"]


def test_store_requires_complete_valid_manifest(tmp_path):
    document = _document()
    chunks = ChunkerRegistry().chunk(document, "def serve():\n    return 'ok'\n")
    store = RetrievalStore(tmp_path / "index.sqlite")
    manifest = RetrievalManifest(repository_fingerprint="fp", document_count=1, chunk_count=len(chunks))
    store.replace([document], chunks, manifest)
    loaded = store.manifest("fp")
    assert loaded.completed is True
    assert [item.chunk_id for item in store.chunks()] == [item.chunk_id for item in chunks]


def test_retrieval_policy_injects_scope_and_blocks_unverified_memory():
    config = HarnessConfig(retrieval={"enabled": True, "sources": ["repository", "verified_memory"]})
    context = {"task_id": "t1", "repository_fingerprint": "fp", "stage": "plan"}
    query = RetrievalQueryBuilder().build(
        "find entrypoint", context=context, purpose="plan_repository",
        sources=["repository"], top_k=8,
    )
    assert RetrievalPolicy(config).validate(query, context).repository_fingerprint == "fp"
    query.sources = ["issue_memory"]
    with pytest.raises(ValueError, match="not allowed|disabled"):
        RetrievalPolicy(config).validate(query, context)
