import pytest

from auto_harness.config import HarnessConfig
from auto_harness.retrieval.embeddings import FakeEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from auto_harness.retrieval.fusion import reciprocal_rank_fusion
from auto_harness.retrieval.schemas import RetrievalChunk
from auto_harness.retrieval.service import RetrievalService
from auto_harness.retrieval.store import RetrievalStore
from auto_harness.retrieval.vector import ExactVectorIndex


def _chunk(chunk_id, text):
    return RetrievalChunk(
        chunk_id=chunk_id, document_id="doc_" + chunk_id,
        chunker_version="test", ordinal=0, text=text,
        text_sha256=chunk_id, token_estimate=10, source_type="repository",
        repository_fingerprint="fp", path=chunk_id + ".py",
    )


def _config(**overrides):
    retrieval = {
        "enabled": True, "mode": "hybrid", "dense_enabled": True,
        "embedding_provider": "fake", "sources": ["repository"],
    }
    retrieval.update(overrides)
    return HarnessConfig(retrieval=retrieval)


def test_fake_embedding_is_deterministic_and_dimension_checked():
    provider = FakeEmbeddingProvider(dimension=16)
    first = provider.embed_query("cuda memory failure")
    second = provider.embed_query("cuda memory failure")
    assert first.vectors == second.vectors
    assert len(first.vectors[0]) == 16


def test_exact_vector_search_and_rrf_are_stable():
    provider = FakeEmbeddingProvider(dimension=32)
    chunks = [_chunk("a", "cuda out of memory"), _chunk("b", "gradio endpoint trace")]
    vectors = provider.embed_documents([item.text for item in chunks]).vectors
    index = ExactVectorIndex()
    index.add(chunks, vectors)
    dense = index.search(provider.embed_query("cuda memory").vectors[0], 2)
    fused = reciprocal_rank_fusion([(chunks[1], 2.0), (chunks[0], 1.0)], dense)
    assert {item[0].chunk_id for item in fused} == {"a", "b"}
    assert all("fusion" in item[2] for item in fused)


def test_hybrid_service_builds_vectors_and_returns_trace(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "serve.py").write_text("def launch_vllm():\n    # cuda memory allocation\n    pass\n", encoding="utf-8")
    provider = FakeEmbeddingProvider(dimension=32)
    service = RetrievalService(
        RetrievalStore(tmp_path / "index.sqlite"), _config(),
        embedding_provider=provider, vector_index=ExactVectorIndex(),
    )
    manifest = service.build_repository_index(repo, "fp")
    assert manifest.embedding["provider"] == "fake"
    hits, trace = service.retrieve(
        text="cuda memory", purpose="diagnose_failure",
        context={"task_id": "t1", "repository_fingerprint": "fp", "stage": "replan"},
        sources=["repository"],
    )
    assert hits and hits[0].requires_exact_read is True
    assert trace.degradation["occurred"] is False
    assert trace.candidate_counts["dense"] >= 1


def test_hybrid_degrades_to_lexical_when_provider_unavailable(tmp_path):
    config = _config(fail_closed=False)
    service = RetrievalService(RetrievalStore(tmp_path / "index.sqlite"), config)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("gradio trace endpoint", encoding="utf-8")
    service.build_repository_index(repo, "fp")
    hits, trace = service.retrieve(
        text="gradio trace", purpose="replan",
        context={"task_id": "t", "repository_fingerprint": "fp", "stage": "replan"},
        sources=["repository"],
    )
    assert hits
    assert trace.degradation["to"] == "lexical"


def test_external_embedding_requires_https_and_never_accepts_key_value():
    with pytest.raises(ValueError, match="https"):
        OpenAICompatibleEmbeddingProvider(
            api_base="http://example.com/v1", model="embed", api_key_env="KEY", dimension=3,
        )

