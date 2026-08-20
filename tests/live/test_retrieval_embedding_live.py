"""Opt-in external Embedding smoke; skipped in normal CI."""

import os

import pytest

from auto_harness.retrieval.embeddings import OpenAICompatibleEmbeddingProvider


@pytest.mark.skipif(
    os.environ.get("RUN_RETRIEVAL_EMBEDDING_LIVE") != "1",
    reason="set RUN_RETRIEVAL_EMBEDDING_LIVE=1 for external smoke",
)
def test_external_embedding_returns_real_dimension():
    dimension = int(os.environ["RETRIEVAL_EMBEDDING_DIMENSION"])
    provider = OpenAICompatibleEmbeddingProvider(
        api_base=os.environ["RETRIEVAL_EMBEDDING_API_BASE"],
        model=os.environ["RETRIEVAL_EMBEDDING_MODEL"],
        api_key_env=os.environ.get(
            "RETRIEVAL_EMBEDDING_API_KEY_ENV",
            "AUTO_HARNESS_RETRIEVAL_EMBEDDING_API_KEY",
        ),
        dimension=dimension,
        timeout_seconds=30,
    )
    result = provider.embed_query("deployment agent evidence retrieval")
    assert result.provider == "openai_compatible"
    assert result.dimension == dimension
    assert result.request_id
