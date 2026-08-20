"""Pluggable retrieval backend protocols."""

from typing import List, Protocol

from auto_harness.retrieval.schemas import EmbeddingResult, RetrievalChunk, RetrievalQuery


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    def embed_documents(self, texts: List[str], *, request_context=None) -> EmbeddingResult:
        ...

    def embed_query(self, text: str, *, request_context=None) -> EmbeddingResult:
        ...


class VectorIndex(Protocol):
    def add(self, chunks: List[RetrievalChunk], vectors: List[List[float]]) -> None:
        ...

    def search(self, query_vector: List[float], limit: int) -> List[tuple]:
        ...


class Reranker(Protocol):
    def rerank(self, query: RetrievalQuery, hits: list) -> list:
        ...
