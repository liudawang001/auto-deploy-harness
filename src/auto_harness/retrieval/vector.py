"""Exact cosine vector index for bounded per-repository corpora."""

import math
from typing import Dict, List


class ExactVectorIndex:
    def __init__(self, maximum_chunks: int = 50000) -> None:
        self.maximum_chunks = maximum_chunks
        self._items: Dict[str, tuple] = {}

    def add(self, chunks, vectors: List[List[float]]) -> None:
        chunks = list(chunks)
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        if len(chunks) > self.maximum_chunks:
            raise ValueError("vector index exceeds configured chunk limit")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValueError("vector dimensions are inconsistent")
        for chunk, vector in zip(chunks, vectors):
            self._items[chunk.chunk_id] = (chunk, list(vector))

    def search(self, query_vector: List[float], limit: int):
        ranked = []
        qnorm = math.sqrt(sum(value * value for value in query_vector))
        if qnorm == 0:
            return []
        for chunk, vector in self._items.values():
            if len(vector) != len(query_vector):
                continue
            norm = math.sqrt(sum(value * value for value in vector))
            score = sum(left * right for left, right in zip(query_vector, vector)) / (qnorm * norm) if norm else 0.0
            ranked.append((chunk, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0].chunk_id))[:limit]

