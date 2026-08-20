"""Deterministic fusion of lexical and dense rankings."""

from typing import Dict, Iterable, List


def reciprocal_rank_fusion(lexical, dense, *, limit: int = 20, k: int = 60, lexical_weight: float = 1.0, dense_weight: float = 1.0):
    items, scores = {}, {}
    component_scores: Dict[str, Dict[str, float]] = {}
    for name, ranked, weight in (("lexical", lexical, lexical_weight), ("dense", dense, dense_weight)):
        for rank, (chunk, raw_score) in enumerate(ranked, 1):
            items[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            component_scores.setdefault(chunk.chunk_id, {})[name] = float(raw_score)
    ordered = sorted(items, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        (items[chunk_id], scores[chunk_id], {
            "lexical": component_scores[chunk_id].get("lexical"),
            "dense": component_scores[chunk_id].get("dense"),
            "fusion": scores[chunk_id],
        }) for chunk_id in ordered
    ]

