"""Offline labeled retrieval metrics; never inferred from production traffic."""

import math
from typing import Any, Dict, Iterable, List, Mapping


def _dcg(labels: List[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(labels))


class RetrievalEvaluator:
    def evaluate(
        self,
        cases: Iterable[Mapping[str, Any]],
        rankings: Mapping[str, List[str]],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        rows = []
        for case in cases:
            case_id = str(case["case_id"])
            relevant = set(case.get("relevant_chunk_labels") or [])
            forbidden = set(case.get("forbidden_chunk_labels") or [])
            ranked = list(rankings.get(case_id, []))
            hits = [1 if item in relevant else 0 for item in ranked]
            found = sum(hits)
            first = next((index + 1 for index, value in enumerate(hits) if value), 0)
            ideal = [1] * min(len(relevant), len(ranked))
            rows.append({
                "case_id": case_id,
                "recall_at_k": found / len(relevant) if relevant else 1.0,
                "precision_at_k": found / len(ranked) if ranked else 0.0,
                "mrr": 1.0 / first if first else 0.0,
                "ndcg_at_k": (_dcg(hits) / _dcg(ideal)) if ideal else 1.0,
                "forbidden_hit_count": sum(item in forbidden for item in ranked),
            })
        count = len(rows)
        average = lambda key: sum(float(row[key]) for row in rows) / count if count else 0.0
        return {
            "schema_version": 1,
            "status": "completed" if rows else "failed",
            "mode": mode,
            "case_count": count,
            "metrics": {
                "recall_at_k": average("recall_at_k"),
                "precision_at_k": average("precision_at_k"),
                "mrr": average("mrr"),
                "ndcg_at_k": average("ndcg_at_k"),
                "forbidden_hit_count": sum(row["forbidden_hit_count"] for row in rows),
            },
            "cases": rows,
        }


def compare_modes(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    base = baseline.get("metrics") or {}
    current = candidate.get("metrics") or {}
    return {
        "baseline_mode": baseline.get("mode", ""),
        "candidate_mode": candidate.get("mode", ""),
        "recall_delta": float(current.get("recall_at_k", 0)) - float(base.get("recall_at_k", 0)),
        "mrr_delta": float(current.get("mrr", 0)) - float(base.get("mrr", 0)),
        "forbidden_hit_delta": int(current.get("forbidden_hit_count", 0)) - int(base.get("forbidden_hit_count", 0)),
    }


__all__ = ["RetrievalEvaluator", "compare_modes"]
