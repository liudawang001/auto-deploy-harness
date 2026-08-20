"""Safe retrieval artifacts and observable-only runtime summaries."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from auto_harness.models.base import write_json


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


class RetrievalArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.retrieval_dir = self.run_dir / "retrieval"
        self.reports_dir = self.run_dir / "reports"

    def write_manifest(self, manifest: Any) -> Path:
        path = self.retrieval_dir / "index_manifest.json"
        write_json(path, manifest.to_dict() if hasattr(manifest, "to_dict") else manifest)
        return path

    def finalize(self, *, requested_mode: str = "lexical") -> Dict[str, Any]:
        manifest = self._read(self.retrieval_dir / "index_manifest.json")
        traces = _jsonl(self.retrieval_dir / "queries.jsonl")
        observations = _jsonl(self.reports_dir / "observation_ledger.jsonl")
        modes = [str((item.get("degradation") or {}).get("to", "")) for item in traces]
        effective = modes[-1] if modes and modes[-1] else requested_mode
        latencies = sorted(int((item.get("latency_ms") or {}).get("total", 0)) for item in traces)
        exact = [item for item in observations if item.get("retrieved_from_query_id") and item.get("status") == "passed"]
        embedding = manifest.get("embedding") or {}
        summary = {
            "schema_version": 1,
            "status": "completed" if manifest.get("completed") else "not_indexed",
            "mode_requested": requested_mode,
            "mode_effective": effective,
            "degraded": any(bool((item.get("degradation") or {}).get("occurred")) for item in traces),
            "degradation_reasons": sorted({
                str((item.get("degradation") or {}).get("reason")) for item in traces
                if (item.get("degradation") or {}).get("reason")
            }),
            "index_manifest_hash": manifest.get("manifest_hash", ""),
            "sources": sorted({source for item in traces for source in ((item.get("query") or {}).get("sources") or [])}),
            "documents": int(manifest.get("document_count", 0) or 0),
            "chunks": int(manifest.get("chunk_count", 0) or 0),
            "queries": len(traces),
            "hits_returned": sum(int((item.get("candidate_counts") or {}).get("returned", 0)) for item in traces),
            "hits_exactly_read": len(exact),
            "hits_grounding_accepted": sum(int(item.get("retrieval_grounding_accepted", 0) or 0) for item in observations),
            "retrieval_tokens": sum(int((item.get("budgets") or {}).get("returned_tokens", 0)) for item in traces),
            "embedding_requests": int(bool(embedding.get("enabled"))),
            "external_embedding_used": embedding.get("provider") not in {None, "", "disabled", "fake"},
            "secret_redactions": sum(int(item.get("redactions", 0) or 0) for item in traces),
            "stale_hits_rejected": sum(int(item.get("stale_hits_rejected", 0) or 0) for item in traces),
            "latency_ms_p50": self._percentile(latencies, 0.50),
            "latency_ms_p95": self._percentile(latencies, 0.95),
            "rag_helped": False,
            "rag_required": False,
        }
        write_json(self.reports_dir / "retrieval_summary.json", summary)
        contribution = self.contribution(summary, observations)
        write_json(self.reports_dir / "retrieval_contribution.json", contribution)
        return summary

    def contribution(self, summary: Dict[str, Any], observations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        accepted = int(summary.get("hits_grounding_accepted", 0) or 0)
        # Retrieval usage alone is never causal proof. A later evaluator may
        # add baseline/outcome evidence and then recompute this artifact.
        return {
            "schema_version": 1,
            "rag_helped": False,
            "rag_required": False,
            "retrieval_used": int(summary.get("queries", 0) or 0) > 0,
            "exact_read_count": int(summary.get("hits_exactly_read", 0) or 0),
            "accepted_grounding_count": accepted,
            "causal_evidence": [],
            "reason": "usage_without_baseline_delta_is_not_causal_evidence",
        }

    @staticmethod
    def _percentile(values: List[int], ratio: float) -> int:
        if not values:
            return 0
        index = max(0, min(len(values) - 1, int(round((len(values) - 1) * ratio))))
        return values[index]

    @staticmethod
    def _read(path: Path) -> Dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}


__all__ = ["RetrievalArtifacts"]
