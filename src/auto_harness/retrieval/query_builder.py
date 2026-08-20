"""Build bounded retrieval queries from trusted runtime state."""

import re
import uuid
from typing import Any, Dict, Iterable

from auto_harness.retrieval.schemas import RetrievalQuery


class RetrievalQueryBuilder:
    def build(self, text: str, *, context: Dict[str, Any], purpose: str, sources=None, top_k: int = 8, mode: str = "lexical", max_context_tokens: int = 3000) -> RetrievalQuery:
        normalized = " ".join(str(text or "").split())[:500]
        if not normalized:
            facts = [context.get("objective"), context.get("failure_category"), context.get("error")]
            normalized = " ".join(str(item) for item in facts if item)[:500]
        if not normalized:
            raise ValueError("retrieval query is empty")
        return RetrievalQuery(
            query_id="qry_" + uuid.uuid4().hex,
            query_text=normalized, purpose=str(purpose),
            task_id=str(context.get("task_id", "")),
            repository_fingerprint=str(context.get("repository_fingerprint", "")),
            stage=str(context.get("stage", "")),
            frameworks=[str(item) for item in context.get("frameworks", [])],
            failure_categories=[str(context.get("failure_category"))] if context.get("failure_category") else [],
            sources=list(sources or ["repository", "verified_memory"]),
            top_k=int(top_k), max_context_tokens=int(max_context_tokens),
            requested_by=str(context.get("requested_by", "runtime")), mode=mode,
        )

