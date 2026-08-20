"""Scope, source, and budget policy for retrieval requests."""

from typing import Any, Dict

from auto_harness.retrieval.schemas import RetrievalQuery


class RetrievalPolicy:
    PURPOSES = {"plan_repository", "diagnose_failure", "select_repair", "select_verify_strategy", "replan"}

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def validate(self, query: RetrievalQuery, context: Dict[str, Any]) -> RetrievalQuery:
        settings = getattr(self.config, "retrieval", {}) if self.config else {}
        if not bool(settings.get("enabled", False)):
            raise ValueError("retrieval is disabled")
        if query.purpose not in self.PURPOSES:
            raise ValueError("unsupported retrieval purpose")
        expected_repo = str(context.get("repository_fingerprint", ""))
        expected_task = str(context.get("task_id", ""))
        if query.repository_fingerprint != expected_repo or query.task_id != expected_task:
            raise ValueError("retrieval scope mismatch")
        allowed_sources = set(settings.get("sources", ["repository", "verified_memory"]))
        if not set(query.sources).issubset(allowed_sources):
            raise ValueError("retrieval source is not allowed")
        query.top_k = min(query.top_k, int(settings.get("max_top_k", 12)))
        query.max_context_tokens = min(query.max_context_tokens, int(settings.get("max_context_tokens", 3000)))
        if "issue_memory" in query.sources and not bool(settings.get("share_unverified_memory", False)):
            raise ValueError("unverified memory retrieval is disabled")
        return query

