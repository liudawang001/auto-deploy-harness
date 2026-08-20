"""Reranker contracts with deterministic no-op implementation."""


class IdentityReranker:
    name = "identity"

    def rerank(self, query, hits):
        return list(hits)

