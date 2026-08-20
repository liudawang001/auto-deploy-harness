"""High-level retrieval service with policy, budget, and trace enforcement."""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from auto_harness.retrieval.ingestion import RetrievalIngestor
from auto_harness.retrieval.lexical import BM25Retriever
from auto_harness.retrieval.policy import RetrievalPolicy
from auto_harness.retrieval.query_builder import RetrievalQueryBuilder
from auto_harness.retrieval.schemas import (
    RetrievalHit,
    RetrievalManifest,
    RetrievalQuery,
    RetrievalTrace,
)
from auto_harness.retrieval.store import RetrievalStore
from auto_harness.utils.time import utc_now_iso


class RetrievalService:
    def __init__(self, store: RetrievalStore, config: Any = None, *, embedding_provider=None, vector_index=None, reranker=None, fault_hook=None) -> None:
        self.store = store
        self.config = config
        self.settings = getattr(config, "retrieval", {}) if config else {"enabled": True, "mode": "lexical"}
        self.policy = RetrievalPolicy(config)
        self.query_builder = RetrievalQueryBuilder()
        self.lexical = BM25Retriever()
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.reranker = reranker
        self.fault_hook = fault_hook

    def build_repository_index(
        self,
        repo_dir: Path,
        repository_fingerprint: str,
        *,
        memory_entries: Optional[Iterable[Dict[str, Any]]] = None,
        runtime_records: Optional[Iterable[Dict[str, Any]]] = None,
        skill_entries: Optional[Iterable[Dict[str, Any]]] = None,
        task_id: str = "",
    ) -> RetrievalManifest:
        ingestor = RetrievalIngestor(self.config)
        documents, chunks = ingestor.ingest_repository(repo_dir, repository_fingerprint)
        if memory_entries:
            memory_documents, memory_chunks = ingestor.ingest_memory_entries(memory_entries)
            documents.extend(memory_documents)
            chunks.extend(memory_chunks)
        if runtime_records and task_id:
            runtime_documents, runtime_chunks = ingestor.ingest_runtime_records(
                runtime_records, task_id,
            )
            documents.extend(runtime_documents)
            chunks.extend(runtime_chunks)
        if skill_entries:
            skill_documents, skill_chunks = ingestor.ingest_skill_entries(skill_entries)
            documents.extend(skill_documents)
            chunks.extend(skill_chunks)
        self._fault("after_document_scan_before_chunk_write")
        maximum = int(self.settings.get("max_repository_chunks", 50000))
        if len(chunks) > maximum:
            raise ValueError("repository chunk count exceeds configured maximum")
        manifest = RetrievalManifest(
            repository_fingerprint=repository_fingerprint,
            document_count=len(documents), chunk_count=len(chunks),
            chunker_versions={"registry": "v1"},
            embedding={"enabled": False, "provider": "disabled", "model": "", "dimension": 0},
            created_at=utc_now_iso(),
        )
        embedding_payload = self._prepare_embeddings(chunks, manifest)
        self.store.replace(documents, chunks, manifest, embedding_payload=embedding_payload)
        return manifest

    def build_memory_index(self, entries: Iterable[Dict[str, Any]]) -> RetrievalManifest:
        ingestor = RetrievalIngestor(self.config)
        documents, chunks = ingestor.ingest_memory_entries(entries)
        manifest = RetrievalManifest(
            repository_fingerprint="global", document_count=len(documents),
            chunk_count=len(chunks), chunker_versions={"memory": "v1"},
            embedding={"enabled": False, "provider": "disabled", "model": "", "dimension": 0},
            created_at=utc_now_iso(),
        )
        embedding_payload = self._prepare_embeddings(chunks, manifest)
        self.store.replace(documents, chunks, manifest, embedding_payload=embedding_payload)
        return manifest

    def _prepare_embeddings(self, chunks, manifest):
        if not bool(self.settings.get("dense_enabled", False)):
            return None
        if self.embedding_provider is None or self.vector_index is None:
            if bool(self.settings.get("fail_closed", False)):
                raise RuntimeError("dense retrieval provider is unavailable")
            return None
        from auto_harness.retrieval.embeddings import embedding_identity
        try:
            result = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        except (OSError, RuntimeError, TypeError, ValueError):
            if bool(self.settings.get("fail_closed", False)):
                raise
            self.embedding_provider = None
            self.vector_index = None
            return None
        identity = embedding_identity(self.embedding_provider)
        self.vector_index.add(chunks, result.vectors)
        manifest.embedding = {
            "enabled": True, "provider": result.provider,
            "model": result.model, "dimension": result.dimension,
            "identity": identity,
        }
        return identity, {chunk.chunk_id: vector for chunk, vector in zip(chunks, result.vectors)}

    def retrieve(self, *, text: str, purpose: str, context: Dict[str, Any], sources=None, top_k: Optional[int] = None, mode: Optional[str] = None) -> Tuple[List[RetrievalHit], RetrievalTrace]:
        started = time.monotonic()
        query = self.query_builder.build(
            text, context={**context, "requested_by": context.get("requested_by", "agent_tool")},
            purpose=purpose, sources=sources or self.settings.get("sources"),
            top_k=top_k or int(self.settings.get("default_top_k", 8)),
            mode=mode or str(self.settings.get("mode", "lexical")),
            max_context_tokens=int(self.settings.get("max_context_tokens", 3000)),
        )
        query = self.policy.validate(query, context)
        if self.store.manifest(query.repository_fingerprint or "global") is None:
            raise ValueError("completed retrieval manifest is unavailable")
        self._fault("after_complete_manifest_before_query")
        chunks = self.store.chunks(query)
        lexical_started = time.monotonic()
        lexical_limit = int(self.settings.get("lexical_top_n", 30))
        backend = str(self.settings.get("lexical_backend", "auto"))
        if backend in {"auto", "fts5"} and self.store.fts5_available:
            lexical = self.store.lexical_search(query.query_text, chunks, lexical_limit)
            lexical_backend = "fts5"
        elif backend == "fts5":
            raise RuntimeError("configured SQLite FTS5 backend is unavailable")
        else:
            lexical = self.lexical.search(query.query_text, chunks, limit=lexical_limit)
            lexical_backend = "python_bm25"
        lexical_ms = int((time.monotonic() - lexical_started) * 1000)
        ranked = [(chunk, score, {"lexical": score, "fusion": score}) for chunk, score in lexical]
        degradation = {"occurred": False, "from": query.mode, "to": query.mode, "reason": ""}
        if query.mode != "lexical":
            ranked, degradation = self._hybrid(query, chunks, lexical)
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked)
        hits = self._bounded_hits(query, ranked)
        trace = RetrievalTrace(
            query=query.to_dict(),
            candidate_counts={
                "metadata_allowed": len(chunks), "lexical": len(lexical),
                "dense": sum(1 for _, _, scores in ranked if scores.get("dense") is not None),
                "fused": len(ranked), "returned": len(hits),
            },
            hits=[hit.to_dict() for hit in hits],
            budgets={
                "requested_tokens": query.max_context_tokens,
                "returned_tokens": sum(max(1, len(hit.preview.encode("utf-8")) // 4) for hit in hits),
                "truncated_hits": sum(bool(hit.metadata.get("truncated")) for hit in hits),
            },
            latency_ms={"lexical": lexical_ms, "total": int((time.monotonic() - started) * 1000)},
            degradation=degradation,
        )
        trace.query["lexical_backend"] = lexical_backend
        return hits, trace

    def _hybrid(self, query, chunks, lexical):
        if self.embedding_provider is None or self.vector_index is None:
            if bool(self.settings.get("fail_closed", False)):
                raise RuntimeError("dense retrieval provider is unavailable")
            fallback = [
                (chunk, score, {"lexical": score, "fusion": score})
                for chunk, score in lexical
            ]
            return fallback, {"occurred": True, "from": query.mode, "to": "lexical", "reason": "embedding_provider_unavailable"}
        from auto_harness.retrieval.fusion import reciprocal_rank_fusion
        embedded = self.embedding_provider.embed_query(query.query_text)
        dense = self.vector_index.search(embedded.vectors[0], int(self.settings.get("dense_top_n", 30)))
        if query.mode == "dense":
            return [
                (chunk, score, {"dense": score, "fusion": score})
                for chunk, score in dense
            ], {"occurred": False, "from": "dense", "to": "dense", "reason": ""}
        return reciprocal_rank_fusion(lexical, dense, limit=int(self.settings.get("fusion_top_n", 20))), {"occurred": False, "from": query.mode, "to": query.mode, "reason": ""}

    def _bounded_hits(self, query: RetrievalQuery, ranked) -> List[RetrievalHit]:
        result, used, per_document, seen = [], 0, defaultdict(int), set()
        max_hit_chars = int(self.settings.get("max_hit_tokens", 640)) * 4
        max_per_doc = int(self.settings.get("max_hits_per_document", 3))
        budget_chars = query.max_context_tokens * 4
        for chunk, _score, scores in ranked:
            if chunk.text_sha256 in seen or per_document[chunk.document_id] >= max_per_doc:
                continue
            preview = chunk.text[:max_hit_chars]
            if result and used + len(preview.encode("utf-8")) > budget_chars:
                continue
            seen.add(chunk.text_sha256)
            per_document[chunk.document_id] += 1
            used += len(preview.encode("utf-8"))
            result.append(RetrievalHit(
                query_id=query.query_id, rank=len(result) + 1,
                chunk_id=chunk.chunk_id, source_type=chunk.source_type,
                path=chunk.path, line_start=chunk.line_start, line_end=chunk.line_end,
                preview=preview, scores=dict(scores), trust_level=chunk.trust_level,
                requires_exact_read=chunk.source_type == "repository",
                metadata={
                    **chunk.metadata, "document_id": chunk.document_id,
                    "text_sha256": chunk.text_sha256,
                    "truncated": len(preview) < len(chunk.text),
                },
            ))
            if len(result) >= query.top_k:
                break
        return result

    def _fault(self, point: str) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point)
