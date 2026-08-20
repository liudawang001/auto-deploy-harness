"""Executor for policy-bounded retrieval tool calls."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.retrieval.service import RetrievalService
from auto_harness.retrieval.store import RetrievalStore
from auto_harness.retrieval.artifacts import RetrievalArtifacts
from auto_harness.utils.atomic import FileLock
from auto_harness.tools.registry import ToolRegistry


class RetrievalToolExecutor:
    def __init__(self, config: Any = None, registry=None, service: Optional[RetrievalService] = None) -> None:
        self.config = config
        self.registry = registry or ToolRegistry()
        self.service = service

    def validate_contract(self) -> None:
        implemented = {
            name for name, schema in self.registry.tools.items()
            if schema.implemented and schema.executor == "retrieval"
        }
        if implemented != {"retrieve_deployment_context"}:
            raise RuntimeError("retrieval tool registry/executor mismatch: %s" % sorted(implemented))

    def execute(self, tool_call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        schema = self.registry.tools.get(tool_call.name)
        if schema is None or not schema.implemented or schema.executor != "retrieval":
            return self._reject(tool_call.name, "tool is not an implemented retrieval tool")
        settings = getattr(self.config, "retrieval", {}) if self.config else {}
        if not bool(settings.get("enabled", False)):
            return self._reject(tool_call.name, "retrieval is disabled")
        if str(context.get("stage", "")) not in schema.stages:
            return self._reject(tool_call.name, "retrieval tool is unavailable in this stage")
        try:
            service = self.service or self._service(context)
            self._check_call_budget(context, settings)
            manifest = self._ensure_index(service, context)
            artifacts = self._artifacts(context)
            if artifacts is not None:
                artifacts.write_manifest(manifest)
            value = dict(tool_call.input or {})
            hits, trace = service.retrieve(
                text=str(value.get("query", "")),
                purpose=str(value.get("purpose", "")),
                context=context,
                sources=value.get("sources"), top_k=value.get("top_k"),
            )
            self._append_trace(context, trace.to_dict())
            if artifacts is not None:
                artifacts.finalize(requested_mode=str(settings.get("mode", "lexical")))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return ToolResult(
                status="failed", tool_name=tool_call.name, category="read_only",
                policy_allowed=True, executed=False, metadata_only=True,
                error=("retrieval failed: %s" % str(exc))[:300],
            )
        evidence = {
            "query_id": trace.query["query_id"],
            "hits": [hit.to_dict() for hit in hits],
            "exact_read_requests": exact_read_requests(
                hits,
                max_files=int(getattr(
                    self.config, "agent_repo_max_requests_per_round", 4,
                ) if self.config else 4),
            ),
            "trace": trace.to_dict(),
            "authority": "candidate_only",
        }
        return ToolResult(
            status="passed", tool_name=tool_call.name, category="read_only",
            policy_allowed=True, executed=False, applied=False,
            metadata_only=True, evidence=evidence,
        )

    def _service(self, context: Dict[str, Any]) -> RetrievalService:
        run_dir = Path(context.get("run_dir") or ".")
        path = Path(context.get("retrieval_store_path") or run_dir / "retrieval" / "index.sqlite")
        provider = None
        vector_index = None
        settings = getattr(self.config, "retrieval", {}) if self.config else {}
        if settings.get("embedding_provider") == "fake":
            from auto_harness.retrieval.embeddings import FakeEmbeddingProvider
            from auto_harness.retrieval.vector import ExactVectorIndex
            provider = FakeEmbeddingProvider()
            vector_index = ExactVectorIndex(
                maximum_chunks=int(settings.get("max_repository_chunks", 50000)),
            )
        elif settings.get("embedding_provider") == "openai_compatible":
            required = (
                settings.get("external_embedding_enabled"),
                settings.get("embedding_api_base"),
                settings.get("embedding_model"),
                settings.get("embedding_api_key_env"),
            )
            if all(required):
                from auto_harness.retrieval.embeddings import OpenAICompatibleEmbeddingProvider
                from auto_harness.retrieval.vector import ExactVectorIndex
                provider = OpenAICompatibleEmbeddingProvider(
                    api_base=str(settings["embedding_api_base"]),
                    model=str(settings["embedding_model"]),
                    api_key_env=str(settings["embedding_api_key_env"]),
                    dimension=int(settings.get("embedding_dimension", 1536)),
                    timeout_seconds=int(settings.get("embedding_timeout_seconds", 30)),
                )
                vector_index = ExactVectorIndex(
                    maximum_chunks=int(settings.get("max_repository_chunks", 50000)),
                )
        return RetrievalService(
            RetrievalStore(path), self.config,
            embedding_provider=provider, vector_index=vector_index,
        )

    @staticmethod
    def _ensure_index(service: RetrievalService, context: Dict[str, Any]):
        fingerprint = str(context.get("repository_fingerprint", ""))
        try:
            manifest = service.store.manifest(fingerprint or "global")
        except ValueError:
            manifest = None
        if manifest is None:
            repo_dir = Path(context.get("repo_dir", ""))
            if not repo_dir.is_dir() or not fingerprint:
                raise ValueError("retrieval index and repository scope are unavailable")
            manifest = service.build_repository_index(
                repo_dir,
                fingerprint,
                memory_entries=RetrievalToolExecutor._memory_entries(service.config),
                runtime_records=context.get("retrieval_runtime_records") or [],
                skill_entries=context.get("retrieval_active_skills") or [],
                task_id=str(context.get("task_id", "")),
            )
        elif service.vector_index is not None and (manifest.embedding or {}).get("enabled"):
            identity = str((manifest.embedding or {}).get("identity", ""))
            chunks = service.store.chunks()
            vectors = service.store.embeddings(identity, [item.chunk_id for item in chunks])
            selected = [item for item in chunks if item.chunk_id in vectors]
            service.vector_index.add(selected, [vectors[item.chunk_id] for item in selected])
        return manifest

    @staticmethod
    def _memory_entries(config: Any):
        settings = getattr(config, "retrieval", {}) if config else {}
        if not ({"verified_memory", "issue_memory"} & set(settings.get("sources", []))):
            return []
        memory_dir = getattr(config, "memory_path", None)
        path = Path(memory_dir) / "deployment_issues.jsonl" if memory_dir else None
        if path is None or not path.is_file():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1000]:
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                entries.append(value)
        return entries

    @staticmethod
    def _artifacts(context: Dict[str, Any]):
        run_dir = context.get("run_dir")
        return RetrievalArtifacts(Path(run_dir)) if run_dir else None

    @staticmethod
    def _append_trace(context: Dict[str, Any], trace: Dict[str, Any]) -> None:
        run_dir = context.get("run_dir")
        if not run_dir:
            return
        path = Path(run_dir) / "retrieval" / "queries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n"
        with FileLock(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    @staticmethod
    def _check_call_budget(context: Dict[str, Any], settings: Dict[str, Any]) -> None:
        run_dir = context.get("run_dir")
        if not run_dir:
            return
        path = Path(run_dir) / "retrieval" / "queries.jsonl"
        if not path.exists():
            return
        task_id = str(context.get("task_id", ""))
        stage = str(context.get("stage", ""))
        used = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                query = (json.loads(line).get("query") or {})
            except (TypeError, ValueError):
                continue
            if str(query.get("task_id", "")) == task_id and str(query.get("stage", "")) == stage:
                used += 1
        if used >= int(settings.get("max_tool_calls_per_stage", 2)):
            raise ValueError("retrieval tool call limit exceeded for stage")

    @staticmethod
    def _reject(name: str, reason: str) -> ToolResult:
        return ToolResult(
            status="rejected", tool_name=name, category="read_only",
            policy_allowed=False, executed=False, applied=False, error=reason,
        )


def exact_read_requests(hits, max_files=4):
    files = []
    chunk_ids = []
    query_id = ""
    for hit in hits:
        if not hit.requires_exact_read or not hit.path:
            continue
        query_id = query_id or hit.query_id
        chunk_ids.append(hit.chunk_id)
        files.append({
            "path": hit.path, "start_line": hit.line_start,
            "end_line": hit.line_end,
        })
        if len(files) >= max(1, int(max_files)):
            break
    if not files:
        return []
    return [{
        "tool": "read_selected_files",
        "input": {
            "files": files,
            "retrieved_from_query_id": query_id,
            "retrieval_chunk_ids": chunk_ids,
        },
    }]


__all__ = ["RetrievalToolExecutor", "exact_read_requests"]
