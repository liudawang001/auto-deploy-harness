"""Safe ingestion of repository, runtime, memory, and skill evidence."""

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from auto_harness.agent.safety import AgentInputSanitizer
from auto_harness.memory.quality import MemoryQualityGate
from auto_harness.retrieval.chunkers import ChunkerRegistry
from auto_harness.retrieval.schemas import RetrievalChunk, RetrievalDocument, stable_hash
from auto_harness.tools.repository_policy import RepositoryReadPolicy
from auto_harness.utils.time import utc_now_iso


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "runs", ".conda"}
SKIP_EXTENSIONS = {
    ".bin", ".pt", ".pth", ".onnx", ".safetensors", ".gguf", ".ckpt",
    ".png", ".jpg", ".jpeg", ".gif", ".mp3", ".mp4", ".zip", ".gz",
    ".whl", ".so", ".dylib", ".dll", ".exe", ".pyc",
}


class RetrievalIngestor:
    def __init__(self, config: Any = None, chunkers=None) -> None:
        self.config = config
        self.chunkers = chunkers or ChunkerRegistry()
        self.sanitizer = AgentInputSanitizer()
        self.redaction_count = 0
        self.risks: List[Dict[str, Any]] = []

    def ingest_repository(self, repo_dir: Path, repository_fingerprint: str) -> Tuple[List[RetrievalDocument], List[RetrievalChunk]]:
        root = Path(repo_dir).resolve()
        if not root.is_dir() or not repository_fingerprint:
            raise ValueError("repository root and fingerprint are required")
        documents, chunks = [], []
        max_chars = int(getattr(self.config, "agent_repo_max_chars_per_read", 12000) if self.config else 12000)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            rel = relative.as_posix()
            if path.suffix.lower() in SKIP_EXTENSIONS or not RepositoryReadPolicy.path_allowed(rel):
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if len(raw) > max_chars * 20:
                raw = raw[: max_chars * 20]
            sanitized = self.sanitizer.sanitize_selected_files({rel: raw})
            if rel not in sanitized:
                continue
            text = sanitized[rel]
            self.redaction_count += sum(int(item.get("count", 0)) for item in self.sanitizer.redactions)
            self.risks.extend(self.sanitizer.risks)
            source_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            identity = "repo:%s:%s" % (repository_fingerprint, rel)
            document = RetrievalDocument(
                document_id="doc_" + stable_hash({"identity": identity, "sha": source_sha}).split(":", 1)[1][:24],
                source_type="repository", source_identity=identity,
                repository_fingerprint=repository_fingerprint, path=rel,
                content_sha256=source_sha,
                mime_type=mimetypes.guess_type(rel)[0] or "text/plain",
                language=path.suffix.lstrip(".") or "text",
                trust_level="untrusted_repository", created_at=utc_now_iso(),
            )
            documents.append(document)
            chunks.extend(self.chunkers.chunk(document, text))
        return documents, chunks

    def ingest_memory_entries(self, entries: Iterable[Dict[str, Any]]) -> Tuple[List[RetrievalDocument], List[RetrievalChunk]]:
        documents, chunks = [], []
        for entry in entries or []:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            verified = bool(MemoryQualityGate().classify(entry).get("eligible"))
            source_type = "verified_memory" if verified else "issue_memory"
            trust = "verified_memory" if verified else "unverified_memory"
            safe = self.sanitizer.redact_value(entry)
            text = json.dumps(safe, ensure_ascii=False, sort_keys=True)
            source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            identity = "memory:%s:%s" % (source_type, entry["id"])
            document = RetrievalDocument(
                document_id="doc_" + stable_hash({"identity": identity, "sha": source_sha}).split(":", 1)[1][:24],
                source_type=source_type, source_identity=identity,
                content_sha256=source_sha, path="memory/%s" % entry["id"],
                language="json", trust_level=trust,
                stage_tags=[str(entry.get("stage", ""))] if entry.get("stage") else [],
                framework_tags=[str(item) for item in entry.get("frameworks", [])],
                created_at=str(entry.get("created_at", "")),
                metadata={
                    "memory_id": str(entry["id"]),
                    "verified": verified,
                    "failure_categories": [str(entry.get("category", ""))] if entry.get("category") else [],
                },
            )
            documents.append(document)
            chunks.extend(self.chunkers.chunk(document, text))
        return documents, chunks

    def ingest_runtime_records(self, records: Iterable[Dict[str, Any]], task_id: str) -> Tuple[List[RetrievalDocument], List[RetrievalChunk]]:
        documents, chunks = [], []
        for ordinal, record in enumerate(records or []):
            source_type = (
                "runtime_log" if record.get("source_type") == "runtime_log"
                else "runtime_evidence"
            )
            settings = getattr(self.config, "retrieval", {}) if self.config else {}
            if source_type == "runtime_log" and not bool(settings.get("index_runtime_logs", True)):
                continue
            safe = self.sanitizer.redact_value(record)
            text = json.dumps(safe, ensure_ascii=False, sort_keys=True)
            identity = "runtime:%s:%s" % (task_id, ordinal)
            source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            document = RetrievalDocument(
                document_id="doc_" + stable_hash({"identity": identity, "sha": source_sha}).split(":", 1)[1][:24],
                source_type=source_type, source_identity=identity,
                content_sha256=source_sha, task_id=task_id,
                path="runtime/%s.json" % ordinal, language="json",
                trust_level=(
                    "untrusted_runtime_log" if source_type == "runtime_log"
                    else "runtime_evidence"
                ), created_at=utc_now_iso(),
                stage_tags=[str(record.get("stage", ""))] if record.get("stage") else [],
                framework_tags=[str(item) for item in record.get("frameworks", [])],
                metadata={"failure_categories": [str(record.get("category", ""))] if record.get("category") else []},
            )
            documents.append(document)
            chunks.extend(self.chunkers.chunk(document, text))
        return documents, chunks

    def ingest_skill_entries(self, entries: Iterable[Dict[str, Any]]) -> Tuple[List[RetrievalDocument], List[RetrievalChunk]]:
        documents, chunks = [], []
        for entry in entries or []:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            safe = self.sanitizer.redact_value(entry)
            text = str(safe.get("content") or json.dumps(safe, ensure_ascii=False, sort_keys=True))
            source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            identity = "skill:%s:%s" % (entry["name"], entry.get("version", ""))
            document = RetrievalDocument(
                document_id="doc_" + stable_hash({"identity": identity, "sha": source_sha}).split(":", 1)[1][:24],
                source_type="active_skill", source_identity=identity,
                content_sha256=source_sha, path="skills/%s" % entry["name"],
                language="markdown", trust_level="reviewed_skill",
                stage_tags=[str(item) for item in entry.get("stages", [])],
                framework_tags=[str(item) for item in entry.get("frameworks", [])],
                created_at=str(entry.get("created_at", "")),
                metadata={"skill_name": str(entry["name"]), "version": str(entry.get("version", ""))},
            )
            documents.append(document)
            chunks.extend(self.chunkers.chunk(document, text))
        return documents, chunks
