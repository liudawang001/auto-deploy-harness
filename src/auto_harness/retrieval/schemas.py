"""Stable, serializable contracts for evidence retrieval."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


SOURCE_TYPES = {
    "repository", "runtime_evidence", "runtime_log",
    "issue_memory", "verified_memory", "active_skill",
}
TRUST_LEVELS = {
    "untrusted_repository", "runtime_evidence", "untrusted_runtime_log",
    "unverified_memory", "verified_memory", "reviewed_skill",
}
RETRIEVAL_MODES = {"lexical", "dense", "hybrid"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _known(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("schema input must be an object")
    allowed = cls.__dataclass_fields__
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError("unknown fields: %s" % ", ".join(unknown))
    return {key: value for key, value in data.items() if key in allowed}


@dataclass
class RetrievalDocument:
    document_id: str
    source_type: str
    source_identity: str
    content_sha256: str
    repository_fingerprint: str = ""
    task_id: str = ""
    path: str = ""
    mime_type: str = "text/plain"
    language: str = "text"
    trust_level: str = "untrusted_repository"
    stage_tags: List[str] = field(default_factory=list)
    framework_tags: List[str] = field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError("invalid source_type: %s" % self.source_type)
        if self.trust_level not in TRUST_LEVELS:
            raise ValueError("invalid trust_level: %s" % self.trust_level)
        if self.source_type == "repository" and (
            not self.repository_fingerprint or not self.path
        ):
            raise ValueError("repository document requires fingerprint and path")
        if self.source_type in {"runtime_evidence", "runtime_log"} and not self.task_id:
            raise ValueError("runtime document requires task_id")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalDocument":
        return cls(**_known(cls, data))


@dataclass
class RetrievalChunk:
    chunk_id: str
    document_id: str
    chunker_version: str
    ordinal: int
    text: str
    text_sha256: str
    token_estimate: int
    path: str = ""
    symbol: str = ""
    line_start: int = 1
    line_end: int = 1
    source_type: str = "repository"
    repository_fingerprint: str = ""
    task_id: str = ""
    trust_level: str = "untrusted_repository"
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.document_id or not self.text_sha256:
            raise ValueError("chunk identity is required")
        if self.ordinal < 0 or self.line_start < 1 or self.line_end < self.line_start:
            raise ValueError("invalid chunk range")
        if self.token_estimate < 0:
            raise ValueError("token_estimate must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalChunk":
        return cls(**_known(cls, data))


@dataclass
class RetrievalQuery:
    query_id: str
    query_text: str
    purpose: str
    task_id: str = ""
    repository_fingerprint: str = ""
    stage: str = ""
    frameworks: List[str] = field(default_factory=list)
    failure_categories: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=lambda: ["repository", "verified_memory"])
    top_k: int = 8
    max_context_tokens: int = 3000
    filters: Dict[str, Any] = field(default_factory=dict)
    requested_by: str = "runtime"
    mode: str = "lexical"
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.query_text = " ".join(str(self.query_text).split())
        if not self.query_id or not self.query_text:
            raise ValueError("query id and text are required")
        if len(self.query_text) > 500:
            raise ValueError("query_text must be <= 500 characters")
        if self.top_k < 1 or self.top_k > 12:
            raise ValueError("top_k must be within [1, 12]")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if self.mode not in RETRIEVAL_MODES:
            raise ValueError("invalid retrieval mode")
        if not self.sources or any(source not in SOURCE_TYPES for source in self.sources):
            raise ValueError("invalid retrieval sources")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalQuery":
        return cls(**_known(cls, data))


@dataclass
class RetrievalHit:
    query_id: str
    rank: int
    chunk_id: str
    source_type: str
    preview: str
    path: str = ""
    line_start: int = 1
    line_end: int = 1
    scores: Dict[str, Any] = field(default_factory=dict)
    trust_level: str = "untrusted_repository"
    freshness: str = "current"
    requires_exact_read: bool = True
    grounding: Dict[str, Any] = field(default_factory=lambda: {"observation_id": "", "validated": False})
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        for value in self.scores.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("scores must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalTrace:
    query: Dict[str, Any]
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    hits: List[Dict[str, Any]] = field(default_factory=list)
    budgets: Dict[str, int] = field(default_factory=dict)
    latency_ms: Dict[str, int] = field(default_factory=dict)
    degradation: Dict[str, Any] = field(default_factory=dict)
    redactions: int = 0
    errors: List[str] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalManifest:
    repository_fingerprint: str
    document_count: int
    chunk_count: int
    index_version: str = "retrieval_v1"
    chunker_versions: Dict[str, str] = field(default_factory=dict)
    redaction_version: str = "v1"
    embedding: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    completed: bool = False
    manifest_hash: str = ""
    schema_version: int = 1

    def payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("manifest_hash", None)
        return data

    def finalize(self) -> "RetrievalManifest":
        self.completed = True
        self.manifest_hash = stable_hash(self.payload())
        return self

    def validate(self) -> None:
        if not self.completed or self.manifest_hash != stable_hash(self.payload()):
            raise ValueError("retrieval manifest is incomplete or invalid")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingResult:
    vectors: List[List[float]]
    provider: str
    model: str
    dimension: int
    request_id: str = ""
    input_count: int = 0
    token_usage: Dict[str, Any] = field(default_factory=dict)
    truncated_count: int = 0
    latency_ms: int = 0

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("embedding dimension must be positive")
        if len(self.vectors) != self.input_count:
            raise ValueError("embedding input count mismatch")
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("embedding dimension mismatch")
        if any(not math.isfinite(value) for vector in self.vectors for value in vector):
            raise ValueError("embedding values must be finite")

