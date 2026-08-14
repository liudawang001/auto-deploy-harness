"""Deterministic model-preparation schemas (Document A).

These dataclasses are the frozen Artifact contract handed to Document B.
They use plain dataclasses (no Pydantic) for Python 3.10 compatibility and
follow the project's existing ``dataclass + to_dict()`` style.

Every hash is computed from a canonical compact JSON form:
    json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
Timestamps, log paths, and local absolute paths are excluded from hashes.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

MODEL_SOURCES = ("huggingface", "modelscope")


def canonical_json(value: Any) -> str:
    """Serialize a value to canonical compact JSON for hashing."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def hash_payload(value: Any) -> str:
    """Return a ``sha256:<hex>`` digest of a canonical JSON value."""
    raw = canonical_json(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reject_unknown_keys(cls, data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("%s.from_dict expects an object" % cls.__name__)
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            "%s.from_dict rejected unknown keys: %s" % (cls.__name__, ", ".join(unknown))
        )


def _require_schema_version(data: Dict[str, Any], cls_name: str) -> None:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            "%s.from_dict rejected schema_version %r" % (cls_name, version)
        )


def _validate_finite_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError("%s must be a finite number" % name)


@dataclass
class ModelReferenceCandidate:
    """A grounded model-reference candidate discovered in a repository."""
    schema_version: int = SCHEMA_VERSION
    source: str = ""
    repo_id: str = ""
    requested_revision: str = "main"
    role: str = "primary_generation_model"
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    discovered_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelReferenceCandidate":
        _reject_unknown_keys(cls, data)
        _require_schema_version(data, cls.__name__)
        source = data.get("source")
        if source not in MODEL_SOURCES:
            raise ValueError("ModelReferenceCandidate.source must be huggingface or modelscope")
        _validate_finite_number("confidence", data.get("confidence"))
        confidence = float(data["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be within [0, 1]")
        evidence = data.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("ModelReferenceCandidate.evidence must be a non-empty list")
        return cls(**data)

    def validate(self) -> List[str]:
        """Return a list of validation problems (empty means valid)."""
        problems = []
        if self.source not in MODEL_SOURCES:
            problems.append("source must be huggingface or modelscope")
        if not self.repo_id or "/" not in self.repo_id:
            problems.append("repo_id must be org/name")
        if self.repo_id.startswith(("/", "\\")) or ".." in self.repo_id:
            problems.append("repo_id must not be an absolute path or contain '..'")
        if not self.role:
            problems.append("role must be non-empty")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            problems.append("confidence must be a number")
        elif not isfinite(float(self.confidence)) or not (0.0 <= float(self.confidence) <= 1.0):
            problems.append("confidence must be within [0, 1]")
        if not self.evidence:
            problems.append("evidence must be non-empty")
        return problems


@dataclass
class ResolvedModelSpec:
    """Immutable, fully-resolved model identity with model architecture facts."""
    schema_version: int = SCHEMA_VERSION
    status: str = ""
    source: str = ""
    repo_id: str = ""
    requested_revision: str = "main"
    resolved_revision: str = ""
    model_identity: str = ""
    model_type: str = ""
    architectures: List[str] = field(default_factory=list)
    task: str = ""
    dtype: str = "float16"
    parameter_count: Optional[int] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    quantization: Optional[str] = None
    requires_remote_code: bool = False
    gated: bool = False
    license: str = ""
    source_metadata_hash: str = ""
    grounding_hash: str = ""
    resolved_at: str = ""

    ALLOWED_STATUSES = frozenset({
        "resolved",
        "ambiguous",
        "needs_human_input",
        "access_required",
        "license_acceptance_required",
        "not_found",
        "unsupported_source",
        "unsupported_architecture",
        "unsupported_quantization",
        "remote_code_required",
        "metadata_invalid",
        "network_failed",
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedModelSpec":
        _reject_unknown_keys(cls, data)
        _require_schema_version(data, cls.__name__)
        status = data.get("status")
        if status not in cls.ALLOWED_STATUSES:
            raise ValueError("ResolvedModelSpec.status %r is not allowed" % status)
        return cls(**data)


@dataclass
class ModelFilePlan:
    """Frozen, immutable Safetensors file closure for one model revision."""
    schema_version: int = SCHEMA_VERSION
    status: str = "planned"
    model_identity: str = ""
    format: str = "safetensors"
    variant: str = "fp16"
    files: List[Dict[str, Any]] = field(default_factory=list)
    total_size_bytes: int = 0
    remaining_download_bytes: int = 0
    integrity_level: str = "strong"
    plan_hash: str = ""

    ALLOWED_STATUSES = frozenset({"planned", "verified", "blocked"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelFilePlan":
        _reject_unknown_keys(cls, data)
        _require_schema_version(data, cls.__name__)
        status = data.get("status")
        if status not in cls.ALLOWED_STATUSES:
            raise ValueError("ModelFilePlan.status %r is not allowed" % status)
        files = data.get("files")
        if not isinstance(files, list):
            raise ValueError("ModelFilePlan.files must be a list")
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("ModelFilePlan.files entries must be objects")
            if isinstance(item.get("size_bytes"), bool) or not isinstance(item.get("size_bytes"), int) or item.get("size_bytes", 0) < 0:
                raise ValueError("ModelFilePlan.files[].size_bytes must be a non-negative integer")
        for key in ("total_size_bytes", "remaining_download_bytes"):
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("ModelFilePlan.%s must be a non-negative integer" % key)
        return cls(**data)

    def plan_payload(self) -> Dict[str, Any]:
        """Deterministic payload for plan_hash (excludes status/hash/timestamps)."""
        return {
            "model_identity": self.model_identity,
            "format": self.format,
            "variant": self.variant,
            "total_size_bytes": self.total_size_bytes,
            "files": sorted(
                (
                    {
                        "path": f.get("path"),
                        "role": f.get("role"),
                        "size_bytes": f.get("size_bytes"),
                        "sha256": f.get("sha256"),
                        "etag": f.get("etag"),
                        "required": f.get("required"),
                    }
                    for f in self.files
                ),
                key=lambda item: item.get("path") or "",
            ),
        }

    def compute_plan_hash(self) -> str:
        return hash_payload(self.plan_payload())


@dataclass
class InferenceResourceDecision:
    """Final GPU/RAM/disk resource decision for a model revision."""
    schema_version: int = SCHEMA_VERSION
    status: str = "uncertain"
    model_identity: str = ""
    runtime: str = "vllm"
    gpu_indexes: List[int] = field(default_factory=list)
    gpu_memory_total_bytes: int = 0
    gpu_memory_free_bytes: int = 0
    weight_bytes: int = 0
    weight_runtime_bytes: int = 0
    kv_cache_bytes: int = 0
    runtime_overhead_bytes: int = 0
    required_vram_bytes: int = 0
    usable_vram_bytes: int = 0
    required_ram_bytes: int = 0
    required_disk_bytes: int = 0
    selected_dtype: str = "float16"
    max_model_len: int = 0
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    decision_hash: str = ""

    ALLOWED_STATUSES = frozenset({
        "allowed",
        "insufficient_gpu_memory",
        "insufficient_system_memory",
        "insufficient_disk",
        "gpu_busy",
        "driver_incompatible",
        "docker_gpu_unavailable",
        "unsupported_model",
        "uncertain",
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResourceDecision":
        _reject_unknown_keys(cls, data)
        _require_schema_version(data, cls.__name__)
        status = data.get("status")
        if status not in cls.ALLOWED_STATUSES:
            raise ValueError("InferenceResourceDecision.status %r is not allowed" % status)
        return cls(**data)

    def decision_payload(self) -> Dict[str, Any]:
        """Deterministic payload for decision_hash (excludes host-fact drift fields)."""
        return {
            "model_identity": self.model_identity,
            "runtime": self.runtime,
            "gpu_indexes": list(self.gpu_indexes),
            "weight_bytes": self.weight_bytes,
            "weight_runtime_bytes": self.weight_runtime_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
            "required_vram_bytes": self.required_vram_bytes,
            "usable_vram_bytes": self.usable_vram_bytes,
            "required_ram_bytes": self.required_ram_bytes,
            "required_disk_bytes": self.required_disk_bytes,
            "selected_dtype": self.selected_dtype,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

    def compute_decision_hash(self) -> str:
        return hash_payload(self.decision_payload())


@dataclass
class CacheCompleteMarker:
    """Atomic marker proving a revision cache is fully verified."""
    schema_version: int = SCHEMA_VERSION
    status: str = "complete"
    model_identity: str = ""
    file_plan_hash: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)
    verified_at: str = ""
    marker_hash: str = ""

    ALLOWED_STATUSES = frozenset({"complete", "invalid"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheCompleteMarker":
        _reject_unknown_keys(cls, data)
        _require_schema_version(data, cls.__name__)
        status = data.get("status")
        if status not in cls.ALLOWED_STATUSES:
            raise ValueError("CacheCompleteMarker.status %r is not allowed" % status)
        return cls(**data)

    def marker_payload(self) -> Dict[str, Any]:
        return {
            "model_identity": self.model_identity,
            "file_plan_hash": self.file_plan_hash,
            "files": sorted(
                (
                    {
                        "path": f.get("path"),
                        "size_bytes": f.get("size_bytes"),
                        "sha256": f.get("sha256"),
                    }
                    for f in self.files
                ),
                key=lambda item: item.get("path") or "",
            ),
        }

    def compute_marker_hash(self) -> str:
        return hash_payload(self.marker_payload())


# Model architecture -> dtype determinism helper (used by A6 resource solver).
DTYPE_ELEMENT_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "half": 2,
    "float32": 4,
    "float": 4,
}
