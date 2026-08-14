"""Serializable schemas for repository-scoped command authorization."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List


def canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sandbox_policy_fingerprint(
    *, phase: str, image: str, network: str, gpus: str,
    model_cache_dir: str = "", security_options: Dict = None,
) -> str:
    """Hash the effective Docker boundary used for repository commands."""
    security = dict(security_options or {})
    security.update({
        "cap_drop_all": True,
        "no_new_privileges": True,
        "repo_mount_mode": "rw" if phase == "install" else "ro",
        "read_only_rootfs": phase != "install",
        "model_cache_mount_mode": "rw" if phase == "install" else "ro",
    })
    if phase != "install":
        security["user"] = security.get("user") or "65532:65532"
    return canonical_hash({
        "backend": "docker",
        "phase": phase,
        "image": str(image),
        "network": str(network),
        "gpus": str(gpus or "none"),
        "model_cache_dir": str(model_cache_dir or ""),
        "security": security,
    })


@dataclass(frozen=True)
class CommandEvidence:
    evidence_id: str
    source_type: str
    path: str
    sha256: str
    line_start: int = 0
    line_end: int = 0
    declaration_key: str = ""
    declared_value: str = ""
    repository_fingerprint: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict):
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass
class CommandCandidate:
    candidate_id: str
    phase: str
    argv: List[str]
    cwd: str = "."
    source_kind: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    declared_executable: str = ""
    resolved_executable: str = ""
    environment_binding: Dict = field(default_factory=dict)
    required_backend: str = "docker"
    network_profile: str = "none"
    filesystem_profile: str = "runtime_read_only"
    risk_level: str = "medium"
    score: float = 0.0
    score_reasons: List[str] = field(default_factory=list)
    fallback_group: str = "run"

    @classmethod
    def build(cls, *, phase: str, argv: List[str], source_kind: str, **values):
        identity = {
            "phase": phase,
            "argv": list(argv),
            "cwd": values.get("cwd", "."),
            "source_kind": source_kind,
            "evidence_ids": sorted(values.get("evidence_ids", [])),
        }
        return cls(
            candidate_id="cmd_%s" % canonical_hash(identity)[:20],
            phase=phase,
            argv=list(argv),
            source_kind=source_kind,
            **values,
        )

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict):
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass
class CommandDecision:
    candidate_id: str
    verdict: str
    reason_code: str
    reasons: List[str] = field(default_factory=list)
    normalized_argv: List[str] = field(default_factory=list)
    effective_backend: str = ""
    required_approval: bool = False
    operation_id: str = ""
    policy_version: str = "1"
    policy_fingerprint: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict):
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass
class CommandRegistry:
    repository_fingerprint: str
    evidence: List[CommandEvidence] = field(default_factory=list)
    candidates: List[CommandCandidate] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "repository_fingerprint": self.repository_fingerprint,
            "evidence": [item.to_dict() for item in self.evidence],
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @classmethod
    def from_dict(cls, value: Dict):
        value = value if isinstance(value, dict) else {}
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            repository_fingerprint=str(value.get("repository_fingerprint", "")),
            evidence=[
                CommandEvidence.from_dict(item)
                for item in value.get("evidence", [])
                if isinstance(item, dict)
            ],
            candidates=[
                CommandCandidate.from_dict(item)
                for item in value.get("candidates", [])
                if isinstance(item, dict)
            ],
        )

    def candidate_for_argv(self, argv: List[str]):
        wanted = list(argv or [])
        for candidate in self.candidates:
            if candidate.argv == wanted:
                return candidate
        return None

    def evidence_by_id(self) -> Dict[str, CommandEvidence]:
        return {item.evidence_id: item for item in self.evidence}
