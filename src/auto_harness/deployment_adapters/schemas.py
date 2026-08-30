"""Side-effect-free schemas shared by deployment adapters."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DetectionContext:
    repo_dir: Path
    files: Tuple[str, ...]
    capabilities: object
    legacy_frameworks: Tuple[str, ...] = ()


@dataclass
class AdapterDetection:
    adapter_id: str
    matched: bool
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EnvironmentProposal:
    adapter_id: str
    backend: str
    confidence: float
    evidence_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunProposal:
    adapter_id: str
    argv: List[str]
    expected_port: int
    confidence: float
    evidence_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VerifyProposal:
    adapter_id: str
    protocol: str
    verify_hint: Dict
    confidence: float
    evidence_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)
