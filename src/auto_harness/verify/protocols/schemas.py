"""Serializable protocol verification contracts."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class Probe:
    verifier_id: str
    protocol: str
    trace_id: str
    method: str
    endpoint: str
    expected_port: int
    request: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProbeEvidence:
    verifier_id: str
    protocol: str
    trace_id: str
    endpoint: str
    expected_port: int
    process_alive: bool
    port_ready: bool
    status: str
    trace_observed: bool = False
    operation_id: str = ""
    artifact_path: str = ""
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VerifyDecision:
    status: str
    reason_code: str
    verifier_id: str
    trace_id: str
    strong_evidence: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProtocolVerifierSelection:
    verifier_id: str
    protocol: str
    source: str
    reason: str
    candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)
