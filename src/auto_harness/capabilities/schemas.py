"""Serializable schemas for project capabilities and deployability."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CapabilityEvidence:
    evidence_id: str
    capability_type: str
    capability_value: str
    source_type: str
    path: str
    sha256: str
    line_start: int = 0
    line_end: int = 0
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DependencyManifest:
    path: str
    ecosystem: str
    status: str = "parsed"
    dependencies: List[str] = field(default_factory=list)
    dependency_names: List[str] = field(default_factory=list)
    sha256: str = ""
    reason_code: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProjectCapabilities:
    languages: List[str] = field(default_factory=list)
    package_ecosystems: List[str] = field(default_factory=list)
    service_frameworks: List[str] = field(default_factory=list)
    ui_frameworks: List[str] = field(default_factory=list)
    ml_libraries: List[str] = field(default_factory=list)
    inference_runtimes: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    workload_types: List[str] = field(default_factory=list)
    build_systems: List[str] = field(default_factory=list)
    evidence: List[CapabilityEvidence] = field(default_factory=list)

    def normalize(self):
        for name in (
            "languages", "package_ecosystems", "service_frameworks",
            "ui_frameworks", "ml_libraries", "inference_runtimes",
            "protocols", "workload_types", "build_systems",
        ):
            setattr(self, name, sorted(set(getattr(self, name))))
        evidence = {item.evidence_id: item for item in self.evidence}
        self.evidence = [evidence[key] for key in sorted(evidence)]
        return self

    def to_dict(self) -> Dict:
        value = asdict(self)
        value["evidence"] = [item.to_dict() for item in self.evidence]
        return value


@dataclass
class DeploymentCandidate:
    candidate_id: str
    source: str
    adapter_ids: List[str] = field(default_factory=list)
    environment_candidate_id: str = ""
    install_candidate_ids: List[str] = field(default_factory=list)
    setup_candidate_ids: List[str] = field(default_factory=list)
    run_candidate_id: str = ""
    expected_port: int = 0
    protocol_hints: List[str] = field(default_factory=list)
    verify_candidate_ids: List[str] = field(default_factory=list)
    required_backend: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    score_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DeployabilityAssessment:
    status: str
    selected_candidate_id: str = ""
    candidate_ids: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    risk_reasons: List[str] = field(default_factory=list)
    next_resolution: str = "human_input"

    def to_dict(self) -> Dict:
        return asdict(self)
