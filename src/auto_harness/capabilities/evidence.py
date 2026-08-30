"""Capability evidence builders with repository boundary checks."""

from pathlib import Path

from auto_harness.capabilities.schemas import CapabilityEvidence
from auto_harness.command_auth.evidence import file_sha256, safe_repository_file
from auto_harness.command_auth.schemas import canonical_hash


def build_capability_evidence(
    repo_dir: Path,
    *,
    capability_type: str,
    capability_value: str,
    source_type: str,
    relative: str,
    confidence: float,
    reason: str,
    line_start: int = 0,
    line_end: int = 0,
) -> CapabilityEvidence:
    path = safe_repository_file(repo_dir, relative)
    sha256 = file_sha256(path)
    identity = {
        "capability_type": str(capability_type),
        "capability_value": str(capability_value),
        "source_type": str(source_type),
        "path": str(relative).replace("\\", "/"),
        "sha256": sha256,
        "line_start": int(line_start or 0),
        "line_end": int(line_end or 0),
    }
    return CapabilityEvidence(
        evidence_id="cap_%s" % canonical_hash(identity)[:20],
        confidence=max(0.0, min(float(confidence), 1.0)),
        reason=str(reason),
        **identity,
    )
