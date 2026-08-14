"""Immutable one-shot approval envelopes for repository commands."""

from datetime import datetime, timezone
from typing import Dict

from auto_harness.command_auth.schemas import CommandCandidate, canonical_hash


def command_operation_id(candidate: CommandCandidate, repository_fingerprint: str) -> str:
    return "op_cmd_%s" % canonical_hash({
        "candidate_id": candidate.candidate_id,
        "argv": candidate.argv,
        "cwd": candidate.cwd,
        "repository_fingerprint": repository_fingerprint,
    })[:24]


def build_command_approval_request(
    candidate: CommandCandidate,
    repository_fingerprint: str,
    evidence,
    sandbox_policy_fingerprint: str,
    *,
    task_id: str = "",
    expires_at: str = "",
) -> Dict:
    operation_id = command_operation_id(candidate, repository_fingerprint)
    payload = {
        "schema_version": 2,
        "approval_id": "approval_%s" % operation_id,
        "approval_kind": "repository_command",
        "operation_id": operation_id,
        "task_id": str(task_id),
        "candidate_id": candidate.candidate_id,
        "phase": candidate.phase,
        "normalized_argv": list(candidate.argv),
        "argv_sha256": canonical_hash(candidate.argv),
        "cwd": candidate.cwd,
        "repository_fingerprint": repository_fingerprint,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "path": item.path,
                "sha256": item.sha256,
                "declaration_key": item.declaration_key,
            }
            for item in evidence
        ],
        "resolved_executable": candidate.resolved_executable,
        "sandbox_policy_fingerprint": sandbox_policy_fingerprint,
        "network_profile": candidate.network_profile,
        "filesystem_profile": candidate.filesystem_profile,
        "risk": candidate.risk_level,
        "reason": "; ".join(candidate.score_reasons)[:2000],
        "requested_action": "execute_repository_command",
        "allowed_decisions": ["approve", "reject"],
        "max_executions": 1,
        "expires_at": str(expires_at or ""),
    }
    payload["request_hash"] = canonical_hash(payload)
    return payload


def approval_valid(
    approval: Dict,
    candidate: CommandCandidate,
    repository_fingerprint: str,
    sandbox_policy_fingerprint: str,
) -> str:
    if not isinstance(approval, dict):
        return "approval_missing"
    request = approval.get("request") if isinstance(approval.get("request"), dict) else approval
    nested_decision = approval.get("decision")
    if isinstance(nested_decision, dict):
        decision = nested_decision
    elif isinstance(approval.get("decision_record"), dict):
        decision = approval["decision_record"]
    else:
        decision = approval
    if decision.get("decision") != "approve":
        return "approval_missing"
    if request.get("operation_id") != command_operation_id(candidate, repository_fingerprint):
        return "approval_operation_mismatch"
    if request.get("candidate_id") != candidate.candidate_id:
        return "approval_candidate_mismatch"
    if request.get("argv_sha256") != canonical_hash(candidate.argv):
        return "approval_command_changed"
    if request.get("repository_fingerprint") != repository_fingerprint:
        return "repository_fingerprint_changed"
    if request.get("sandbox_policy_fingerprint") != sandbox_policy_fingerprint:
        return "sandbox_profile_changed"
    if decision.get("request_hash") != request.get("request_hash"):
        return "approval_request_hash_mismatch"
    if decision.get("approval_id") != request.get("approval_id"):
        return "approval_id_mismatch"
    if decision.get("operation_id") != request.get("operation_id"):
        return "approval_operation_mismatch"
    if int(approval.get("execution_count", 0) or 0) >= int(request.get("max_executions", 1) or 1):
        return "approval_already_consumed"
    expires_at = str(request.get("expires_at") or "")
    if expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= expiry:
                return "approval_expired"
        except ValueError:
            return "approval_expiry_invalid"
    return ""
