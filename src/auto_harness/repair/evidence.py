"""Repair evidence helpers.

Shared definitions for effective repair actions, fresh trace verification,
and repair_verified computation. Used by repair apply nodes, contribution
analyzers, and reporters to avoid rule duplication.

Key invariants:
- A repair is only 'verified' if it has effective actions, a rerun/resume,
  fresh strong verify with current trace evidence.
- metadata_only actions never count as effective.
- repair_verified is only computed in finalize/report, never in apply.
"""
from typing import Any, Dict, Optional


def is_effective_repair_action(result: Dict) -> bool:
    """Determine if a repair action result is effective.

    An action is effective only if:
    - It is NOT metadata_only
    - It was executed with exit_code == 0, OR
    - It has a tool_result with strong_verify_pass == True

    Args:
        result: Repair action result dict.

    Returns:
        True if the action is effective, False otherwise.
    """
    if not isinstance(result, dict):
        return False
    if result.get("metadata_only"):
        return False
    if result.get("executed") is True:
        return int(result.get("exit_code", 1)) == 0
    tool_result = result.get("tool_result")
    if isinstance(tool_result, dict):
        return tool_result.get("strong_verify_pass") is True
    return False


def compute_fresh_trace(
    before_trace: Optional[str],
    after_trace: Optional[str],
) -> bool:
    """Determine if the trace evidence is fresh (changed after repair).

    Args:
        before_trace: Trace ID captured before repair.
        after_trace: Trace ID captured after repair.

    Returns:
        True if before and after traces exist and differ.
    """
    return bool(
        before_trace
        and after_trace
        and before_trace != after_trace
    )


def compute_repair_verified(
    *,
    effective_action_count: int,
    resume_executed: bool,
    verify_status_after: str,
    evidence_contains_after_trace: bool,
    fresh_trace: bool,
) -> bool:
    """Compute repair_verified.

    Only computed in finalize/report stage, never in repair apply.
    A repair is verified only if ALL conditions hold:
    - effective_action_count > 0
    - resume_executed (a rerun/resume happened)
    - verify_status_after in (pass, passed)
    - evidence_contains_after_trace
    - fresh_trace (before != after)

    Args:
        effective_action_count: Number of effective repair actions.
        resume_executed: Whether a resume/rerun happened.
        verify_status_after: Verify status after repair.
        evidence_contains_after_trace: Whether after-trace is in evidence.
        fresh_trace: Whether the trace changed after repair.

    Returns:
        True if the repair is verified.
    """
    return bool(
        effective_action_count > 0
        and resume_executed
        and verify_status_after in ("pass", "passed")
        and evidence_contains_after_trace
        and fresh_trace
    )


def build_repair_attempt(
    *,
    attempt: int,
    failure_signature_before: str = "",
    diagnosis_path: str = "",
    plan_path: str = "",
    policy_path: str = "",
    apply_path: str = "",
    resume_from_stage: str = "",
    effective_action_count: int = 0,
    metadata_only_count: int = 0,
    verify_status_after: str = "",
    verification_trace_id: str = "",
    fresh_trace: bool = False,
    repair_verified: bool = False,
) -> Dict[str, Any]:
    """Build a RepairAttempt schema dict.

    Each repair attempt produces this artifact, saved to
    repairs/attempt_<N>.json.

    Args:
        attempt: Attempt number (1-based).
        failure_signature_before: Failure signature before this repair.
        diagnosis_path: Path to diagnosis artifact.
        plan_path: Path to repair plan artifact.
        policy_path: Path to repair policy artifact.
        apply_path: Path to repair apply artifact.
        resume_from_stage: Stage to resume from after repair.
        effective_action_count: Number of effective actions in this attempt.
        metadata_only_count: Number of metadata-only actions.
        verify_status_after: Verify status after repair rerun.
        verification_trace_id: Trace ID from post-repair verify.
        fresh_trace: Whether post-repair trace differs from pre-repair.
        repair_verified: Whether this repair attempt is verified.

    Returns:
        RepairAttempt dict with schema_version=1.
    """
    return {
        "schema_version": 1,
        "attempt": attempt,
        "failure_signature_before": failure_signature_before,
        "diagnosis_path": diagnosis_path,
        "plan_path": plan_path,
        "policy_path": policy_path,
        "apply_path": apply_path,
        "resume_from_stage": resume_from_stage,
        "effective_action_count": effective_action_count,
        "metadata_only_count": metadata_only_count,
        "verify_status_after": verify_status_after,
        "verification_trace_id": verification_trace_id,
        "fresh_trace": fresh_trace,
        "repair_verified": repair_verified,
    }
