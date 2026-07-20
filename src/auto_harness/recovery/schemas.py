"""Recovery schemas: operation record, reconcile result, stable ID computation.

OperationRecord tracks the lifecycle of a side-effect operation.
ReconcileResult captures the reconciler's decision about external state.
compute_operation_id produces a stable, deterministic operation ID.
"""
import hashlib
import json
from typing import Any, Dict, List, TypedDict


# Valid operation statuses and their allowed transitions
OPERATION_STATUSES = frozenset({
    "planned", "running", "committed", "failed", "unknown",
    "retryable", "conflict", "manual",
})

# Valid reconcile decisions
RECONCILE_DECISIONS = frozenset({
    "reuse", "continue", "retry", "cleanup_then_retry",
    "conflict", "manual",
})

# Unknown decisions are treated as "manual"
DEFAULT_DECISION = "manual"


class OperationRecord(TypedDict, total=False):
    """Record of a side-effect operation in the journal.

    Lifecycle: planned -> running -> committed / failed
    After crash: running -> unknown -> committed / retryable / conflict / manual
    """
    schema_version: int
    operation_id: str
    task_id: str
    stage: str
    action: str
    resource_type: str
    resource_identity: Dict[str, str]
    observed_resource: Dict[str, Any]
    normalized_input_hash: str
    status: str
    attempt: int
    started_at: str
    committed_at: str
    last_checked_at: str
    result_artifacts: List[str]
    error: str
    created_at: str
    updated_at: str
    reconcile_result: Dict[str, Any]
    cleanup: Dict[str, Any]


class ReconcileResult(TypedDict, total=False):
    """Result of reconciling an operation against external state.

    decision: what to do next (reuse/continue/retry/cleanup_then_retry/conflict/manual)
    observed_state: what the reconciler found externally
    reason: human-readable explanation
    evidence_paths: paths to evidence files
    """
    decision: str
    observed_state: Dict[str, Any]
    reason: str
    evidence_paths: List[str]


def canonical_json(value: Any) -> str:
    """Produce a canonical JSON string for hashing.

    Keys are sorted, ASCII-only, no whitespace. This ensures
    deterministic output regardless of dict insertion order.
    """
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_operation_id(
    task_id: str,
    stage: str,
    action: str,
    normalized_input: Dict[str, Any],
    resource_identity: Dict[str, str],
) -> str:
    """Compute a stable operation ID from deterministic inputs.

    The ID is a truncated SHA-256 hash of the canonical JSON of:
    task_id, stage, action, normalized_input, resource_identity.

    Rules:
    - Same inputs always produce the same ID.
    - Changed inputs produce a different ID.
    - Secret values must NOT be in normalized_input or resource_identity
      (callers must sanitize before calling this function).
    - PID, container ID, process start time, etc. are execution-time
      observations and must NOT be part of the ID.
    - Step index is NOT used (unstable across replans).
    """
    payload = {
        "task_id": task_id,
        "stage": stage,
        "action": action,
        "input": normalized_input,
        "resource": resource_identity,
    }
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()[:24]
