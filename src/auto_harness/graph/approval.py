"""Approval node and store for human-in-the-loop interrupt.

Provides:
- ApprovalStore: persistent approval request/response records
- build_approval_request: creates a deterministic approval request
- approval_node: LangGraph node that interrupts for approval
- cleanup_node: LangGraph node that executes approved cleanup
- resume_approval: resumes the graph with a decision
- sanitize_approval: strips non-allowed fields from approval decisions

Rules:
- Approval only authorizes one operation_id + requested_action
- Input changes require re-approval
- Reject means executor call count = 0
- Side effects must be AFTER the approval node, not before
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from auto_harness.utils.atomic import FileLock, atomic_write_text


class ApprovalStore:
    """Persistent store for approval requests and decisions.

    Storage: runs/<task-id>/approvals/<approval-id>.json
    """

    def __init__(self, run_dir):
        self.root = Path(run_dir) / "approvals"

    def save(self, approval_id, payload):
        """Save an approval record atomically."""
        if not approval_id or not approval_id.replace("-", "").isalnum():
            raise ValueError("invalid approval_id")
        path = self.root / (approval_id + ".json")
        with FileLock(path):
            atomic_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            )
        return path

    def load(self, approval_id):
        """Load an approval record by ID."""
        if not approval_id or not approval_id.replace("-", "").isalnum():
            raise ValueError("invalid approval_id")
        path = self.root / (approval_id + ".json")
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


def build_approval_request(operation, requested_action, reason, risk="high"):
    """Build an approval request dict.

    The approval_id is deterministic based on operation_id + action.
    """
    seed = "%s:%s" % (operation["operation_id"], requested_action)
    approval_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return {
        "approval_id": approval_id,
        "task_id": operation["task_id"],
        "operation_id": operation["operation_id"],
        "requested_action": requested_action,
        "risk": risk,
        "reason": str(reason)[:2000],
        "allowed_decisions": ["approve", "reject"],
        "normalized_input_hash": operation.get("normalized_input_hash", ""),
    }


ALLOWED_APPROVAL_FIELDS = (
    "approval_id", "operation_id", "decision",
    "reviewer", "note", "resolved_at",
)


def sanitize_approval(decision):
    """Strip non-allowed fields from an approval decision.

    Prevents injection of extra fields into the approval record.
    """
    return {
        key: str(decision.get(key, ""))[:2000]
        for key in ALLOWED_APPROVAL_FIELDS
        if key in decision
    }


def approval_node(state):
    """LangGraph node that interrupts for human approval.

    Saves the approval request, then calls interrupt() to pause
    the graph. When resumed, validates the decision and updates state.

    The interrupt is BEFORE any side-effect execution. When the
    node resumes after interrupt, it re-runs from the top, so
    the approval request is re-saved (idempotent).
    """
    from langgraph.types import interrupt

    request = state.get("pending_approval")
    if not request:
        return {"stop_reason": "approval_request_missing"}

    store = ApprovalStore(state["run_dir"])
    approval_id = request.get("approval_id", "")
    store.save(approval_id, {"status": "pending", "request": request})

    # Interrupt — pauses graph execution until resume is called
    decision = interrupt(request)

    if not isinstance(decision, dict):
        return {"stop_reason": "invalid_approval_response"}

    if decision.get("operation_id") != request.get("operation_id"):
        return {"stop_reason": "approval_operation_mismatch"}

    if decision.get("approval_id") != approval_id:
        return {"stop_reason": "approval_id_mismatch"}

    if decision.get("decision") not in request.get("allowed_decisions", []):
        return {"stop_reason": "approval_decision_not_allowed"}

    safe_decision = sanitize_approval(decision)
    store.save(approval_id, {
        "status": "resolved",
        "request": request,
        "decision": safe_decision,
    })

    return {
        "approval_history": [safe_decision],
        "pending_approval": None,
        "approved_operation_id": (
            request["operation_id"] if decision["decision"] == "approve" else ""
        ),
        "approved_action": (
            request.get("requested_action", "retry")
            if decision["decision"] == "approve" else ""
        ),
        "stop_reason": "" if decision["decision"] == "approve" else "operator_rejected",
    }


def cleanup_node(state, recovery, cleanup_executor):
    """LangGraph node that executes an approved cleanup.

    Only runs when approved_action is "cleanup_then_retry".
    Re-verifies ownership before cleanup.
    """
    operation_id = state.get("approved_operation_id", "")
    if state.get("approved_action") != "cleanup_then_retry" or not operation_id:
        return {"stop_reason": "cleanup_not_approved"}

    operation = recovery.journal.load(operation_id)
    if not operation or operation.get("status") != "manual":
        return {"stop_reason": "cleanup_operation_state_changed"}

    # Re-verify ownership before cleanup
    reconciler = recovery.reconcilers.get(operation["resource_type"])
    if reconciler is None or not hasattr(reconciler, "verify_cleanup_target"):
        return {"stop_reason": "cleanup_verification_not_available"}

    check = reconciler.verify_cleanup_target(operation)
    if not check.get("owned"):
        return {"stop_reason": "cleanup_ownership_conflict"}

    cleanup_result = cleanup_executor.remove_owned_resource(operation, check)
    if not cleanup_result.get("success"):
        return {"stop_reason": "approved_cleanup_failed"}

    recovery.journal.transition(operation_id, "retryable", cleanup=cleanup_result)
    return {
        "approved_operation_id": "",
        "approved_action": "",
        "stop_reason": "",
        "recovery_events": [{"operation_id": operation_id, "event": "cleaned"}],
    }


def resume_approval(graph, task_id, decision, checkpointer=None):
    """Resume a graph that was interrupted for approval.

    Uses Command(resume=decision) to pass the approval decision
    back into the graph.
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": task_id}}
    return graph.invoke(Command(resume=decision), config=config)


# -------------------------------------------------------------------
# Route functions for approval flow
# -------------------------------------------------------------------

def route_after_recovery(state):
    """Route after recovery decision.

    If pending_approval is set, go to approval node.
    If stop_reason is set, go to stop.
    Otherwise continue execution.
    """
    if state.get("pending_approval"):
        return "approval"
    if state.get("stop_reason"):
        return "stop"
    return "continue"


def route_after_approval(state):
    """Route after approval decision.

    If rejected, go to stop.
    If approved for cleanup, go to cleanup node.
    If approved for retry, go back to the recovery flow.
    """
    if state.get("stop_reason"):
        return "stop"
    if state.get("approved_action") == "cleanup_then_retry":
        return "cleanup"
    return "retry"
