"""Tests for approval node, ApprovalStore, and approval CLI flow.

Phase 5 tests: approval request building, store persistence,
sanitize, route functions, and approval node behavior.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.graph.approval import (
    ApprovalStore,
    build_approval_request,
    sanitize_approval,
    approval_node,
    cleanup_node,
    route_after_recovery,
    route_after_approval,
    ALLOWED_APPROVAL_FIELDS,
)
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_operation(
    task_id="test_task",
    operation_id="op123",
    normalized_input_hash="hash123",
):
    return {
        "operation_id": operation_id,
        "task_id": task_id,
        "normalized_input_hash": normalized_input_hash,
    }


# -------------------------------------------------------------------
# ApprovalStore Tests
# -------------------------------------------------------------------

class TestApprovalStore:
    def test_save_and_load(self, tmp_path):
        store = ApprovalStore(tmp_path)
        path = store.save("abc123", {"status": "pending", "request": {"approval_id": "abc123"}})
        assert path.exists()
        loaded = store.load("abc123")
        assert loaded["status"] == "pending"

    def test_load_nonexistent(self, tmp_path):
        store = ApprovalStore(tmp_path)
        assert store.load("nonexistent") is None

    def test_invalid_approval_id(self, tmp_path):
        store = ApprovalStore(tmp_path)
        with pytest.raises(ValueError, match="invalid approval_id"):
            store.save("", {})
        with pytest.raises(ValueError, match="invalid approval_id"):
            store.save("../../../etc/passwd", {})

    def test_save_overwrites(self, tmp_path):
        store = ApprovalStore(tmp_path)
        store.save("abc123", {"status": "pending"})
        store.save("abc123", {"status": "resolved", "decision": {"decision": "approve"}})
        loaded = store.load("abc123")
        assert loaded["status"] == "resolved"


# -------------------------------------------------------------------
# Build Approval Request Tests
# -------------------------------------------------------------------

class TestBuildApprovalRequest:
    def test_deterministic_id(self):
        op = make_operation()
        req1 = build_approval_request(op, "cleanup_then_retry", "config changed")
        req2 = build_approval_request(op, "cleanup_then_retry", "config changed")
        assert req1["approval_id"] == req2["approval_id"]

    def test_different_action_different_id(self):
        op = make_operation()
        req1 = build_approval_request(op, "cleanup_then_retry", "reason1")
        req2 = build_approval_request(op, "retry", "reason2")
        assert req1["approval_id"] != req2["approval_id"]

    def test_required_fields(self):
        op = make_operation()
        req = build_approval_request(op, "cleanup_then_retry", "config changed")
        assert "approval_id" in req
        assert req["task_id"] == "test_task"
        assert req["operation_id"] == "op123"
        assert req["requested_action"] == "cleanup_then_retry"
        assert req["risk"] == "high"
        assert "approve" in req["allowed_decisions"]
        assert "reject" in req["allowed_decisions"]
        assert req["normalized_input_hash"] == "hash123"

    def test_custom_risk(self):
        op = make_operation()
        req = build_approval_request(op, "retry", "low risk", risk="low")
        assert req["risk"] == "low"

    def test_reason_truncated(self):
        op = make_operation()
        long_reason = "x" * 5000
        req = build_approval_request(op, "retry", long_reason)
        assert len(req["reason"]) <= 2000


# -------------------------------------------------------------------
# Sanitize Approval Tests
# -------------------------------------------------------------------

class TestSanitizeApproval:
    def test_strips_extra_fields(self):
        decision = {
            "approval_id": "abc",
            "operation_id": "op1",
            "decision": "approve",
            "reviewer": "cli",
            "note": "looks good",
            "resolved_at": "2025-01-01",
            "injected_field": "should be removed",
        }
        safe = sanitize_approval(decision)
        assert "injected_field" not in safe
        assert safe["decision"] == "approve"

    def test_all_allowed_fields_present(self):
        decision = {field: "value" for field in ALLOWED_APPROVAL_FIELDS}
        safe = sanitize_approval(decision)
        assert set(safe.keys()) == set(ALLOWED_APPROVAL_FIELDS)

    def test_missing_fields_omitted(self):
        decision = {"approval_id": "abc", "decision": "approve"}
        safe = sanitize_approval(decision)
        assert "reviewer" not in safe
        assert safe["decision"] == "approve"

    def test_values_truncated(self):
        decision = {"decision": "x" * 5000}
        safe = sanitize_approval(decision)
        assert len(safe["decision"]) <= 2000


# -------------------------------------------------------------------
# Approval Node Tests (without real LangGraph interrupt)
# -------------------------------------------------------------------

class TestApprovalNode:
    def test_missing_request(self):
        """No pending_approval → stop_reason set."""
        state = {"run_dir": "/tmp/test", "pending_approval": None}
        result = approval_node(state)
        assert result["stop_reason"] == "approval_request_missing"

    def test_approve_decision(self, tmp_path):
        """Approval with 'approve' decision sets approved fields."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "cleanup_then_retry",
            "risk": "high",
            "reason": "config changed",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "hash123",
        }
        state = {
            "run_dir": str(tmp_path),
            "pending_approval": request,
        }
        # Mock interrupt to return an approve decision
        decision = {
            "approval_id": "abc123",
            "operation_id": "op123",
            "decision": "approve",
            "reviewer": "cli",
            "note": "approved",
            "resolved_at": "2025-01-01",
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["approved_operation_id"] == "op123"
        assert result["approved_action"] == "cleanup_then_retry"
        assert result["stop_reason"] == ""
        assert result["pending_approval"] is None
        assert len(result["approval_history"]) == 1

    def test_reject_decision(self, tmp_path):
        """Approval with 'reject' decision sets stop_reason."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "cleanup_then_retry",
            "risk": "high",
            "reason": "config changed",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "hash123",
        }
        state = {
            "run_dir": str(tmp_path),
            "pending_approval": request,
        }
        decision = {
            "approval_id": "abc123",
            "operation_id": "op123",
            "decision": "reject",
            "reviewer": "cli",
            "note": "rejected",
            "resolved_at": "2025-01-01",
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["approved_operation_id"] == ""
        assert result["stop_reason"] == "operator_rejected"

    def test_operation_mismatch(self, tmp_path):
        """Decision with wrong operation_id → stop."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "retry",
            "risk": "high",
            "reason": "test",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "",
        }
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": "abc123",
            "operation_id": "WRONG_OP",
            "decision": "approve",
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["stop_reason"] == "approval_operation_mismatch"

    def test_approval_id_mismatch(self, tmp_path):
        """Decision with wrong approval_id → stop."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "retry",
            "risk": "high",
            "reason": "test",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "",
        }
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": "WRONG_ID",
            "operation_id": "op123",
            "decision": "approve",
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["stop_reason"] == "approval_id_mismatch"

    def test_disallowed_decision(self, tmp_path):
        """Decision not in allowed_decisions → stop."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "retry",
            "risk": "high",
            "reason": "test",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "",
        }
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": "abc123",
            "operation_id": "op123",
            "decision": "maybe",  # Not in allowed_decisions
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["stop_reason"] == "approval_decision_not_allowed"

    def test_non_dict_response(self, tmp_path):
        """Non-dict interrupt response → stop."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "retry",
            "risk": "high",
            "reason": "test",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "",
        }
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        with patch("langgraph.types.interrupt", return_value="not a dict"):
            result = approval_node(state)
        assert result["stop_reason"] == "invalid_approval_response"

    def test_approval_saved_to_store(self, tmp_path):
        """Approval record is saved to ApprovalStore."""
        request = {
            "approval_id": "abc123",
            "task_id": "test_task",
            "operation_id": "op123",
            "requested_action": "retry",
            "risk": "high",
            "reason": "test",
            "allowed_decisions": ["approve", "reject"],
            "normalized_input_hash": "",
        }
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": "abc123",
            "operation_id": "op123",
            "decision": "approve",
            "reviewer": "cli",
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            approval_node(state)
        # Check the store
        store = ApprovalStore(tmp_path)
        record = store.load("abc123")
        assert record is not None
        assert record["status"] == "resolved"


# -------------------------------------------------------------------
# Cleanup Node Tests
# -------------------------------------------------------------------

class TestCleanupNode:
    def test_not_approved(self):
        """No approved_action → stop."""
        state = {"approved_operation_id": "", "approved_action": ""}
        result = cleanup_node(state, MagicMock(), MagicMock())
        assert result["stop_reason"] == "cleanup_not_approved"

    def test_wrong_action(self):
        """Approved action is not cleanup_then_retry → stop."""
        state = {"approved_operation_id": "op1", "approved_action": "retry"}
        result = cleanup_node(state, MagicMock(), MagicMock())
        assert result["stop_reason"] == "cleanup_not_approved"


# -------------------------------------------------------------------
# Route Function Tests
# -------------------------------------------------------------------

class TestRouteAfterRecovery:
    def test_pending_approval_goes_to_approval(self):
        state = {"pending_approval": {"approval_id": "abc"}}
        assert route_after_recovery(state) == "approval"

    def test_stop_reason_goes_to_stop(self):
        state = {"pending_approval": None, "stop_reason": "conflict"}
        assert route_after_recovery(state) == "stop"

    def test_continue_when_no_issues(self):
        state = {"pending_approval": None, "stop_reason": ""}
        assert route_after_recovery(state) == "continue"


class TestRouteAfterApproval:
    def test_rejected_goes_to_stop(self):
        state = {"stop_reason": "operator_rejected", "approved_action": ""}
        assert route_after_approval(state) == "stop"

    def test_cleanup_goes_to_cleanup(self):
        state = {"stop_reason": "", "approved_action": "cleanup_then_retry"}
        assert route_after_approval(state) == "cleanup"

    def test_approved_retry_goes_to_retry(self):
        state = {"stop_reason": "", "approved_action": "retry"}
        assert route_after_approval(state) == "retry"

    def test_approved_empty_action_goes_to_retry(self):
        state = {"stop_reason": "", "approved_action": ""}
        assert route_after_approval(state) == "retry"
