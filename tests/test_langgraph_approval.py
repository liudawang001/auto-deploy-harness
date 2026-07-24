"""Tests for approval node, ApprovalStore, and approval CLI flow.

Phase 5 tests: approval request building, store persistence,
sanitize, route functions, and approval node behavior.
Updated for schema-versioned build_approval_request factory.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.graph.approval import (
    ApprovalStore,
    build_approval_request,
    canonical_hash,
    sanitize_approval,
    approval_node,
    cleanup_node,
    route_after_recovery,
    route_after_approval,
    ALLOWED_APPROVAL_FIELDS,
    APPROVAL_SCHEMA_VERSION,
)
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_approval_request(**overrides):
    """Build a valid approval request using the factory."""
    defaults = {
        "approval_id": "test-approval-001",
        "operation_id": "op123",
        "approval_kind": "recovery",
        "requested_action": "cleanup_then_retry",
        "risk": "high",
        "reason": "config changed",
    }
    defaults.update(overrides)
    return build_approval_request(**defaults)


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
# Build Approval Request Tests (schema-versioned)
# -------------------------------------------------------------------

class TestBuildApprovalRequest:
    def test_schema_version(self):
        req = make_approval_request()
        assert req["schema_version"] == APPROVAL_SCHEMA_VERSION

    def test_deterministic_request_hash(self):
        req1 = make_approval_request()
        req2 = make_approval_request()
        # Same inputs -> same hash
        assert req1["request_hash"] == req2["request_hash"]

    def test_different_inputs_different_hash(self):
        req1 = make_approval_request(requested_action="cleanup_then_retry")
        req2 = make_approval_request(requested_action="apply_repair")
        assert req1["request_hash"] != req2["request_hash"]

    def test_required_fields(self):
        req = make_approval_request()
        assert "approval_id" in req
        assert req["operation_id"] == "op123"
        assert req["approval_kind"] == "recovery"
        assert req["requested_action"] == "cleanup_then_retry"
        assert req["risk"] == "high"
        assert "approve" in req["allowed_decisions"]
        assert "reject" in req["allowed_decisions"]
        assert "request_hash" in req
        assert "created_at" in req

    def test_custom_risk(self):
        req = make_approval_request(risk="low")
        assert req["risk"] == "low"

    def test_reason_truncated(self):
        long_reason = "x" * 5000
        req = make_approval_request(reason=long_reason)
        assert len(req["reason"]) <= 2000

    def test_empty_operation_id_rejected(self):
        """Empty operation_id is still allowed (string of empty)."""
        req = make_approval_request(operation_id="")
        assert req["operation_id"] == ""


# -------------------------------------------------------------------
# Canonical Hash Tests
# -------------------------------------------------------------------

class TestCanonicalHash:
    def test_deterministic(self):
        payload = {"a": 1, "b": 2}
        assert canonical_hash(payload) == canonical_hash(payload)

    def test_key_order_independent(self):
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


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

    def test_request_hash_allowed(self):
        """request_hash is in ALLOWED_APPROVAL_FIELDS."""
        decision = {"request_hash": "abc123", "decision": "approve"}
        safe = sanitize_approval(decision)
        assert "request_hash" in safe


# -------------------------------------------------------------------
# Approval Node Tests (without real LangGraph interrupt)
# -------------------------------------------------------------------

class TestApprovalNode:
    def test_missing_request(self):
        """No pending_approval → stop_reason set."""
        state = {"run_dir": "/tmp/test", "pending_approval": None}
        result = approval_node(state)
        assert result["stop_reason"] == "approval_request_missing"

    def test_missing_required_fields(self, tmp_path):
        """Missing required fields → stop."""
        request = {"approval_id": "abc123"}  # Missing most required fields
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        result = approval_node(state)
        assert "approval_request_invalid" in result["stop_reason"]

    def test_approve_decision(self, tmp_path):
        """Approval with 'approve' decision sets approved fields."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": request["operation_id"],
            "decision": "approve",
            "reviewer": "cli",
            "note": "approved",
            "resolved_at": "2025-01-01",
            "request_hash": request["request_hash"],
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["approved_operation_id"] == request["operation_id"]
        assert result["approved_action"] == "cleanup_then_retry"
        assert result["stop_reason"] == ""
        assert result["pending_approval"] is None
        assert len(result["approval_history"]) == 1

    def test_reject_decision(self, tmp_path):
        """Approval with 'reject' decision sets stop_reason."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": request["operation_id"],
            "decision": "reject",
            "reviewer": "cli",
            "note": "rejected",
            "resolved_at": "2025-01-01",
            "request_hash": request["request_hash"],
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["approved_operation_id"] == ""
        assert result["stop_reason"] == "operator_rejected"

    def test_request_hash_mismatch(self, tmp_path):
        """Decision with wrong request_hash → stop."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": request["operation_id"],
            "decision": "approve",
            "request_hash": "wrong_hash",
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert "request_hash_mismatch" in result["stop_reason"]

    def test_operation_mismatch(self, tmp_path):
        """Decision with wrong operation_id → stop."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": "WRONG_OP",
            "decision": "approve",
            "request_hash": request["request_hash"],
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["stop_reason"] == "approval_operation_mismatch"

    def test_approval_id_mismatch(self, tmp_path):
        """Decision with wrong approval_id → stop."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": "WRONG_ID",
            "operation_id": request["operation_id"],
            "decision": "approve",
            "request_hash": request["request_hash"],
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["stop_reason"] == "approval_id_mismatch"

    def test_disallowed_decision(self, tmp_path):
        """Decision not in allowed_decisions → stop."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": request["operation_id"],
            "decision": "maybe",  # Not in allowed_decisions
            "request_hash": request["request_hash"],
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert result["stop_reason"] == "approval_decision_not_allowed"

    def test_non_dict_response(self, tmp_path):
        """Non-dict interrupt response → stop."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        with patch("langgraph.types.interrupt", return_value="not a dict"):
            result = approval_node(state)
        assert result["stop_reason"] == "invalid_approval_response"

    def test_approval_saved_to_store(self, tmp_path):
        """Approval record is saved to ApprovalStore."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": request["operation_id"],
            "decision": "approve",
            "reviewer": "cli",
            "request_hash": request["request_hash"],
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            approval_node(state)
        store = ApprovalStore(tmp_path)
        record = store.load(request["approval_id"])
        assert record is not None
        assert record["status"] == "resolved"

    def test_same_checkpoint_rerun_same_hash(self, tmp_path):
        """Same checkpoint rerun produces same request_hash."""
        request1 = make_approval_request()
        request2 = make_approval_request()
        assert request1["request_hash"] == request2["request_hash"]

    def test_decision_must_bind_request_hash(self, tmp_path):
        """Decision without request_hash is rejected."""
        request = make_approval_request()
        state = {"run_dir": str(tmp_path), "pending_approval": request}
        decision = {
            "approval_id": request["approval_id"],
            "operation_id": request["operation_id"],
            "decision": "approve",
            # No request_hash
        }
        with patch("langgraph.types.interrupt", return_value=decision):
            result = approval_node(state)
        assert "request_hash_mismatch" in result["stop_reason"]


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
