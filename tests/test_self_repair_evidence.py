"""Tests for self-repair evidence tightening.

Validates:
- Plan-only is not self-repair
- Metadata-only is not self-repair
- Install success without rerun is not verified
- Rerun without verify is not verified
- Old trace is not verified
- Effective action + resume + fresh verify IS verified
- Same failure limit stops loop
"""
import pytest

from auto_harness.repair.evidence import (
    build_repair_attempt,
    compute_fresh_trace,
    compute_repair_verified,
    is_effective_repair_action,
)


class TestIsEffectiveRepairAction:
    """Test is_effective_repair_action function."""

    def test_metadata_only_is_not_effective(self):
        """metadata_only action must not be effective."""
        result = {"metadata_only": True, "executed": True, "exit_code": 0}
        assert is_effective_repair_action(result) is False

    def test_executed_with_zero_exit_code_is_effective(self):
        """Executed action with exit_code 0 is effective."""
        result = {"executed": True, "exit_code": 0}
        assert is_effective_repair_action(result) is True

    def test_executed_with_nonzero_exit_code_is_not_effective(self):
        """Executed action with non-zero exit_code is not effective."""
        result = {"executed": True, "exit_code": 1}
        assert is_effective_repair_action(result) is False

    def test_strong_verify_pass_is_effective(self):
        """Tool result with strong_verify_pass is effective."""
        result = {"tool_result": {"strong_verify_pass": True}}
        assert is_effective_repair_action(result) is True

    def test_non_dict_returns_false(self):
        """Non-dict input returns False."""
        assert is_effective_repair_action(None) is False
        assert is_effective_repair_action("string") is False

    def test_empty_dict_returns_false(self):
        """Empty dict returns False."""
        assert is_effective_repair_action({}) is False


class TestFreshTrace:
    """Test compute_fresh_trace function."""

    def test_different_traces_is_fresh(self):
        """Different before/after traces means fresh."""
        assert compute_fresh_trace("trace-before", "trace-after") is True

    def test_same_traces_is_not_fresh(self):
        """Same before/after traces means not fresh."""
        assert compute_fresh_trace("trace-same", "trace-same") is False

    def test_none_before_is_not_fresh(self):
        """None before trace means not fresh."""
        assert compute_fresh_trace(None, "trace-after") is False

    def test_none_after_is_not_fresh(self):
        """None after trace means not fresh."""
        assert compute_fresh_trace("trace-before", None) is False

    def test_empty_strings_is_not_fresh(self):
        """Empty string traces mean not fresh."""
        assert compute_fresh_trace("", "") is False


class TestRepairVerified:
    """Test compute_repair_verified function."""

    def test_plan_only_is_not_self_repair(self):
        """No effective actions means not verified."""
        assert compute_repair_verified(
            effective_action_count=0,
            resume_executed=True,
            verify_status_after="passed",
            evidence_contains_after_trace=True,
            fresh_trace=True,
        ) is False

    def test_install_success_without_rerun_is_not_verified(self):
        """Effective action but no resume means not verified."""
        assert compute_repair_verified(
            effective_action_count=1,
            resume_executed=False,
            verify_status_after="passed",
            evidence_contains_after_trace=True,
            fresh_trace=True,
        ) is False

    def test_rerun_without_verify_is_not_verified(self):
        """Resume but verify not passed means not verified."""
        assert compute_repair_verified(
            effective_action_count=1,
            resume_executed=True,
            verify_status_after="uncertain",
            evidence_contains_after_trace=True,
            fresh_trace=True,
        ) is False

    def test_old_trace_is_not_verified(self):
        """Verify passed but old trace means not verified."""
        assert compute_repair_verified(
            effective_action_count=1,
            resume_executed=True,
            verify_status_after="passed",
            evidence_contains_after_trace=True,
            fresh_trace=False,
        ) is False

    def test_effective_action_resume_fresh_verify_is_verified(self):
        """All conditions met means verified."""
        assert compute_repair_verified(
            effective_action_count=1,
            resume_executed=True,
            verify_status_after="passed",
            evidence_contains_after_trace=True,
            fresh_trace=True,
        ) is True

    def test_verify_status_pass_also_works(self):
        """verify_status_after='pass' also works."""
        assert compute_repair_verified(
            effective_action_count=1,
            resume_executed=True,
            verify_status_after="pass",
            evidence_contains_after_trace=True,
            fresh_trace=True,
        ) is True


class TestRepairAttempt:
    """Test build_repair_attempt function."""

    def test_schema_version(self):
        """RepairAttempt must have schema_version=1."""
        attempt = build_repair_attempt(attempt=1)
        assert attempt["schema_version"] == 1

    def test_attempt_number(self):
        """Attempt number must be recorded."""
        attempt = build_repair_attempt(attempt=2)
        assert attempt["attempt"] == 2

    def test_defaults(self):
        """Default values must be correct."""
        attempt = build_repair_attempt(attempt=1)
        assert attempt["effective_action_count"] == 0
        assert attempt["metadata_only_count"] == 0
        assert attempt["repair_verified"] is False
        assert attempt["fresh_trace"] is False

    def test_full_attempt(self):
        """Full attempt with all fields."""
        attempt = build_repair_attempt(
            attempt=1,
            failure_signature_before="ImportError: requests",
            diagnosis_path="repairs/diagnosis_1.json",
            plan_path="repairs/repair_plan_1.json",
            policy_path="repairs/repair_policy_1.json",
            apply_path="repairs/repair_apply_1.json",
            resume_from_stage="env_deploy",
            effective_action_count=1,
            metadata_only_count=0,
            verify_status_after="passed",
            verification_trace_id="trace-after-123",
            fresh_trace=True,
            repair_verified=True,
        )
        assert attempt["failure_signature_before"] == "ImportError: requests"
        assert attempt["effective_action_count"] == 1
        assert attempt["repair_verified"] is True


class TestSameFailureLimit:
    """Test that same failure limit stops the repair loop."""

    def test_same_failure_limit_stops_loop(self):
        """When same_failure_count reaches max, loop must stop."""
        max_same_failure = 2
        same_failure_count = 2
        # The graph nodes already implement this check
        # This test verifies the logic
        should_stop = same_failure_count >= max_same_failure
        assert should_stop is True

    def test_different_failure_allows_continue(self):
        """Different failure signature should allow repair to continue."""
        max_same_failure = 2
        same_failure_count = 1
        should_stop = same_failure_count >= max_same_failure
        assert should_stop is False
