"""Tests for MemoryQualityGate (Phase 1).

Verifies that:
- metadata_only entries are not eligible
- verified_success=false entries are not eligible
- Missing verification_trace_id entries are not eligible
- policy_rejected_high_risk entries are not eligible
- Valid verified memory entries are eligible
- regression_proven entries are classified correctly
- Secret-like values in suggested actions are rejected
- Absolute tmp paths in suggested actions are rejected
"""
import unittest

from auto_harness.memory.quality import MemoryQualityGate


class TestMemoryQualityGate(unittest.TestCase):
    """Test the MemoryQualityGate classify and filter logic."""

    def setUp(self):
        self.gate = MemoryQualityGate()

    def _make_verified_entry(self, **overrides) -> dict:
        """Create a minimal verified_resolution entry."""
        entry = {
            "verified_success": True,
            "verification_trace_id": "trace-abc123",
            "repair_action_hash": "hash_def456",
            "repair_action_status": "executed",
            "policy_rejected_high_risk": False,
            "metadata_only": False,
            "regression_status": "passed",
            "regression_case_ids": ["case_1"],
            "suggested_next_action": "Promote after regression.",
        }
        entry.update(overrides)
        return entry

    def test_metadata_only_not_eligible(self):
        """metadata_only=true should not be eligible for evolution."""
        entry = self._make_verified_entry(metadata_only=True)
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("metadata_only", " ".join(result["reject_reasons"]).lower())

    def test_verified_success_false_not_eligible(self):
        """verified_success=false should not be eligible."""
        entry = self._make_verified_entry(verified_success=False)
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("verified_success", " ".join(result["reject_reasons"]).lower())

    def test_missing_trace_id_not_eligible(self):
        """Missing verification_trace_id should not be eligible."""
        entry = self._make_verified_entry(verification_trace_id="")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("trace_id", " ".join(result["reject_reasons"]).lower())

    def test_missing_repair_hash_not_eligible(self):
        """Missing repair_action_hash should not be eligible."""
        entry = self._make_verified_entry(repair_action_hash="")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("repair_action_hash", " ".join(result["reject_reasons"]).lower())

    def test_policy_rejected_high_risk_not_eligible(self):
        """policy_rejected_high_risk=true should not be eligible."""
        entry = self._make_verified_entry(policy_rejected_high_risk=True)
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("policy_rejected_high_risk", " ".join(result["reject_reasons"]).lower())

    def test_rejected_high_risk_action_not_eligible(self):
        """rejected_high_risk_action=true should not be eligible."""
        entry = self._make_verified_entry(rejected_high_risk_action=True)
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])

    def test_valid_verified_memory_eligible(self):
        """Valid verified memory should be eligible."""
        entry = self._make_verified_entry()
        result = self.gate.classify(entry)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["level"], "regression_proven")

    def test_verified_resolution_without_regression(self):
        """Verified but no regression case_ids → verified_resolution level."""
        entry = self._make_verified_entry(regression_case_ids=[], regression_status="")
        result = self.gate.classify(entry)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["level"], "verified_resolution")

    def test_regression_failed_not_eligible(self):
        """regression_status=failed should not be eligible."""
        entry = self._make_verified_entry(regression_status="failed")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("regression", " ".join(result["reject_reasons"]).lower())

    def test_secret_in_suggested_action_not_eligible(self):
        """Secret-like value in suggested_next_action should not be eligible."""
        entry = self._make_verified_entry(suggested_next_action="Set api_key=sk-xxx and retry")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("secret", " ".join(result["reject_reasons"]).lower())

    def test_absolute_tmp_path_not_eligible(self):
        """Absolute /tmp/ path in suggested_next_action should not be eligible."""
        entry = self._make_verified_entry(suggested_next_action="Copy from /tmp/build_output to project")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])
        self.assertIn("tmp", " ".join(result["reject_reasons"]).lower())

    def test_raw_issue_classification(self):
        """Non-verified entry without root_cause → raw_issue level."""
        entry = {"verified_success": False, "root_cause": ""}
        result = self.gate.classify(entry)
        self.assertEqual(result["level"], "raw_issue")
        self.assertFalse(result["eligible"])

    def test_diagnosed_issue_classification(self):
        """Non-verified entry with root_cause → diagnosed_issue level."""
        entry = {"verified_success": False, "root_cause": "missing dependency"}
        result = self.gate.classify(entry)
        self.assertEqual(result["level"], "diagnosed_issue")
        self.assertFalse(result["eligible"])

    def test_eligible_for_evolution_method(self):
        """eligible_for_evolution() returns bool."""
        entry = self._make_verified_entry()
        self.assertTrue(self.gate.eligible_for_evolution(entry))
        entry_bad = self._make_verified_entry(verified_success=False)
        self.assertFalse(self.gate.eligible_for_evolution(entry_bad))

    def test_filter_verified(self):
        """filter_verified() returns only eligible entries."""
        good = self._make_verified_entry()
        bad = self._make_verified_entry(verified_success=False)
        result = self.gate.filter_verified([good, bad])
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["verified_success"])

    def test_bearer_in_suggested_action_rejected(self):
        """Bearer token pattern in suggested action should be rejected."""
        entry = self._make_verified_entry(suggested_next_action="Use Bearer xyz123 for auth")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])

    def test_users_path_in_suggested_action_rejected(self):
        """/Users/name path in suggested action should be rejected."""
        entry = self._make_verified_entry(suggested_next_action="Load model from /Users/alice/models")
        result = self.gate.classify(entry)
        self.assertFalse(result["eligible"])


if __name__ == "__main__":
    unittest.main()
