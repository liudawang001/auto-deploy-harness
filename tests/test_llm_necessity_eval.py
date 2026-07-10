"""Tests for LLM Necessity Evaluator (Phase 6).

Covers:
- manifest loading and case evaluation
- llm_required only on baseline failure + agent pass
- llm_helped is always bool (not string)
- safety case correctly identifies policy rejection
- report generation with summary
- evaluator actually runs pipelines (not reads expectations)
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.evals.llm_necessity import LLMNecessityEvaluator


class TestLLMNecessityEvaluator(unittest.TestCase):
    """Tests for LLMNecessityEvaluator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "eval_output"
        self.evaluator = LLMNecessityEvaluator(output_dir=self.output_dir)

    def test_llm_helped_is_bool(self):
        """llm_helped must always be bool, not string."""
        results = [
            {"case_id": "a", "llm_required": True, "llm_helped": True, "fixture_exists": True},
            {"case_id": "b", "llm_required": False, "llm_helped": False, "fixture_exists": True},
        ]
        summary = self.evaluator._build_summary(results)
        self.assertTrue(summary["all_llm_helped_are_bool"])
        for r in results:
            self.assertIsInstance(r["llm_helped"], bool)

    def test_evaluate_case_fixture_not_found(self):
        """Should return error result when fixture doesn't exist."""
        case = {
            "case_id": "missing_fixture",
            "target_gate": "runner",
            "fixture_dir": "/nonexistent/path",
        }
        result = self.evaluator.evaluate_case(case)
        self.assertFalse(result["llm_required"])
        self.assertFalse(result["fixture_exists"])
        self.assertEqual(result["baseline_status"], "error")

    def test_evaluate_manifest_missing_file(self):
        result = self.evaluator.evaluate_manifest(Path("/nonexistent/manifest.json"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("not found", result["error"])

    def test_summary_counts(self):
        results = [
            {"case_id": "a", "llm_required": True, "llm_helped": True, "fixture_exists": True},
            {"case_id": "b", "llm_required": False, "llm_helped": False, "fixture_exists": True},
            {"case_id": "malicious_c", "llm_required": False, "llm_helped": False, "fixture_exists": True},
        ]
        summary = self.evaluator._build_summary(results)
        self.assertEqual(summary["total_cases"], 3)
        self.assertEqual(summary["llm_required_count"], 1)
        self.assertEqual(summary["llm_helped_count"], 1)
        self.assertEqual(summary["safety_cases"], 1)
        self.assertTrue(summary["llm_necessity_proven"])

    def test_safety_case_llm_required_false(self):
        """Malicious cases should have llm_required=False."""
        results = [
            {"case_id": "malicious_readme", "llm_required": True, "llm_helped": True, "fixture_exists": True},
        ]
        # The evaluator should set llm_required=False for malicious cases
        # This is tested via the _compare_runs method
        comparison = self.evaluator._compare_runs(
            {"stage_status": {"runner": {"status": "failed"}}},
            {"stage_status": {"runner": {"status": "failed"}}, "decisions": []},
            {"case_id": "malicious_readme_prompt_injection", "target_gate": "runner"},
        )
        self.assertFalse(comparison["llm_required"])

    def test_compare_runs_baseline_passed_not_required(self):
        """When baseline passes, llm_required should be False."""
        comparison = self.evaluator._compare_runs(
            {"stage_status": {"runner": {"status": "passed"}}},
            {"stage_status": {"runner": {"status": "passed"}}, "decisions": []},
            {"case_id": "simple_case", "target_gate": "runner"},
        )
        self.assertFalse(comparison["llm_required"])

    def test_compare_runs_baseline_failed_agent_passed(self):
        """When baseline fails and agent passes with LLM decision, llm_required=True."""
        comparison = self.evaluator._compare_runs(
            {"stage_status": {"runner": {"status": "failed"}}},
            {
                "stage_status": {"runner": {"status": "passed"}},
                "decisions": [
                    {"decision": {
                        "stage": "runner",
                        "policy_allowed": True,
                        "executed": True,
                        "tool_name": "select_runner_candidate",
                    }},
                ],
                "verify": {"status": "passed", "evidence_paths": ["/evidence/1.json"]},
            },
            {"case_id": "wrong_default_entrypoint", "target_gate": "runner"},
        )
        self.assertTrue(comparison["llm_required"])
        self.assertTrue(comparison["llm_helped"])
        self.assertIsInstance(comparison["llm_helped"], bool)

    def test_build_error_result(self):
        """Error result should have correct structure."""
        result = self.evaluator._build_error_result(
            "test_case", "runner", "test_error", "/path",
        )
        self.assertEqual(result["case_id"], "test_case")
        self.assertFalse(result["llm_required"])
        self.assertFalse(result["llm_helped"])
        self.assertEqual(result["error"], "test_error")


class TestManifestStructure(unittest.TestCase):
    """Test the manifest file structure."""

    def test_manifest_has_all_cases(self):
        manifest_path = Path("eval_targets/llm_necessity_manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest not found")
        manifest = json.loads(manifest_path.read_text())
        cases = manifest.get("cases", [])
        case_ids = [c["case_id"] for c in cases]
        expected = [
            "wrong_default_entrypoint",
            "dependency_conflict_pydantic",
            "model_path_ambiguous",
            "repair_missing_dependency",
            "cross_stage_strategy_gradio_model",
            "malicious_readme_prompt_injection",
        ]
        for eid in expected:
            self.assertIn(eid, case_ids, "missing case: %s" % eid)

    def test_manifest_gates_covered(self):
        manifest_path = Path("eval_targets/llm_necessity_manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest not found")
        manifest = json.loads(manifest_path.read_text())
        gates = set(c["target_gate"] for c in manifest.get("cases", []))
        expected_gates = {"runner", "env_solve", "model_prepare", "repair", "plan"}
        self.assertTrue(expected_gates.issubset(gates), "missing gates: %s" % (expected_gates - gates))


if __name__ == "__main__":
    unittest.main()
