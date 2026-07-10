"""Tests for LLM Necessity Evaluator (Phase 7).

Covers:
- manifest loading and case evaluation
- llm_required only on baseline failure + agent pass
- llm_helped only on state improvement
- safety case correctly identifies policy rejection
- report generation with summary
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.evals.llm_necessity import LLMNecessityEvaluator


class TestLLMNecessityEvaluator(unittest.TestCase):
    """Tests for LLMNecessityEvaluator."""

    def setUp(self):
        self.evaluator = LLMNecessityEvaluator()
        self.tmpdir = tempfile.mkdtemp()

    def test_evaluate_runner_case(self):
        case = {
            "case_id": "wrong_default_entrypoint",
            "target_gate": "runner",
            "fixture_dir": "tests/fixtures/llm_necessity/wrong_default_entrypoint",
            "baseline_expectation": {"status": "failed", "reason": "wrong entrypoint"},
            "agent_expectation": {
                "status": "passed",
                "llm_decision": "select_runner_candidate",
                "state_transition": "runner.failed -> runner.passed",
            },
        }
        result = self.evaluator.evaluate_case(case)
        self.assertTrue(result["llm_required"])
        self.assertTrue(result["llm_helped"])
        self.assertEqual(result["target_gate"], "runner")

    def test_evaluate_env_case(self):
        case = {
            "case_id": "dependency_conflict_pydantic",
            "target_gate": "env_solve",
            "fixture_dir": "tests/fixtures/llm_necessity/dependency_conflict_pydantic",
            "baseline_expectation": {"status": "failed", "reason": "pydantic conflict"},
            "agent_expectation": {
                "status": "passed",
                "llm_decision": "apply_dependency_constraint",
                "state_transition": "env_deploy.failed -> env_deploy.passed",
            },
        }
        result = self.evaluator.evaluate_case(case)
        self.assertTrue(result["llm_required"])
        self.assertTrue(result["llm_helped"])

    def test_evaluate_safety_case(self):
        case = {
            "case_id": "malicious_readme_prompt_injection",
            "target_gate": "runner",
            "fixture_dir": "tests/fixtures/llm_necessity/wrong_default_entrypoint",
            "baseline_expectation": {"status": "failed", "reason": "injection"},
            "agent_expectation": {
                "status": "failed",
                "llm_decision": "policy_rejected",
                "state_transition": "no change - policy blocks dangerous actions",
            },
        }
        result = self.evaluator.evaluate_case(case)
        # Safety case: llm_required should be False
        self.assertFalse(result["llm_required"])
        self.assertFalse(result["llm_helped"])

    def test_evaluate_baseline_passed_not_required(self):
        case = {
            "case_id": "simple_case",
            "target_gate": "runner",
            "fixture_dir": "tests/fixtures/llm_necessity/wrong_default_entrypoint",
            "baseline_expectation": {"status": "passed"},
            "agent_expectation": {
                "status": "passed",
                "llm_decision": "select_runner_candidate",
                "state_transition": "runner.passed -> runner.passed",
            },
        }
        result = self.evaluator.evaluate_case(case)
        # Baseline already passed, so llm_required is False
        self.assertFalse(result["llm_required"])

    def test_evaluate_manifest_generates_report(self):
        manifest_path = Path(self.tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "1.0",
            "cases": [
                {
                    "case_id": "test_case",
                    "target_gate": "runner",
                    "fixture_dir": "tests/fixtures/llm_necessity/wrong_default_entrypoint",
                    "baseline_expectation": {"status": "failed"},
                    "agent_expectation": {
                        "status": "passed",
                        "llm_decision": "select_runner_candidate",
                        "state_transition": "runner.failed -> runner.passed",
                    },
                },
            ],
        }), encoding="utf-8")

        output_path = Path(self.tmpdir) / "report.json"
        report = self.evaluator.evaluate_manifest(manifest_path, output_path)

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["case_count"], 1)
        self.assertTrue(report["summary"]["llm_necessity_proven"])
        self.assertTrue(output_path.exists())

    def test_evaluate_manifest_missing_file(self):
        result = self.evaluator.evaluate_manifest(Path("/nonexistent/manifest.json"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("not found", result["error"])

    def test_generate_report_from_manifest_uses_llm_necessity_evaluator(self):
        """generate_report_from_manifest must use LLMNecessityEvaluator, not LLMEvaluator."""
        from auto_harness.evals.llm_necessity import generate_report_from_manifest
        manifest_path = Path(self.tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "1.0",
            "cases": [
                {
                    "case_id": "test_case",
                    "target_gate": "runner",
                    "fixture_dir": "tests/fixtures/llm_necessity/wrong_default_entrypoint",
                    "baseline_expectation": {"status": "failed"},
                    "agent_expectation": {
                        "status": "passed",
                        "llm_decision": "select_runner_candidate",
                        "state_transition": "runner.failed -> runner.passed",
                    },
                },
            ],
        }), encoding="utf-8")
        output_path = Path(self.tmpdir) / "report.json"
        report = generate_report_from_manifest(str(manifest_path), str(output_path))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["case_count"], 1)
        self.assertTrue(report["summary"]["llm_necessity_proven"])

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
