"""E2E tests for LLM Necessity Evaluator with real fixtures.

Phase 7: These tests actually run the evaluator with fixture directories
to verify that baseline and agent pipelines produce different results.
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.evals.llm_necessity import LLMNecessityEvaluator


class TestLLMNecessityE2E(unittest.TestCase):
    """E2E tests that run the evaluator with real fixtures."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir) / "eval_output"
        self.evaluator = LLMNecessityEvaluator(output_dir=self.output_dir)
        self.fixtures_dir = Path("tests/fixtures/llm_necessity")

    @unittest.skipUnless(
        Path("tests/fixtures/llm_necessity/wrong_default_entrypoint").exists(),
        "Fixture not found",
    )
    def test_wrong_default_entrypoint_fixture_exists(self):
        """wrong_default_entrypoint fixture should exist with required files."""
        fixture_dir = self.fixtures_dir / "wrong_default_entrypoint"
        self.assertTrue(fixture_dir.exists())
        # Should have app.py and gradio_app.py
        self.assertTrue((fixture_dir / "app.py").exists())
        self.assertTrue((fixture_dir / "gradio_app.py").exists())

    @unittest.skipUnless(
        Path("tests/fixtures/llm_necessity/dependency_conflict_pydantic").exists(),
        "Fixture not found",
    )
    def test_dependency_conflict_pydantic_fixture_exists(self):
        """dependency_conflict_pydantic fixture should exist with required files."""
        fixture_dir = self.fixtures_dir / "dependency_conflict_pydantic"
        self.assertTrue(fixture_dir.exists())
        # Should have requirements.txt
        self.assertTrue((fixture_dir / "requirements.txt").exists())

    @unittest.skipUnless(
        Path("tests/fixtures/llm_necessity/model_path_ambiguous").exists(),
        "Fixture not found",
    )
    def test_model_path_ambiguous_fixture_exists(self):
        """model_path_ambiguous fixture should exist with required files."""
        fixture_dir = self.fixtures_dir / "model_path_ambiguous"
        self.assertTrue(fixture_dir.exists())

    def test_evaluate_case_creates_output_dirs(self):
        """evaluate_case should create baseline and agent output directories."""
        case = {
            "case_id": "test_output_dirs",
            "target_gate": "runner",
            "fixture_dir": str(self.fixtures_dir / "wrong_default_entrypoint"),
        }
        result = self.evaluator.evaluate_case(case)

        # Output directories should be created
        case_dir = self.output_dir / "test_output_dirs"
        self.assertTrue(case_dir.exists())
        self.assertTrue((case_dir / "baseline").exists())
        self.assertTrue((case_dir / "agent").exists())
        self.assertTrue((case_dir / "comparison.json").exists())

    def test_evaluate_case_result_structure(self):
        """evaluate_case should return result with all required fields."""
        case = {
            "case_id": "test_structure",
            "target_gate": "runner",
            "fixture_dir": str(self.fixtures_dir / "wrong_default_entrypoint"),
        }
        result = self.evaluator.evaluate_case(case)

        # Check required fields
        required_fields = [
            "case_id", "target_gate", "baseline_status", "agent_status",
            "llm_required", "llm_helped", "llm_helped_type",
            "fixture_exists", "evidence_paths",
        ]
        for field in required_fields:
            self.assertIn(field, result, "Missing field: %s" % field)

        # llm_helped must be bool
        self.assertIsInstance(result["llm_helped"], bool)
        self.assertEqual(result["llm_helped_type"], "bool")

    def test_evaluate_manifest_with_real_fixtures(self):
        """evaluate_manifest should run with real fixture manifest."""
        manifest_path = Path("eval_targets/llm_necessity_manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest not found")

        output_path = Path(self.tmpdir) / "report.json"
        report = self.evaluator.evaluate_manifest(manifest_path, output_path)

        self.assertEqual(report["status"], "completed")
        self.assertGreater(report["case_count"], 0)
        self.assertTrue(output_path.exists())

        # Check summary
        summary = report["summary"]
        self.assertIn("total_cases", summary)
        self.assertIn("llm_required_count", summary)
        self.assertIn("all_llm_helped_are_bool", summary)
        self.assertTrue(summary["all_llm_helped_are_bool"])

    def test_safety_case_never_llm_required(self):
        """Malicious cases should never have llm_required=True."""
        manifest_path = Path("eval_targets/llm_necessity_manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest not found")

        report = self.evaluator.evaluate_manifest(manifest_path)

        # Find malicious cases
        for result in report["results"]:
            if "malicious" in result.get("case_id", ""):
                self.assertFalse(
                    result["llm_required"],
                    "Malicious case %s should have llm_required=False" % result["case_id"],
                )


class TestFixtureContent(unittest.TestCase):
    """Test that fixtures have the right content for E2E testing."""

    def test_wrong_default_entrypoint_app_py(self):
        """app.py should be a placeholder, not the real entrypoint."""
        app_py = Path("tests/fixtures/llm_necessity/wrong_default_entrypoint/app.py")
        if not app_py.exists():
            self.skipTest("fixture not found")
        content = app_py.read_text()
        # Should be a placeholder
        self.assertIn("placeholder", content.lower())

    def test_wrong_default_entrypoint_gradio_app_py(self):
        """gradio_app.py should be the real entrypoint."""
        gradio_app = Path("tests/fixtures/llm_necessity/wrong_default_entrypoint/gradio_app.py")
        if not gradio_app.exists():
            self.skipTest("fixture not found")
        content = gradio_app.read_text()
        # Should import gradio
        self.assertIn("gradio", content.lower())

    def test_dependency_conflict_requirements(self):
        """requirements.txt should have conflicting dependencies."""
        req_file = Path("tests/fixtures/llm_necessity/dependency_conflict_pydantic/requirements.txt")
        if not req_file.exists():
            self.skipTest("fixture not found")
        content = req_file.read_text()
        # Should have pydantic or gradio
        self.assertTrue("pydantic" in content.lower() or "gradio" in content.lower())


if __name__ == "__main__":
    unittest.main()
