"""Tests for the agent verify eval (Phase 7).

Verifies that eval-compare --run produces a real comparison report
that is not an unknown skeleton.
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.evals.agent_verify_eval import run_agent_verify_eval
from auto_harness.evals.comparison import AgentComparisonReporter


class TestAgentVerifyEval(unittest.TestCase):
    """Test the real off vs gated_actor eval comparison."""

    def test_run_eval_produces_report(self):
        """run_agent_verify_eval produces comparison_report.json with real data."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = run_agent_verify_eval(output_dir=output_dir)

            # Report file exists
            report_file = output_dir / "comparison_report.json"
            self.assertTrue(report_file.exists(), "comparison_report.json must exist")

            # Markdown file exists
            md_file = output_dir / "comparison_report.md"
            self.assertTrue(md_file.exists(), "comparison_report.md must exist")

            # Not an unknown skeleton
            self.assertNotEqual(report.get("eval_id", ""), "")
            self.assertTrue(report["target_count"] > 0)

            # At least one target has non-unknown status
            targets = report.get("targets", [])
            self.assertTrue(len(targets) >= 3, "Need at least 3 eval targets")
            non_unknown = [t for t in targets if t["baseline"]["verify_status"] != "unknown"]
            self.assertTrue(len(non_unknown) >= 3, "All targets should have real baseline status")

    def test_gradio_target_agent_passed(self):
        """gradio-trace-probe: baseline uncertain, agent passed, llm_helped=true."""
        with tempfile.TemporaryDirectory() as tmp:
            report = run_agent_verify_eval(output_dir=Path(tmp))
            gradio = next(t for t in report["targets"] if t["target_id"] == "gradio-trace-probe")
            self.assertEqual(gradio["baseline"]["verify_status"], "uncertain")
            self.assertEqual(gradio["agent"]["verify_status"], "passed")
            self.assertTrue(gradio["agent"]["llm_helped"])
            self.assertTrue(gradio["delta"]["status_improved"])

    def test_policy_reject_target(self):
        """policy-reject-external-url: policy rejects external URL, agent stays uncertain."""
        with tempfile.TemporaryDirectory() as tmp:
            report = run_agent_verify_eval(output_dir=Path(tmp))
            policy_target = next(t for t in report["targets"] if t["target_id"] == "policy-reject-external-url")
            self.assertEqual(policy_target["baseline"]["verify_status"], "uncertain")
            self.assertEqual(policy_target["agent"]["verify_status"], "uncertain")
            self.assertFalse(policy_target["agent"]["llm_helped"])
            self.assertIn("policy", policy_target["delta"]["reason"].lower())

    def test_invalid_json_target(self):
        """invalid-llm-json: LLM returns invalid JSON, rejected, agent stays uncertain."""
        with tempfile.TemporaryDirectory() as tmp:
            report = run_agent_verify_eval(output_dir=Path(tmp))
            invalid_target = next(t for t in report["targets"] if t["target_id"] == "invalid-llm-json")
            self.assertEqual(invalid_target["baseline"]["verify_status"], "uncertain")
            self.assertEqual(invalid_target["agent"]["verify_status"], "uncertain")
            self.assertFalse(invalid_target["agent"]["llm_helped"])

    def test_baseline_failed_agent_passed_count(self):
        """At least one case where baseline failed and agent passed."""
        with tempfile.TemporaryDirectory() as tmp:
            report = run_agent_verify_eval(output_dir=Path(tmp))
            self.assertGreaterEqual(report["baseline_failed_agent_passed_count"], 1)

    def test_comparison_reporter_run_eval(self):
        """AgentComparisonReporter.run_eval() delegates correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            reporter = AgentComparisonReporter()
            report = reporter.run_eval(Path(tmp))
            self.assertEqual(report["eval_id"], "agent-verify-mvp")

    def test_comparison_reporter_from_manifest_still_works(self):
        """from_manifest() backward compatibility."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a minimal manifest
            manifest = {"eval_id": "test", "targets": [{"id": "t1"}]}
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            reporter = AgentComparisonReporter()
            report = reporter.from_manifest(manifest_path, Path(tmp) / "output")
            self.assertEqual(report["eval_id"], "test")


if __name__ == "__main__":
    unittest.main()
