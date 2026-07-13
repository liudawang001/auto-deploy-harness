"""Tests for SkillMetricsReporter: aggregate skill selection/influence/pass/harm metrics."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from auto_harness.skills.metrics import SkillMetricsReporter
from auto_harness.models.base import write_json


class TestSkillMetricsReporter(unittest.TestCase):
    """Test SkillMetricsReporter."""

    def setUp(self):
        self.reporter = SkillMetricsReporter()

    def test_selection_count(self):
        effects = {"effects": [
            {"skill_name": "verify-evidence", "skill_sha256": "abc123", "field_changed": "verify.request", "accepted_by_policy": True},
        ]}
        pipeline = {"verify": {"status": "passed"}}
        result = self.reporter.compute_run_metrics(effects, pipeline)
        self.assertTrue(len(result["skills"]) > 0)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["selection_count"], 1)

    def test_influence_count(self):
        effects = {"effects": [
            {"skill_name": "deploy-python-webui", "skill_sha256": "def456", "field_changed": "run.candidates", "accepted_by_policy": True},
            {"skill_name": "deploy-python-webui", "skill_sha256": "def456", "field_changed": "environment.install_commands", "accepted_by_policy": True},
        ]}
        pipeline = {"verify": {"status": "passed"}}
        result = self.reporter.compute_run_metrics(effects, pipeline)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["influence_count"], 2)

    def test_policy_accept_rate(self):
        effects = {"effects": [
            {"skill_name": "test-skill", "skill_sha256": "abc", "field_changed": "verify.request", "accepted_by_policy": True},
            {"skill_name": "test-skill", "skill_sha256": "abc", "field_changed": "run.candidates", "accepted_by_policy": False},
        ]}
        pipeline = {"verify": {"status": "passed"}}
        result = self.reporter.compute_run_metrics(effects, pipeline)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["policy_accept_count"], 1)
        self.assertEqual(result["skills"][skill_key]["policy_reject_count"], 1)
        self.assertAlmostEqual(result["skills"][skill_key]["policy_accept_rate"], 0.5)

    def test_verify_pass_rate(self):
        effects = {"effects": [
            {"skill_name": "verify-skill", "skill_sha256": "xyz", "field_changed": "verify.request", "accepted_by_policy": True},
        ]}
        pipeline = {"verify": {"status": "passed"}}
        result = self.reporter.compute_run_metrics(effects, pipeline)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["verify_pass_count"], 1)
        self.assertAlmostEqual(result["skills"][skill_key]["verify_pass_rate"], 1.0)

    def test_harm_rate_when_verify_failed(self):
        effects = {"effects": [
            {"skill_name": "harm-skill", "skill_sha256": "bad", "field_changed": "run.candidates", "accepted_by_policy": True},
        ]}
        pipeline = {"verify": {"status": "failed"}}
        result = self.reporter.compute_run_metrics(effects, pipeline)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["harm_count"], 1)
        self.assertAlmostEqual(result["skills"][skill_key]["harm_rate"], 1.0)

    def test_no_harm_when_verify_passed(self):
        effects = {"effects": [
            {"skill_name": "good-skill", "skill_sha256": "good", "field_changed": "run.candidates", "accepted_by_policy": True},
        ]}
        pipeline = {"verify": {"status": "passed"}}
        result = self.reporter.compute_run_metrics(effects, pipeline)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["harm_count"], 0)
        self.assertAlmostEqual(result["skills"][skill_key]["harm_rate"], 0.0)

    def test_llm_helped_count(self):
        effects = {"effects": [
            {"skill_name": "test-skill", "skill_sha256": "abc", "field_changed": "verify.request", "accepted_by_policy": True},
        ]}
        pipeline = {"verify": {"status": "passed"}}
        agent_metrics = {"agent_metrics": {"llm_helped": True}}
        result = self.reporter.compute_run_metrics(effects, pipeline, agent_metrics)
        skill_key = list(result["skills"].keys())[0]
        self.assertEqual(result["skills"][skill_key]["llm_helped_count"], 1)

    def test_aggregate_across_runs(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            # Create two mock runs
            for run_id in ("run-1", "run-2"):
                run_dir = tmpdir / run_id
                reports = run_dir / "reports"
                reports.mkdir(parents=True, exist_ok=True)
                write_json(reports / "skill_effects.json", {
                    "effects": [
                        {"skill_name": "verify-evidence", "skill_sha256": "abc123", "field_changed": "verify.request", "accepted_by_policy": True},
                    ],
                })
                write_json(reports / "pipeline_results.json", {"verify": {"status": "passed"}})

            result = self.reporter.aggregate_metrics(tmpdir)
            skill_key = list(result["skills"].keys())[0]
            self.assertEqual(result["skills"][skill_key]["selection_count"], 2)
            self.assertEqual(result["skills"][skill_key]["influence_count"], 2)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_aggregate_with_output_path(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            run_dir = tmpdir / "run-1"
            reports = run_dir / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            write_json(reports / "skill_effects.json", {
                "effects": [
                    {"skill_name": "test", "skill_sha256": "abc", "field_changed": "x", "accepted_by_policy": True},
                ],
            })
            write_json(reports / "pipeline_results.json", {"verify": {"status": "passed"}})

            output = tmpdir / "skill_metrics.json"
            result = self.reporter.aggregate_metrics(tmpdir, output_path=output)
            self.assertTrue(output.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
