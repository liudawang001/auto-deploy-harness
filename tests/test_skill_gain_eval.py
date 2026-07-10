"""Tests for Skill Gain Evaluator (Phase 7).

Verifies:
- Candidate with better tool shows gain
- Candidate with same tool shows no gain
- Baseline already passing shows no gain
- Candidate with no recommendation shows no gain
- Gain report is written to output path
- Heuristic fallback works without run_dir
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.evals.skill_gain import SkillGainEvaluator


def _make_candidate(
    base_sha: str = "abc123",
    source_memory_ids: list = None,
    reusable_rule: dict = None,
) -> dict:
    """Create a minimal candidate for gain evaluation."""
    return {
        "candidate_id": "skillcand_gain_test",
        "target_skill": "verify-evidence/SKILL.md",
        "base_skill_sha256": base_sha,
        "source_memory_ids": source_memory_ids or ["mem_001", "mem_002", "mem_003"],
        "pattern": {
            "stage": "verify",
            "frameworks": ["gradio"],
            "failure_signature": "HTTP 200 but no trace_id",
        },
        "reusable_rule": reusable_rule or {
            "when": "verify uncertain and framework_hint=gradio",
            "do": ["discover /config with discover_gradio_api", "send current trace_id"],
            "do_not": ["do not mark success on HTTP 200 alone"],
        },
        "patch": {
            "section_title": "Gradio API shape discovery",
            "markdown": "When Gradio verify is uncertain, inspect /config and probe the inferred callable endpoint with the current trace_id. Do not mark success on HTTP 200 alone.",
        },
    }


class TestSkillGainEvaluator(unittest.TestCase):
    """Test SkillGainEvaluator."""

    def test_candidate_with_better_tool_shows_gain(self):
        """Candidate selecting a better tool shows gain over baseline."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _make_candidate()
            candidate_path = Path(tmp) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            evaluator = SkillGainEvaluator()
            report = evaluator.evaluate_candidate(
                candidate_path,
                observation={"status": "uncertain", "baseline_tool": "probe_http"},
            )

            self.assertTrue(report["gain"]["improved"])
            self.assertIn("discover_gradio_api", report["gain"]["reason"])

    def test_candidate_same_tool_no_gain(self):
        """Candidate selecting same tool as baseline shows no gain."""
        with tempfile.TemporaryDirectory() as tmp:
            # Candidate recommends probe_http (same as baseline)
            candidate = _make_candidate(reusable_rule={
                "when": "verify uncertain",
                "do": ["probe HTTP endpoint"],
                "do_not": [],
            })
            candidate_path = Path(tmp) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            evaluator = SkillGainEvaluator()
            report = evaluator.evaluate_candidate(
                candidate_path,
                observation={"status": "uncertain", "baseline_tool": "probe_http"},
            )

            self.assertFalse(report["gain"]["improved"])
            self.assertIn("same tool", report["gain"]["reason"])

    def test_baseline_already_passing_no_gain(self):
        """When baseline already passes, no gain is needed."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _make_candidate()
            candidate_path = Path(tmp) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            evaluator = SkillGainEvaluator()
            report = evaluator.evaluate_candidate(
                candidate_path,
                observation={"status": "passed", "baseline_tool": "probe_http"},
            )

            self.assertFalse(report["gain"]["improved"])
            self.assertIn("already passes", report["gain"]["reason"])

    def test_no_recommendation_no_gain(self):
        """Candidate with no recommended tools shows no gain."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _make_candidate(reusable_rule={
                "when": "verify uncertain",
                "do": [],
                "do_not": [],
            })
            candidate_path = Path(tmp) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            evaluator = SkillGainEvaluator()
            report = evaluator.evaluate_candidate(candidate_path)

            self.assertFalse(report["gain"]["improved"])
            self.assertEqual(report["candidate"]["status"], "no_recommendation")

    def test_gain_report_written_to_file(self):
        """Gain report is written to output path."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _make_candidate()
            candidate_path = Path(tmp) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            output_path = Path(tmp) / "gain_report.json"

            evaluator = SkillGainEvaluator()
            report = evaluator.evaluate_candidate(
                candidate_path,
                observation={"status": "uncertain", "baseline_tool": "probe_http"},
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["candidate_id"], "skillcand_gain_test")
            self.assertTrue(loaded["gain"]["improved"])

    def test_heuristic_fallback_without_run_dir(self):
        """Heuristic fallback works when no run_dir is provided."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _make_candidate()
            candidate_path = Path(tmp) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            evaluator = SkillGainEvaluator()
            report = evaluator.evaluate_candidate(
                candidate_path,
                observation={"status": "uncertain"},
            )

            # Should use heuristic from reusable_rule
            self.assertEqual(report["candidate"]["shadow_decision"], "discover_gradio_api")
            self.assertFalse(report["candidate"]["would_execute"])

    def test_candidate_not_found(self):
        """Non-existent candidate returns failed."""
        evaluator = SkillGainEvaluator()
        report = evaluator.evaluate_candidate(Path("/nonexistent"))
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
