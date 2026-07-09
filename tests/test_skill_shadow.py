"""Tests for Shadow Skill Evaluation (Phase 5).

Verifies:
- Shadow evaluation does not change real deployment results
- Candidate helped/harmful counts accumulate
- Shadow artifact is written
- Shadow threshold updates candidate status
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.skills.shadow import ShadowSkillEvaluator


def _make_candidate() -> dict:
    """Create a minimal candidate for shadow eval."""
    return {
        "candidate_id": "skillcand_shadow_test",
        "status": "candidate",
        "target_skill": "verify-evidence/SKILL.md",
        "base_skill_sha256": "abc123",
        "pattern": {
            "stage": "verify",
            "frameworks": ["gradio"],
            "failure_signature": "HTTP 200 but no trace_id",
        },
        "reusable_rule": {
            "when": "verify uncertain and framework_hint=gradio",
            "do": ["discover /config", "send current trace_id"],
            "do_not": ["do not mark success on HTTP 200 alone"],
        },
        "shadow": {"enabled": False, "helped_count": 0, "harmful_count": 0},
    }


def _make_run_dir(tmp: Path, with_steps: bool = True) -> Path:
    """Create a minimal run directory with agent verify results."""
    run_dir = tmp / "runs" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    # Write agent verify result
    result = {
        "stage": "verify",
        "frameworks": ["gradio"],
        "final_status": "passed",
        "llm_helped": True,
    }
    (run_dir / "reports" / "agent_verify_result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )

    if with_steps:
        # Write agent verify steps
        steps = [
            {"step": 1, "tool_name": "discover_gradio_api", "status": "passed"},
            {"step": 2, "tool_name": "probe_http", "status": "ok"},
        ]
        (run_dir / "agent_verify_steps.jsonl").write_text(
            "\n".join(json.dumps(s) for s in steps) + "\n",
            encoding="utf-8",
        )

    return run_dir


class TestShadowSkillEvaluator(unittest.TestCase):
    """Test ShadowSkillEvaluator."""

    def test_evaluate_run_matched_helped(self):
        """Shadow eval matches and detects would_help."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_dir = _make_run_dir(tmp)
            candidate_path = tmp / "candidate.json"
            candidate_path.write_text(json.dumps(_make_candidate()), encoding="utf-8")

            evaluator = ShadowSkillEvaluator()
            result = evaluator.evaluate_run(run_dir, candidate_path)

            self.assertTrue(result["matched"])
            self.assertTrue(result["would_help"])
            self.assertFalse(result["would_harm"])

    def test_evaluate_run_not_matched(self):
        """Shadow eval does not match when context differs."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_dir = _make_run_dir(tmp)

            # Modify candidate to have different stage
            candidate = _make_candidate()
            candidate["pattern"]["stage"] = "runner"
            candidate_path = tmp / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            evaluator = ShadowSkillEvaluator()
            result = evaluator.evaluate_run(run_dir, candidate_path)

            self.assertFalse(result["matched"])

    def test_shadow_does_not_modify_deployment(self):
        """Shadow evaluation does not modify deployment results."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_dir = _make_run_dir(tmp)
            candidate_path = tmp / "candidate.json"
            candidate_path.write_text(json.dumps(_make_candidate()), encoding="utf-8")

            # Read original result before shadow eval
            original_result = json.loads(
                (run_dir / "reports" / "agent_verify_result.json").read_text(encoding="utf-8")
            )

            evaluator = ShadowSkillEvaluator()
            evaluator.evaluate_run(run_dir, candidate_path)

            # Read result after shadow eval — should be unchanged
            after_result = json.loads(
                (run_dir / "reports" / "agent_verify_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(original_result, after_result)

    def test_record_accumulates_counts(self):
        """Recording shadow results accumulates helped/harmful counts."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            candidate_path = tmp / "candidate.json"
            candidate_path.write_text(json.dumps(_make_candidate()), encoding="utf-8")

            evaluator = ShadowSkillEvaluator()

            # Record first would_help result
            result1 = {"would_help": True, "would_harm": False, "run_id": "run_001", "matched": True, "reason": "ok", "evaluated_at": "2026-07-09T00:00:00+00:00"}
            shadow = evaluator.record(candidate_path, result1)
            self.assertEqual(shadow["helped_count"], 1)
            self.assertEqual(shadow["harmful_count"], 0)

            # Record second would_help result
            result2 = {"would_help": True, "would_harm": False, "run_id": "run_002", "matched": True, "reason": "ok", "evaluated_at": "2026-07-09T00:01:00+00:00"}
            shadow = evaluator.record(candidate_path, result2)
            self.assertEqual(shadow["helped_count"], 2)
            self.assertEqual(shadow["harmful_count"], 0)

            # Check candidate status updated to shadow_passed
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(candidate["status"], "shadow_passed")

    def test_record_harmful_blocks_promotion(self):
        """Harmful shadow result blocks promotion."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            candidate_path = tmp / "candidate.json"
            candidate_path.write_text(json.dumps(_make_candidate()), encoding="utf-8")

            evaluator = ShadowSkillEvaluator()

            # Record a harmful result
            result = {"would_help": False, "would_harm": True, "run_id": "run_001", "matched": True, "reason": "bypass trace", "evaluated_at": "2026-07-09T00:00:00+00:00"}
            shadow = evaluator.record(candidate_path, result)
            self.assertEqual(shadow["harmful_count"], 1)

            # Check candidate status is shadow_failed
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(candidate["status"], "shadow_failed")

    def test_shadow_artifact_written(self):
        """Shadow artifact file is written alongside candidate."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            candidate_path = tmp / "candidate.json"
            candidate_path.write_text(json.dumps(_make_candidate()), encoding="utf-8")

            evaluator = ShadowSkillEvaluator()
            result = {"would_help": True, "would_harm": False, "run_id": "run_001", "matched": True, "reason": "ok", "evaluated_at": "2026-07-09T00:00:00+00:00"}
            evaluator.record(candidate_path, result)

            # Check shadow artifact exists
            shadow_artifact = candidate_path.with_suffix(".shadow.json")
            self.assertTrue(shadow_artifact.exists())

    def test_candidate_not_found(self):
        """Non-existent candidate returns failed."""
        evaluator = ShadowSkillEvaluator()
        result = evaluator.evaluate_run(Path("/nonexistent"), Path("/nonexistent_candidate"))
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
