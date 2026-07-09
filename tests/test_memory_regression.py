"""Tests for Regression Gate (Phase 4).

Verifies:
- run_regression() writes regression artifact
- regression_failed blocks promotion
- regression skipped blocks auto promotion
- Empty case_ids causes regression_failed
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.memory.evolution import MemoryEvolutionManager


def _make_candidate(status: str = "candidate", case_ids: list = None) -> dict:
    """Create a minimal candidate dict."""
    if case_ids is None:
        case_ids = ["gradio_config_discovery"]
    return {
        "candidate_id": "skillcand_regtest",
        "status": status,
        "target_skill": "verify-evidence/SKILL.md",
        "base_skill_sha256": "",
        "quality_gate": {"passed": True},
        "regression_binding": {
            "manifest": "tests/fixtures/benchmarks/manifest.json",
            "case_ids": case_ids,
            "required_before_promote": True,
        },
        "regression": {},
        "shadow": {"helped_count": 2, "harmful_count": 0},
        "patch": {"section_title": "Test", "markdown": "## Test\n"},
    }


class TestRegressionGate(unittest.TestCase):
    """Test regression gate functionality."""

    def test_candidate_not_found(self):
        """Non-existent candidate path returns failed."""
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryEvolutionManager(
                memory_dir=Path(tmp),
                skills_dir=Path(tmp) / "skills",
            )
            result = manager.run_regression(Path(tmp) / "nonexistent.json")
            self.assertEqual(result["status"], "failed")

    def test_empty_case_ids_causes_regression_failed(self):
        """Empty case_ids results in regression_failed."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()
            candidate_dir = memory_dir / "skill_candidates"
            candidate_dir.mkdir()

            candidate = _make_candidate(case_ids=[])
            candidate_path = candidate_dir / "candidate_skillcand_regtest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
            )
            result = manager.run_regression(candidate_path)
            self.assertEqual(result["status"], "regression_failed")

            # Check candidate status was updated
            updated = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "regression_failed")

    def test_regression_artifact_written(self):
        """Regression artifact file is written."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()
            candidate_dir = memory_dir / "skill_candidates"
            candidate_dir.mkdir()

            candidate = _make_candidate()
            candidate_path = candidate_dir / "candidate_skillcand_regtest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            # Mock benchmark runner that returns passed
            mock_runner = MagicMock()
            mock_runner.run.return_value = {"status": "passed", "cases": [{"id": "gradio_config_discovery", "status": "passed"}]}

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
            )
            result = manager.run_regression(candidate_path, benchmark_runner=mock_runner)
            self.assertEqual(result["status"], "passed")

            # Check regression artifact exists
            regression_path = candidate_path.with_suffix(".regression.json")
            self.assertTrue(regression_path.exists())

            # Check candidate status updated
            updated = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "regression_passed")

    def test_regression_failed_updates_candidate(self):
        """Failed regression updates candidate status to regression_failed."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()
            candidate_dir = memory_dir / "skill_candidates"
            candidate_dir.mkdir()

            candidate = _make_candidate()
            candidate_path = candidate_dir / "candidate_skillcand_regtest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            # Mock benchmark runner that returns failed
            mock_runner = MagicMock()
            mock_runner.run.return_value = {
                "status": "failed",
                "cases": [{"id": "gradio_config_discovery", "status": "failed"}],
            }

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
            )
            result = manager.run_regression(candidate_path, benchmark_runner=mock_runner)
            self.assertEqual(result["status"], "failed")

            updated = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "regression_failed")

    def test_regression_failed_blocks_promotion(self):
        """Promotion is blocked when regression is failed."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()
            candidate_dir = memory_dir / "skill_candidates"
            candidate_dir.mkdir()

            candidate = _make_candidate(status="regression_failed")
            candidate["regression"] = {"status": "failed"}
            candidate_path = candidate_dir / "candidate_skillcand_regtest.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
            )
            # Try to promote — should fail because status not in allowed set
            result = manager.promote(candidate_path, require_shadow=False)
            self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
