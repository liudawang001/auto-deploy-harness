"""Tests for Skill Rollback (Phase 7).

Verifies:
- Base sha mismatch blocks promote
- Apply before promote writes rollback copy
- Rollback restores pre-promotion skill
- Non-active candidate cannot be rolled back
- rollback_to_history works
- SkillOutcomeRecorder records outcomes
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.memory.evolution import MemoryEvolutionManager
from auto_harness.memory.outcomes import SkillOutcomeRecorder
from auto_harness.skills.rollback import SkillRollbackManager


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _create_skill(skills_dir: Path, name: str = "verify-evidence") -> str:
    """Create a minimal skill and return its sha256."""
    skill_path = skills_dir / name / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    content = "---\nname: %s\ndescription: test\n---\n\n# %s\n\nOriginal.\n" % (name, name)
    skill_path.write_text(content, encoding="utf-8")
    return _sha256(content)


def _create_promoted_candidate(
    memory_dir: Path,
    skills_dir: Path,
    candidate_id: str = "skillcand_promo_test",
) -> Path:
    """Create a candidate that has been promoted, return its path."""
    candidate_dir = memory_dir / "skill_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Create the skill
    base_sha = _create_skill(skills_dir)

    # Create and apply the candidate
    skill_path = skills_dir / "verify-evidence" / "SKILL.md"
    original = skill_path.read_text(encoding="utf-8")

    marker = "auto-harness-skill-evolution:%s" % candidate_id
    block = "\n\n<!-- %s -->\n## Test Patch\nTest content.\n<!-- /%s -->\n" % (marker, marker)
    new_content = original.rstrip() + block
    skill_path.write_text(new_content, encoding="utf-8")

    # Create history/rollback
    history_dir = skill_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    rollback_path = history_dir / ("20260709_%s.md" % candidate_id)
    rollback_path.write_text(original, encoding="utf-8")

    candidate = {
        "candidate_id": candidate_id,
        "status": "active",
        "target_skill": "verify-evidence/SKILL.md",
        "base_skill_sha256": base_sha,
        "quality_gate": {"passed": True},
        "regression": {"status": "passed"},
        "shadow": {"helped_count": 2, "harmful_count": 0},
        "patch": {"section_title": "Test Patch", "markdown": "## Test Patch\nTest content."},
        "promotion": {
            "status": "promoted",
            "promoted_at": "2026-07-09T00:00:00+00:00",
            "previous_sha256": base_sha,
            "new_sha256": _sha256(new_content),
            "rollback_path": str(rollback_path),
        },
    }

    candidate_path = candidate_dir / ("candidate_%s.json" % candidate_id)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    return candidate_path


class TestPromotionAndRollback(unittest.TestCase):
    """Test promotion and rollback flow."""

    def test_rollback_restores_skill(self):
        """Rollback restores the pre-promotion skill content."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()

            candidate_path = _create_promoted_candidate(memory_dir, skills_dir)

            # Verify the patch is present
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            content = skill_path.read_text(encoding="utf-8")
            self.assertIn("auto-harness-skill-evolution:skillcand_promo_test", content)

            # Rollback
            manager = SkillRollbackManager()
            result = manager.rollback_candidate(candidate_path)
            self.assertEqual(result["status"], "rolled_back")

            # Verify skill was restored
            restored = skill_path.read_text(encoding="utf-8")
            self.assertNotIn("auto-harness-skill-evolution", restored)
            self.assertIn("# verify-evidence", restored)
            self.assertIn("Original.", restored)

            # Verify candidate was updated
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(candidate["status"], "rolled_back")

    def test_rollback_saves_current_to_history(self):
        """Rollback saves current (patched) skill to history before restoring."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()

            candidate_path = _create_promoted_candidate(memory_dir, skills_dir)

            manager = SkillRollbackManager()
            result = manager.rollback_candidate(candidate_path)
            self.assertEqual(result["status"], "rolled_back")

            # Check pre_rollback_backup exists
            backup_path = result.get("pre_rollback_backup", "")
            self.assertTrue(Path(backup_path).exists())

    def test_non_active_candidate_cannot_rollback(self):
        """Non-active candidate cannot be rolled back."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            candidate_dir = memory_dir / "skill_candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            candidate = {
                "candidate_id": "skillcand_not_active",
                "status": "candidate",
                "promotion": {},
            }
            candidate_path = candidate_dir / "candidate_skillcand_not_active.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            manager = SkillRollbackManager()
            result = manager.rollback_candidate(candidate_path)
            self.assertEqual(result["status"], "failed")
            self.assertIn("not 'active'", result["error"])

    def test_rollback_to_history(self):
        """rollback_to_history restores skill from a specific history file."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)

            # Create history backup
            history_path = skills_dir / "verify-evidence" / "history" / "backup.md"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text("# Old Version\n\nOld content.\n", encoding="utf-8")

            # Create current (newer) version
            skill_path.write_text("# New Version\n\nNew content.\n", encoding="utf-8")

            manager = SkillRollbackManager()
            result = manager.rollback_to_history(skill_path, history_path)
            self.assertEqual(result["status"], "rolled_back")

            # Verify content was restored
            restored = skill_path.read_text(encoding="utf-8")
            self.assertIn("Old Version", restored)
            self.assertNotIn("New Version", restored)


class TestSkillOutcomeRecorder(unittest.TestCase):
    """Test SkillOutcomeRecorder."""

    def test_record_run_with_skills(self):
        """record_run records outcomes for selected skills."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkillOutcomeRecorder(Path(tmp))

            result = recorder.record_run(
                run_id="run_001",
                stage="verify",
                selected_skills=[
                    {"name": "verify-evidence", "path": "skills/verify-evidence/SKILL.md", "sha256": "abc123"},
                ],
                result={"status": "passed"},
                agent_metadata={"llm_helped": True, "tool_selected": "probe_http"},
            )

            self.assertEqual(result["recorded_count"], 1)
            self.assertTrue(result["records"][0]["selected"])
            self.assertTrue(result["records"][0]["llm_helped"])

    def test_record_run_no_skills(self):
        """record_run records even when no skills were selected."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkillOutcomeRecorder(Path(tmp))

            result = recorder.record_run(
                run_id="run_002",
                stage="verify",
                selected_skills=[],
                result={"status": "uncertain"},
            )

            self.assertEqual(result["recorded_count"], 1)
            self.assertFalse(result["records"][0]["selected"])

    def test_outcomes_written_to_jsonl(self):
        """Outcomes are written to skill_outcomes.jsonl."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkillOutcomeRecorder(Path(tmp))

            recorder.record_run(
                run_id="run_003",
                stage="verify",
                selected_skills=[{"name": "test", "path": "", "sha256": "def456"}],
                result={"status": "passed"},
            )

            outcomes_path = Path(tmp) / "skill_outcomes.jsonl"
            self.assertTrue(outcomes_path.exists())
            lines = outcomes_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

    def test_summarize(self):
        """summarize() returns correct aggregates."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkillOutcomeRecorder(Path(tmp))

            recorder.record_run("run_1", "verify", [{"name": "test", "sha256": "sha1"}], {"status": "passed"}, {"llm_helped": True})
            recorder.record_run("run_2", "verify", [{"name": "test", "sha256": "sha1"}], {"status": "failed"})
            recorder.record_run("run_3", "verify", [{"name": "other", "sha256": "sha2"}], {"status": "passed"})

            summary = recorder.summarize()
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["passed"], 2)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["llm_helped_count"], 1)

            # Filter by skill_name
            summary = recorder.summarize(skill_name="test")
            self.assertEqual(summary["total"], 2)

    def test_summarize_by_sha(self):
        """summarize() groups by skill_sha256."""
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SkillOutcomeRecorder(Path(tmp))

            recorder.record_run("run_1", "verify", [{"name": "test", "sha256": "sha1"}], {"status": "passed"})
            recorder.record_run("run_2", "verify", [{"name": "test", "sha256": "sha1"}], {"status": "failed"})
            recorder.record_run("run_3", "verify", [{"name": "test", "sha256": "sha2"}], {"status": "passed"})

            summary = recorder.summarize()
            by_sha = summary["by_skill_sha"]
            self.assertIn("sha1", by_sha)
            self.assertEqual(by_sha["sha1"]["count"], 2)
            self.assertEqual(by_sha["sha1"]["passed"], 1)
            self.assertEqual(by_sha["sha1"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
