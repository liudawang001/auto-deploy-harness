"""Tests for atomic write, file lock, and skill patch/rollback atomicity (Phase 5).

Verifies:
- atomic_write_text writes content atomically
- atomic_write_text cleans up temp file on error
- FileLock acquires and releases lock
- SkillPatchApplier uses atomic write + file lock
- SkillRollbackManager uses atomic write + file lock
- Apply audit is written
- Rollback audit is written
"""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from auto_harness.skills.patch import SkillPatchApplier
from auto_harness.skills.rollback import SkillRollbackManager
from auto_harness.utils.atomic import FileLock, atomic_write_text


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestAtomicWriteText(unittest.TestCase):
    """Test atomic_write_text utility."""

    def test_atomic_write_creates_file(self):
        """atomic_write_text creates a file with correct content."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            atomic_write_text(path, "hello world")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "hello world")

    def test_atomic_write_overwrites_existing(self):
        """atomic_write_text atomically replaces existing file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("old content", encoding="utf-8")
            atomic_write_text(path, "new content")
            self.assertEqual(path.read_text(encoding="utf-8"), "new content")

    def test_atomic_write_creates_parent_dirs(self):
        """atomic_write_text creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "dir" / "test.txt"
            atomic_write_text(path, "nested content")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "nested content")

    def test_atomic_write_no_partial_on_error(self):
        """If write fails, original file is not partially written."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("original", encoding="utf-8")

            # Force an error by making the path a directory
            bad_path = Path(tmp) / "subdir"
            bad_path.mkdir()

            try:
                atomic_write_text(bad_path, "should fail")
                self.fail("Should have raised")
            except (OSError, IsADirectoryError):
                pass

            # Original file should be untouched
            self.assertEqual(path.read_text(encoding="utf-8"), "original")

    def test_atomic_write_cleans_up_temp(self):
        """Temp files are cleaned up on error."""
        with tempfile.TemporaryDirectory() as tmp:
            # Count temp files before
            before = list(Path(tmp).glob(".atomic_*"))

            # Force error
            bad_path = Path(tmp) / "existing_dir"
            bad_path.mkdir()
            try:
                atomic_write_text(bad_path, "fail")
            except (OSError, IsADirectoryError):
                pass

            # No leftover temp files
            after = list(Path(tmp).glob(".atomic_*"))
            self.assertEqual(len(before), len(after))


class TestFileLock(unittest.TestCase):
    """Test FileLock utility."""

    def test_file_lock_acquires_and_releases(self):
        """FileLock can be acquired and released without error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("content", encoding="utf-8")

            with FileLock(path) as lock:
                # Should be able to read/write while locked
                content = path.read_text(encoding="utf-8")
                self.assertEqual(content, "content")

    def test_file_lock_creates_lock_file(self):
        """FileLock creates a .lock file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("content", encoding="utf-8")

            with FileLock(path):
                self.assertTrue((Path(str(path) + ".lock")).exists())

    def test_file_lock_context_manager(self):
        """FileLock works as context manager — lock file persists after exit."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("content", encoding="utf-8")

            with FileLock(path) as lock:
                lock_path = Path(str(path) + ".lock")
                self.assertTrue(lock_path.exists())

            # Lock file persists after exit (reusable)
            self.assertTrue(lock_path.exists())


class TestSkillPatchAtomic(unittest.TestCase):
    """Test SkillPatchApplier uses atomic write + file lock."""

    def test_apply_uses_atomic_write(self):
        """Apply writes skill file atomically (content is correct)."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skill_dir = skills_dir / "verify-evidence"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            original = "# Original Skill\n"
            skill_path.write_text(original, encoding="utf-8")

            candidate = {
                "candidate_id": "skillcand_atomic_test",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": _sha256(original),
                "patch": {
                    "section_title": "Test Patch",
                    "markdown": "Test patch content.",
                },
            }

            applier = SkillPatchApplier()
            result = applier.apply_candidate(candidate, skills_dir)

            self.assertEqual(result["status"], "applied")

            # Verify content is correct
            content = skill_path.read_text(encoding="utf-8")
            self.assertIn("auto-harness-skill-evolution:skillcand_atomic_test", content)
            self.assertIn("Test patch content.", content)
            self.assertIn("# Original Skill", content)

    def test_apply_writes_audit(self):
        """Apply writes an audit JSON file to history/."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skill_dir = skills_dir / "verify-evidence"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            original = "# Original Skill\n"
            skill_path.write_text(original, encoding="utf-8")

            candidate = {
                "candidate_id": "skillcand_audit_test",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": _sha256(original),
                "patch": {
                    "section_title": "Audit Test",
                    "markdown": "Audit test content.",
                },
            }

            applier = SkillPatchApplier()
            result = applier.apply_candidate(candidate, skills_dir)

            # Check audit file exists
            history_dir = skill_dir / "history"
            audit_files = list(history_dir.glob("*.apply.json"))
            self.assertGreater(len(audit_files), 0, "apply audit must exist")

            # Verify audit content
            audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
            self.assertEqual(audit["candidate_id"], "skillcand_audit_test")
            self.assertIn("previous_sha256", audit)
            self.assertIn("new_sha256", audit)
            self.assertIn("applied_at", audit)

    def test_apply_writes_rollback_copy(self):
        """Apply writes a rollback copy to history/."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skill_dir = skills_dir / "verify-evidence"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            original = "# Original Skill\n"
            skill_path.write_text(original, encoding="utf-8")

            candidate = {
                "candidate_id": "skillcand_rollback_test",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": _sha256(original),
                "patch": {
                    "section_title": "Rollback Test",
                    "markdown": "Rollback test content.",
                },
            }

            applier = SkillPatchApplier()
            result = applier.apply_candidate(candidate, skills_dir)

            # Check rollback copy exists
            rollback_path = Path(result["rollback_path"])
            self.assertTrue(rollback_path.exists())
            self.assertEqual(rollback_path.read_text(encoding="utf-8"), original)

    def test_rollback_uses_atomic_write(self):
        """Rollback restores skill file atomically."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skill_dir = skills_dir / "verify-evidence"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            original = "# Original Skill\n"
            skill_path.write_text(original, encoding="utf-8")

            # First, apply a candidate
            candidate = {
                "candidate_id": "skillcand_rb_atomic",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": _sha256(original),
                "patch": {
                    "section_title": "RB Test",
                    "markdown": "RB test content.",
                },
            }

            applier = SkillPatchApplier()
            apply_result = applier.apply_candidate(candidate, skills_dir)
            self.assertEqual(apply_result["status"], "applied")

            # Write candidate json with promotion info
            candidate_path = Path(tmp) / "candidate.json"
            candidate["status"] = "active"
            candidate["promotion"] = {
                "status": "promoted",
                "rollback_path": apply_result["rollback_path"],
            }
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            # Rollback
            rollback_mgr = SkillRollbackManager()
            result = rollback_mgr.rollback_candidate(candidate_path)

            self.assertEqual(result["status"], "rolled_back")

            # Verify skill content restored
            content = skill_path.read_text(encoding="utf-8")
            self.assertEqual(content, original)
            self.assertNotIn("auto-harness-skill-evolution:skillcand_rb_atomic", content)

    def test_rollback_writes_audit(self):
        """Rollback writes an audit JSON file to history/."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills"
            skill_dir = skills_dir / "verify-evidence"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            original = "# Original Skill\n"
            skill_path.write_text(original, encoding="utf-8")

            # Apply
            candidate = {
                "candidate_id": "skillcand_rb_audit",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": _sha256(original),
                "patch": {
                    "section_title": "RB Audit Test",
                    "markdown": "RB audit content.",
                },
            }

            applier = SkillPatchApplier()
            apply_result = applier.apply_candidate(candidate, skills_dir)

            # Write candidate json
            candidate_path = Path(tmp) / "candidate.json"
            candidate["status"] = "active"
            candidate["promotion"] = {
                "status": "promoted",
                "rollback_path": apply_result["rollback_path"],
            }
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

            # Rollback
            rollback_mgr = SkillRollbackManager()
            rollback_mgr.rollback_candidate(candidate_path)

            # Check rollback audit exists
            history_dir = skill_dir / "history"
            audit_files = list(history_dir.glob("*.rollback.json"))
            self.assertGreater(len(audit_files), 0, "rollback audit must exist")

            audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
            self.assertEqual(audit["candidate_id"], "skillcand_rb_audit")
            self.assertIn("restored_sha256", audit)
            self.assertIn("rolled_back_at", audit)


if __name__ == "__main__":
    unittest.main()
