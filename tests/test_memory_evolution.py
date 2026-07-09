"""Tests for MemoryEvolutionManager and SkillPatchValidator/Applier (Phase 3).

Verifies:
- propose() generates candidates without modifying skills/*
- candidate contains source_memory_ids, base_skill_sha256, regression_binding
- SkillPatchValidator rejects secrets, paths, HTTP 200 false success
- SkillPatchApplier uses markers and checks base sha
- reject() marks candidate as rejected
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.memory.evolution import MemoryEvolutionManager
from auto_harness.skills.patch import SkillPatchValidator, SkillPatchApplier


# Helper: mock LLM provider that returns valid curator response
class _MockCuratorProvider:
    """Mock LLM provider for MemoryCurator that returns a valid candidate draft."""

    def complete(self, messages):
        response = {
            "status": "ok",
            "pattern": {
                "stage": "verify",
                "frameworks": ["gradio"],
                "failure_signature": "HTTP 200 but current trace_id absent",
                "root_cause_generalized": "Non-default Gradio API shape",
            },
            "reusable_rule": {
                "when": "verify uncertain and framework_hint=gradio",
                "do": ["discover /config", "send current trace_id"],
                "do_not": ["do not mark success on HTTP 200 alone"],
            },
            "skill_patch": {
                "target_skill": "verify-evidence/SKILL.md",
                "section_title": "Gradio API shape discovery",
                "markdown": "## Gradio API shape discovery\n\nWhen verify uncertain, discover /config endpoint.",
            },
            "regression_proposal": {
                "case_ids": ["gradio_api_shape_variation"],
                "new_case_suggestions": [],
            },
            "risk": {"level": "low", "overfit_risk": "medium", "failure_modes": []},
        }
        return MagicMock(text=json.dumps(response, ensure_ascii=False))


def _write_verified_memory(memory_dir: Path, count: int = 5) -> None:
    """Write verified success memory entries to deployment_issues.jsonl."""
    issue_path = memory_dir / "deployment_issues.jsonl"
    entries = []
    for i in range(count):
        entries.append(json.dumps({
            "id": "mem_success_%03d" % i,
            "memory_type": "verified_success",
            "created_at": "2026-07-09T00:00:00+00:00",
            "task_id": "task_%03d" % i,
            "stage": "verify",
            "category": "verification_gap",
            "frameworks": ["gradio"],
            "signature": "sig_%03d" % i,
            "symptom": "HTTP 200 but no trace_id observed",
            "root_cause": "non-default Gradio API shape",
            "repair_action_hash": "hash_%03d" % i,
            "repair_actions": ["discover /config"],
            "repair_action_status": "executed",
            "verification_trace_id": "trace_%03d" % i,
            "verify_status": "passed",
            "regression_case_ids": ["gradio_config_discovery"],
            "regression_status": "passed",
            "verified_success": True,
            "policy_rejected_high_risk": False,
            "suggested_next_action": "Promote after regression.",
        }, ensure_ascii=False))
    issue_path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _write_skill(skills_dir: Path, name: str = "verify-evidence") -> str:
    """Write a minimal SKILL.md and return its sha256."""
    skill_path = skills_dir / name / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    content = "---\nname: %s\ndescription: test skill\n---\n\n# %s\n\nOriginal content.\n" % (name, name)
    skill_path.write_text(content, encoding="utf-8")
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TestSkillPatchValidator(unittest.TestCase):
    """Test SkillPatchValidator."""

    def setUp(self):
        self.validator = SkillPatchValidator()

    def test_valid_markdown_passes(self):
        """Valid markdown passes validation."""
        result = self.validator.validate("## Gradio API discovery\n\nDiscover /config endpoint.")
        self.assertTrue(result["valid"])

    def test_empty_markdown_rejected(self):
        """Empty markdown is rejected."""
        result = self.validator.validate("")
        self.assertFalse(result["valid"])

    def test_secret_rejected(self):
        """Markdown with api_key is rejected."""
        result = self.validator.validate("Set api_key=sk-xxx")
        self.assertFalse(result["valid"])

    def test_absolute_path_rejected(self):
        """Markdown with /tmp/ path is rejected."""
        result = self.validator.validate("Copy from /tmp/model_cache")
        self.assertFalse(result["valid"])

    def test_http_200_success_rejected(self):
        """HTTP 200 means success rule is rejected."""
        result = self.validator.validate("HTTP 200 means success")
        self.assertFalse(result["valid"])

    def test_privilege_escalation_rejected(self):
        """Allow arbitrary shell is rejected."""
        result = self.validator.validate("allow arbitrary shell commands")
        self.assertFalse(result["valid"])

    def test_delete_original_rejected(self):
        """Delete existing section is rejected."""
        result = self.validator.validate("Delete the existing section and replace")
        self.assertFalse(result["valid"])

    def test_overlength_rejected(self):
        """Markdown exceeding max length is rejected."""
        result = self.validator.validate("x" * 10001)
        self.assertFalse(result["valid"])


class TestSkillPatchApplier(unittest.TestCase):
    """Test SkillPatchApplier."""

    def setUp(self):
        self.applier = SkillPatchApplier()

    def test_apply_writes_marker_block(self):
        """Applying a patch adds marker block to skill."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp)
            # Create a target skill
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            original = "# Original\n\nSome content.\n"
            skill_path.write_text(original, encoding="utf-8")
            import hashlib
            base_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()

            candidate = {
                "candidate_id": "skillcand_test001",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": base_sha,
                "patch": {
                    "section_title": "Gradio Discovery",
                    "markdown": "## Gradio Discovery\n\nDiscover /config endpoint.",
                },
            }
            result = self.applier.apply_candidate(candidate, skills_dir)
            self.assertEqual(result["status"], "applied")

            # Check marker is present
            new_content = skill_path.read_text(encoding="utf-8")
            self.assertIn("auto-harness-skill-evolution:skillcand_test001", new_content)
            self.assertIn("Gradio Discovery", new_content)
            # Original content preserved
            self.assertIn("# Original", new_content)

    def test_base_sha_mismatch_blocks_apply(self):
        """Base sha mismatch blocks apply."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp)
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("# Original\n", encoding="utf-8")

            candidate = {
                "candidate_id": "skillcand_test002",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": "0000000000000000",
                "patch": {"section_title": "Test", "markdown": "## Test\n"},
            }
            result = self.applier.apply_candidate(candidate, skills_dir)
            self.assertEqual(result["status"], "base_changed")

    def test_target_not_exist_fails(self):
        """Non-existent target skill fails."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self.applier.apply_candidate(
                {"candidate_id": "test", "target_skill": "nonexistent/SKILL.md", "base_skill_sha256": "", "patch": {}},
                Path(tmp),
            )
            self.assertEqual(result["status"], "failed")

    def test_rollback_copy_created(self):
        """Rollback copy is created before applying."""
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp)
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            original = "# Original\n"
            skill_path.write_text(original, encoding="utf-8")
            import hashlib
            base_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()

            candidate = {
                "candidate_id": "skillcand_test003",
                "target_skill": "verify-evidence/SKILL.md",
                "base_skill_sha256": base_sha,
                "patch": {"section_title": "Test", "markdown": "## Test\n"},
            }
            result = self.applier.apply_candidate(candidate, skills_dir)
            self.assertEqual(result["status"], "applied")
            # Rollback path exists
            rollback_path = Path(result["rollback_path"])
            self.assertTrue(rollback_path.exists())
            self.assertEqual(rollback_path.read_text(encoding="utf-8"), original)


class TestMemoryEvolutionPropose(unittest.TestCase):
    """Test propose() flow."""

    def test_propose_generates_candidates(self):
        """propose() generates candidate json + md without modifying skills/*."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()

            # Write verified memory entries
            _write_verified_memory(memory_dir, count=5)
            # Write target skill
            _write_skill(skills_dir)

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
                provider=_MockCuratorProvider(),
            )

            result = manager.propose(min_verified_count=3)
            self.assertEqual(result["status"], "proposed")
            self.assertGreater(len(result["candidates"]), 0)

            # Check candidate was written
            candidate = result["candidates"][0]
            candidate_id = candidate["candidate_id"]
            candidate_json = memory_dir / "skill_candidates" / ("candidate_%s.json" % candidate_id)
            candidate_md = memory_dir / "skill_candidates" / ("candidate_%s.md" % candidate_id)
            self.assertTrue(candidate_json.exists())
            self.assertTrue(candidate_md.exists())

            # Check candidate has required fields
            loaded = json.loads(candidate_json.read_text(encoding="utf-8"))
            self.assertIn("source_memory_ids", loaded)
            self.assertTrue(len(loaded["source_memory_ids"]) > 0)
            self.assertIn("base_skill_sha256", loaded)
            self.assertTrue(loaded["base_skill_sha256"])  # non-empty
            self.assertIn("regression_binding", loaded)
            self.assertIn("quality_gate", loaded)
            self.assertTrue(loaded["quality_gate"]["passed"])

            # Check skills were NOT modified
            skill_content = (skills_dir / "verify-evidence" / "SKILL.md").read_text(encoding="utf-8")
            self.assertNotIn("auto-harness-skill-evolution", skill_content)

    def test_propose_no_verified_memory(self):
        """propose() returns no_candidates when there's no verified memory."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()

            # No entries
            (memory_dir / "deployment_issues.jsonl").write_text("", encoding="utf-8")

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
                provider=_MockCuratorProvider(),
            )

            result = manager.propose(min_verified_count=1)
            self.assertEqual(result["status"], "no_candidates")

    def test_reject_marks_candidate(self):
        """reject() marks candidate as rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            skills_dir = Path(tmp) / "skills"
            memory_dir.mkdir()
            skills_dir.mkdir()

            _write_verified_memory(memory_dir, count=5)
            _write_skill(skills_dir)

            manager = MemoryEvolutionManager(
                memory_dir=memory_dir,
                skills_dir=skills_dir,
                provider=_MockCuratorProvider(),
            )

            result = manager.propose(min_verified_count=3)
            candidate = result["candidates"][0]
            candidate_id = candidate["candidate_id"]
            candidate_path = memory_dir / "skill_candidates" / ("candidate_%s.json" % candidate_id)

            reject_result = manager.reject(candidate_path, reason="bad pattern")
            self.assertEqual(reject_result["status"], "rejected")

            # Verify candidate was updated on disk
            loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
