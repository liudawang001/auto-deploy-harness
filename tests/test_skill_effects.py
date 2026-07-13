"""Tests for SkillEffectRecorder: record skill influences on plan fields."""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.skills.effects import SkillEffect, SkillEffectRecorder


class TestSkillEffect(unittest.TestCase):
    """Test SkillEffect dataclass."""

    def test_basic_creation(self):
        effect = SkillEffect(
            skill_name="verify-evidence",
            skill_sha256="abc123",
            stage="plan_first",
            effect_type="verify_hint_generation",
            field_changed="verify.request",
            accepted_by_policy=True,
        )
        self.assertEqual(effect.skill_name, "verify-evidence")
        self.assertTrue(effect.accepted_by_policy)


class TestSkillEffectRecorder(unittest.TestCase):
    """Test SkillEffectRecorder."""

    def setUp(self):
        self.recorder = SkillEffectRecorder()

    def test_verification_skill_influences_verify_request(self):
        routed_skills = [{
            "name": "verify-evidence",
            "sha256": "abc123",
            "type": "verification_skill",
            "stage": "plan_first",
            "allowed_tools": ["probe_http"],
        }]
        compiled_plan = {
            "analysis": {
                "verify_hint": {"method": "POST", "path": "/api/predict"},
            },
            "effective_plan": {},
        }
        original_analysis = {"verify_hint": {}}

        result = self.recorder.record_effects(
            task_id="test-123",
            routed_skills=routed_skills,
            compiled_plan=compiled_plan,
            policy_result={"allowed": True},
            original_analysis=original_analysis,
        )
        self.assertEqual(result["task_id"], "test-123")
        self.assertTrue(len(result["effects"]) > 0)
        effect = result["effects"][0]
        self.assertEqual(effect["skill_name"], "verify-evidence")
        self.assertEqual(effect["effect_type"], "verify_hint_generation")
        self.assertEqual(effect["field_changed"], "verify.request")
        self.assertTrue(effect["accepted_by_policy"])

    def test_execution_skill_influences_run_candidates(self):
        routed_skills = [{
            "name": "deploy-python-webui",
            "sha256": "def456",
            "type": "execution_skill",
            "stage": "plan_first",
            "allowed_tools": ["add_runner_candidate"],
        }]
        compiled_plan = {
            "analysis": {
                "run_candidates": [
                    {"cmd": ["python", "app.py"], "selected_by": "llm_plan_first"},
                ],
            },
            "effective_plan": {},
        }
        original_analysis = {"run_candidates": []}

        result = self.recorder.record_effects(
            task_id="test-456",
            routed_skills=routed_skills,
            compiled_plan=compiled_plan,
            policy_result={"allowed": True},
            original_analysis=original_analysis,
        )
        self.assertTrue(len(result["effects"]) > 0)
        effect = result["effects"][0]
        self.assertEqual(effect["skill_name"], "deploy-python-webui")
        self.assertEqual(effect["effect_type"], "runner_candidate_selection")
        self.assertEqual(effect["field_changed"], "run.candidates")

    def test_policy_rejected_records_accepted_by_policy_false(self):
        routed_skills = [{
            "name": "verify-evidence",
            "sha256": "abc123",
            "type": "verification_skill",
            "stage": "plan_first",
            "allowed_tools": ["probe_http"],
        }]
        compiled_plan = {
            "analysis": {
                "verify_hint": {"method": "POST", "path": "/api/predict"},
            },
            "effective_plan": {},
        }
        original_analysis = {"verify_hint": {}}

        result = self.recorder.record_effects(
            task_id="test-789",
            routed_skills=routed_skills,
            compiled_plan=compiled_plan,
            policy_result={"allowed": False, "rejected_items": [{"section": "verify", "reason": "unsafe"}]},
            original_analysis=original_analysis,
        )
        self.assertTrue(len(result["effects"]) > 0)
        effect = result["effects"][0]
        self.assertFalse(effect["accepted_by_policy"])

    def test_write_effects_creates_file(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            effects_data = {
                "task_id": "test-123",
                "effects": [
                    {
                        "skill_name": "verify-evidence",
                        "skill_sha256": "abc123",
                        "stage": "plan_first",
                        "effect_type": "verify_hint_generation",
                        "field_changed": "verify.request",
                        "accepted_by_policy": True,
                        "evidence": {},
                    },
                ],
                "created_at": "2026-07-13T00:00:00Z",
            }
            path = self.recorder.write_effects(tmpdir, effects_data)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "skill_effects.json")
            # Verify content
            from auto_harness.models.base import read_json
            data = read_json(path)
            self.assertEqual(data["task_id"], "test-123")
            self.assertEqual(len(data["effects"]), 1)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_no_effects_when_no_skill_influence(self):
        routed_skills = [{
            "name": "some-skill",
            "sha256": "xyz",
            "type": "analysis_skill",
            "stage": "plan_first",
            "allowed_tools": [],
        }]
        compiled_plan = {"analysis": {}, "effective_plan": {}}

        result = self.recorder.record_effects(
            task_id="test-empty",
            routed_skills=routed_skills,
            compiled_plan=compiled_plan,
            policy_result={"allowed": True},
        )
        self.assertEqual(len(result["effects"]), 0)


if __name__ == "__main__":
    unittest.main()
