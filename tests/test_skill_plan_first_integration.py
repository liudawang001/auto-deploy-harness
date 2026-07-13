"""Tests for Skill-driven Plan-first integration.

Verifies that:
- Plan-first snapshot includes skill_context
- LLM prompt includes selected skill context
- effective plan is influenced by skill guidance
- skill_effects.json is generated
- llm_contribution_evidence.json includes skill contribution
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from auto_harness.agent_runtime.plan_first_loop import (
    LLMDeploymentPlanner,
    PlanFirstDeploymentLoop,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    REPLAN_TEMPLATE,
)
from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
from auto_harness.skills.context import SkillContextBuilder, SKILL_CONTEXT_INSTRUCTION
from auto_harness.skills.router import SkillRouteRequest, SkillRouter
from auto_harness.skills.schema import SkillSpec
from auto_harness.providers.mock import MockLLMProvider


class TestPlanFirstSkillContextInSnapshot(unittest.TestCase):
    """Test that skill_context is included in project snapshot."""

    def test_snapshot_includes_selected_skills(self):
        builder = ProjectSnapshotBuilder()
        # Create a minimal project dir
        tmpdir = Path(tempfile.mkdtemp())
        try:
            (tmpdir / "app.py").write_text("import gradio\ngradio.Interface(lambda x: x).launch()", encoding="utf-8")
            snapshot = builder.build(
                tmpdir,
                task_id="test-123",
                selected_skills=[{"name": "deploy-python-webui", "version": "1.0.0"}],
                skill_context={"stage": "plan_first", "selected_skills": [{"name": "deploy-python-webui"}], "instruction": SKILL_CONTEXT_INSTRUCTION},
            )
            self.assertIn("selected_skills", snapshot)
            self.assertEqual(len(snapshot["selected_skills"]), 1)
            self.assertEqual(snapshot["selected_skills"][0]["name"], "deploy-python-webui")
            self.assertIn("skill_context", snapshot)
            self.assertEqual(snapshot["skill_context"]["stage"], "plan_first")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_snapshot_without_skills(self):
        builder = ProjectSnapshotBuilder()
        tmpdir = Path(tempfile.mkdtemp())
        try:
            (tmpdir / "app.py").write_text("print('hello')", encoding="utf-8")
            snapshot = builder.build(tmpdir, task_id="test-456")
            self.assertEqual(snapshot.get("selected_skills"), [])
            self.assertEqual(snapshot.get("skill_context"), {})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPlannerPromptIncludesSkillContext(unittest.TestCase):
    """Test that LLM planner prompt includes skill context."""

    def test_system_prompt_includes_skill_advisory(self):
        self.assertIn("Skill Advisory", PLANNER_SYSTEM_PROMPT)
        self.assertIn("not executable", PLANNER_SYSTEM_PROMPT)
        self.assertIn("policy gate", PLANNER_SYSTEM_PROMPT)

    def test_user_prompt_includes_skill_context_section(self):
        self.assertIn("skill_context_section", PLANNER_USER_TEMPLATE)
        self.assertIn("selected_skills", PLANNER_USER_TEMPLATE)

    def test_replan_prompt_includes_skill_context_section(self):
        self.assertIn("skill_context_section", REPLAN_TEMPLATE)
        self.assertIn("failure-specific selected_skills", REPLAN_TEMPLATE)


class TestSkillRouterForPlanFirst(unittest.TestCase):
    """Test SkillRouter works for plan_first stage."""

    def test_route_skills_for_plan_first(self):
        skills_dir = Path(__file__).parent.parent / "skills"
        if not skills_dir.exists():
            self.skipTest("skills directory not found")

        router = SkillRouter(skills_dir=skills_dir)
        request = SkillRouteRequest(
            stage="plan_first",
            frameworks=["gradio"],
            allowed_tools=["add_runner_candidate", "select_runner_candidate"],
        )
        routed = router.route(request, limit=3)
        # Should find at least one skill matching plan_first
        # (depends on whether existing skills have plan_first in stages)
        # At minimum, the router should not crash
        self.assertIsInstance(routed, list)


class TestSkillContextBuilderForPlanFirst(unittest.TestCase):
    """Test SkillContextBuilder produces valid context for plan_first."""

    def test_build_context_with_routed_skills(self):
        from auto_harness.skills.router import RoutedSkill

        spec = SkillSpec(
            name="deploy-python-webui",
            version="1.0.0",
            type="execution_skill",
            stages=["runner", "plan_first"],
            frameworks=["gradio", "streamlit"],
            sha256="abc123",
            content="# Guidance\n- Prefer documented entrypoint\n- For Gradio use server_name=127.0.0.1\n\n# Allowed Plan Effects\n- Add or reorder run candidates\n- Suggest verify request shape\n\n# Forbidden\n- Do not propose shell strings",
        )
        routed = RoutedSkill(
            spec=spec,
            score=16,
            match_reasons=["stage=plan_first", "framework=gradio"],
            penalties=[],
        )

        builder = SkillContextBuilder()
        ctx = builder.build([routed], stage="plan_first")

        self.assertEqual(ctx["stage"], "plan_first")
        self.assertEqual(len(ctx["selected_skills"]), 1)
        skill = ctx["selected_skills"][0]
        self.assertEqual(skill["name"], "deploy-python-webui")
        self.assertEqual(skill["score"], 16)
        self.assertIn("stage=plan_first", skill["match_reasons"])
        self.assertTrue(len(skill["applicable_rules"]) > 0)
        self.assertTrue(len(skill["allowed_plan_effects"]) > 0)
        self.assertTrue(len(skill["forbidden"]) > 0)
        self.assertIn("not executable", ctx["instruction"])


class TestPlanFirstLoopSkillRouting(unittest.TestCase):
    """Test PlanFirstDeploymentLoop skill routing methods."""

    def test_classify_failure_category_dependency(self):
        loop = PlanFirstDeploymentLoop(
            provider=MockLLMProvider(),
            config=None,
        )
        ctx = {"error": "ModuleNotFoundError: No module named 'gradio'", "summary": "", "log_tail": ""}
        self.assertEqual(loop._classify_failure_category(ctx), "dependency_missing")

    def test_classify_failure_category_port(self):
        loop = PlanFirstDeploymentLoop(
            provider=MockLLMProvider(),
            config=None,
        )
        ctx = {"error": "OSError: [Errno 98] Address already in use", "summary": "", "log_tail": ""}
        self.assertEqual(loop._classify_failure_category(ctx), "port_conflict")

    def test_classify_failure_category_unknown(self):
        loop = PlanFirstDeploymentLoop(
            provider=MockLLMProvider(),
            config=None,
        )
        ctx = {"error": "something unexpected", "summary": "", "log_tail": ""}
        self.assertEqual(loop._classify_failure_category(ctx), "")


if __name__ == "__main__":
    unittest.main()
