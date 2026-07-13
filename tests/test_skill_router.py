"""Tests for SkillRouter: select the most relevant skills for a deployment stage."""
import os
import tempfile
import unittest
from pathlib import Path

from auto_harness.skills.router import (
    PENALTY_DEPRECATED,
    PENALTY_RECENT_HARMFUL,
    PENALTY_SIDE_EFFECT_IN_PLANNER,
    SCORE_FAILURE_CATEGORY_MATCH,
    SCORE_FRAMEWORK_MATCH,
    SCORE_STAGE_MATCH,
    SCORE_TOOL_OVERLAP,
    RoutedSkill,
    SkillRouteRequest,
    SkillRouter,
)
from auto_harness.skills.schema import SkillSpec


def _write_skill(tmpdir: Path, name: str, frontmatter_extra: str = "", body: str = "# Purpose\nTest."):
    """Helper to write a SKILL.md file in a temp directory."""
    skill_dir = tmpdir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = "---\nname: %s\nversion: 1.0.0\ntype: execution_skill\nstages: [runner]\nrisk_level: low\nside_effects: false\nallowed_tools: []\nsuccess_signals: []\nregression_cases: []\n%s\n---\n\n%s" % (name, frontmatter_extra, body)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


class TestSkillRouteRequest(unittest.TestCase):
    """Test SkillRouteRequest dataclass."""

    def test_default_values(self):
        req = SkillRouteRequest(stage="verify")
        self.assertEqual(req.stage, "verify")
        self.assertEqual(req.analysis, {})
        self.assertEqual(req.failure_category, "")
        self.assertEqual(req.frameworks, [])
        self.assertEqual(req.allowed_tools, [])
        self.assertEqual(req.mode, "planner")
        self.assertEqual(req.history, {})


class TestRoutedSkill(unittest.TestCase):
    """Test RoutedSkill dataclass."""

    def test_to_context(self):
        spec = SkillSpec(name="test", version="1.0.0", type="execution_skill", stages=["runner"], sha256="abc123")
        routed = RoutedSkill(spec=spec, score=10, match_reasons=["stage=runner"], penalties=[])
        ctx = routed.to_context()
        self.assertEqual(ctx["name"], "test")
        self.assertEqual(ctx["score"], 10)
        self.assertEqual(ctx["match_reasons"], ["stage=runner"])


class TestSkillRouterScoring(unittest.TestCase):
    """Test SkillRouter scoring logic with temp skill files."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.router = SkillRouter(skills_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_skill(self, name, frontmatter_extra="", body="# Purpose\nTest."):
        skill_dir = self.tmpdir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        default_fm = "name: %s\nversion: 1.0.0\ntype: execution_skill\nstages: [runner]\nrisk_level: low\nside_effects: false\nallowed_tools: []\nsuccess_signals: []\nregression_cases: []" % name
        content = "---\n%s\n%s\n---\n\n%s" % (default_fm, frontmatter_extra, body)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    def test_stage_match_scores_highest(self):
        self._write_skill("runner-skill", "stages: [runner]")
        self._write_skill("verify-skill", "stages: [verify]")

        req = SkillRouteRequest(stage="runner")
        results = self.router.route(req, limit=3)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].spec.name, "runner-skill")
        self.assertIn("stage=runner", results[0].match_reasons)
        self.assertEqual(results[0].score, SCORE_STAGE_MATCH)

    def test_framework_match_adds_score(self):
        self._write_skill("gradio-skill", "stages: [runner]\nframeworks: [gradio]")
        self._write_skill("generic-skill", "stages: [runner]")

        req = SkillRouteRequest(stage="runner", frameworks=["gradio"])
        results = self.router.route(req, limit=3)
        # gradio-skill should score higher
        gradio = next((r for r in results if r.spec.name == "gradio-skill"), None)
        generic = next((r for r in results if r.spec.name == "generic-skill"), None)
        self.assertIsNotNone(gradio)
        self.assertIsNotNone(generic)
        self.assertGreater(gradio.score, generic.score)
        self.assertIn("framework=gradio", gradio.match_reasons)

    def test_failure_category_match_adds_score(self):
        self._write_skill("dep-repair", "stages: [repair]\ntype: repair_skill\nfailure_categories: [dependency_missing]")
        self._write_skill("generic-repair", "stages: [repair]\ntype: repair_skill")

        req = SkillRouteRequest(stage="repair", failure_category="dependency_missing")
        results = self.router.route(req, limit=3)
        dep_repair = next((r for r in results if r.spec.name == "dep-repair"), None)
        self.assertIsNotNone(dep_repair)
        self.assertIn("failure_category=dependency_missing", dep_repair.match_reasons)

    def test_tool_overlap_adds_score(self):
        self._write_skill("tool-skill", "stages: [runner]\nallowed_tools: [add_runner_candidate, select_runner_candidate]")
        self._write_skill("no-tool-skill", "stages: [runner]")

        req = SkillRouteRequest(stage="runner", allowed_tools=["add_runner_candidate"])
        results = self.router.route(req, limit=3)
        tool_skill = next((r for r in results if r.spec.name == "tool-skill"), None)
        self.assertIsNotNone(tool_skill)
        self.assertIn("tool_overlap=1", tool_skill.match_reasons)

    def test_deprecated_penalty(self):
        self._write_skill("old-skill", "stages: [runner]\ndeprecated: true\nreplacement: new-skill")

        req = SkillRouteRequest(stage="runner")
        results = self.router.route(req, limit=3)
        old = next((r for r in results if r.spec.name == "old-skill"), None)
        if old:
            self.assertIn("deprecated", old.penalties)
            self.assertTrue(old.score < SCORE_STAGE_MATCH)

    def test_harmful_history_penalty(self):
        self._write_skill("harm-skill", "stages: [runner]")

        req = SkillRouteRequest(stage="runner", history={"harm-skill": {"recent_harmful": True}})
        results = self.router.route(req, limit=3)
        harm = next((r for r in results if r.spec.name == "harm-skill"), None)
        if harm:
            self.assertIn("recent_harmful", harm.penalties)

    def test_side_effect_in_planner_mode_penalty(self):
        self._write_skill("side-effect-skill", "stages: [runner]\nside_effects: true")

        req = SkillRouteRequest(stage="runner", mode="planner")
        results = self.router.route(req, limit=3)
        side_effect = next((r for r in results if r.spec.name == "side-effect-skill"), None)
        if side_effect:
            self.assertIn("side_effect_in_planner_mode", side_effect.penalties)

    def test_no_side_effect_penalty_in_gated_actor(self):
        self._write_skill("side-effect-skill", "stages: [runner]\nside_effects: true")

        req = SkillRouteRequest(stage="runner", mode="gated_actor")
        results = self.router.route(req, limit=3)
        side_effect = next((r for r in results if r.spec.name == "side-effect-skill"), None)
        if side_effect:
            self.assertNotIn("side_effect_in_planner_mode", side_effect.penalties)

    def test_match_reasons_correctly_written(self):
        self._write_skill("multi-skill", "stages: [runner, plan_first]\nframeworks: [gradio, streamlit]\nallowed_tools: [add_runner_candidate]")

        req = SkillRouteRequest(stage="runner", frameworks=["gradio"], allowed_tools=["add_runner_candidate"])
        results = self.router.route(req, limit=3)
        multi = next((r for r in results if r.spec.name == "multi-skill"), None)
        self.assertIsNotNone(multi)
        self.assertIn("stage=runner", multi.match_reasons)
        self.assertIn("framework=gradio", multi.match_reasons)
        self.assertIn("tool_overlap=1", multi.match_reasons)

    def test_limit_results(self):
        for i in range(5):
            self._write_skill("skill-%d" % i, "stages: [runner]")

        req = SkillRouteRequest(stage="runner")
        results = self.router.route(req, limit=2)
        self.assertLessEqual(len(results), 2)

    def test_empty_skills_dir(self):
        empty_dir = Path(tempfile.mkdtemp())
        try:
            router = SkillRouter(skills_dir=empty_dir)
            req = SkillRouteRequest(stage="runner")
            results = router.route(req)
            self.assertEqual(results, [])
        finally:
            import shutil
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_verified_success_history_bonus(self):
        self._write_skill("good-skill", "stages: [runner]")

        req = SkillRouteRequest(
            stage="runner",
            history={"good-skill": {"recent_verified_success": True}},
        )
        results = self.router.route(req, limit=3)
        good = next((r for r in results if r.spec.name == "good-skill"), None)
        if good:
            self.assertIn("recent_verified_success", good.match_reasons)

    def test_regression_failed_penalty(self):
        self._write_skill("broken-skill", "stages: [runner]")

        req = SkillRouteRequest(
            stage="runner",
            history={"broken-skill": {"regression_failed": True}},
        )
        results = self.router.route(req, limit=3)
        broken = next((r for r in results if r.spec.name == "broken-skill"), None)
        if broken:
            self.assertIn("regression_failed", broken.penalties)


if __name__ == "__main__":
    unittest.main()
