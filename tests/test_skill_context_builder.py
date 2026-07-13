"""Tests for SkillContextBuilder: compress skills into LLM-usable context."""
import unittest
from pathlib import Path

from auto_harness.skills.context import (
    DEFAULT_MAX_RULES,
    SKILL_CONTEXT_INSTRUCTION,
    SkillContextBuilder,
)
from auto_harness.skills.router import RoutedSkill
from auto_harness.skills.schema import SkillSpec


def _make_routed_skill(
    name="test-skill",
    version="1.0.0",
    skill_type="execution_skill",
    stages=None,
    content="",
    score=10,
    match_reasons=None,
) -> RoutedSkill:
    """Helper to create a RoutedSkill for testing."""
    spec = SkillSpec(
        name=name,
        version=version,
        type=skill_type,
        stages=stages or ["runner"],
        sha256="abc123def456",
        content=content,
    )
    return RoutedSkill(
        spec=spec,
        score=score,
        match_reasons=match_reasons or ["stage=runner"],
        penalties=[],
    )


class TestSkillContextBuilder(unittest.TestCase):
    """Test SkillContextBuilder."""

    def setUp(self):
        self.builder = SkillContextBuilder()

    def test_build_returns_stage_and_instruction(self):
        skills = [_make_routed_skill()]
        ctx = self.builder.build(skills, stage="verify")
        self.assertEqual(ctx["stage"], "verify")
        self.assertIn("selected_skills", ctx)
        self.assertEqual(ctx["instruction"], SKILL_CONTEXT_INSTRUCTION)

    def test_instruction_says_skill_not_executable(self):
        self.assertIn("not executable", SKILL_CONTEXT_INSTRUCTION)

    def test_build_includes_skill_metadata(self):
        skills = [_make_routed_skill(name="verify-evidence", version="1.0.0", skill_type="verification_skill")]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        self.assertEqual(skill["name"], "verify-evidence")
        self.assertEqual(skill["version"], "1.0.0")
        self.assertEqual(skill["type"], "verification_skill")
        self.assertEqual(skill["sha256"], "abc123def456")
        self.assertEqual(skill["score"], 10)
        self.assertEqual(skill["match_reasons"], ["stage=runner"])

    def test_extract_guidance_section(self):
        content = """# Purpose
Verify deployment.

# Guidance

- HTTP 200 is not success
- Response must contain current trace_id
- For Gradio inspect /config before probing

# Forbidden

- Do not mark success without trace evidence
"""
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        self.assertIn("HTTP 200 is not success", skill["applicable_rules"])
        self.assertIn("Response must contain current trace_id", skill["applicable_rules"])
        self.assertIn("For Gradio inspect /config before probing", skill["applicable_rules"])

    def test_extract_allowed_plan_effects(self):
        content = """# Guidance

- Use trace-based verification

# Allowed Plan Effects

- update_verify_hint
- discover_gradio_api
- probe_http
"""
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        self.assertIn("update_verify_hint", skill["allowed_plan_effects"])
        self.assertIn("discover_gradio_api", skill["allowed_plan_effects"])
        self.assertIn("probe_http", skill["allowed_plan_effects"])

    def test_extract_forbidden(self):
        content = """# Guidance

- Use trace-based verification

# Forbidden

- Do not mark success without trace evidence
- Do not bypass policy gate
"""
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        self.assertIn("Do not mark success without trace evidence", skill["forbidden"])
        self.assertIn("Do not bypass policy gate", skill["forbidden"])

    def test_max_rules_limit(self):
        rules = "\n".join("- Rule %d" % i for i in range(20))
        content = "# Guidance\n\n%s" % rules
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify", max_rules=5)
        skill = ctx["selected_skills"][0]
        self.assertLessEqual(len(skill["applicable_rules"]), 5)

    def test_fallback_to_first_lines_when_no_sections(self):
        content = "This is a simple skill.\nIt has no sections.\nJust plain text."
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        self.assertTrue(len(skill["applicable_rules"]) > 0)

    def test_multiple_skills(self):
        skills = [
            _make_routed_skill(name="skill-a", score=15),
            _make_routed_skill(name="skill-b", score=10),
        ]
        ctx = self.builder.build(skills, stage="verify")
        self.assertEqual(len(ctx["selected_skills"]), 2)
        self.assertEqual(ctx["selected_skills"][0]["name"], "skill-a")
        self.assertEqual(ctx["selected_skills"][1]["name"], "skill-b")

    def test_empty_skills_list(self):
        ctx = self.builder.build([], stage="verify")
        self.assertEqual(ctx["selected_skills"], [])
        self.assertEqual(ctx["instruction"], SKILL_CONTEXT_INSTRUCTION)

    def test_when_to_use_section(self):
        content = """# When To Use

- Use when project imports gradio or streamlit
- Use for HTTP demo services

# Guidance

- Prefer documented entrypoint
"""
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        # When To Use is extracted but not directly in the output
        # It feeds into applicable_rules if Guidance is present
        self.assertTrue(len(skill["applicable_rules"]) > 0)

    def test_numbered_list_extraction(self):
        content = """# Guidance

1. First rule
2. Second rule
3. Third rule
"""
        skills = [_make_routed_skill(content=content)]
        ctx = self.builder.build(skills, stage="verify")
        skill = ctx["selected_skills"][0]
        self.assertIn("First rule", skill["applicable_rules"])
        self.assertIn("Second rule", skill["applicable_rules"])


if __name__ == "__main__":
    unittest.main()
