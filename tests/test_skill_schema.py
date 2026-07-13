"""Tests for SkillSchemaParser: parse and validate SKILL.md frontmatter."""
import unittest
from pathlib import Path

from auto_harness.skills.schema import (
    SEMVER_PATTERN,
    VALID_STAGES,
    VALID_TYPES,
    SkillSchemaParser,
    SkillSpec,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestSkillSpec(unittest.TestCase):
    """Test SkillSpec dataclass."""

    def test_to_context_returns_required_fields(self):
        spec = SkillSpec(name="test", version="1.0.0", type="execution_skill", stages=["runner"])
        ctx = spec.to_context()
        self.assertEqual(ctx["name"], "test")
        self.assertEqual(ctx["version"], "1.0.0")
        self.assertEqual(ctx["type"], "execution_skill")
        self.assertEqual(ctx["stages"], ["runner"])
        self.assertIn("sha256", ctx)
        self.assertIn("deprecated", ctx)

    def test_default_values(self):
        spec = SkillSpec(name="test", version="1.0.0", type="analysis_skill", stages=["analyze"])
        self.assertEqual(spec.frameworks, [])
        self.assertEqual(spec.failure_categories, [])
        self.assertEqual(spec.risk_level, "low")
        self.assertFalse(spec.side_effects)
        self.assertEqual(spec.allowed_tools, [])
        self.assertFalse(spec.deprecated)


class TestSkillSchemaParserParseText(unittest.TestCase):
    """Test SkillSchemaParser.parse_text."""

    def setUp(self):
        self.parser = SkillSchemaParser()

    def test_parse_valid_frontmatter(self):
        raw = """---
name: deploy-python-webui
version: 1.0.0
type: execution_skill
stages: [analyze, runner]
frameworks: [gradio, streamlit]
risk_level: low
side_effects: false
allowed_tools: [add_runner_candidate, select_runner_candidate]
success_signals: [runner process alive]
regression_cases: [gradio_tiny_local]
---

# Purpose

Guide deployment planning.
"""
        spec = self.parser.parse_text(raw)
        self.assertEqual(spec.name, "deploy-python-webui")
        self.assertEqual(spec.version, "1.0.0")
        self.assertEqual(spec.type, "execution_skill")
        self.assertEqual(spec.stages, ["analyze", "runner"])
        self.assertEqual(spec.frameworks, ["gradio", "streamlit"])
        self.assertEqual(spec.risk_level, "low")
        self.assertFalse(spec.side_effects)
        self.assertEqual(spec.allowed_tools, ["add_runner_candidate", "select_runner_candidate"])
        self.assertEqual(spec.success_signals, ["runner process alive"])
        self.assertEqual(spec.regression_cases, ["gradio_tiny_local"])
        self.assertIn("Purpose", spec.content)

    def test_parse_comma_separated_list(self):
        raw = """---
name: verify-evidence
version: 2.0.0
type: verification_skill
stages: verify, plan_first
frameworks: gradio, fastapi
---

# Purpose
Verify deployment.
"""
        spec = self.parser.parse_text(raw)
        self.assertEqual(spec.stages, ["verify", "plan_first"])
        self.assertEqual(spec.frameworks, ["gradio", "fastapi"])

    def test_parse_boolean_fields(self):
        raw = """---
name: risky-skill
version: 1.0.0
type: execution_skill
stages: [runner]
side_effects: true
deprecated: true
replacement: safe-skill
---

# Purpose
Risky skill.
"""
        spec = self.parser.parse_text(raw)
        self.assertTrue(spec.side_effects)
        self.assertTrue(spec.deprecated)
        self.assertEqual(spec.replacement, "safe-skill")

    def test_parse_no_frontmatter(self):
        raw = "# Just a body\n\nNo frontmatter here."
        spec = self.parser.parse_text(raw)
        self.assertEqual(spec.name, "")
        self.assertEqual(spec.version, "")
        self.assertEqual(spec.type, "")

    def test_parse_failure_categories(self):
        raw = """---
name: repair-dep
version: 1.0.0
type: repair_skill
stages: [repair]
failure_categories: [dependency_missing, version_conflict]
---

# Purpose
Repair dependencies.
"""
        spec = self.parser.parse_text(raw)
        self.assertEqual(spec.failure_categories, ["dependency_missing", "version_conflict"])

    def test_parse_optional_fields(self):
        raw = """---
name: test-skill
version: 1.0.0
type: analysis_skill
stages: [analyze]
model_sources: [huggingface, modelscope]
env_backends: [venv, conda]
owners: [team-a]
---

# Purpose
Test.
"""
        spec = self.parser.parse_text(raw)
        self.assertEqual(spec.model_sources, ["huggingface", "modelscope"])
        self.assertEqual(spec.env_backends, ["venv", "conda"])
        self.assertEqual(spec.owners, ["team-a"])


class TestSkillSchemaParserValidate(unittest.TestCase):
    """Test SkillSchemaParser.validate."""

    def setUp(self):
        self.parser = SkillSchemaParser()

    def test_valid_spec_no_errors(self):
        spec = SkillSpec(
            name="deploy-python-webui",
            version="1.0.0",
            type="execution_skill",
            stages=["runner", "plan_first"],
            risk_level="low",
        )
        errors = self.parser.validate(spec)
        self.assertEqual(errors, [])

    def test_missing_name(self):
        spec = SkillSpec(name="", version="1.0.0", type="execution_skill", stages=["runner"])
        errors = self.parser.validate(spec)
        self.assertTrue(any("name" in e for e in errors))

    def test_missing_version(self):
        spec = SkillSpec(name="test", version="", type="execution_skill", stages=["runner"])
        errors = self.parser.validate(spec)
        self.assertTrue(any("version" in e for e in errors))

    def test_invalid_version_format(self):
        spec = SkillSpec(name="test", version="v1", type="execution_skill", stages=["runner"])
        errors = self.parser.validate(spec)
        self.assertTrue(any("semver" in e for e in errors))

    def test_valid_version_formats(self):
        for v in ("1.0.0", "0.1.0", "2.3.14"):
            spec = SkillSpec(name="test", version=v, type="execution_skill", stages=["runner"])
            errors = self.parser.validate(spec)
            self.assertFalse(any("version" in e for e in errors), "version %s should be valid" % v)

    def test_invalid_type(self):
        spec = SkillSpec(name="test", version="1.0.0", type="plugin", stages=["runner"])
        errors = self.parser.validate(spec)
        self.assertTrue(any("type" in e for e in errors))

    def test_all_valid_types(self):
        for t in VALID_TYPES:
            spec = SkillSpec(name="test", version="1.0.0", type=t, stages=["analyze"])
            errors = self.parser.validate(spec)
            self.assertFalse(any("type" in e for e in errors), "type %s should be valid" % t)

    def test_empty_stages(self):
        spec = SkillSpec(name="test", version="1.0.0", type="execution_skill", stages=[])
        errors = self.parser.validate(spec)
        self.assertTrue(any("stages" in e for e in errors))

    def test_invalid_stage(self):
        spec = SkillSpec(name="test", version="1.0.0", type="execution_skill", stages=["runner", "invalid_stage"])
        errors = self.parser.validate(spec)
        self.assertTrue(any("invalid_stage" in e for e in errors))

    def test_plan_first_and_replan_are_valid_stages(self):
        spec = SkillSpec(name="test", version="1.0.0", type="execution_skill", stages=["plan_first", "replan"])
        errors = self.parser.validate(spec)
        self.assertFalse(any("stages" in e for e in errors))

    def test_invalid_risk_level(self):
        spec = SkillSpec(name="test", version="1.0.0", type="execution_skill", stages=["runner"], risk_level="critical")
        errors = self.parser.validate(spec)
        self.assertTrue(any("risk_level" in e for e in errors))

    def test_deprecated_without_replacement(self):
        spec = SkillSpec(
            name="old-skill", version="1.0.0", type="analysis_skill",
            stages=["analyze"], deprecated=True, replacement="",
        )
        errors = self.parser.validate(spec)
        self.assertTrue(any("replacement" in e for e in errors))

    def test_deprecated_with_replacement(self):
        spec = SkillSpec(
            name="old-skill", version="1.0.0", type="analysis_skill",
            stages=["analyze"], deprecated=True, replacement="new-skill",
        )
        errors = self.parser.validate(spec)
        self.assertFalse(any("replacement" in e for e in errors))


class TestSkillSchemaParserParseFile(unittest.TestCase):
    """Test SkillSchemaParser.parse_file with real skill files."""

    def setUp(self):
        self.parser = SkillSchemaParser()

    def test_parse_existing_skill_file(self):
        skills_dir = Path(__file__).parent.parent / "skills"
        skill_path = skills_dir / "verify-evidence" / "SKILL.md"
        if not skill_path.exists():
            self.skipTest("verify-evidence SKILL.md not found")
        spec = self.parser.parse_file(skill_path)
        self.assertEqual(spec.name, "verify-evidence")
        self.assertTrue(spec.sha256)
        self.assertTrue(len(spec.content) > 0)

    def test_parse_nonexistent_file(self):
        spec = self.parser.parse_file(Path("/nonexistent/SKILL.md"))
        self.assertEqual(spec.name, "")


if __name__ == "__main__":
    unittest.main()
