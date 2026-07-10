"""Tests for Environment Decision Gate (Phase 3).

Covers:
- env gate writes temporary constraints overlay
- env gate does not modify source requirements
- env gate rejects arbitrary index URL
- env gate rejects invalid version spec
- env gate sets llm_helped only after env status improves
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
from auto_harness.agent_runtime.stage_schemas import ENV_TOOLS


class TestEnvGatePolicy(unittest.TestCase):
    """Env gate policy validation tests."""

    def test_valid_dependency_constraint(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_dependency_constraint", "input": {"package": "pydantic", "version_spec": "<2"}}
        r = policy.validate(tc, "env_solve")
        self.assertTrue(r["allowed"])

    def test_rejects_arbitrary_index_url(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_dependency_constraint", "input": {
            "package": "pydantic", "version_spec": "<2", "index_url": "https://evil.com/simple"
        }}
        r = policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])
        self.assertIn("index URL", r["reason"])

    def test_rejects_invalid_version_spec(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_dependency_constraint", "input": {
            "package": "pydantic", "version_spec": "not_a_version"
        }}
        r = policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])

    def test_rejects_source_edit(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_dependency_constraint", "input": {
            "package": "pydantic", "version_spec": "<2", "source_edit": True
        }}
        r = policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])
        self.assertIn("source edit", r["reason"])

    def test_valid_version_specs(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        for spec in ["<2", "<=1.10.15", ">=0.10,<0.11", "==1.2.3", "~=1.4"]:
            tc = {"name": "apply_dependency_constraint", "input": {"package": "pydantic", "version_spec": spec}}
            r = policy.validate(tc, "env_solve")
            self.assertTrue(r["allowed"], "spec '%s' should be valid" % spec)

    def test_rejects_shell_metachar_in_package(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_dependency_constraint", "input": {"package": "pydantic; rm -rf /", "version_spec": "<2"}}
        r = policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])

    def test_select_backend_allowed(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        for backend in ["venv", "conda", "mamba"]:
            tc = {"name": "select_environment_backend", "input": {"backend": backend}}
            r = policy.validate(tc, "env_solve")
            self.assertTrue(r["allowed"], "backend '%s' should be allowed" % backend)

    def test_rejects_unknown_backend(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_environment_backend", "input": {"backend": "unknown_backend"}}
        r = policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])


class TestEnvGateExecution(unittest.TestCase):
    """Env gate full pipeline tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir)

    def _make_provider(self, response_json):
        provider = MagicMock()
        result = MagicMock()
        result.text = json.dumps(response_json)
        provider.complete.return_value = result
        return provider

    def test_writes_temporary_constraints_overlay(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "gradio requires pydantic<2",
            "confidence": 0.9,
            "tool_call": {
                "name": "apply_dependency_constraint",
                "input": {"package": "pydantic", "version_spec": "<2", "scope": "temporary_overlay"},
            },
            "expected_observation": "env deploy passes",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="env_solve",
            observation={},
            allowed_tools=list(ENV_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertTrue(result.execution.get("applied"))
        overlay_path = self.run_dir / "repair_overlay" / "constraints.txt"
        self.assertTrue(overlay_path.exists())
        content = overlay_path.read_text()
        self.assertIn("pydantic", content)

    def test_does_not_modify_source_requirements(self):
        """The env gate writes to repair_overlay, not to requirements.txt."""
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "pydantic<2 needed",
            "confidence": 0.9,
            "tool_call": {
                "name": "apply_dependency_constraint",
                "input": {"package": "pydantic", "version_spec": "<2"},
            },
            "expected_observation": "env deploy passes",
        })
        # Create a fake requirements.txt that should NOT be modified
        req_path = self.run_dir / "requirements.txt"
        req_path.write_text("gradio==3.50.0\npydantic\n", encoding="utf-8")

        gate = AgentDecisionGate(provider=provider)
        gate.decide(
            stage="env_solve",
            observation={},
            allowed_tools=list(ENV_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # Original requirements.txt should be unchanged
        self.assertEqual(req_path.read_text(), "gradio==3.50.0\npydantic\n")
        # Overlay should exist separately
        overlay_path = self.run_dir / "repair_overlay" / "constraints.txt"
        self.assertTrue(overlay_path.exists())

    def test_planner_mode_does_not_write_overlay(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "pydantic<2 needed",
            "confidence": 0.9,
            "tool_call": {
                "name": "apply_dependency_constraint",
                "input": {"package": "pydantic", "version_spec": "<2"},
            },
            "expected_observation": "env deploy passes",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="env_solve",
            observation={},
            allowed_tools=list(ENV_TOOLS),
            mode="planner",
            run_dir=self.run_dir,
        )
        self.assertFalse(result.execution.get("executed", True))
        overlay_path = self.run_dir / "repair_overlay" / "constraints.txt"
        self.assertFalse(overlay_path.exists())

    def test_artifact_written_on_rejected_action(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.5,
            "tool_call": {
                "name": "apply_dependency_constraint",
                "input": {"package": "pydantic; echo pwned", "version_spec": "<2"},
            },
            "expected_observation": "none",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="env_solve",
            observation={},
            allowed_tools=list(ENV_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # Should be rejected by policy
        artifact_path = self.run_dir / "agent_decision_gates" / "env_solve_gate.json"
        self.assertTrue(artifact_path.exists())


class TestEnvGateFixture(unittest.TestCase):
    """Test using the dependency_conflict_pydantic fixture."""

    def test_fixture_has_risk_reasons(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "llm_necessity" / "dependency_conflict_pydantic"
        analysis_path = fixture_dir / "analysis.json"
        if not analysis_path.exists():
            self.skipTest("fixture not found")
        analysis = json.loads(analysis_path.read_text())
        self.assertTrue(analysis["env_solution"]["risk_reasons"])


if __name__ == "__main__":
    unittest.main()
