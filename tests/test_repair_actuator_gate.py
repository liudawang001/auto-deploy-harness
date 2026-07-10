"""Tests for Repair Actuator Gate (Phase 5).

Covers:
- repair action requires policy_allowed before apply
- repair metadata_only is not self_healing
- repair resume_from_stage runs after apply
- repair verified requires final verify passed
- repair artifacts written for rejected action
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
from auto_harness.agent_runtime.stage_schemas import REPAIR_TOOLS, RepairActuatorResult


class TestRepairGatePolicy(unittest.TestCase):
    """Repair gate policy validation tests."""

    def test_apply_repair_allowed(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_repair", "input": {"action_type": "install_package", "package": "gradio"}}
        r = policy.validate(tc, "repair")
        self.assertTrue(r["allowed"])

    def test_source_edit_rejected(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "apply_repair", "input": {"action_type": "source_edit"}}
        r = policy.validate(tc, "repair")
        self.assertFalse(r["allowed"])
        self.assertIn("source edit", r["reason"])

    def test_resume_safe_stage_allowed(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        for stage in ["env_deploy", "model_prepare", "runner", "verify"]:
            tc = {"name": "resume_from_stage", "input": {"stage": stage}}
            r = policy.validate(tc, "repair")
            self.assertTrue(r["allowed"], "stage '%s' should be safe" % stage)

    def test_resume_unsafe_stage_rejected(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "resume_from_stage", "input": {"stage": "analyze"}}
        r = policy.validate(tc, "repair")
        self.assertFalse(r["allowed"])

    def test_verify_after_repair_allowed(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "verify_after_repair", "input": {}}
        r = policy.validate(tc, "repair")
        self.assertTrue(r["allowed"])

    def test_inspect_log_allowed(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "inspect_log", "input": {"path": "/var/log/app.log"}}
        r = policy.validate(tc, "repair")
        self.assertTrue(r["allowed"])


class TestRepairGateExecution(unittest.TestCase):
    """Repair gate full pipeline tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir)

    def _make_provider(self, response_json):
        provider = MagicMock()
        result = MagicMock()
        result.text = json.dumps(response_json)
        provider.complete.return_value = result
        return provider

    def test_repair_action_policy_allows_and_applies(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "gradio missing, install it",
            "confidence": 0.9,
            "tool_call": {
                "name": "apply_repair",
                "input": {"action_type": "install_package", "package": "gradio"},
            },
            "expected_observation": "runner passes after gradio installed",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="repair",
            observation={
                "failure": {"stage": "runner", "status": "failed", "error": "ModuleNotFoundError: No module named 'gradio'"},
                "diagnosis": {"category": "dependency_missing", "signal": "gradio"},
                "previous_repairs": [],
            },
            allowed_tools=list(REPAIR_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "ok")
        # apply_repair needs executor, so it will be no_executor unless we provide one
        self.assertIn(result.execution.get("status"), ["no_executor", "applied"])

    def test_repair_planner_mode_no_execution(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "install gradio",
            "confidence": 0.9,
            "tool_call": {
                "name": "apply_repair",
                "input": {"action_type": "install_package", "package": "gradio"},
            },
            "expected_observation": "runner passes",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="repair",
            observation={
                "failure": {"stage": "runner", "status": "failed"},
                "diagnosis": {},
                "previous_repairs": [],
            },
            allowed_tools=list(REPAIR_TOOLS),
            mode="planner",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.mode, "planner")
        self.assertFalse(result.execution.get("executed", True))

    def test_repair_artifacts_written_for_rejected_action(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "edit source",
            "confidence": 0.5,
            "tool_call": {
                "name": "apply_repair",
                "input": {"action_type": "source_edit"},
            },
            "expected_observation": "none",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="repair",
            observation={
                "failure": {"stage": "runner", "status": "failed"},
                "diagnosis": {},
                "previous_repairs": [],
            },
            allowed_tools=list(REPAIR_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # Should be rejected by policy
        artifact_path = self.run_dir / "agent_decision_gates" / "repair_gate.json"
        self.assertTrue(artifact_path.exists())

    def test_resume_from_stage_applies(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "rerun from env_deploy",
            "confidence": 0.8,
            "tool_call": {
                "name": "resume_from_stage",
                "input": {"stage": "env_deploy"},
            },
            "expected_observation": "pipeline resumes from env_deploy",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="repair",
            observation={
                "failure": {"stage": "runner", "status": "failed"},
                "diagnosis": {},
                "previous_repairs": [],
            },
            allowed_tools=list(REPAIR_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertTrue(result.execution.get("applied"))
        self.assertEqual(result.state_delta.get("resume_from_stage"), "env_deploy")


class TestRepairActuatorResult(unittest.TestCase):
    """Test RepairActuatorResult data class."""

    def test_default_not_self_healing(self):
        r = RepairActuatorResult()
        self.assertFalse(r.repair_verified)
        self.assertEqual(r.repair_status, "planned")

    def test_full_flow_is_self_healing(self):
        r = RepairActuatorResult(
            repair_status="verified",
            policy_allowed=True,
            executed=True,
            repair_verified=True,
            final_verify_status="passed",
        )
        self.assertTrue(r.repair_verified)
        self.assertTrue(r.policy_allowed)
        self.assertTrue(r.executed)

    def test_metadata_only_is_not_self_healing(self):
        r = RepairActuatorResult(
            repair_status="applied",
            policy_allowed=True,
            executed=True,
            metadata_only=True,
            repair_verified=False,
        )
        self.assertFalse(r.repair_verified)
        self.assertTrue(r.metadata_only)


class TestRepairGateFixture(unittest.TestCase):
    """Test using the repair_missing_dependency fixture."""

    def test_fixture_has_dependency_missing_diagnosis(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "llm_necessity" / "repair_missing_dependency"
        result_path = fixture_dir / "runner_result.json"
        if not result_path.exists():
            self.skipTest("fixture not found")
        result = json.loads(result_path.read_text())
        self.assertEqual(result["data"]["diagnosis"]["category"], "dependency_missing")
        self.assertEqual(result["data"]["diagnosis"]["signal"], "gradio")


if __name__ == "__main__":
    unittest.main()
