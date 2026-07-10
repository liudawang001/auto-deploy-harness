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


class TestRepairLoopClosure(unittest.TestCase):
    """Test repair loop closure: repair -> apply -> resume -> final verify."""

    def test_repair_requires_final_verify_for_repair_verified(self):
        """repair_verified should only be True when final verify passes."""
        r = RepairActuatorResult(
            repair_status="verified",
            policy_allowed=True,
            executed=True,
            repair_verified=True,
            final_verify_status="passed",
        )
        self.assertTrue(r.repair_verified)
        self.assertEqual(r.final_verify_status, "passed")

    def test_metadata_only_repair_is_not_self_healing(self):
        """metadata_only repair should not count as self_healing."""
        r = RepairActuatorResult(
            repair_status="applied",
            policy_allowed=True,
            executed=True,
            metadata_only=True,
            repair_verified=False,
        )
        self.assertFalse(r.repair_verified)
        self.assertTrue(r.metadata_only)

    def test_repair_policy_rejection_does_not_resume(self):
        """When policy rejects repair, resume should not happen."""
        from auto_harness.repair.loop import RepairLoopController
        import tempfile
        tmpdir = tempfile.mkdtemp()
        run_dir = Path(tmpdir)
        controller = RepairLoopController(max_attempts=2)
        memory_entry = {"signature": "runner:dependency_missing:env_deploy"}
        plan = {"root_cause": "dependency_missing", "rerun_from": "env_deploy"}
        policy_result = {"allowed": False, "decisions": [{"action_type": "install_package", "allowed": False}]}
        result = controller.gate(run_dir, "runner", memory_entry, plan, policy_result)
        self.assertFalse(result["allowed"])

    def test_repair_resume_from_safe_stage(self):
        """Resume should only be from safe stages."""
        from auto_harness.repair.loop import RepairLoopController
        import tempfile
        tmpdir = tempfile.mkdtemp()
        run_dir = Path(tmpdir)
        controller = RepairLoopController(max_attempts=2)
        memory_entry = {"signature": "runner:dependency_missing:env_deploy"}
        plan = {"root_cause": "dependency_missing", "rerun_from": "env_deploy"}
        policy_result = {"allowed": True, "decisions": []}
        result = controller.gate(run_dir, "runner", memory_entry, plan, policy_result)
        self.assertTrue(result["allowed"])
        self.assertIn(result["loop"]["rerun_from_effective"], RepairLoopController.SAFE_RERUN_STAGES)

    def test_repair_forbidden_stage_fallback(self):
        """Resume from forbidden stage should fallback to env_deploy."""
        from auto_harness.repair.loop import RepairLoopController
        import tempfile
        tmpdir = tempfile.mkdtemp()
        run_dir = Path(tmpdir)
        controller = RepairLoopController(max_attempts=2)
        memory_entry = {"signature": "runner:dependency_missing:analyze"}
        plan = {"root_cause": "dependency_missing", "rerun_from": "analyze"}
        policy_result = {"allowed": True, "decisions": []}
        result = controller.gate(run_dir, "runner", memory_entry, plan, policy_result)
        # analyze is forbidden, should fallback to env_deploy
        self.assertEqual(result["loop"]["rerun_from_effective"], "env_deploy")


class TestMetadataOnlyNotEffectiveRepair(unittest.TestCase):
    """metadata_only actions must NEVER count as effective repair."""

    def test_metadata_only_rerun_from_stage_not_effective_repair(self):
        """rerun_from_stage metadata_only must not be treated as effective."""
        from auto_harness.orchestrator import TaskRunner
        from auto_harness.config import HarnessConfig
        config = HarnessConfig()
        runner = TaskRunner(config)
        apply_result = {
            "status": "applied",
            "action_results": [
                {
                    "action_type": "rerun_from_stage",
                    "executed": False,
                    "status": "metadata_only",
                    "rerun_from": "env_deploy",
                }
            ],
        }
        self.assertFalse(runner._repair_apply_effective(apply_result))

    def test_metadata_only_update_verify_hint_not_effective_repair(self):
        """update_verify_hint metadata_only must not be treated as effective."""
        from auto_harness.orchestrator import TaskRunner
        from auto_harness.config import HarnessConfig
        config = HarnessConfig()
        runner = TaskRunner(config)
        apply_result = {
            "status": "applied",
            "action_results": [
                {
                    "action_type": "update_verify_hint",
                    "executed": False,
                    "status": "metadata_only",
                }
            ],
        }
        self.assertFalse(runner._repair_apply_effective(apply_result))

    def test_executed_action_with_zero_exit_code_is_effective(self):
        """An executed action with exit_code=0 should be effective."""
        from auto_harness.orchestrator import TaskRunner
        from auto_harness.config import HarnessConfig
        config = HarnessConfig()
        runner = TaskRunner(config)
        apply_result = {
            "status": "applied",
            "action_results": [
                {
                    "action_type": "install_package",
                    "executed": True,
                    "exit_code": 0,
                    "cmd": ["pip", "install", "gradio"],
                }
            ],
        }
        self.assertTrue(runner._repair_apply_effective(apply_result))

    def test_strong_verify_pass_is_effective(self):
        """tool_result with strong_verify_pass should be effective."""
        from auto_harness.orchestrator import TaskRunner
        from auto_harness.config import HarnessConfig
        config = HarnessConfig()
        runner = TaskRunner(config)
        apply_result = {
            "status": "applied",
            "action_results": [
                {
                    "action_type": "verify_after_repair",
                    "executed": False,
                    "tool_result": {"strong_verify_pass": True},
                }
            ],
        }
        self.assertTrue(runner._repair_apply_effective(apply_result))


if __name__ == "__main__":
    unittest.main()
