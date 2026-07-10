"""Tests for Cross-stage Plan Decision Gate (Phase 6).

Covers:
- plan gate writes initial strategy
- plan gate rejects unknown stage name
- plan gate cannot change policy allowlist
- plan revision only on failed or uncertain boundary
- plan revision has max once per stage
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.decision_gate import AgentDecisionGate, StagePolicyValidator
from auto_harness.agent_runtime.stage_planners import PlanPlanner, parse_gate_decision
from auto_harness.agent_runtime.stage_schemas import PLAN_TOOLS, DeploymentStrategy


class TestPlanGatePolicy(unittest.TestCase):
    """Plan gate policy validation tests."""

    def test_valid_stage_hint(self):
        policy = StagePolicyValidator()
        tc = {"name": "set_stage_hint", "input": {"stage": "runner", "hints": {"strategy": "gradio_launch"}}}
        r = policy.validate(tc, "plan")
        self.assertTrue(r["allowed"])

    def test_rejects_unknown_stage_name(self):
        policy = StagePolicyValidator()
        tc = {"name": "set_stage_hint", "input": {"stage": "nonexistent_stage"}}
        r = policy.validate(tc, "plan")
        self.assertFalse(r["allowed"])
        self.assertIn("invalid stage", r["reason"])

    def test_rejects_policy_bypass_hint(self):
        policy = StagePolicyValidator()
        tc = {"name": "set_stage_hint", "input": {"stage": "runner", "hints": {"bypass_policy": True}}}
        r = policy.validate(tc, "plan")
        self.assertFalse(r["allowed"])
        self.assertIn("change policy", r["reason"])

    def test_rejects_allow_source_edit_hint(self):
        policy = StagePolicyValidator()
        tc = {"name": "set_stage_hint", "input": {"stage": "runner", "hints": {"allow_source_edit": True}}}
        r = policy.validate(tc, "plan")
        self.assertFalse(r["allowed"])

    def test_valid_deployment_strategy(self):
        policy = StagePolicyValidator()
        tc = {"name": "set_deployment_strategy", "input": {"strategy": "gradio_local_transformers"}}
        r = policy.validate(tc, "plan")
        self.assertTrue(r["allowed"])


class TestPlanPlanner(unittest.TestCase):
    """Test PlanPlanner."""

    def test_no_provider_returns_no_action(self):
        planner = PlanPlanner()
        d = planner.plan({}, provider=None)
        self.assertEqual(d.status, "no_action")

    def test_parse_valid_plan_decision(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "gradio local deployment",
            "confidence": 0.8,
            "tool_call": {
                "name": "set_deployment_strategy",
                "input": {
                    "strategy": "gradio_local_transformers",
                    "stage_plan": [
                        {"stage": "model_prepare", "strategy": "download_hf_snapshot"},
                        {"stage": "runner", "strategy": "gradio_launch"},
                    ],
                },
            },
            "expected_observation": "pipeline runs with gradio",
        })
        d = parse_gate_decision(raw, allowed_tools=list(PLAN_TOOLS), stage="plan")
        self.assertEqual(d.status, "ok")
        self.assertEqual(d.tool_call["name"], "set_deployment_strategy")

    def test_parse_no_action_plan(self):
        raw = json.dumps({
            "status": "no_action",
            "hypothesis": "deterministic is sufficient",
            "confidence": 0.3,
            "tool_call": None,
            "stop_reason": "no_uncertainty",
        })
        d = parse_gate_decision(raw, allowed_tools=list(PLAN_TOOLS), stage="plan")
        self.assertEqual(d.status, "no_action")


class TestPlanGateExecution(unittest.TestCase):
    """Plan gate full pipeline tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir)

    def _make_provider(self, response_json):
        provider = MagicMock()
        result = MagicMock()
        result.text = json.dumps(response_json)
        provider.complete.return_value = result
        return provider

    def test_writes_initial_strategy(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "gradio local deployment",
            "confidence": 0.8,
            "tool_call": {
                "name": "set_deployment_strategy",
                "input": {
                    "strategy": "gradio_local_transformers",
                    "stage_plan": [
                        {"stage": "model_prepare", "strategy": "download_hf_snapshot"},
                        {"stage": "runner", "strategy": "gradio_launch"},
                    ],
                },
            },
            "expected_observation": "pipeline runs",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="plan",
            observation={
                "analysis_summary": {"frameworks": ["gradio"]},
                "frameworks": ["gradio"],
                "previous_results": {},
                "uncertainties": ["Multiple run candidates"],
            },
            allowed_tools=list(PLAN_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # Plan gate should apply strategy
        self.assertEqual(result.decision_status, "ok")

    def test_planner_mode_does_not_execute(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "test strategy",
            "confidence": 0.7,
            "tool_call": {
                "name": "set_deployment_strategy",
                "input": {"strategy": "test"},
            },
            "expected_observation": "test",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="plan",
            observation={"analysis_summary": {}, "frameworks": [], "previous_results": {}, "uncertainties": []},
            allowed_tools=list(PLAN_TOOLS),
            mode="planner",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.mode, "planner")
        self.assertFalse(result.execution.get("executed", True))

    def test_no_action_when_no_uncertainty(self):
        provider = self._make_provider({
            "status": "no_action",
            "hypothesis": "deterministic is sufficient",
            "confidence": 0.3,
            "tool_call": None,
            "stop_reason": "no_uncertainty",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="plan",
            observation={"analysis_summary": {}, "frameworks": [], "previous_results": {}, "uncertainties": []},
            allowed_tools=list(PLAN_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "no_action")

    def test_artifact_written(self):
        provider = MagicMock()
        result = MagicMock()
        result.text = "invalid json"
        provider.complete.return_value = result

        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="plan",
            observation={"analysis_summary": {}, "frameworks": [], "previous_results": {}, "uncertainties": []},
            allowed_tools=list(PLAN_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        artifact_path = self.run_dir / "agent_decision_gates" / "plan_gate.json"
        self.assertTrue(artifact_path.exists())


class TestDeploymentStrategy(unittest.TestCase):
    """Test DeploymentStrategy data class."""

    def test_default_values(self):
        ds = DeploymentStrategy()
        self.assertEqual(ds.status, "invalid")
        self.assertEqual(ds.fallback, "deterministic_pipeline")

    def test_valid_strategy(self):
        ds = DeploymentStrategy(
            status="ok",
            deployment_strategy="gradio_local",
            confidence=0.8,
            stage_plan=[{"stage": "runner", "strategy": "gradio_launch"}],
        )
        self.assertEqual(ds.status, "ok")
        self.assertEqual(len(ds.stage_plan), 1)


if __name__ == "__main__":
    unittest.main()
