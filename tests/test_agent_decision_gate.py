"""Tests for AgentDecisionGate core framework (Phase 1).

Covers:
- schema validation (parse_gate_decision)
- critic rejection
- policy rejection
- planner mode no execution
- gated_actor applies state delta
- llm_helped only on improvement
- artifact written
- safety prompt injection rejected
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.decision_gate import (
    AgentDecisionGate,
    GateArtifactWriter,
    GateCritic,
    StagePolicyValidator,
)
from auto_harness.agent_runtime.stage_planners import parse_gate_decision
from auto_harness.agent_runtime.stage_schemas import (
    GateDecision,
    GateResult,
    RUNNER_TOOLS,
    ENV_TOOLS,
)


class TestParseGateDecision(unittest.TestCase):
    """Tests for parse_gate_decision schema validation."""

    def test_valid_ok_decision(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "cand_1 is better",
            "confidence": 0.8,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "candidate reordered",
        })
        d = parse_gate_decision(raw, allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "ok")
        self.assertEqual(d.hypothesis, "cand_1 is better")
        self.assertAlmostEqual(d.confidence, 0.8)
        self.assertEqual(d.tool_call["name"], "select_runner_candidate")

    def test_valid_no_action(self):
        raw = json.dumps({
            "status": "no_action",
            "hypothesis": "current candidate is fine",
            "confidence": 0.3,
            "tool_call": None,
            "stop_reason": "no_safe_action",
        })
        d = parse_gate_decision(raw, allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "no_action")
        self.assertIsNone(d.tool_call)

    def test_invalid_json(self):
        d = parse_gate_decision("not json", allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "invalid")
        self.assertIn("invalid", d.stop_reason)

    def test_empty_response(self):
        d = parse_gate_decision("", allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "invalid")
        self.assertEqual(d.stop_reason, "empty_response")

    def test_invalid_status_value(self):
        raw = json.dumps({"status": "success", "confidence": 0.5})
        d = parse_gate_decision(raw, allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "invalid")
        self.assertEqual(d.stop_reason, "invalid_status")

    def test_unknown_tool_rejected(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.5,
            "tool_call": {"name": "unknown_tool", "input": {}},
        })
        d = parse_gate_decision(raw, allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "invalid")
        self.assertEqual(d.stop_reason, "unknown_tool")

    def test_missing_tool_call_name(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.5,
            "tool_call": {"input": {}},
        })
        d = parse_gate_decision(raw, allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "invalid")
        self.assertEqual(d.stop_reason, "missing_tool_call_name")

    def test_no_action_with_tool_call_rejected(self):
        raw = json.dumps({
            "status": "no_action",
            "hypothesis": "test",
            "confidence": 0.5,
            "tool_call": {"name": "select_runner_candidate", "input": {}},
        })
        d = parse_gate_decision(raw, allowed_tools=list(RUNNER_TOOLS), stage="runner")
        self.assertEqual(d.status, "invalid")
        self.assertEqual(d.stop_reason, "no_action_with_tool_call")


class TestGateCritic(unittest.TestCase):
    """Tests for GateCritic."""

    def setUp(self):
        self.critic = GateCritic()

    def test_valid_tool_passes(self):
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_0"}}
        r = self.critic.evaluate(tc, "runner")
        self.assertTrue(r["allowed"])

    def test_secret_in_input_rejected(self):
        tc = {"name": "select_model_source", "input": {"api_key": "sk-123456"}}
        r = self.critic.evaluate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("secret", r["reason"])

    def test_stage_mismatch_rejected(self):
        tc = {"name": "apply_repair", "input": {}}
        r = self.critic.evaluate(tc, "runner")
        self.assertFalse(r["allowed"])
        self.assertIn("not relevant", r["reason"])


class TestStagePolicyValidator(unittest.TestCase):
    """Tests for StagePolicyValidator."""

    def setUp(self):
        self.policy = StagePolicyValidator()

    def test_runner_select_valid_candidate(self):
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_0"}}
        r = self.policy.validate(tc, "runner", obs)
        self.assertTrue(r["allowed"])

    def test_runner_select_unknown_candidate_rejected(self):
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_99"}}
        r = self.policy.validate(tc, "runner", obs)
        self.assertFalse(r["allowed"])
        self.assertIn("not found", r["reason"])

    def test_runner_shell_metachar_rejected(self):
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py; rm -rf /"]}]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_0"}}
        r = self.policy.validate(tc, "runner", obs)
        self.assertFalse(r["allowed"])
        self.assertIn("metachar", r["reason"])

    def test_env_valid_package_constraint(self):
        tc = {"name": "apply_dependency_constraint", "input": {"package": "pydantic", "version_spec": "<2"}}
        r = self.policy.validate(tc, "env_solve")
        self.assertTrue(r["allowed"])

    def test_env_invalid_package_name_rejected(self):
        tc = {"name": "apply_dependency_constraint", "input": {"package": "pydantic; rm -rf /", "version_spec": "<2"}}
        r = self.policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])

    def test_env_index_url_rejected(self):
        tc = {"name": "apply_dependency_constraint", "input": {
            "package": "pydantic", "version_spec": "<2", "index_url": "https://evil.com/simple"
        }}
        r = self.policy.validate(tc, "env_solve")
        self.assertFalse(r["allowed"])
        self.assertIn("index URL", r["reason"])

    def test_model_external_url_rejected(self):
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "https://evil.com/model"
        }}
        r = self.policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("external URL", r["reason"])

    def test_model_path_traversal_rejected(self):
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "org/model", "target_path": "../../etc/passwd"
        }}
        r = self.policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("traversal", r["reason"])

    def test_model_secret_field_rejected(self):
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "org/model", "token": "hf_abc123"
        }}
        r = self.policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("secret", r["reason"])

    def test_repair_resume_unsafe_stage_rejected(self):
        tc = {"name": "resume_from_stage", "input": {"stage": "analyze"}}
        r = self.policy.validate(tc, "repair")
        self.assertFalse(r["allowed"])
        self.assertIn("not in safe stages", r["reason"])

    def test_repair_resume_safe_stage_allowed(self):
        tc = {"name": "resume_from_stage", "input": {"stage": "env_deploy"}}
        r = self.policy.validate(tc, "repair")
        self.assertTrue(r["allowed"])

    def test_plan_invalid_stage_rejected(self):
        tc = {"name": "set_stage_hint", "input": {"stage": "nonexistent_stage"}}
        r = self.policy.validate(tc, "plan")
        self.assertFalse(r["allowed"])
        self.assertIn("invalid stage", r["reason"])

    def test_plan_policy_bypass_rejected(self):
        tc = {"name": "set_stage_hint", "input": {"stage": "runner", "hints": {"bypass_policy": True}}}
        r = self.policy.validate(tc, "plan")
        self.assertFalse(r["allowed"])
        self.assertIn("change policy", r["reason"])


class TestAgentDecisionGate(unittest.TestCase):
    """Tests for AgentDecisionGate full pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir)

    def _make_mock_provider(self, response_json):
        provider = MagicMock()
        result = MagicMock()
        result.text = json.dumps(response_json)
        provider.complete.return_value = result
        return provider

    def test_planner_mode_does_not_execute(self):
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "cand_1 is better",
            "confidence": 0.8,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "reordered",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [
            {"id": "cand_0", "cmd": ["python", "app.py"]},
            {"id": "cand_1", "cmd": ["python", "gradio_app.py"]},
        ]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="planner",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.mode, "planner")
        self.assertFalse(result.execution.get("executed", True))
        self.assertEqual(result.execution.get("status"), "planner_mode_would_execute")

    def test_gated_actor_applies_state_delta(self):
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "cand_1 is better",
            "confidence": 0.8,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "reordered",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [
            {"id": "cand_0", "cmd": ["python", "app.py"]},
            {"id": "cand_1", "cmd": ["python", "gradio_app.py"]},
        ]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.mode, "gated_actor")
        self.assertTrue(result.execution.get("executed") or result.execution.get("applied"))
        self.assertTrue(result.state_delta.get("changed"))

    def test_policy_rejected_no_state_change(self):
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.5,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_99"}},
            "expected_observation": "none",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # Should loop and eventually return without executing
        self.assertNotEqual(result.execution.get("status"), "applied")

    def test_artifact_written_on_invalid_json(self):
        provider = self._make_mock_provider("not valid json at all")
        # Need to make provider return something with .text
        provider = MagicMock()
        result = MagicMock()
        result.text = "not valid json"
        provider.complete.return_value = result

        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="runner",
            observation={},
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "invalid")
        # Artifact should be written
        artifact_path = self.run_dir / "agent_decision_gates" / "runner_gate.json"
        self.assertTrue(artifact_path.exists())
        artifact = json.loads(artifact_path.read_text())
        self.assertEqual(artifact["decision_status"], "invalid")

    def test_env_gate_constraint_overlay_written(self):
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "pydantic<2 needed",
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

    def test_no_action_returns_no_action_status(self):
        provider = self._make_mock_provider({
            "status": "no_action",
            "hypothesis": "current is fine",
            "confidence": 0.3,
            "tool_call": None,
            "stop_reason": "no_safe_action",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="runner",
            observation={"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]},
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "no_action")

    def test_no_provider_returns_no_action(self):
        gate = AgentDecisionGate(provider=None)
        result = gate.decide(
            stage="runner",
            observation={},
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "no_action")

    def test_llm_helped_false_at_gate_level_even_when_applied(self):
        """GateResult must NOT self-declare llm_helped=true.

        llm_helped must be computed by AgentContributionAnalyzer after
        observing actual stage status improvement, not at gate level.
        """
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "cand_1 is better",
            "confidence": 0.8,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "reordered",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [
            {"id": "cand_0", "cmd": ["python", "app.py"]},
            {"id": "cand_1", "cmd": ["python", "gradio_app.py"]},
        ]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # GateResult NEVER self-declares llm_helped=true
        self.assertFalse(result.llm_helped, "GateResult must not self-declare llm_helped")
        # But policy should have allowed and execution should have applied
        self.assertTrue(result.policy.get("allowed", False))
        self.assertTrue(result.execution.get("applied") or result.execution.get("executed"))

    def test_llm_helped_false_when_planner_mode(self):
        """llm_helped should be False in planner mode (no execution)."""
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "cand_1 is better",
            "confidence": 0.8,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "reordered",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [
            {"id": "cand_0", "cmd": ["python", "app.py"]},
            {"id": "cand_1", "cmd": ["python", "gradio_app.py"]},
        ]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="planner",
            run_dir=self.run_dir,
        )
        self.assertFalse(result.llm_helped, "llm_helped should be False in planner mode")

    def test_llm_helped_false_when_policy_rejected(self):
        """llm_helped should be False when policy rejects the action."""
        provider = self._make_mock_provider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.5,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_99"}},
            "expected_observation": "none",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertFalse(result.llm_helped, "llm_helped should be False when policy rejects")


class TestComputeLlmHelped(unittest.TestCase):
    """Tests for the compute_llm_helped function."""

    def test_helped_when_failed_to_passed(self):
        from auto_harness.agent_runtime.contribution import compute_llm_helped
        self.assertTrue(compute_llm_helped(
            before_status="failed",
            after_status="passed",
            policy_allowed=True,
            applied=True,
            executed=False,
            final_verify_status="passed",
        ))

    def test_not_helped_when_policy_rejected(self):
        from auto_harness.agent_runtime.contribution import compute_llm_helped
        self.assertFalse(compute_llm_helped(
            before_status="failed",
            after_status="passed",
            policy_allowed=False,
            applied=True,
            executed=False,
        ))

    def test_not_helped_when_not_applied(self):
        from auto_harness.agent_runtime.contribution import compute_llm_helped
        self.assertFalse(compute_llm_helped(
            before_status="failed",
            after_status="passed",
            policy_allowed=True,
            applied=False,
            executed=False,
        ))

    def test_not_helped_when_before_was_passed(self):
        from auto_harness.agent_runtime.contribution import compute_llm_helped
        self.assertFalse(compute_llm_helped(
            before_status="passed",
            after_status="passed",
            policy_allowed=True,
            applied=True,
            executed=False,
        ))

    def test_helped_with_improved_needs_final_verify(self):
        from auto_harness.agent_runtime.contribution import compute_llm_helped
        # improved without final verify -> False
        self.assertFalse(compute_llm_helped(
            before_status="failed",
            after_status="improved",
            policy_allowed=True,
            applied=True,
            executed=False,
            final_verify_status="",
        ))
        # improved with final verify passed -> True
        self.assertTrue(compute_llm_helped(
            before_status="failed",
            after_status="improved",
            policy_allowed=True,
            applied=True,
            executed=False,
            final_verify_status="passed",
        ))


class TestGateArtifactWriter(unittest.TestCase):
    """Tests for GateArtifactWriter."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = GateArtifactWriter(Path(self.tmpdir))

    def test_write_gate_result(self):
        result = GateResult(stage="runner", mode="gated_actor", decision_status="ok")
        path = self.writer.write_gate_result("runner", result)
        self.assertTrue(path.exists())
        data = json.loads(path.read_text())
        self.assertEqual(data["stage"], "runner")
        self.assertEqual(data["mode"], "gated_actor")

    def test_write_step_appends_jsonl(self):
        self.writer.write_step("runner", 1, {"step": 1, "action": "test"})
        self.writer.write_step("runner", 2, {"step": 2, "action": "test2"})
        path = self.writer.gates_dir / "runner_steps.jsonl"
        lines = path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)

    def test_write_summary(self):
        results = {"enabled_gates": ["runner"], "llm_required": True}
        path = self.writer.write_summary(results)
        self.assertTrue(path.exists())
        data = json.loads(path.read_text())
        self.assertTrue(data["llm_required"])


if __name__ == "__main__":
    unittest.main()
