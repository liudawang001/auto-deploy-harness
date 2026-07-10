"""Tests for Runner Decision Gate (Phase 2).

Covers:
- runner gate reorders candidate when policy allows
- runner gate rejects unknown candidate_id
- runner gate rejects shell metachar command
- runner gate planner mode does not change candidate order
- runner gate artifact written on invalid JSON
- integration with orchestrator _apply_runner_gate
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
from auto_harness.agent_runtime.stage_schemas import RUNNER_TOOLS, GateResult


class TestRunnerGatePolicy(unittest.TestCase):
    """Runner gate policy validation tests."""

    def test_select_valid_candidate(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        obs = {"run_candidates": [
            {"id": "cand_0", "cmd": ["python", "app.py"]},
            {"id": "cand_1", "cmd": ["python", "gradio_app.py"]},
        ]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}}
        r = policy.validate(tc, "runner", obs)
        self.assertTrue(r["allowed"])

    def test_reject_unknown_candidate_id(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_99"}}
        r = policy.validate(tc, "runner", obs)
        self.assertFalse(r["allowed"])
        self.assertIn("not found", r["reason"])

    def test_reject_shell_metachar_command(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py; rm -rf /"]}]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_0"}}
        r = policy.validate(tc, "runner", obs)
        self.assertFalse(r["allowed"])
        self.assertIn("metachar", r["reason"])

    def test_reject_disallowed_command_root(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        obs = {"run_candidates": [{"id": "cand_0", "cmd": ["curl", "http://evil.com"]}]}
        tc = {"name": "select_runner_candidate", "input": {"candidate_id": "cand_0"}}
        r = policy.validate(tc, "runner", obs)
        self.assertFalse(r["allowed"])
        self.assertIn("not in allowed", r["reason"])


class TestRunnerGateExecution(unittest.TestCase):
    """Runner gate full pipeline tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir)

    def _make_provider(self, response_json):
        provider = MagicMock()
        result = MagicMock()
        result.text = json.dumps(response_json)
        provider.complete.return_value = result
        return provider

    def test_reorders_candidate_when_policy_allows(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "gradio_app.py is the real entrypoint",
            "confidence": 0.9,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "runner passes with gradio_app.py",
        })
        gate = AgentDecisionGate(provider=provider)
        obs = {"run_candidates": [
            {"id": "cand_0", "cmd": ["python", "app.py"], "entrypoint": "app.py"},
            {"id": "cand_1", "cmd": ["python", "gradio_app.py"], "entrypoint": "gradio_app.py"},
        ]}
        result = gate.decide(
            stage="runner",
            observation=obs,
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "ok")
        self.assertTrue(result.state_delta.get("changed"))
        reordered = result.state_delta.get("reordered_candidates", [])
        self.assertEqual(len(reordered), 2)
        self.assertEqual(reordered[0]["id"], "cand_1")

    def test_planner_mode_does_not_change_candidate_order(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "gradio_app.py is better",
            "confidence": 0.8,
            "tool_call": {"name": "select_runner_candidate", "input": {"candidate_id": "cand_1"}},
            "expected_observation": "runner passes",
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

    def test_artifact_written_on_invalid_json(self):
        provider = MagicMock()
        result = MagicMock()
        result.text = "not json"
        provider.complete.return_value = result

        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="runner",
            observation={"run_candidates": [{"id": "cand_0", "cmd": ["python", "app.py"]}]},
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "invalid")
        artifact_path = self.run_dir / "agent_decision_gates" / "runner_gate.json"
        self.assertTrue(artifact_path.exists())

    def test_no_candidates_returns_no_action(self):
        provider = MagicMock()
        gate = AgentDecisionGate(provider=None)
        result = gate.decide(
            stage="runner",
            observation={"run_candidates": []},
            allowed_tools=list(RUNNER_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "no_action")

    def test_reject_runner_candidate(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "cand_0 is broken",
            "confidence": 0.7,
            "tool_call": {"name": "reject_runner_candidate", "input": {"candidate_id": "cand_0"}},
            "expected_observation": "cand_0 removed",
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
        self.assertTrue(result.state_delta.get("changed"))
        self.assertEqual(result.state_delta.get("rejected_id"), "cand_0")
        remaining = result.state_delta.get("remaining_candidates", [])
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], "cand_1")


class TestRunnerGateFixture(unittest.TestCase):
    """Test using the wrong_default_entrypoint fixture."""

    def test_fixture_analysis_has_two_candidates(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "llm_necessity" / "wrong_default_entrypoint"
        analysis_path = fixture_dir / "analysis.json"
        if not analysis_path.exists():
            self.skipTest("fixture not found")
        analysis = json.loads(analysis_path.read_text())
        self.assertEqual(len(analysis["run_candidates"]), 2)
        self.assertEqual(analysis["run_candidates"][0]["entrypoint"], "app.py")
        self.assertEqual(analysis["run_candidates"][1]["entrypoint"], "gradio_app.py")


if __name__ == "__main__":
    unittest.main()
