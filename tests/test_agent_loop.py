"""Tests for DeploymentAgentLoop (Phase 2).

Covers:
- AgentState data structure
- StopCondition logic
- AgentArtifactWriter
- DeploymentAgentLoop basic flow
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.state import AgentState
from auto_harness.agent_runtime.stop import StopCondition
from auto_harness.agent_runtime.artifacts import AgentArtifactWriter
from auto_harness.agent_runtime.loop import DeploymentAgentLoop


class TestAgentState(unittest.TestCase):
    """Tests for AgentState data structure."""

    def test_default_state(self):
        state = AgentState(task_id="test", run_dir="/tmp/test")
        self.assertEqual(state.task_id, "test")
        self.assertEqual(state.mode, "planner")
        self.assertEqual(state.current_stage, "analyze")
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.max_iterations, 5)
        self.assertEqual(state.stop_reason, "")

    def test_to_dict(self):
        state = AgentState(task_id="test", run_dir="/tmp/test")
        d = state.to_dict()
        self.assertEqual(d["task_id"], "test")
        self.assertIn("updated_at", d)
        self.assertIsInstance(d["observations"], list)
        self.assertIsInstance(d["decisions"], list)

    def test_record_observation(self):
        state = AgentState(task_id="test", run_dir="/tmp/test")
        state.record_observation("runner", {"candidates": []})
        self.assertEqual(len(state.observations), 1)
        self.assertEqual(state.observations[0]["stage"], "runner")

    def test_record_decision(self):
        state = AgentState(task_id="test", run_dir="/tmp/test")
        state.record_decision("runner", {"tool_call": "select_candidate"})
        self.assertEqual(len(state.decisions), 1)

    def test_record_tool_result(self):
        state = AgentState(task_id="test", run_dir="/tmp/test")
        state.record_tool_result("runner", {"executed": True})
        self.assertEqual(len(state.tool_results), 1)

    def test_update_stage_status(self):
        state = AgentState(task_id="test", run_dir="/tmp/test")
        state.update_stage_status("runner", "improved")
        self.assertEqual(state.stage_status["runner"]["status"], "improved")

    def test_save(self):
        tmpdir = tempfile.mkdtemp()
        state = AgentState(task_id="test", run_dir=tmpdir)
        path = state.save()
        self.assertTrue(path.exists())
        data = json.loads(path.read_text())
        self.assertEqual(data["task_id"], "test")


class TestStopCondition(unittest.TestCase):
    """Tests for StopCondition."""

    def test_stop_on_verify_passed(self):
        stop = StopCondition(max_iterations=5, stop_on_verify_pass=True)
        should, reason = stop.check(
            iteration=1, verify_status="passed",
            policy_results=[], stage_status={},
        )
        self.assertTrue(should)
        self.assertEqual(reason, "verify_passed")

    def test_stop_on_max_iterations(self):
        stop = StopCondition(max_iterations=3)
        should, reason = stop.check(
            iteration=3, verify_status="uncertain",
            policy_results=[], stage_status={},
        )
        self.assertTrue(should)
        self.assertEqual(reason, "max_iterations_reached")

    def test_stop_on_all_policy_rejected(self):
        stop = StopCondition(max_iterations=5)
        should, reason = stop.check(
            iteration=1, verify_status="uncertain",
            policy_results=[{"allowed": False}, {"allowed": False}],
            stage_status={},
        )
        self.assertTrue(should)
        self.assertEqual(reason, "all_actions_policy_rejected")

    def test_stop_on_same_failure_twice(self):
        stop = StopCondition(max_iterations=5)
        stop.check(iteration=1, verify_status="uncertain",
                   policy_results=[{"allowed": True}], stage_status={},
                   last_error="timeout")
        should, reason = stop.check(
            iteration=2, verify_status="uncertain",
            policy_results=[{"allowed": True}], stage_status={},
            last_error="timeout",
        )
        self.assertTrue(should)
        self.assertEqual(reason, "same_failure_repeats_twice")

    def test_no_stop_when_ok(self):
        stop = StopCondition(max_iterations=5)
        should, reason = stop.check(
            iteration=1, verify_status="uncertain",
            policy_results=[{"allowed": True}], stage_status={},
        )
        self.assertFalse(should)

    def test_reset_clears_history(self):
        stop = StopCondition(max_iterations=5)
        stop.check(iteration=1, verify_status="uncertain",
                   policy_results=[{"allowed": True}], stage_status={},
                   last_error="timeout")
        stop.reset()
        should, _ = stop.check(
            iteration=2, verify_status="uncertain",
            policy_results=[{"allowed": True}], stage_status={},
            last_error="timeout",
        )
        self.assertFalse(should)

    def test_stop_on_human_intervention(self):
        stop = StopCondition(max_iterations=5)
        should, reason = stop.check(
            iteration=1, verify_status="uncertain",
            policy_results=[{"allowed": True}],
            stage_status={"env_deploy": {"status": "requires_secret"}},
        )
        self.assertTrue(should)
        self.assertEqual(reason, "requires_human_intervention")


class TestAgentArtifactWriter(unittest.TestCase):
    """Tests for AgentArtifactWriter."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = AgentArtifactWriter(Path(self.tmpdir))

    def test_write_step(self):
        path = self.writer.write_step({"step_id": 1, "stage": "runner"})
        self.assertTrue(path.exists())
        lines = path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 1)

    def test_write_state(self):
        path = self.writer.write_state({"task_id": "test"})
        self.assertTrue(path.exists())

    def test_write_plan(self):
        path = self.writer.write_plan({"strategy": "gradio"})
        self.assertTrue(path.exists())

    def test_write_plan_revision(self):
        path = self.writer.write_plan_revision({"revision": 1})
        self.assertTrue(path.exists())

    def test_write_decision(self):
        path = self.writer.write_decision("runner", 1, {"tool_call": "test"})
        self.assertTrue(path.exists())

    def test_write_policy_check(self):
        path = self.writer.write_policy_check("runner", 1, {"allowed": True})
        self.assertTrue(path.exists())

    def test_write_tool_result(self):
        path = self.writer.write_tool_result("runner", 1, {"executed": True})
        self.assertTrue(path.exists())

    def test_write_repair_artifact(self):
        path = self.writer.write_repair_artifact("repair_hypothesis", {"hypothesis": "test"})
        self.assertTrue(path.exists())
        self.assertIn("repairs", str(path))


class TestDeploymentAgentLoop(unittest.TestCase):
    """Tests for DeploymentAgentLoop."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "run"
        self.run_dir.mkdir()

    def test_import(self):
        """Verify DeploymentAgentLoop can be imported."""
        self.assertIsNotNone(DeploymentAgentLoop)

    def test_determine_start_stage_with_failure(self):
        loop = DeploymentAgentLoop()
        results = {
            "analyze": {"status": "passed"},
            "env_solve": {"status": "failed"},
        }
        stage = loop._determine_start_stage(results)
        self.assertEqual(stage, "env_solve")

    def test_determine_start_stage_all_passed(self):
        loop = DeploymentAgentLoop()
        results = {
            "analyze": {"status": "passed"},
            "env_solve": {"status": "passed"},
            "runner": {"status": "passed"},
        }
        stage = loop._determine_start_stage(results)
        self.assertEqual(stage, "verify")

    def test_next_stage_progression(self):
        loop = DeploymentAgentLoop()
        state = AgentState(current_stage="analyze")
        result = MagicMock()
        result.execution = {}
        next_stage = loop._next_stage(state, result)
        self.assertEqual(next_stage, "resource_plan")

    def test_next_stage_from_repair_resume(self):
        loop = DeploymentAgentLoop()
        state = AgentState(current_stage="repair")
        result = MagicMock()
        result.execution = {"resume_from_stage": "env_deploy"}
        next_stage = loop._next_stage(state, result)
        self.assertEqual(next_stage, "env_deploy")

    def test_build_result(self):
        loop = DeploymentAgentLoop()
        state = AgentState(task_id="test", run_dir=str(self.run_dir))
        state.iteration = 2
        state.stop_reason = "verify_passed"
        result = loop._build_result(state, self.run_dir)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["iteration_count"], 3)
        self.assertEqual(result["stop_reason"], "verify_passed")


if __name__ == "__main__":
    unittest.main()
