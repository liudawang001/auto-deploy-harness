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
        # When all passed, start from beginning for full execution
        self.assertEqual(stage, "analyze")

    def test_build_result(self):
        loop = DeploymentAgentLoop()
        state = AgentState(task_id="test", run_dir=str(self.run_dir))
        state.iteration = 2
        state.stop_reason = "verify_passed"
        result = loop._build_result(state, self.run_dir)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["iteration_count"], 3)
        self.assertEqual(result["stop_reason"], "verify_passed")


# ------------------------------------------------------------------
# Phase 3: AgentLoop as Primary Controller Tests
# ------------------------------------------------------------------

from auto_harness.agent_runtime.loop import STAGE_TOOLS
from auto_harness.agent_runtime.stage_executor import StageExecutionResult
from auto_harness.agent_runtime.stage_schemas import PIPELINE_STAGES


def _make_stage_executor_mock(results_map=None):
    """Create a mock AgentStageExecutor that returns predefined results."""
    mock = MagicMock()
    default_results = {
        "analyze": StageExecutionResult(
            stage="analyze", before_status="", after_status="passed",
            result={"framework": "gradio"}, changed=True,
        ),
        "resource_plan": StageExecutionResult(
            stage="resource_plan", before_status="", after_status="passed",
            result={"gpu": "T4"}, changed=True,
        ),
        "env_solve": StageExecutionResult(
            stage="env_solve", before_status="", after_status="passed",
            result={"solution": "pip install"}, changed=True,
        ),
        "env_deploy": StageExecutionResult(
            stage="env_deploy", before_status="", after_status="passed",
            result={"deployed": True}, changed=True,
        ),
        "model_prepare": StageExecutionResult(
            stage="model_prepare", before_status="", after_status="passed",
            result={"model_path": "/models/test"}, changed=True,
        ),
        "runner": StageExecutionResult(
            stage="runner", before_status="", after_status="passed",
            result={"candidates": []}, changed=True,
        ),
        "verify": StageExecutionResult(
            stage="verify", before_status="", after_status="passed",
            result={"verified": True}, changed=True,
            evidence_paths=["/evidence/1.json"],
        ),
    }
    if results_map:
        default_results.update(results_map)

    def execute_side_effect(**kwargs):
        stage = kwargs.get("stage", "")
        return default_results.get(stage, StageExecutionResult(
            stage=stage, before_status="", after_status="passed",
            result={}, changed=False,
        ))

    mock.execute_stage.side_effect = execute_side_effect
    return mock


class TestAgentLoopPrimaryController(unittest.TestCase):
    """Phase 3: AgentLoop is the primary deployment controller."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "run"
        self.run_dir.mkdir()

    def test_agent_runtime_loop_calls_stage_executor_in_order(self):
        """AgentLoop should call stage_executor.execute_stage for each pipeline stage."""
        executor = _make_stage_executor_mock()
        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=10, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-order",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        # Should have called execute_stage for each stage in order
        calls = executor.execute_stage.call_args_list
        called_stages = [call.kwargs.get("stage") for call in calls]

        # Should call at least analyze and verify
        self.assertIn("analyze", called_stages)
        self.assertIn("verify", called_stages)
        # Should be in pipeline order
        analyze_idx = called_stages.index("analyze")
        verify_idx = called_stages.index("verify")
        self.assertLess(analyze_idx, verify_idx)

    def test_agent_runtime_loop_stops_on_verify_passed(self):
        """AgentLoop should stop when verify stage returns passed."""
        executor = _make_stage_executor_mock()
        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=10, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-verify-stop",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        self.assertEqual(result["stop_reason"], "verify_passed")
        self.assertEqual(result["verify"]["status"], "passed")

    def test_agent_runtime_loop_records_before_after_status(self):
        """AgentLoop should record before/after status for each stage."""
        executor = _make_stage_executor_mock({
            "analyze": StageExecutionResult(
                stage="analyze", before_status="failed", after_status="passed",
                result={"framework": "gradio"}, changed=True,
            ),
        })
        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=10, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-status",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={"analyze": {"status": "failed"}},
            dry_run=True,
        )

        # stage_status should have analyze with passed
        self.assertEqual(result["stage_status"]["analyze"]["status"], "passed")

    def test_agent_runtime_loop_enters_repair_on_failed_stage(self):
        """AgentLoop should enter repair loop when a stage fails."""
        # Make env_deploy fail
        executor = _make_stage_executor_mock({
            "env_deploy": StageExecutionResult(
                stage="env_deploy", before_status="", after_status="failed",
                result={}, changed=False, error="pip install failed",
            ),
        })

        # Mock provider to return a repair decision
        provider = MagicMock()
        provider.chat.return_value = json.dumps({
            "action": "apply_repair",
            "tool": "apply_repair",
            "tool_input": {"fix": "install dependency"},
            "reason": "fix pip install",
        })

        loop = DeploymentAgentLoop(
            provider=provider,
            stage_executor=executor,
            max_iterations=10,
            stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-repair",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        # Should have recorded a repair
        self.assertGreaterEqual(result["repair_count"], 1)

    def test_agent_runtime_loop_stops_on_failed_stage_without_provider(self):
        """AgentLoop should stop when a stage fails and no provider for repair."""
        executor = _make_stage_executor_mock({
            "env_deploy": StageExecutionResult(
                stage="env_deploy", before_status="", after_status="failed",
                result={}, changed=False, error="pip install failed",
            ),
        })

        loop = DeploymentAgentLoop(
            stage_executor=executor,
            max_iterations=10,
            stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-fail-no-provider",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        # Should have stopped due to failed stage
        self.assertIn("stop_reason", result)

    def test_agent_runtime_loop_does_not_run_full_pipeline_first(self):
        """When position=primary, AgentLoop should use stage_executor directly."""
        executor = _make_stage_executor_mock()
        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=15, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-primary",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        # Should have completed through verify
        self.assertEqual(result["stop_reason"], "verify_passed")

        # All stages should have been executed through stage_executor
        self.assertGreaterEqual(executor.execute_stage.call_count, 7)


class TestAgentLoopStageTools(unittest.TestCase):
    """Test STAGE_TOOLS mapping."""

    def test_verify_has_verify_tools(self):
        from auto_harness.agent_runtime.stage_schemas import VERIFY_TOOLS
        self.assertIn("verify", STAGE_TOOLS)
        for tool in VERIFY_TOOLS:
            self.assertIn(tool, STAGE_TOOLS["verify"])

    def test_all_pipeline_stages_have_tools(self):
        for stage in PIPELINE_STAGES:
            if stage == "report":
                continue
            self.assertIn(stage, STAGE_TOOLS, f"Stage {stage} missing from STAGE_TOOLS")

    def test_repair_tools_include_resume(self):
        self.assertIn("resume_from_stage", STAGE_TOOLS["repair"])


class TestAgentLoopEdgeCases(unittest.TestCase):
    """Edge cases for the agent loop."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "run"
        self.run_dir.mkdir()

    def test_loop_with_no_stage_executor(self):
        """Loop should handle missing stage executor gracefully."""
        loop = DeploymentAgentLoop(max_iterations=5, stop_on_verify_pass=True)

        result = loop.run(
            task_id="test-no-executor",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        self.assertEqual(result["status"], "completed")

    def test_loop_with_max_iterations(self):
        """Loop should stop at max_iterations."""
        def execute_side_effect(**kwargs):
            stage = kwargs.get("stage", "")
            return StageExecutionResult(
                stage=stage, before_status="", after_status="uncertain",
                result={}, changed=False,
            )

        executor = MagicMock()
        executor.execute_stage.side_effect = execute_side_effect

        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=3, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-max-iter",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        self.assertLessEqual(result["iteration_count"], 3)

    def test_loop_records_llm_helped_false_without_provider(self):
        """Loop should track llm_helped=False when no provider."""
        executor = _make_stage_executor_mock()
        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=10, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-llm-helped",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        self.assertFalse(result["llm_helped"])

    def test_loop_creates_artifacts(self):
        """Loop should create agent artifacts."""
        executor = _make_stage_executor_mock()
        loop = DeploymentAgentLoop(
            stage_executor=executor, max_iterations=10, stop_on_verify_pass=True,
        )

        result = loop.run(
            task_id="test-artifacts",
            run_dir=self.run_dir,
            repo_dir=self.run_dir,
            initial_results={},
            dry_run=True,
        )

        self.assertIn("agent_steps", result["artifacts"])
        self.assertIn("agent_state", result["artifacts"])


if __name__ == "__main__":
    unittest.main()
