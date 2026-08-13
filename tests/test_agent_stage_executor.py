"""Tests for AgentStageExecutor (Phase 2).

Covers:
- execute_stage calls correct module for each stage
- before/after status returned correctly
- dry_run vs execute mode
- error handling when module not available
- repair overlay merged into env_deploy
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.agent_runtime.stage_executor import AgentStageExecutor, StageExecutionResult


class TestStageExecutionResult(unittest.TestCase):
    """Test StageExecutionResult data class."""

    def test_default_values(self):
        r = StageExecutionResult(stage="runner", before_status="", after_status="passed", result={}, changed=True)
        self.assertEqual(r.stage, "runner")
        self.assertTrue(r.changed)
        self.assertEqual(r.evidence_paths, [])
        self.assertEqual(r.error, "")

    def test_with_evidence(self):
        r = StageExecutionResult(
            stage="verify", before_status="uncertain", after_status="passed",
            result={}, changed=True, evidence_paths=["/tmp/evidence.json"],
        )
        self.assertEqual(len(r.evidence_paths), 1)


class TestStageExecutorCallsModules(unittest.TestCase):
    """Test that AgentStageExecutor delegates to real stage modules."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir) / "run"
        self.run_dir.mkdir(parents=True)
        self.repo_dir = Path(self.tmpdir) / "repo"
        self.repo_dir.mkdir(parents=True)

    def test_stage_executor_calls_env_solve_module(self):
        """env_solve should delegate to EnvSolveModule.solve()."""
        (self.repo_dir / "requirements.txt").write_text("gradio\n", encoding="utf-8")
        executor = AgentStageExecutor()
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="env_solve",
            state={},
            analysis={
                "frameworks": ["gradio"],
                "install_plan": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ],
            },
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.stage, "env_solve")
        self.assertIn(result.after_status, ("passed", "uncertain"))
        self.assertIn("constraints", result.result.get("data", {}))

    def test_resource_plan_defers_optional_models_when_plan_marks_not_required(self):
        (self.repo_dir / "README.md").write_text(
            "Optional local model: CUDA https://huggingface.co/org/demo-model\n",
            encoding="utf-8",
        )
        result = AgentStageExecutor().execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="resource_plan",
            state={},
            analysis={
                "frameworks": ["transformers"],
                "model_assets": {"required": False, "strategy": "none"},
            },
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        data = result.result["data"]
        self.assertEqual(result.after_status, "passed")
        self.assertFalse(data["gpu_required"])
        self.assertEqual(data["model_assets"], [])
        self.assertEqual(data["external_tokens"], [])

    def test_resource_plan_treats_empty_model_assets_as_no_required_download(self):
        (self.repo_dir / "README.md").write_text(
            "Optional CUDA model https://huggingface.co/org/demo-model\n",
            encoding="utf-8",
        )
        result = AgentStageExecutor().execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="resource_plan",
            state={},
            analysis={"frameworks": ["transformers"], "model_assets": {}},
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertFalse(result.result["data"]["gpu_required"])
        self.assertEqual(result.result["data"]["model_assets"], [])

    def test_stage_executor_calls_env_deploy_module(self):
        """env_deploy should delegate to EnvDeployModule.deploy()."""
        executor = AgentStageExecutor()
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="env_deploy",
            state={},
            analysis={},
            resource_data={},
            deploy_analysis={
                "install_plan": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "gradio"],
                ],
            },
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.stage, "env_deploy")
        self.assertIn(result.after_status, ("passed", "uncertain"))

    def test_stage_executor_calls_model_prepare_module(self):
        """model_prepare should delegate to ModelPrepareModule.prepare()."""
        from auto_harness.assets import ModelCache, HuggingFaceDownloader, ModelScopeDownloader
        cache = ModelCache(Path(self.tmpdir) / "cache")
        model_prepare_module = type("MP", (), {
            "prepare": lambda self, run_dir, resource_plan, **kw: type("R", (), {
                "status": "passed", "data": {"assets": []}, "summary": "ok", "evidence": [],
                "__dict__": {"status": "passed", "data": {"assets": []}, "summary": "ok"},
            })()
        })()
        executor = AgentStageExecutor(model_prepare=model_prepare_module)
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="model_prepare",
            state={},
            analysis={},
            resource_data={"model_assets": []},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.stage, "model_prepare")
        self.assertEqual(result.after_status, "passed")

    def test_stage_executor_calls_runner_module(self):
        """runner should delegate to RunnerModule.run()."""
        executor = AgentStageExecutor()
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="runner",
            state={},
            analysis={},
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.stage, "runner")
        # No candidates -> uncertain
        self.assertEqual(result.after_status, "uncertain")

    def test_stage_executor_calls_verify_module(self):
        """verify should delegate to VerifyModule.verify()."""
        executor = AgentStageExecutor()
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="verify",
            state={},
            analysis={},
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.stage, "verify")
        # No service running -> uncertain
        self.assertEqual(result.after_status, "uncertain")

    def test_stage_executor_returns_before_after_status(self):
        """StageExecutionResult should have correct before/after status."""
        executor = AgentStageExecutor()
        state = {"stage_results": {"runner": {"status": "failed"}}}
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="runner",
            state=state,
            analysis={},
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.before_status, "failed")
        self.assertEqual(result.after_status, "uncertain")
        self.assertTrue(result.changed)

    def test_stage_executor_handles_unknown_stage(self):
        """Unknown stage should return failed status with error."""
        executor = AgentStageExecutor()
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="nonexistent",
            state={},
            analysis={},
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.after_status, "failed")
        self.assertTrue(result.changed)
        self.assertIn("unknown stage", result.error)

    def test_stage_executor_handles_module_exception(self):
        """Module exception should be caught and returned as error."""
        class BrokenAnalyzer:
            def analyze(self, repo_dir):
                raise RuntimeError("module broken")

        executor = AgentStageExecutor()
        # Patch the _execute_analyze to use a broken module
        original = executor._execute_analyze
        executor._execute_analyze = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("module broken"))
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="analyze",
            state={},
            analysis={},
            resource_data={},
            deploy_analysis={},
            runner_data={},
            dry_run=True,
        )
        self.assertEqual(result.after_status, "failed")
        self.assertIn("module broken", result.error)

    def test_stage_executor_merges_repair_overlay(self):
        """env_deploy should merge repair overlay install commands."""
        executor = AgentStageExecutor()
        repair_overlay = {
            "active": True,
            "install_commands": [["pip", "install", "pydantic<2"]],
        }
        result = executor.execute_stage(
            task_id="test",
            run_dir=self.run_dir,
            repo_dir=self.repo_dir,
            stage="env_deploy",
            state={},
            analysis={},
            resource_data={},
            deploy_analysis={
                "install_plan": [
                    ["python3", "-m", "venv", ".venv"],
                ],
            },
            runner_data={},
            dry_run=True,
            repair_overlay=repair_overlay,
        )
        self.assertEqual(result.stage, "env_deploy")
        # The overlay should have been merged into the install plan
        commands = result.result.get("data", {}).get("commands", [])
        self.assertTrue(any("pydantic" in str(cmd) for cmd in commands))


if __name__ == "__main__":
    unittest.main()
