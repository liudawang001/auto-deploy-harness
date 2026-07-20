"""LegacyController tests: verify routing logic and result adaptation.

The LegacyController must preserve the existing TaskRunner priority:
1. agent_plan_first → plan_first
2. agent_enable_runtime_loop + gated_actor + primary → agent_loop
3. else → pipeline
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.controllers.legacy import LegacyController


def make_context(tmp_path, task_id="test_task", dry_run=True):
    return DeploymentContext(
        task_id=task_id,
        run_dir=str(tmp_path / "runs" / task_id),
        repo_dir=str(tmp_path / "runs" / task_id / "workspace" / "repo"),
        dry_run=dry_run,
        runtime_policy={"allow_dependency_install": False, "allow_service_start": False},
    )


class TestLegacyControllerRouting:
    def test_plan_first_highest_priority(self, tmp_path):
        """When agent_plan_first is True, plan_first callable is used."""
        config = MagicMock()
        config.agent_plan_first = True
        config.agent_enable_runtime_loop = True
        config.agent_mode = "gated_actor"
        config.agent_runtime_loop_position = "primary"

        run_plan_first = MagicMock()
        run_agent_loop = MagicMock()
        run_pipeline = MagicMock()
        resume_existing = MagicMock()
        result_adapter = MagicMock(return_value=DeploymentResult(
            task_id="test_task", status="completed", stop_reason="ok", controller="legacy",
        ))

        ctrl = LegacyController(
            config=config,
            run_plan_first=run_plan_first,
            run_agent_loop=run_agent_loop,
            run_pipeline=run_pipeline,
            resume_existing=resume_existing,
            result_adapter=result_adapter,
        )

        ctx = make_context(tmp_path)
        ctrl.run(ctx)

        run_plan_first.assert_called_once_with("test_task", dry_run=True)
        run_agent_loop.assert_not_called()
        run_pipeline.assert_not_called()
        result_adapter.assert_called_once_with(ctx, controller="legacy", strategy="plan_first")

    def test_agent_loop_second_priority(self, tmp_path):
        """When agent runtime loop is primary, agent_loop callable is used."""
        config = MagicMock()
        config.agent_plan_first = False
        config.agent_enable_runtime_loop = True
        config.agent_mode = "gated_actor"
        config.agent_runtime_loop_position = "primary"

        run_plan_first = MagicMock()
        run_agent_loop = MagicMock()
        run_pipeline = MagicMock()
        resume_existing = MagicMock()
        result_adapter = MagicMock(return_value=DeploymentResult(
            task_id="test_task", status="completed", stop_reason="ok", controller="legacy",
        ))

        ctrl = LegacyController(
            config=config,
            run_plan_first=run_plan_first,
            run_agent_loop=run_agent_loop,
            run_pipeline=run_pipeline,
            resume_existing=resume_existing,
            result_adapter=result_adapter,
        )

        ctx = make_context(tmp_path)
        ctrl.run(ctx)

        run_plan_first.assert_not_called()
        run_agent_loop.assert_called_once_with("test_task", dry_run=True)
        run_pipeline.assert_not_called()
        result_adapter.assert_called_once_with(ctx, controller="legacy", strategy="agent_loop")

    def test_pipeline_fallback(self, tmp_path):
        """When neither plan_first nor agent_loop conditions are met, pipeline is used."""
        config = MagicMock()
        config.agent_plan_first = False
        config.agent_enable_runtime_loop = False
        config.agent_mode = "off"
        config.agent_runtime_loop_position = "primary"

        run_plan_first = MagicMock()
        run_agent_loop = MagicMock()
        run_pipeline = MagicMock()
        resume_existing = MagicMock()
        result_adapter = MagicMock(return_value=DeploymentResult(
            task_id="test_task", status="completed", stop_reason="ok", controller="legacy",
        ))

        ctrl = LegacyController(
            config=config,
            run_plan_first=run_plan_first,
            run_agent_loop=run_agent_loop,
            run_pipeline=run_pipeline,
            resume_existing=resume_existing,
            result_adapter=result_adapter,
        )

        ctx = make_context(tmp_path)
        ctrl.run(ctx)

        run_plan_first.assert_not_called()
        run_agent_loop.assert_not_called()
        run_pipeline.assert_called_once_with("test_task", dry_run=True)
        result_adapter.assert_called_once_with(ctx, controller="legacy", strategy="pipeline")

    def test_resume_delegates_to_resume_existing(self, tmp_path):
        """Resume calls resume_existing with task_id, dry_run, and resume_input."""
        config = MagicMock()
        resume_existing = MagicMock()
        result_adapter = MagicMock(return_value=DeploymentResult(
            task_id="test_task", status="completed", stop_reason="ok", controller="legacy",
        ))

        ctrl = LegacyController(
            config=config,
            run_plan_first=MagicMock(),
            run_agent_loop=MagicMock(),
            run_pipeline=MagicMock(),
            resume_existing=resume_existing,
            result_adapter=result_adapter,
        )

        ctx = make_context(tmp_path)
        resume_input = {"start_stage": "runner"}
        ctrl.resume(ctx, resume_input=resume_input)

        resume_existing.assert_called_once_with(
            "test_task", dry_run=True, resume_input=resume_input,
        )
        result_adapter.assert_called_once_with(ctx, controller="legacy", strategy="resume")

    def test_resume_without_input(self, tmp_path):
        """Resume with no resume_input passes None."""
        config = MagicMock()
        resume_existing = MagicMock()
        result_adapter = MagicMock(return_value=DeploymentResult(
            task_id="test_task", status="completed", stop_reason="ok", controller="legacy",
        ))

        ctrl = LegacyController(
            config=config,
            run_plan_first=MagicMock(),
            run_agent_loop=MagicMock(),
            run_pipeline=MagicMock(),
            resume_existing=resume_existing,
            result_adapter=result_adapter,
        )

        ctx = make_context(tmp_path)
        ctrl.resume(ctx)

        resume_existing.assert_called_once_with(
            "test_task", dry_run=True, resume_input=None,
        )
