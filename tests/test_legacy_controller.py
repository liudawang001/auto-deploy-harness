"""LegacyController tests: verify routing logic and result adaptation.

The LegacyController must preserve the existing TaskRunner priority:
1. agent_plan_first → plan_first
2. agent_enable_runtime_loop + gated_actor + primary → agent_loop
3. else → pipeline
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.controllers.legacy import LegacyController
from auto_harness.orchestrator import TaskRunner


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
        config.agent_plan_first = False
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
        config.agent_plan_first = False
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

    def test_plan_first_resume_forwards_approval_input(self, tmp_path):
        config = MagicMock()
        config.agent_plan_first = True
        run_plan_first = MagicMock()
        resume_existing = MagicMock()
        result_adapter = MagicMock(return_value=DeploymentResult(
            task_id="test_task", status="interrupted",
            stop_reason="repository_command_approval_required",
            controller="legacy",
        ))
        ctrl = LegacyController(
            config=config,
            run_plan_first=run_plan_first,
            run_agent_loop=MagicMock(),
            run_pipeline=MagicMock(),
            resume_existing=resume_existing,
            result_adapter=result_adapter,
        )
        ctx = make_context(tmp_path, dry_run=False)
        decision = {"decision": "approve", "approval_id": "approval_cmd"}

        ctrl.resume(ctx, resume_input=decision)

        run_plan_first.assert_called_once_with(
            "test_task", dry_run=False, resume_input=decision,
        )
        resume_existing.assert_not_called()
        result_adapter.assert_called_once_with(
            ctx, controller="legacy", strategy="plan_first",
        )


def test_legacy_command_approval_is_bound_and_persisted(tmp_path):
    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    request = {
        "approval_id": "approval_op_cmd_test",
        "operation_id": "op_cmd_test",
        "request_hash": "request_hash_test",
        "candidate_id": "cmd_test",
        "allowed_decisions": ["approve", "reject"],
    }
    (reports / "plan_first_result.json").write_text(
        json.dumps({"approval_request": request}), encoding="utf-8",
    )
    decision = {
        "approval_id": request["approval_id"],
        "operation_id": request["operation_id"],
        "request_hash": request["request_hash"],
        "decision": "approve",
        "reviewer": "tester",
    }

    approval, rejected = TaskRunner._legacy_command_approval(run_dir, decision)

    assert rejected == []
    assert approval["request"] == request
    assert approval["decision"]["decision"] == "approve"
    stored = json.loads(
        (run_dir / "approvals" / (request["approval_id"] + ".json")).read_text(
            encoding="utf-8"
        )
    )
    assert stored["status"] == "resolved"


def test_legacy_command_rejection_excludes_candidate(tmp_path):
    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    request = {
        "approval_id": "approval_op_cmd_test",
        "operation_id": "op_cmd_test",
        "request_hash": "request_hash_test",
        "candidate_id": "cmd_test",
        "allowed_decisions": ["approve", "reject"],
    }
    (reports / "plan_first_result.json").write_text(
        json.dumps({"approval_request": request}), encoding="utf-8",
    )

    approval, rejected = TaskRunner._legacy_command_approval(run_dir, {
        "approval_id": request["approval_id"],
        "operation_id": request["operation_id"],
        "request_hash": request["request_hash"],
        "decision": "reject",
    })

    assert approval is None
    assert rejected == ["cmd_test"]
