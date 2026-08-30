"""Controller contract tests: verify the DeploymentController Protocol.

Both LegacyController and LangGraphController must satisfy the same
interface contract. These tests verify the Protocol is correctly
implemented and the factory works.
"""
import pytest
from pathlib import Path

from auto_harness.controllers.base import DeploymentContext, DeploymentController, DeploymentResult
from auto_harness.controllers.factory import ControllerUnavailableError, create_controller
from auto_harness.controllers.legacy import LegacyController


class FakeConfig:
    """Minimal config stub for LegacyController."""
    agent_plan_first = False
    agent_enable_runtime_loop = False
    agent_mode = "off"
    agent_runtime_loop_position = "primary"


class FakeDependencies:
    """Stub dependencies for the factory."""
    def build_legacy_controller(self):
        return LegacyController(
            config=FakeConfig(),
            run_plan_first=lambda task_id, dry_run=True: None,
            run_agent_loop=lambda task_id, dry_run=True: None,
            run_pipeline=lambda task_id, dry_run=True, start_stage="analyze": task_id,
            resume_existing=lambda task_id, dry_run=True, resume_input=None: None,
            result_adapter=lambda ctx, controller="legacy", strategy="pipeline": DeploymentResult(
                task_id=ctx.task_id, status="completed", stop_reason="ok", controller=controller,
            ),
        )

    def graph_dependencies(self):
        from auto_harness.controllers.langgraph_deps import LangGraphControllerDependencies
        # Return a mock that satisfies LangGraphController.__init__
        from unittest.mock import MagicMock
        return MagicMock()


def make_context(tmp_path, dry_run=True, task_id="test_task", resume_input=None):
    return DeploymentContext(
        task_id=task_id,
        run_dir=str(tmp_path / "runs" / task_id),
        repo_dir=str(tmp_path / "runs" / task_id / "workspace" / "repo"),
        dry_run=dry_run,
        runtime_policy={"allow_dependency_install": False, "allow_service_start": False},
        resume_input=resume_input,
    )


class TestDeploymentContext:
    def test_context_fields(self, tmp_path):
        ctx = make_context(tmp_path)
        assert ctx.task_id == "test_task"
        assert ctx.dry_run is True
        assert ctx.resume_input is None

    def test_context_with_resume_input(self, tmp_path):
        ctx = make_context(tmp_path, resume_input={"start_stage": "runner"})
        assert ctx.resume_input == {"start_stage": "runner"}


class TestDeploymentResult:
    def test_result_defaults(self):
        result = DeploymentResult(
            task_id="t1", status="completed", stop_reason="ok", controller="legacy",
        )
        assert result.verify_status == ""
        assert result.artifacts == {}
        assert result.metrics == {}

    def test_result_with_all_fields(self):
        result = DeploymentResult(
            task_id="t1", status="completed", stop_reason="verify_passed",
            controller="langgraph", verify_status="passed",
            artifacts={"plan": "reports/plan.json"},
            metrics={"replan_count": 1},
        )
        assert result.verify_status == "passed"
        assert result.metrics["replan_count"] == 1


class TestControllerFactory:
    def test_create_legacy(self):
        deps = FakeDependencies()
        ctrl = create_controller("legacy", deps)
        assert ctrl.name == "legacy"

    def test_create_langgraph_raises_unavailable(self):
        """When langgraph is not installed, factory raises ControllerUnavailableError.

        Note: if langgraph IS installed (as in CI), this test verifies
        that the import succeeds and the controller is created.
        """
        deps = FakeDependencies()
        try:
            ctrl = create_controller("langgraph", deps)
            # If langgraph is installed, we get a LangGraphController
            assert ctrl.name == "langgraph"
        except ControllerUnavailableError:
            # If langgraph is not installed, we get the expected error
            pass

    def test_create_unknown_raises_value_error(self):
        deps = FakeDependencies()
        with pytest.raises(ValueError, match="unsupported controller"):
            create_controller("unknown", deps)


class TestLangGraphCompletedResult:
    def test_completed_checkpoint_requires_verified_pass(self):
        from auto_harness.controllers.langgraph_deps import LangGraphControllerDependencies

        result = LangGraphControllerDependencies(object()).completed_result({
            "task_id": "t1",
            "verify_status": "uncertain",
            "dry_run": False,
        })

        assert result.status == "stopped"
        assert result.stop_reason == "graph_ended_without_verify_pass"

    def test_verified_checkpoint_is_completed(self):
        from auto_harness.controllers.langgraph_deps import LangGraphControllerDependencies

        result = LangGraphControllerDependencies(object()).completed_result({
            "task_id": "t1",
            "verify_status": "pass",
            "dry_run": False,
        })

        assert result.status == "completed"
        assert result.stop_reason == "already_completed"


class TestLegacyControllerContract:
    def test_implements_protocol(self):
        """LegacyController must satisfy the DeploymentController Protocol."""
        ctrl = FakeDependencies().build_legacy_controller()
        assert hasattr(ctrl, "name")
        assert hasattr(ctrl, "run")
        assert hasattr(ctrl, "resume")
        assert ctrl.name == "legacy"

    def test_run_returns_deployment_result(self, tmp_path):
        ctrl = FakeDependencies().build_legacy_controller()
        ctx = make_context(tmp_path)
        result = ctrl.run(ctx)
        assert isinstance(result, DeploymentResult)
        assert result.controller == "legacy"

    def test_resume_returns_deployment_result(self, tmp_path):
        ctrl = FakeDependencies().build_legacy_controller()
        ctx = make_context(tmp_path)
        result = ctrl.resume(ctx)
        assert isinstance(result, DeploymentResult)
        assert result.controller == "legacy"
