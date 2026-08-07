"""Task 2 tests: CLI default controller and resolve_controller.

Verifies:
1. CLI deploy without --controller passes None to runner
2. create_spec resolves config default to langgraph
3. create_spec explicit legacy overrides default
4. resume rejects controller switch
5. resolve_controller enforces all rules
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from auto_harness.config import HarnessConfig
from auto_harness.controllers.factory import resolve_controller, VALID_CONTROLLERS
from auto_harness.orchestrator import TaskRunner
from auto_harness.cli import build_parser, main


class TestResolveController:
    """Unit tests for resolve_controller."""

    def test_deploy_no_explicit_uses_default(self):
        assert resolve_controller(explicit=None, configured_default="langgraph") == "langgraph"

    def test_deploy_explicit_overrides_default(self):
        assert resolve_controller(explicit="legacy", configured_default="langgraph") == "legacy"

    def test_resume_no_explicit_uses_stored(self):
        assert resolve_controller(
            explicit=None, configured_default="langgraph", stored="legacy", is_resume=True
        ) == "legacy"

    def test_resume_explicit_matches_stored(self):
        assert resolve_controller(
            explicit="langgraph", configured_default="langgraph", stored="langgraph", is_resume=True
        ) == "langgraph"

    def test_resume_switch_rejected(self):
        with pytest.raises(ValueError, match="controller_switch_on_resume"):
            resolve_controller(
                explicit="legacy", configured_default="langgraph", stored="langgraph", is_resume=True
            )

    def test_unsupported_configured_default(self):
        with pytest.raises(ValueError, match="unsupported configured controller"):
            resolve_controller(explicit=None, configured_default="invalid")

    def test_unsupported_explicit(self):
        with pytest.raises(ValueError, match="unsupported controller"):
            resolve_controller(explicit="invalid", configured_default="langgraph")

    def test_resume_without_stored_raises(self):
        with pytest.raises(ValueError, match="stored controller is required"):
            resolve_controller(explicit=None, configured_default="langgraph", stored=None, is_resume=True)

    def test_unsupported_stored(self):
        with pytest.raises(ValueError, match="unsupported stored controller"):
            resolve_controller(explicit=None, configured_default="langgraph", stored="bad", is_resume=True)


class TestCreateSpecResolvesDefault:
    """create_spec uses resolve_controller to determine controller."""

    def test_create_spec_resolves_config_default_langgraph(self, tmp_path):
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")
        runner = TaskRunner(config)
        spec = runner.create_spec("https://example.com/repo", "demo")
        assert spec.controller == "langgraph"

    def test_create_spec_explicit_legacy_overrides_default(self, tmp_path):
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")
        runner = TaskRunner(config)
        spec = runner.create_spec("https://example.com/repo", "demo", controller="legacy")
        assert spec.controller == "legacy"

    def test_create_spec_explicit_langgraph(self, tmp_path):
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")
        runner = TaskRunner(config)
        spec = runner.create_spec("https://example.com/repo", "demo", controller="langgraph")
        assert spec.controller == "langgraph"


class TestResumeRejectsControllerSwitch:
    """resume() rejects switching controller via resolve_controller."""

    def test_resume_rejects_controller_switch(self, tmp_path):
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")
        runner = TaskRunner(config)
        # Create a task with langgraph controller
        spec = runner.create_spec("https://example.com/repo", "demo")
        runner.store.create_task(spec)
        # Resume with legacy should fail
        with pytest.raises(ValueError, match="controller_switch_on_resume"):
            runner.resume(spec.task_id, dry_run=True, controller="legacy")


class TestCLIDeployWithoutController:
    """CLI deploy without --controller passes None to runner."""

    def test_cli_deploy_without_controller_passes_none(self, tmp_path):
        """When --controller is not specified, None is passed (not "legacy")."""
        config = HarnessConfig(
            runs_dir=str(tmp_path / "runs"),
            default_controller="langgraph",
            agent_provider="mock",
            agent_plan_first_provider="mock",
        )
        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.deploy.return_value = "task_123"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main(["deploy", "--repo", "https://example.com/repo"])
            assert exit_code == 0
            # Verify controller=None was passed (not "legacy")
            deploy_kwargs = mock_runner.deploy.call_args.kwargs
            assert deploy_kwargs.get("controller") is None

    def test_cli_deploy_with_explicit_controller(self, tmp_path):
        """When --controller legacy is specified, "legacy" is passed."""
        config = HarnessConfig(
            runs_dir=str(tmp_path / "runs"),
            default_controller="langgraph",
            agent_provider="mock",
            agent_plan_first_provider="mock",
        )
        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.deploy.return_value = "task_123"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main(["deploy", "--repo", "https://example.com/repo", "--controller", "legacy"])
            assert exit_code == 0
            deploy_kwargs = mock_runner.deploy.call_args.kwargs
            assert deploy_kwargs.get("controller") == "legacy"
