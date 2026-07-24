"""Task 6: CLI approval-resolve E2E tests.

Verifies:
1. CLI approval-resolve calls runner.resume with correct resume_input
2. Decision includes request_hash bound to the original request
3. Multiple pending approvals require --approval-id
4. No pending approval returns error
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.cli import build_parser, main
from auto_harness.config import HarnessConfig
from auto_harness.graph.approval import build_approval_request


class TestCLIApprovalResolve:
    """CLI approval-resolve calls runner.resume."""

    def _setup_pending_approval(self, tmp_path, request=None):
        """Create a pending approval file in the run directory."""
        if request is None:
            request = build_approval_request(
                approval_id="test-app-001",
                operation_id="op-001",
                approval_kind="repair",
                requested_action="apply_repair",
                risk="high",
                reason="source_edit requires approval",
            )
        runs_dir = tmp_path / "runs"
        task_dir = runs_dir / "test_task"
        task_dir.mkdir(parents=True)
        approvals_dir = task_dir / "approvals"
        approvals_dir.mkdir()
        approval_file = approvals_dir / (request["approval_id"] + ".json")
        approval_file.write_text(json.dumps({
            "status": "pending",
            "request": request,
        }), encoding="utf-8")
        # Also write task.json for StateStore
        from auto_harness.models.task import TaskSpec, ProjectSpec, RuntimePolicy
        from auto_harness.utils.time import utc_now_iso
        spec = TaskSpec(
            task_id="test_task",
            project=ProjectSpec(name="test", repo_url="https://example.com/repo"),
            runtime=RuntimePolicy(workspace_root="/tmp"),
            controller="langgraph",
            created_at=utc_now_iso(),
        )
        (task_dir / "task.json").write_text(json.dumps({
            "task_id": spec.task_id,
            "project": {"name": "test", "repo_url": "https://example.com/repo", "branch": "main"},
            "runtime": {"workspace_root": "/tmp", "allow_dependency_install": False, "allow_service_start": False},
            "created_at": spec.created_at,
            "controller": "langgraph",
        }), encoding="utf-8")
        return request

    def test_approval_resolve_calls_runner_resume(self, tmp_path):
        """approval-resolve calls runner.resume with resume_input."""
        request = self._setup_pending_approval(tmp_path)
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")

        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.resume.return_value = "test_task"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main([
                    "approval-resolve",
                    "--task-id", "test_task",
                    "--decision", "approve",
                ])

        assert exit_code == 0
        # Verify runner.resume was called
        mock_runner.resume.assert_called_once()
        resume_kwargs = mock_runner.resume.call_args.kwargs
        assert resume_kwargs["resume_input"]["decision"] == "approve"
        assert resume_kwargs["resume_input"]["operation_id"] == request["operation_id"]
        assert resume_kwargs["resume_input"]["request_hash"] == request["request_hash"]
        assert resume_kwargs["resume_input"]["approval_id"] == request["approval_id"]

    def test_approval_resolve_reject(self, tmp_path):
        """approval-resolve with reject also calls runner.resume."""
        request = self._setup_pending_approval(tmp_path)
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")

        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.resume.return_value = "test_task"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main([
                    "approval-resolve",
                    "--task-id", "test_task",
                    "--decision", "reject",
                ])

        assert exit_code == 0
        resume_kwargs = mock_runner.resume.call_args.kwargs
        assert resume_kwargs["resume_input"]["decision"] == "reject"

    def test_approval_resolve_includes_reviewer(self, tmp_path):
        """approval-resolve --reviewer is passed in resume_input."""
        request = self._setup_pending_approval(tmp_path)
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")

        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.resume.return_value = "test_task"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main([
                    "approval-resolve",
                    "--task-id", "test_task",
                    "--decision", "approve",
                    "--reviewer", "admin",
                ])

        assert exit_code == 0
        resume_kwargs = mock_runner.resume.call_args.kwargs
        assert resume_kwargs["resume_input"]["reviewer"] == "admin"

    def test_no_pending_approval_returns_error(self, tmp_path):
        """No pending approvals returns error."""
        runs_dir = tmp_path / "runs"
        task_dir = runs_dir / "test_task"
        task_dir.mkdir(parents=True)
        (task_dir / "approvals").mkdir()
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")

        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                # Need to also set up task.json
                from auto_harness.utils.time import utc_now_iso
                (task_dir / "task.json").write_text(json.dumps({
                    "task_id": "test_task",
                    "project": {"name": "test", "repo_url": "x", "branch": "main"},
                    "runtime": {"workspace_root": "/tmp", "allow_dependency_install": False, "allow_service_start": False},
                    "created_at": utc_now_iso(),
                    "controller": "langgraph",
                }), encoding="utf-8")
                exit_code = main([
                    "approval-resolve",
                    "--task-id", "test_task",
                    "--decision", "approve",
                ])

        assert exit_code == 2

    def test_multiple_pending_approvals_require_approval_id(self, tmp_path):
        """Multiple pending approvals require --approval-id."""
        # Create two pending approvals
        request1 = build_approval_request(
            approval_id="app-001", operation_id="op-001",
            approval_kind="repair", requested_action="apply_repair",
            risk="high", reason="repair 1",
        )
        request2 = build_approval_request(
            approval_id="app-002", operation_id="op-002",
            approval_kind="recovery", requested_action="cleanup_then_retry",
            risk="high", reason="recovery 2",
        )

        runs_dir = tmp_path / "runs"
        task_dir = runs_dir / "test_task"
        task_dir.mkdir(parents=True)
        approvals_dir = task_dir / "approvals"
        approvals_dir.mkdir()

        for req in (request1, request2):
            f = approvals_dir / (req["approval_id"] + ".json")
            f.write_text(json.dumps({"status": "pending", "request": req}), encoding="utf-8")

        from auto_harness.utils.time import utc_now_iso
        (task_dir / "task.json").write_text(json.dumps({
            "task_id": "test_task",
            "project": {"name": "test", "repo_url": "x", "branch": "main"},
            "runtime": {"workspace_root": "/tmp", "allow_dependency_install": False, "allow_service_start": False},
            "created_at": utc_now_iso(),
            "controller": "langgraph",
        }), encoding="utf-8")

        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")

        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                exit_code = main([
                    "approval-resolve",
                    "--task-id", "test_task",
                    "--decision", "approve",
                ])

        assert exit_code == 2  # Error: multiple pending approvals

    def test_dry_run_default_when_no_execute(self, tmp_path):
        """Without --execute, dry_run=True is passed to runner.resume."""
        request = self._setup_pending_approval(tmp_path)
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")

        with patch("auto_harness.cli.TaskRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.resume.return_value = "test_task"
            MockRunner.return_value = mock_runner
            with patch("auto_harness.cli.HarnessConfig.load", return_value=config):
                main([
                    "approval-resolve",
                    "--task-id", "test_task",
                    "--decision", "approve",
                ])

        resume_kwargs = mock_runner.resume.call_args.kwargs
        assert resume_kwargs["dry_run"] is True
