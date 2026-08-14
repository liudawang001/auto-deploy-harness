"""E2E test for LLM Plan-first Deployment Agent.

Phase 4 of LLM Plan-first Deployment Agent.

Tests the full plan-first flow with mock LLM:
- Project snapshot built
- LLM generates deployment plan
- Policy gate accepts
- Framework executes stages
- Verify passes with trace evidence
- Artifacts are written to disk
"""
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.orchestrator import TaskRunner


class TestLLMPlanFirstE2E(unittest.TestCase):
    """E2E test for plan-first deployment with mock provider."""

    def test_execute_pipeline_plan_first_requires_approval(self):
        """Unknown repository entrypoints stop before execution for approval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_src = Path(__file__).parent / "fixtures" / "e2e" / "llm_plan_first_http_trace"

            # Copy fixture to temp repo
            repo_dir = root / "repo"
            shutil.copytree(str(fixture_src), str(repo_dir))

            # Replace port 8917 with dynamic port
            port = self._free_port()
            app_py = repo_dir / "app.py"
            app_py.write_text(app_py.read_text(encoding="utf-8").replace("8917", str(port)))

            # Configure with plan-first mode
            config = HarnessConfig(
                default_controller="legacy",
                runs_dir=str(root / "runs"),
                skills_dir=str(root / "skills"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                allowed_commands=["python", "python3", "pip"],
                allow_dependency_install=True,
                allow_service_start=True,
                agent_plan_first=True,
                agent_plan_first_provider="mock",
                agent_plan_first_mode="gated_actor",
                agent_plan_first_max_replans=2,
            )

            runner = TaskRunner(config)
            run_dir = None

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    task_id = runner.deploy(
                        str(repo_dir),
                        name="plan-first-http",
                        dry_run=False,
                        skip_clone=False,  # Let deploy copy files to workspace/repo
                        allow_install=True,
                        allow_start=True,
                    )

                self.assertTrue(task_id.startswith("plan-first-http"))
                run_dir = Path(config.runs_dir) / task_id
                reports_dir = run_dir / "reports"
                state = runner.store.load_state(task_id)
                self.assertEqual(state.status, "waiting_for_approval")
                pending = json.loads(
                    (reports_dir / "plan_first_result.json").read_text(encoding="utf-8")
                )["approval_request"]
                approval_record = json.loads(
                    (run_dir / "approvals" / (pending["approval_id"] + ".json")).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(approval_record["status"], "pending")

                # Planning artifacts exist, but no command or verify stage ran.
                self.assertTrue((reports_dir / "project_snapshot.json").exists(), "project_snapshot.json missing")
                self.assertTrue((reports_dir / "llm_deployment_plan.raw.json").exists(), "raw plan missing")
                self.assertTrue((reports_dir / "llm_deployment_plan.parsed.json").exists(), "parsed plan missing")
                self.assertTrue((reports_dir / "llm_plan_policy.json").exists(), "policy result missing")
                self.assertFalse((reports_dir / "effective_deployment_plan.json").exists())
                self.assertFalse((reports_dir / "pipeline_results.json").exists())

                # Check policy requested immutable human approval.
                policy = json.loads((reports_dir / "llm_plan_policy.json").read_text(encoding="utf-8"))
                self.assertFalse(policy.get("allowed"))
                self.assertEqual(policy.get("status"), "approval_required")
                self.assertEqual(pending.get("approval_kind"), "repository_command")

            finally:
                if run_dir is not None:
                    self._terminate_runner_pid(run_dir)

    def _free_port(self) -> int:
        """Find a free port."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _terminate_runner_pid(self, run_dir: Path) -> None:
        """Terminate the runner process if still alive."""
        pipeline_path = run_dir / "reports" / "pipeline_results.json"
        if not pipeline_path.exists():
            return
        try:
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            runner_data = pipeline.get("runner", {}).get("data", {})
            pid = runner_data.get("pid") or runner_data.get("process_id")
            if pid:
                import os
                import signal
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
        except (json.JSONDecodeError, OSError, ValueError):
            pass


if __name__ == "__main__":
    unittest.main()
