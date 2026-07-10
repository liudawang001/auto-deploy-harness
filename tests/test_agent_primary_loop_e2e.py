import json
import os
import shutil
import signal
import socket
import tempfile
import unittest
import warnings
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.orchestrator import TaskRunner


class TestAgentPrimaryLoopE2E(unittest.TestCase):
    """Agent primary loop E2E: DeploymentAgentLoop as primary controller."""

    def test_agent_primary_loop_full_deployment(self):
        """Test that agent primary loop can complete full deployment cycle.

        Validates:
        - agent_steps.jsonl exists and contains all expected stages
        - agent_state.json exists
        - agent_plan.json exists
        - reports/agent_loop_result.json exists
        - reports/pipeline_results.json exists with verify passed
        - HTTP trace evidence contains current trace_id
        - runner pid is cleaned up
        """
        fixture = Path("tests/fixtures/e2e/http_trace_echo")
        if not fixture.exists():
            self.skipTest("http_trace_echo fixture not found")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            shutil.copytree(fixture, repo)
            port = self._free_port()
            app_path = repo / "app.py"
            app_path.write_text(
                app_path.read_text(encoding="utf-8").replace("8917", str(port)),
                encoding="utf-8",
            )

            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                allowed_commands=["python", "python3", "pip"],
                env_backend="venv",
                # Enable agent runtime loop as primary controller
                agent_mode="gated_actor",
                agent_enable_runtime_loop=True,
                agent_runtime_loop_position="primary",
                agent_runtime_loop_max_iterations=15,
                agent_runtime_loop_stop_on_verify_pass=True,
                agent_auto_resume_after_repair=False,
            )
            runner = TaskRunner(config)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ResourceWarning)
                task_id = runner.deploy(
                    str(repo),
                    "agent-primary-e2e",
                    dry_run=False,
                    allow_install=True,
                    allow_start=True,
                )

            run_dir = root / "runs" / task_id

            try:
                # 1. Verify agent_loop_result.json exists
                agent_loop_result_path = run_dir / "reports" / "agent_loop_result.json"
                self.assertTrue(
                    agent_loop_result_path.exists(),
                    "agent_loop_result.json should exist",
                )
                agent_loop_result = json.loads(agent_loop_result_path.read_text(encoding="utf-8"))
                self.assertIn("stop_reason", agent_loop_result)
                self.assertIn("iteration_count", agent_loop_result)

                # 2. Verify agent_steps.jsonl exists and contains expected stages
                agent_steps_path = run_dir / "agent_steps.jsonl"
                self.assertTrue(
                    agent_steps_path.exists(),
                    "agent_steps.jsonl should exist",
                )
                steps = [
                    json.loads(line)
                    for line in agent_steps_path.read_text(encoding="utf-8").strip().split("\n")
                    if line.strip()
                ]
                self.assertGreater(len(steps), 0, "agent_steps.jsonl should not be empty")

                # Extract stages from steps
                step_stages = [s.get("stage") for s in steps]
                expected_stages = ["analyze", "env_deploy", "runner", "verify"]
                for stage in expected_stages:
                    self.assertIn(
                        stage,
                        step_stages,
                        f"agent_steps.jsonl should contain {stage} stage",
                    )

                # 3. Verify agent_state.json exists
                agent_state_path = run_dir / "agent_state.json"
                self.assertTrue(
                    agent_state_path.exists(),
                    "agent_state.json should exist",
                )
                agent_state = json.loads(agent_state_path.read_text(encoding="utf-8"))
                self.assertIn("task_id", agent_state)
                self.assertIn("stage_status", agent_state)

                # 4. Verify agent_plan.json exists
                agent_plan_path = run_dir / "agent_plan.json"
                self.assertTrue(
                    agent_plan_path.exists(),
                    "agent_plan.json should exist",
                )

                # 5. Verify pipeline_results.json has verify passed
                pipeline_path = run_dir / "reports" / "pipeline_results.json"
                self.assertTrue(
                    pipeline_path.exists(),
                    "pipeline_results.json should exist",
                )
                pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
                self.assertIn("verify", pipeline)
                self.assertEqual(
                    pipeline["verify"]["status"],
                    "passed",
                    "verify status should be passed",
                )

                # 6. Verify HTTP trace evidence
                verify_data = pipeline["verify"].get("data", {})
                trace_id = verify_data.get("trace_id", "")
                self.assertTrue(
                    trace_id.startswith("verify_"),
                    f"trace_id should start with verify_, got: {trace_id}",
                )

                # Check evidence files contain trace_id
                evidence_dir = run_dir / "evidence"
                if evidence_dir.exists():
                    evidence_files = list(evidence_dir.glob("*trace*.json"))
                    self.assertGreater(
                        len(evidence_files),
                        0,
                        "Should have trace evidence files",
                    )
                    for evidence_file in evidence_files:
                        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
                        # Verify trace_id is in the evidence
                        evidence_str = json.dumps(evidence)
                        self.assertIn(
                            trace_id,
                            evidence_str,
                            f"Evidence file {evidence_file.name} should contain trace_id",
                        )

                # 7. Verify report.md exists and contains trace id
                report_path = run_dir / "reports" / "report.md"
                self.assertTrue(report_path.exists(), "report.md should exist")
                report_content = report_path.read_text(encoding="utf-8")
                self.assertIn(
                    trace_id,
                    report_content,
                    "report.md should contain trace_id",
                )

            finally:
                self._terminate_runner_pid(run_dir)

    def test_agent_primary_loop_writes_all_artifacts(self):
        """Test that agent primary loop writes all required artifacts."""
        fixture = Path("tests/fixtures/e2e/http_trace_echo")
        if not fixture.exists():
            self.skipTest("http_trace_echo fixture not found")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            shutil.copytree(fixture, repo)
            port = self._free_port()
            app_path = repo / "app.py"
            app_path.write_text(
                app_path.read_text(encoding="utf-8").replace("8917", str(port)),
                encoding="utf-8",
            )

            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                allowed_commands=["python", "python3", "pip"],
                env_backend="venv",
                agent_mode="gated_actor",
                agent_enable_runtime_loop=True,
                agent_runtime_loop_position="primary",
                agent_runtime_loop_max_iterations=15,
                agent_auto_resume_after_repair=False,
            )
            runner = TaskRunner(config)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ResourceWarning)
                task_id = runner.deploy(
                    str(repo),
                    "agent-artifacts-e2e",
                    dry_run=False,
                    allow_install=True,
                    allow_start=True,
                )

            run_dir = root / "runs" / task_id

            try:
                # Check all required artifacts
                required_artifacts = [
                    "agent_steps.jsonl",
                    "agent_state.json",
                    "agent_plan.json",
                    "reports/agent_loop_result.json",
                    "reports/pipeline_results.json",
                    "reports/report.md",
                ]

                for artifact in required_artifacts:
                    artifact_path = run_dir / artifact
                    self.assertTrue(
                        artifact_path.exists(),
                        f"Required artifact {artifact} should exist",
                    )

                # Verify agent_steps.jsonl has step records with full context
                steps_path = run_dir / "agent_steps.jsonl"
                steps = [
                    json.loads(line)
                    for line in steps_path.read_text(encoding="utf-8").strip().split("\n")
                    if line.strip()
                ]
                for step in steps:
                    self.assertIn("step_id", step)
                    self.assertIn("stage", step)
                    self.assertIn("before_status", step)
                    self.assertIn("after_status", step)
                    self.assertIn("observation", step)
                    self.assertIn("recorded_at", step)

            finally:
                self._terminate_runner_pid(run_dir)

    def _terminate_runner_pid(self, run_dir: Path) -> None:
        """Terminate runner process if it exists."""
        pipeline_path = run_dir / "reports" / "pipeline_results.json"
        if not pipeline_path.exists():
            return
        try:
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
            pid = int(pipeline.get("runner", {}).get("data", {}).get("pid") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        if pid <= 0:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def _free_port(self) -> int:
        """Get a free port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
