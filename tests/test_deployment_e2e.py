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


class TestDeploymentE2E(unittest.TestCase):
    """Full local deployment E2E using a trace-echoing HTTP demo."""

    def test_execute_pipeline_starts_service_and_writes_trace_evidence(self):
        fixture = Path("tests/fixtures/e2e/http_trace_echo")
        if not fixture.exists():
            self.skipTest("http_trace_echo fixture not found")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            shutil.copytree(fixture, repo)
            port = self._free_port()
            app_path = repo / "app.py"
            app_path.write_text(app_path.read_text(encoding="utf-8").replace("8917", str(port)), encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                allowed_commands=["python", "python3", "pip"],
                env_backend="venv",
                agent_auto_resume_after_repair=False,
            )
            runner = TaskRunner(config)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ResourceWarning)
                task_id = runner.deploy(
                    str(repo),
                    "http-trace-e2e",
                    dry_run=False,
                    allow_install=True,
                    allow_start=True,
                )
            run_dir = root / "runs" / task_id

            try:
                pipeline = json.loads((run_dir / "reports" / "pipeline_results.json").read_text(encoding="utf-8"))
                self.assertEqual(pipeline["analyze"]["status"], "passed")
                self.assertEqual(pipeline["env_deploy"]["status"], "passed")
                self.assertEqual(pipeline["runner"]["status"], "passed")
                self.assertEqual(pipeline["verify"]["status"], "passed")
                self.assertEqual(pipeline["report"]["status"], "passed")

                runner_data = pipeline["runner"]["data"]
                self.assertTrue(runner_data["service_ready"])
                self.assertEqual(runner_data["expected_port"], port)
                self.assertTrue(Path(runner_data["log_path"]).exists())

                verify_data = pipeline["verify"]["data"]
                trace_id = verify_data["trace_id"]
                self.assertTrue(trace_id.startswith("verify_"))
                self.assertEqual(verify_data["status"], "pass")
                self.assertTrue(
                    any(
                        check["name"] == "http_trace_response" and check["status"] == "pass"
                        for check in verify_data["checks"]
                    )
                )

                http_evidence_paths = [
                    Path(path)
                    for path in verify_data["evidence"]
                    if path.endswith("_http_trace_initial.json")
                ]
                self.assertEqual(len(http_evidence_paths), 1)
                http_evidence = json.loads(http_evidence_paths[0].read_text(encoding="utf-8"))
                self.assertEqual(http_evidence["request"]["trace_id"], trace_id)
                self.assertIn("_auto_harness_trace=", http_evidence["request"]["url"])
                self.assertIn(trace_id, http_evidence["response"]["body_tail"])
                self.assertEqual(http_evidence["check"]["status"], "pass")

                report = (run_dir / "reports" / "report.md").read_text(encoding="utf-8")
                self.assertIn("Final status: `pass`", report)
                self.assertIn("Trace ID: `%s`" % trace_id, report)

                events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
                self.assertIn("copy_local_repo", events)
                self.assertIn('"stage": "verify"', events)
            finally:
                self._terminate_runner_pid(run_dir)

    def _terminate_runner_pid(self, run_dir: Path) -> None:
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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
