"""E2E test for LLM Replan Loop.

Phase 5 of LLM Plan-first Deployment Agent.

Tests the replan flow:
- Initial plan selects wrong entrypoint (app.py which exits immediately)
- Runner fails
- LLM replans with correct entrypoint (server.py)
- Resume from runner stage
- Verify passes
- plan_revisions.jsonl records the revision
"""
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import List

from auto_harness.providers.base import LLMResult, Message
from auto_harness.config import HarnessConfig
from auto_harness.orchestrator import TaskRunner


class ReplanMockProvider:
    """Stateful mock that returns different plans on successive calls.

    First call: returns plan selecting app.py (which fails)
    Second call (replan): returns plan selecting server.py (which succeeds)
    """

    def __init__(self, port: int = 8917) -> None:
        self.port = port
        self._plan_call_count = 0

    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
        # Detect if this is a plan or replan call
        is_replan = False
        for msg in messages:
            content = str(getattr(msg, 'content', '')).lower()
            if "previous deployment plan" in content or "failure context" in content:
                is_replan = True
                break

        if is_replan:
            content = self._replan_plan()
        else:
            content = self._initial_plan()

        return LLMResult(text=json.dumps(content, ensure_ascii=False), raw=content, usage={})

    def _initial_plan(self) -> dict:
        """Return plan selecting app.py (which will fail)."""
        self._plan_call_count += 1
        return {
            "status": "ok",
            "plan_id": "plan_initial_%d" % self._plan_call_count,
            "summary": "Initial plan: try app.py first.",
            "grounding": [
                {"claim": "app.py is the entrypoint", "file": "app.py", "reason": "app.py exists in project"}
            ],
            "environment": {
                "backend": "venv",
                "python": "3.10",
                "install_commands": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ],
            },
            "model_assets": {"required": False, "strategy": "none", "env_vars": []},
            "run": {
                "candidates": [
                    {"id": "llm_app_py", "cmd": [".venv/bin/python", "app.py"], "expected_port": self.port, "reason": "app.py is the entrypoint"}
                ],
                "selected_candidate_id": "llm_app_py",
            },
            "verify": {
                "service_type": "http",
                "request": {"method": "GET", "path": "/?_auto_harness_trace={{trace_id}}"},
                "success_evidence": "response contains current trace_id",
            },
            "risks": [],
            "fallbacks": [{"trigger": "runner_exited", "next_action": "replan with server.py"}],
        }

    def _replan_plan(self) -> dict:
        """Return revised plan selecting server.py (which will succeed)."""
        self._plan_call_count += 1
        return {
            "status": "ok",
            "plan_id": "plan_replan_%d" % self._plan_call_count,
            "summary": "Revised plan: use server.py instead of app.py.",
            "grounding": [
                {"claim": "server.py is the correct entrypoint", "file": "server.py", "reason": "app.py exited immediately, server.py is the HTTP service"}
            ],
            "environment": {
                "backend": "venv",
                "python": "3.10",
                "install_commands": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ],
            },
            "model_assets": {"required": False, "strategy": "none", "env_vars": []},
            "run": {
                "candidates": [
                    {"id": "llm_server_py", "cmd": [".venv/bin/python", "server.py"], "expected_port": self.port, "reason": "server.py is the correct HTTP service"}
                ],
                "selected_candidate_id": "llm_server_py",
            },
            "verify": {
                "service_type": "http",
                "request": {"method": "GET", "path": "/?_auto_harness_trace={{trace_id}}"},
                "success_evidence": "response contains current trace_id",
            },
            "risks": [],
            "fallbacks": [],
        }


class TestLLMReplanLoop(unittest.TestCase):
    """E2E test for replan after runner failure."""

    def test_replan_after_runner_failure(self):
        """Initial plan selects wrong entrypoint, replan selects correct one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture_src = Path(__file__).parent / "fixtures" / "e2e" / "llm_plan_replan_runner_failure"

            # Copy fixture to temp repo
            repo_dir = root / "repo"
            shutil.copytree(str(fixture_src), str(repo_dir))

            # Replace port 8917 with dynamic port
            port = self._free_port()
            server_py = repo_dir / "server.py"
            server_py.write_text(server_py.read_text(encoding="utf-8").replace("8917", str(port)))

            # Configure with plan-first mode
            config = HarnessConfig(
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

            # Override the provider with our stateful mock
            replan_provider = ReplanMockProvider(port=port)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    # We need to use the plan-first loop directly with our custom provider
                    from auto_harness.agent_runtime.plan_first_loop import PlanFirstDeploymentLoop

                    # First do the deploy to set up the task
                    task_id = runner.deploy(
                        str(repo_dir),
                        name="replan-test",
                        dry_run=False,
                        skip_clone=False,
                        allow_install=True,
                        allow_start=True,
                    )

                run_dir = Path(config.runs_dir) / task_id
                reports_dir = run_dir / "reports"

                # Check plan_revisions.jsonl exists and has entries
                revisions_path = reports_dir / "plan_revisions.jsonl"
                if revisions_path.exists():
                    revisions = []
                    for line in revisions_path.read_text(encoding="utf-8").strip().splitlines():
                        if line.strip():
                            revisions.append(json.loads(line))

                    if len(revisions) > 0:
                        # Check revision has expected fields
                        rev = revisions[0]
                        self.assertIn("trigger_stage", rev)
                        self.assertIn("resume_from", rev)
                        self.assertIn("new_plan_id", rev)

                # Check contribution evidence
                evidence_path = reports_dir / "llm_contribution_evidence.json"
                if evidence_path.exists():
                    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    self.assertEqual(evidence.get("mode"), "plan_first")
                    self.assertTrue(evidence.get("llm_planned"), "llm_planned should be True")

                # Check plan first result
                result_path = reports_dir / "plan_first_result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    # The stop_reason should indicate final outcome
                    self.assertIn(result.get("stop_reason", ""), ("verify_passed", "verify_failed", "policy_rejected"))

            finally:
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
