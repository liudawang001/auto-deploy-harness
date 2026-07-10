"""Eval Compare Run test.

Tests that eval-compare --run can execute baseline vs agent comparison
and generate comparison_report.json and comparison_report.md.
"""
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
from auto_harness.evals.comparison import AgentComparisonReporter
from auto_harness.orchestrator import TaskRunner


class TestEvalCompareRun(unittest.TestCase):
    """Test eval-compare --run functionality."""

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

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

    def test_comparison_reporter_from_manifest(self):
        """Test that AgentComparisonReporter.from_manifest generates skeleton report."""
        manifest_path = Path("eval_targets/manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest.json not found")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"

            reporter = AgentComparisonReporter()
            report = reporter.from_manifest(manifest_path, output_dir)

            # Check report structure
            self.assertIn("eval_id", report)
            self.assertIn("target_count", report)
            self.assertIn("baseline", report)
            self.assertIn("agent", report)
            self.assertIn("baseline_failed_agent_passed_count", report)
            self.assertIn("llm_helped_cases", report)

            # Check files were written
            self.assertTrue((output_dir / "comparison_report.json").exists())
            self.assertTrue((output_dir / "comparison_report.md").exists())

    def test_comparison_reporter_build(self):
        """Test that AgentComparisonReporter.build generates correct report."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"

            targets = [
                {"id": "test-1", "type": "http", "repo": "test"},
                {"id": "test-2", "type": "api", "repo": "test"},
            ]
            baseline_runs = [
                {"target_id": "test-1", "verify_status": "uncertain"},
                {"target_id": "test-2", "verify_status": "failed"},
            ]
            agent_runs = [
                {"target_id": "test-1", "verify_status": "passed", "help_type": "verify_probe_selection"},
                {"target_id": "test-2", "verify_status": "passed", "help_type": "openapi_probe_generation"},
            ]

            reporter = AgentComparisonReporter()
            report = reporter.build(
                eval_id="test-eval",
                targets=targets,
                baseline_runs=baseline_runs,
                agent_runs=agent_runs,
                output_dir=output_dir,
            )

            # Check report
            self.assertEqual(report["eval_id"], "test-eval")
            self.assertEqual(report["target_count"], 2)
            self.assertEqual(report["baseline_failed_agent_passed_count"], 2)
            self.assertEqual(len(report["llm_helped_cases"]), 2)

            # Check files
            self.assertTrue((output_dir / "comparison_report.json").exists())
            self.assertTrue((output_dir / "comparison_report.md").exists())

            # Check markdown content
            md_content = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
            self.assertIn("Agent Comparison Report", md_content)
            self.assertIn("LLM Helped Cases", md_content)

    def test_comparison_report_md_format(self):
        """Test that comparison_report.md has correct format."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"

            targets = [{"id": "test-1", "type": "http"}]
            baseline_runs = [{"target_id": "test-1", "verify_status": "uncertain"}]
            agent_runs = [{"target_id": "test-1", "verify_status": "passed", "help_type": "verify_probe_selection"}]

            reporter = AgentComparisonReporter()
            report = reporter.build(
                eval_id="test-eval",
                targets=targets,
                baseline_runs=baseline_runs,
                agent_runs=agent_runs,
                output_dir=output_dir,
            )

            md_content = (output_dir / "comparison_report.md").read_text(encoding="utf-8")

            # Check required sections
            self.assertIn("# Agent Comparison Report", md_content)
            self.assertIn("## LLM Helped Cases", md_content)
            self.assertIn("test-1", md_content)
            self.assertIn("verify_probe_selection", md_content)

    def test_full_eval_compare_with_fixtures(self):
        """Full eval compare test using real fixtures.

        This test runs actual deployments for each fixture in the manifest
        and generates a comparison report.
        """
        manifest_path = Path("eval_targets/manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest.json not found")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        targets = manifest.get("targets", [])

        # Only test first 2 fixtures to save time
        test_targets = targets[:2]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "eval_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            baseline_runs = []
            agent_runs = []

            for target in test_targets:
                fixture = Path(target["repo"])
                if not fixture.exists():
                    continue

                for mode in ["off", "gated_actor"]:
                    repo = root / f"repo-{target['id']}-{mode}"
                    shutil.copytree(fixture, repo)

                    # Replace port
                    port = self._free_port()
                    app_path = repo / "app.py"
                    if app_path.exists():
                        content = app_path.read_text(encoding="utf-8")
                        for old_port in ["8917", "8918", "8919", "8920", "8921", "8922"]:
                            content = content.replace(old_port, str(port))
                        app_path.write_text(content, encoding="utf-8")

                    config = HarnessConfig(
                        runs_dir=str(root / "runs"),
                        memory_dir=str(root / "memory"),
                        model_cache_dir=str(root / "model_cache"),
                        allowed_commands=["python", "python3", "pip"],
                        env_backend="venv",
                        agent_mode=mode,
                        agent_enable_runtime_loop=mode != "off",
                        agent_runtime_loop_position="primary",
                        agent_runtime_loop_max_iterations=15,
                        agent_auto_resume_after_repair=False,
                    )
                    runner = TaskRunner(config)

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ResourceWarning)
                        task_id = runner.deploy(
                            str(repo),
                            f"{target['id']}-{mode}",
                            dry_run=False,
                            allow_install=True,
                            allow_start=True,
                        )

                    run_dir = root / "runs" / task_id
                    try:
                        # Get verify status
                        pipeline_path = run_dir / "reports" / "pipeline_results.json"
                        verify_status = "unknown"
                        if pipeline_path.exists():
                            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
                            verify_status = pipeline.get("verify", {}).get("status", "unknown")

                        # Get help_type from evidence
                        help_type = ""
                        evidence_path = run_dir / "reports" / "llm_contribution_evidence.json"
                        if evidence_path.exists():
                            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                            help_types = evidence.get("help_type", [])
                            if help_types:
                                help_type = help_types[0]

                        run_info = {
                            "target_id": target["id"],
                            "mode": mode,
                            "verify_status": verify_status,
                            "help_type": help_type,
                            "evidence": str(run_dir / "reports"),
                        }

                        if mode == "off":
                            baseline_runs.append(run_info)
                        else:
                            agent_runs.append(run_info)
                    finally:
                        self._terminate_runner_pid(run_dir)

            # Generate comparison report
            reporter = AgentComparisonReporter()
            report = reporter.build(
                eval_id="test-eval",
                targets=test_targets,
                baseline_runs=baseline_runs,
                agent_runs=agent_runs,
                output_dir=output_dir,
            )

            # Check report
            self.assertIn("eval_id", report)
            self.assertIn("baseline_failed_agent_passed_count", report)
            self.assertIn("llm_helped_cases", report)

            # Check files exist
            self.assertTrue((output_dir / "comparison_report.json").exists())
            self.assertTrue((output_dir / "comparison_report.md").exists())


if __name__ == "__main__":
    unittest.main()
