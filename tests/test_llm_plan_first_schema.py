"""Tests for DeploymentPlan schema and ProjectSnapshotBuilder.

Phase 1 of LLM Plan-first Deployment Agent.

Covers:
- Valid plan parse success
- Invalid JSON rejected
- Missing run candidates rejected
- Missing verify trace_id rejected
- Command as string rejected
- status=no_safe_plan accepted
- selected_candidate_id mismatch rejected
- Snapshot build collects files
- Snapshot redacts secrets
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import copy

from auto_harness.agent_runtime.deployment_plan import DeploymentPlan, DeploymentPlanParser
from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder


# A valid minimal deployment plan matching the spec
VALID_PLAN = {
    "status": "ok",
    "plan_id": "plan_http_trace",
    "summary": "Run the local HTTP trace echo app in a venv.",
    "grounding": [
        {
            "claim": "app.py is the service entrypoint",
            "file": "app.py",
            "reason": "contains HTTPServer(('127.0.0.1', 8917), Handler).serve_forever()",
        }
    ],
    "environment": {
        "backend": "venv",
        "python": "3.10",
        "install_commands": [
            ["python3", "-m", "venv", ".venv"],
            [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
        ],
    },
    "model_assets": {
        "required": False,
        "strategy": "none",
        "env_vars": [],
    },
    "run": {
        "candidates": [
            {
                "id": "llm_app_py",
                "cmd": [".venv/bin/python", "app.py"],
                "expected_port": 8917,
                "reason": "app.py starts HTTPServer on 8917",
            }
        ],
        "selected_candidate_id": "llm_app_py",
    },
    "verify": {
        "service_type": "http",
        "request": {
            "method": "GET",
            "path": "/?_auto_harness_trace={{trace_id}}",
        },
        "success_evidence": "response contains current trace_id",
    },
    "risks": [],
    "fallbacks": [
        {
            "trigger": "runner_exited",
            "next_action": "inspect runner log and replan",
        }
    ],
}


class TestDeploymentPlanParser(unittest.TestCase):
    """Test DeploymentPlanParser validation logic."""

    def setUp(self):
        self.parser = DeploymentPlanParser()

    def test_valid_plan_parse_success(self):
        """A valid DeploymentPlan JSON should parse without errors."""
        plan = self.parser.parse(json.dumps(VALID_PLAN))
        self.assertEqual(plan.status, "ok")
        self.assertEqual(plan.plan_id, "plan_http_trace")
        self.assertEqual(plan.summary, "Run the local HTTP trace echo app in a venv.")
        self.assertEqual(len(plan.grounding), 1)
        self.assertEqual(plan.grounding[0]["file"], "app.py")
        self.assertEqual(len(plan.environment["install_commands"]), 2)
        self.assertEqual(len(plan.run["candidates"]), 1)
        self.assertEqual(plan.run["selected_candidate_id"], "llm_app_py")
        self.assertIn("{{trace_id}}", plan.verify["request"]["path"])

    def test_invalid_json_rejected(self):
        """Non-JSON text should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse("not json at all")
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_missing_run_candidates_rejected(self):
        """Empty run.candidates should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["run"]["candidates"] = []
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("run.candidates", str(ctx.exception))

    def test_missing_verify_trace_id_rejected(self):
        """Verify request without {{trace_id}} should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["verify"] = {
            "service_type": "http",
            "request": {"method": "GET", "path": "/health"},
            "success_evidence": "HTTP 200",
        }
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("{{trace_id}}", str(ctx.exception))

    def test_command_as_string_rejected(self):
        """install_commands containing a string instead of list should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["environment"]["install_commands"] = ["python app.py"]
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("list of strings, not a shell string", str(ctx.exception))

    def test_status_no_safe_plan_accepted(self):
        """status=no_safe_plan should parse without requiring other fields."""
        plan_json = json.dumps({"status": "no_safe_plan", "summary": "Cannot determine a safe plan"})
        plan = self.parser.parse(plan_json)
        self.assertEqual(plan.status, "no_safe_plan")
        self.assertEqual(plan.summary, "Cannot determine a safe plan")
        self.assertEqual(plan.grounding, [])
        self.assertEqual(plan.environment, {})

    def test_selected_candidate_id_mismatch_rejected(self):
        """selected_candidate_id not matching any candidate should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["run"]["selected_candidate_id"] = "nonexistent_id"
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("does not match any candidate id", str(ctx.exception))

    def test_missing_install_commands_rejected(self):
        """Empty install_commands should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["environment"]["install_commands"] = []
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("install_commands", str(ctx.exception))

    def test_candidate_missing_id_rejected(self):
        """Candidate without id should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["run"]["candidates"] = [
            {"cmd": [".venv/bin/python", "app.py"], "expected_port": 8917}
        ]
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("must have 'id'", str(ctx.exception))

    def test_candidate_missing_expected_port_rejected(self):
        """Candidate without expected_port should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["run"]["candidates"] = [
            {"id": "llm_1", "cmd": [".venv/bin/python", "app.py"]}
        ]
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("expected_port", str(ctx.exception))

    def test_verify_external_url_rejected(self):
        """Verify request with external URL path should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["verify"] = {
            "service_type": "http",
            "request": {"method": "GET", "path": "https://evil.com/health?trace={{trace_id}}"},
            "success_evidence": "response ok",
        }
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("external URL", str(ctx.exception))

    def test_grounding_missing_reason_rejected(self):
        """Grounding entry without reason should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["grounding"] = [{"claim": "entrypoint", "file": "app.py"}]
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("reason", str(ctx.exception))

    def test_verify_invalid_method_rejected(self):
        """Verify request with invalid method should raise ValueError."""
        plan = copy.deepcopy(VALID_PLAN)
        plan["verify"] = {
            "service_type": "http",
            "request": {"method": "DELETE", "path": "/?_auto_harness_trace={{trace_id}}"},
            "success_evidence": "deleted",
        }
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse(json.dumps(plan))
        self.assertIn("GET or POST", str(ctx.exception))


class TestProjectSnapshotBuilder(unittest.TestCase):
    """Test ProjectSnapshotBuilder file collection and redaction."""

    def test_snapshot_build_collects_files(self):
        """Snapshot should collect file tree, selected files, and detected signals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create test project files
            (root / "app.py").write_text(
                "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        self.wfile.write(b'hello')\n"
                "HTTPServer(('127.0.0.1', 8917), Handler).serve_forever()\n"
            )
            (root / "requirements.txt").write_text("\n")
            (root / "README.md").write_text("# Test Project\nA test demo.\n")

            builder = ProjectSnapshotBuilder(max_files=80, max_file_chars=6000)
            snapshot = builder.build(root, task_id="test_123")

            # Check file tree
            self.assertIn("app.py", snapshot["file_tree"])
            self.assertIn("requirements.txt", snapshot["file_tree"])
            self.assertIn("README.md", snapshot["file_tree"])

            # Check selected files have content
            self.assertIn("app.py", snapshot["selected_files"])
            self.assertIn("requirements.txt", snapshot["selected_files"])
            self.assertIn("README.md", snapshot["selected_files"])
            self.assertIn("HTTPServer", snapshot["selected_files"]["app.py"]["content"])

            # Check detected signals
            signals = snapshot["detected_signals"]
            self.assertIn("http.server", signals["frameworks"])
            self.assertIn("app.py", signals["entrypoint_candidates"])
            self.assertIn("requirements.txt", signals["dependency_files"])
            self.assertIn(8917, signals["ports"])

            # Check sha256
            self.assertTrue(snapshot["selected_files"]["app.py"]["sha256"])

    def test_snapshot_detects_console_script_and_documented_run_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "demo"\nrequires-python = ">=3.12"\n'
                '[project.scripts]\ndemo = "demo.cli:main"\n'
            )
            (root / "README.md").write_text(
                "```bash\ndemo init\ndemo run --host 0.0.0.0 --port 8088 # dashboard\n```\n"
            )
            snapshot = ProjectSnapshotBuilder().build(root)
            signals = snapshot["detected_signals"]
            self.assertEqual(signals["python_requires"], ">=3.12")
            self.assertEqual(signals["console_scripts"][0]["name"], "demo")
            self.assertEqual(
                signals["documented_run_commands"][0]["cmd"],
                ["demo", "run", "--host", "0.0.0.0", "--port", "8088"],
            )
            self.assertEqual(signals["documented_run_commands"][0]["expected_port"], 8088)

    def test_snapshot_detects_required_locked_frontend_source_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "dashboard").mkdir()
            (root / "dashboard" / "package.json").write_text('{}\n')
            (root / "dashboard" / "package-lock.json").write_text('{}\n')
            (root / "Makefile").write_text(
                "DASHBOARD_DIR := dashboard\n"
                "build-frontend:\n"
                "\tcd $(DASHBOARD_DIR) && npm ci\n"
                "\tcd $(DASHBOARD_DIR) && npm run build\n"
            )
            snapshot = ProjectSnapshotBuilder().build(root)
            assert snapshot["detected_signals"]["source_build_commands"] == [
                {
                    "cmd": ["npm", "--prefix", "dashboard", "ci"],
                    "source": "Makefile",
                    "reason": "lockfile-backed frontend dependencies required by build-frontend",
                },
                {
                    "cmd": ["npm", "--prefix", "dashboard", "run", "build"],
                    "source": "Makefile",
                    "reason": "build missing production dashboard artifact",
                },
            ]

            built = root / "src" / "octop" / "dashboard"
            built.mkdir(parents=True)
            (built / "index.html").write_text("built\n")
            snapshot = ProjectSnapshotBuilder().build(root)
            assert snapshot["detected_signals"]["source_build_commands"] == []

    def test_snapshot_detects_console_build_setup_and_app_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "console").mkdir()
            (root / "deploy").mkdir()
            (root / "console" / "package.json").write_text('{}\n')
            (root / "console" / "package-lock.json").write_text('{}\n')
            (root / "deploy" / "entrypoint.sh").write_text(
                "#!/bin/sh\npaw init --defaults --accept-security\n",
            )
            (root / "pyproject.toml").write_text(
                '[project]\nname = "paw"\nrequires-python = ">=3.11,<3.14"\n'
                '[project.scripts]\npaw = "paw.cli:main"\n'
            )
            (root / "README.md").write_text(
                "```bash\n"
                "paw init --defaults\n"
                "paw app\n"
                "```\n"
                "Open http://127.0.0.1:8088/ after startup.\n\n"
                "From source: `cd console && npm ci && npm run build`.\n"
            )
            signals = ProjectSnapshotBuilder(
                context_mode="layered", core_budget_tokens=12000,
            ).build(root)["detected_signals"]
            assert signals["console_scripts"][0]["name"] == "paw"
            assert signals["documented_setup_commands"][0]["cmd"] == [
                "paw", "init", "--defaults", "--accept-security",
            ]
            assert signals["documented_setup_commands"][0]["source"] == (
                "deploy/entrypoint.sh"
            )
            assert signals["documented_run_commands"][0]["cmd"] == ["paw", "app"]
            assert signals["documented_run_commands"][0]["expected_port"] == 8088
            assert signals["source_build_commands"][-1]["cmd"] == [
                "npm", "--prefix", "console", "run", "build",
            ]

    def test_long_readme_keeps_late_deployment_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "paw"\nrequires-python = ">=3.11"\n'
                '[project.scripts]\npaw = "paw.cli:main"\n'
            )
            (root / "README.md").write_text(
                "# Paw\n" + ("feature overview text\n" * 1000)
                + "## Quick Start\n```bash\npaw init --defaults\npaw app\n```\n"
                + "Open http://127.0.0.1:8088/ after startup.\n"
            )
            snapshot = ProjectSnapshotBuilder(
                max_file_chars=6000,
                context_mode="layered",
                core_budget_tokens=12000,
            ).build(root)
            content = snapshot["selected_files"]["README.md"]["content"]
            assert "paw init --defaults" in content
            assert "paw app" in content
            signals = snapshot["detected_signals"]
            assert signals["documented_setup_commands"][0]["cmd"] == [
                "paw", "init", "--defaults",
            ]
            assert signals["documented_setup_commands"][0]["line"] > 1000
            assert signals["documented_run_commands"][0]["expected_port"] == 8088

    def test_snapshot_redacts_secrets(self):
        """Snapshot should redact secret patterns from file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "app.py").write_text(
                "api_key=sk-abcdefghij123456\n"
                "print('hello')\n"
            )
            (root / "requirements.txt").write_text("\n")

            builder = ProjectSnapshotBuilder()
            snapshot = builder.build(root, task_id="test_secret")

            # The content should have the secret redacted
            content = snapshot["selected_files"]["app.py"]["content"]
            self.assertIn("[REDACTED_SECRET]", content)
            self.assertNotIn("sk-abcdefghij123456", content)

            # Redactions list should be non-empty
            self.assertTrue(len(snapshot["redactions"]) > 0)

    def test_snapshot_skips_git_and_binaries(self):
        """Snapshot should skip .git directory and binary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "app.py").write_text("print('hello')\n")
            (root / "requirements.txt").write_text("\n")
            # Create .git directory with files
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
            # Create a binary file
            (root / "model.bin").write_bytes(b"\x00" * 100)

            builder = ProjectSnapshotBuilder()
            snapshot = builder.build(root, task_id="test_skip")

            self.assertNotIn(".git/HEAD", snapshot["file_tree"])
            self.assertNotIn("model.bin", snapshot["file_tree"])

    def test_snapshot_truncates_large_files(self):
        """Snapshot should truncate files exceeding max_file_chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            long_content = "x" * 10000
            (root / "README.md").write_text(long_content)
            (root / "requirements.txt").write_text("\n")

            builder = ProjectSnapshotBuilder(max_file_chars=100)
            snapshot = builder.build(root, task_id="test_trunc")

            content = snapshot["selected_files"]["README.md"]["content"]
            self.assertTrue(content.endswith("[truncated]"))
            self.assertLessEqual(len(content), 200)  # 100 chars + truncation marker

    def test_snapshot_task_id_preserved(self):
        """Snapshot should preserve the task_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "app.py").write_text("print('hello')\n")

            builder = ProjectSnapshotBuilder()
            snapshot = builder.build(root, task_id="my_task_42")

            self.assertEqual(snapshot["task_id"], "my_task_42")


if __name__ == "__main__":
    unittest.main()
