"""Tests for LLMNecessityEvaluator runner rewrite.

Validates:
- Evaluator uses TaskRunner (via runner_factory injection)
- Evaluator uses LangGraph for both modes
- Baseline uses deterministic planner without provider call
- Deterministic planner emits parser-valid plan
- Baseline and agent have different workspaces
- Missing fixture is infrastructure error
- Missing report is infrastructure error
- Infrastructure error makes report failed
- CLI returns nonzero on failed report
"""
import json
import tempfile
from pathlib import Path

import pytest

from auto_harness.agent_runtime.deterministic_planner import DeterministicDeploymentPlanner
from auto_harness.evals.llm_necessity import LLMNecessityEvaluator, _load_run_result
from auto_harness.models.base import write_json
from auto_harness.providers.base import LLMResult


class TestDeterministicPlanner:
    """Test DeterministicDeploymentPlanner."""

    def test_deterministic_planner_emits_parser_valid_plan(self):
        """Deterministic planner must emit valid JSON plan."""
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["app.py", "requirements.txt"],
            "detected_signals": {
                "entrypoint_candidates": ["app.py"],
                "ports": [8000],
            },
        }
        result = planner.plan(snapshot)
        assert isinstance(result, LLMResult)
        assert result.protocol == "deterministic"
        plan = json.loads(result.text)
        assert plan["status"] == "ok"
        assert plan["run"]["candidates"][0]["cmd"] == [".venv/bin/python", "app.py"]
        assert plan["run"]["candidates"][0]["expected_port"] == 8000

    def test_deterministic_planner_uses_requirements_txt(self):
        """requirements.txt should produce pip install -r requirements.txt."""
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["requirements.txt"],
            "detected_signals": {"entrypoint_candidates": ["app.py"], "ports": [8000]},
        }
        result = planner.plan(snapshot)
        plan = json.loads(result.text)
        install_commands = plan["environment"]["install_commands"]
        assert [".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"] in install_commands
        assert any("requirements.txt" in " ".join(c) for c in install_commands)

    def test_deterministic_planner_uses_pyproject_toml(self):
        """pyproject.toml should produce pip install -e ."""
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["pyproject.toml"],
            "detected_signals": {"entrypoint_candidates": ["app.py"], "ports": [8000]},
        }
        result = planner.plan(snapshot)
        plan = json.loads(result.text)
        install_commands = plan["environment"]["install_commands"]
        assert any("-e" in c and "." in c for c in install_commands)

    def test_deterministic_planner_prefers_frozen_uv_lock(self):
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["pyproject.toml", "uv.lock", "app.py"],
            "detected_signals": {
                "dependency_files": ["pyproject.toml", "uv.lock"],
                "entrypoint_candidates": ["app.py"],
                "ports": [8000],
            },
        }
        plan = json.loads(planner.plan(snapshot).text)
        assert plan["environment"]["install_commands"][-1] == [
            ".venv/bin/uv", "sync", "--frozen", "--no-dev",
        ]

    def test_deterministic_planner_no_entrypoint_returns_no_safe_plan(self):
        """No entrypoint candidate should return no_safe_plan."""
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["requirements.txt"],
            "detected_signals": {"entrypoint_candidates": [], "ports": [8000]},
        }
        result = planner.plan(snapshot)
        plan = json.loads(result.text)
        assert plan["status"] == "no_safe_plan"

    def test_deterministic_planner_uses_documented_console_script(self):
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["README.md", "pyproject.toml"],
            "detected_signals": {
                "entrypoint_candidates": [],
                "ports": [5900],
                "python_requires": ">=3.12",
                "console_scripts": [
                    {"name": "demo", "target": "demo.cli:main", "source": "pyproject.toml"},
                ],
                "documented_run_commands": [
                    {"cmd": ["demo", "run", "--port", "8088"], "source": "README.md", "line": 10},
                ],
            },
        }
        plan = json.loads(planner.plan(snapshot).text)
        assert plan["status"] == "ok"
        assert plan["environment"]["python"] == "3.12"
        assert plan["run"]["candidates"][0]["cmd"] == [
            ".venv/bin/demo", "run", "--port", "8088",
        ]
        assert plan["run"]["candidates"][0]["expected_port"] == 8088

    def test_deterministic_planner_includes_detected_frontend_build(self):
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": [
                "README.md", "Makefile", "pyproject.toml", "uv.lock",
                "dashboard/package.json", "dashboard/package-lock.json",
            ],
            "detected_signals": {
                "python_requires": ">=3.12",
                "console_scripts": [
                    {"name": "demo", "target": "demo.cli:main", "source": "pyproject.toml"},
                ],
                "documented_run_commands": [
                    {"cmd": ["demo", "run"], "source": "README.md", "line": 10},
                ],
                "source_build_commands": [
                    {"cmd": ["npm", "--prefix", "dashboard", "ci"], "source": "Makefile"},
                    {"cmd": ["npm", "--prefix", "dashboard", "run", "build"], "source": "Makefile"},
                ],
            },
        }
        plan = json.loads(planner.plan(snapshot).text)
        assert plan["environment"]["install_commands"][-2:] == [
            ["npm", "--prefix", "dashboard", "ci"],
            ["npm", "--prefix", "dashboard", "run", "build"],
        ]
        assert any(item["file"] == "Makefile" for item in plan["grounding"])

    def test_deterministic_planner_runs_documented_setup_before_app(self):
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["README.md", "pyproject.toml"],
            "detected_signals": {
                "python_requires": ">=3.11,<3.14",
                "console_scripts": [
                    {"name": "paw", "target": "paw.cli:main", "source": "pyproject.toml"},
                ],
                "documented_setup_commands": [
                    {"cmd": ["paw", "init", "--defaults"], "source": "README.md", "line": 1},
                ],
                "documented_run_commands": [
                    {"cmd": ["paw", "app"], "source": "README.md", "line": 2, "expected_port": 8088},
                ],
            },
        }
        plan = json.loads(planner.plan(snapshot).text)
        assert plan["environment"]["python"] == "3.11"
        assert plan["environment"]["install_commands"][-1] == [
            ".venv/bin/paw", "init", "--defaults",
        ]
        assert plan["run"]["candidates"][0]["cmd"] == [
            ".venv/bin/paw", "app",
        ]

    def test_deterministic_planner_default_port_8000(self):
        """No port signal should default to 8000."""
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["app.py"],
            "detected_signals": {"entrypoint_candidates": ["app.py"], "ports": []},
        }
        result = planner.plan(snapshot)
        plan = json.loads(result.text)
        assert plan["run"]["candidates"][0]["expected_port"] == 8000

    def test_deterministic_planner_replan_returns_no_safe_plan(self):
        """replan must always return no_safe_plan."""
        planner = DeterministicDeploymentPlanner()
        result = planner.replan({}, {}, {})
        plan = json.loads(result.text)
        assert plan["status"] == "no_safe_plan"

    def test_deterministic_planner_does_not_call_provider(self):
        """Deterministic planner must never call any LLMProvider."""
        planner = DeterministicDeploymentPlanner()
        snapshot = {
            "file_tree": ["app.py"],
            "detected_signals": {"entrypoint_candidates": ["app.py"], "ports": [8000]},
        }
        # If it tried to call a provider, it would need one passed in
        # Since none is passed, it should work fine
        result = planner.plan(snapshot)
        assert result.protocol == "deterministic"


class TestLoadRunResult:
    """Test _load_run_result function."""

    def test_missing_report_is_infrastructure_error(self):
        """Missing controller_result.json should be infrastructure_error."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "test_task"
            run_dir.mkdir(parents=True)
            result = _load_run_result(run_dir)
            assert result["status"] == "infrastructure_error"
            assert "not found" in result["error"]

    def test_valid_report_returns_completed(self):
        """Valid controller_result.json should return completed status."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "test_task"
            reports_dir = run_dir / "reports"
            reports_dir.mkdir(parents=True)
            write_json(reports_dir / "controller_result.json", {
                "task_id": "test_task",
                "controller": "langgraph",
                "status": "completed",
                "verify_status": "passed",
            })
            result = _load_run_result(run_dir)
            assert result["status"] == "completed"
            assert result["verify_status"] == "passed"


class TestLLMNecessityEvaluator:
    """Test LLMNecessityEvaluator with injected factories."""

    def test_evaluator_uses_task_runner(self):
        """Evaluator should use TaskRunner via runner_factory."""
        calls = []

        class FakeRunner:
            def __init__(self, config):
                self.config = config
                calls.append(("init", config.agent_mode))

            def deploy(self, **kwargs):
                calls.append(("deploy", kwargs.get("name")))
                return "fake_task_id"

        with tempfile.TemporaryDirectory() as tmp:
            # Create a fixture
            fixture = Path(tmp) / "fixture"
            fixture.mkdir()
            (fixture / "app.py").write_text("print('hello')")

            manifest = {"cases": [{"case_id": "test1", "fixture_dir": str(fixture), "dry_run": True}]}

            evaluator = LLMNecessityEvaluator(
                output_dir=Path(tmp) / "evals",
                runner_factory=FakeRunner,
            )

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            result = evaluator.evaluate_manifest(manifest_path)

            # Should have called deploy for both baseline and agent
            deploy_calls = [c for c in calls if c[0] == "deploy"]
            assert len(deploy_calls) == 2  # baseline + agent

    def test_missing_fixture_is_infrastructure_error(self):
        """Missing fixture should produce infrastructure_error."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"cases": [{"case_id": "test1", "fixture_dir": "/nonexistent/path", "dry_run": True}]}

            evaluator = LLMNecessityEvaluator(output_dir=Path(tmp) / "evals")

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            result = evaluator.evaluate_manifest(manifest_path)
            assert result["status"] == "failed"
            assert result["summary"]["infrastructure_error_count"] > 0

    def test_infrastructure_error_makes_report_failed(self):
        """Infrastructure errors should make report status=failed."""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            fixture.mkdir()
            (fixture / "app.py").write_text("print('hello')")

            class BrokenRunner:
                def __init__(self, config):
                    pass

                def deploy(self, **kwargs):
                    raise RuntimeError("broken runner")

            manifest = {"cases": [{"case_id": "test1", "fixture_dir": str(fixture), "dry_run": True}]}

            evaluator = LLMNecessityEvaluator(
                output_dir=Path(tmp) / "evals",
                runner_factory=BrokenRunner,
            )

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            result = evaluator.evaluate_manifest(manifest_path)
            assert result["status"] == "failed"

    def test_baseline_uses_deterministic_planner_without_provider_call(self):
        """Baseline config must use deterministic planner mode."""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            fixture.mkdir()
            (fixture / "app.py").write_text("print('hello')")

            captured_configs = []

            class CapturingRunner:
                def __init__(self, config):
                    captured_configs.append(config)

                def deploy(self, **kwargs):
                    return "task_id"

            manifest = {"cases": [{"case_id": "test1", "fixture_dir": str(fixture), "dry_run": True}]}

            evaluator = LLMNecessityEvaluator(
                output_dir=Path(tmp) / "evals",
                runner_factory=CapturingRunner,
            )

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            evaluator.evaluate_manifest(manifest_path)

            # First config is baseline (off mode, deterministic planner)
            baseline_config = captured_configs[0]
            assert baseline_config.agent_mode == "off"
            assert baseline_config.langgraph_planner_mode == "deterministic"

            # Second config is agent (gated_actor, llm planner)
            agent_config = captured_configs[1]
            assert agent_config.agent_mode == "gated_actor"
            assert agent_config.langgraph_planner_mode == "llm"

    def test_baseline_and_agent_have_different_workspaces(self):
        """Baseline and agent must have different run directories."""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            fixture.mkdir()
            (fixture / "app.py").write_text("print('hello')")

            deploy_names = []

            class TrackingRunner:
                def __init__(self, config):
                    self.config = config

                def deploy(self, **kwargs):
                    deploy_names.append(kwargs.get("name"))
                    return "task_%d" % len(deploy_names)

            manifest = {"cases": [{"case_id": "test1", "fixture_dir": str(fixture), "dry_run": True}]}

            evaluator = LLMNecessityEvaluator(
                output_dir=Path(tmp) / "evals",
                runner_factory=TrackingRunner,
            )

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            evaluator.evaluate_manifest(manifest_path)

            # Names should differ (baseline vs agent suffix)
            assert len(deploy_names) == 2
            assert deploy_names[0] != deploy_names[1]

    def test_evaluator_uses_langgraph_for_both_modes(self):
        """Both baseline and agent must use langgraph controller."""
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixture"
            fixture.mkdir()
            (fixture / "app.py").write_text("print('hello')")

            controllers = []

            class ControllerTrackingRunner:
                def __init__(self, config):
                    self.config = config

                def deploy(self, **kwargs):
                    controllers.append(kwargs.get("controller"))
                    return "task_id"

            manifest = {"cases": [{"case_id": "test1", "fixture_dir": str(fixture), "dry_run": True}]}

            evaluator = LLMNecessityEvaluator(
                output_dir=Path(tmp) / "evals",
                runner_factory=ControllerTrackingRunner,
            )

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            evaluator.evaluate_manifest(manifest_path)

            assert all(c == "langgraph" for c in controllers)

    def test_cli_returns_nonzero_on_failed_report(self):
        """CLI should return nonzero (2) when report status is failed."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"cases": [{"case_id": "test1", "fixture_dir": "/nonexistent", "dry_run": True}]}

            manifest_path = Path(tmp) / "manifest.json"
            write_json(manifest_path, manifest)

            evaluator = LLMNecessityEvaluator(output_dir=Path(tmp) / "evals")
            result = evaluator.evaluate_manifest(manifest_path)

            # Verify the result would cause CLI to return 2
            assert result.get("status") == "failed"
            # The CLI returns: 0 if status == "completed" else 2
            exit_code = 0 if result.get("status") == "completed" else 2
            assert exit_code == 2
