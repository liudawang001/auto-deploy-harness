import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import pytest

from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.runner import RunnerModule
from auto_harness.runtime.environment import (
    ChildEnvironmentPolicy,
    is_secret_environment_name,
)


def test_child_environment_policy_removes_provider_and_cloud_secrets(tmp_path):
    source = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
        "DEEPSEEK_API_KEY": "TEST_ONLY_DEEPSEEK_SECRET",
        "HF_TOKEN": "TEST_ONLY_HF_SECRET",
        "AWS_SECRET_ACCESS_KEY": "TEST_ONLY_AWS_SECRET",
        "GITHUB_TOKEN": "TEST_ONLY_GITHUB_SECRET",
        "SAFE_VALUE": "not inherited by default",
    }
    child = ChildEnvironmentPolicy(source).build_for_service(
        home_dir=tmp_path / "home",
        extra={"AUTO_HARNESS_TRACE_ID": "trace-test"},
    )

    assert child["PATH"] == source["PATH"]
    assert child["LANG"] == "C.UTF-8"
    assert child["HOME"] == str(tmp_path / "home")
    assert child["AUTO_HARNESS_TRACE_ID"] == "trace-test"
    assert "SAFE_VALUE" not in child
    assert not any("TEST_ONLY" in value for value in child.values())
    assert "DEEPSEEK_API_KEY" not in child
    assert "HF_TOKEN" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert "GITHUB_TOKEN" not in child


@pytest.mark.parametrize(
    "name",
    [
        "DEEPSEEK_API_KEY",
        "AUTO_HARNESS_LLM_API_KEY",
        "HF_TOKEN",
        "XUNFEI_API_SECRET",
        "AWS_SESSION_TOKEN",
        "MY_PRIVATE_KEY",
        "DATABASE_PASSWORD",
    ],
)
def test_secret_environment_names_are_rejected(name):
    assert is_secret_environment_name(name)
    with pytest.raises(ValueError):
        ChildEnvironmentPolicy({"PATH": os.defpath}).build_for_install(
            extra={name: "TEST_ONLY_SECRET"},
        )


def test_runner_target_process_cannot_read_parent_provider_secret(monkeypatch):
    secret = "TEST_ONLY_PARENT_PROVIDER_SECRET"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "workspace" / "repo"
        repo.mkdir(parents=True)
        output = repo / "observed.json"
        script = repo / "app.py"
        script.write_text(
            "import json, os, time\n"
            "from pathlib import Path\n"
            "Path('observed.json').write_text(json.dumps({\n"
            "  'has_deepseek': 'DEEPSEEK_API_KEY' in os.environ,\n"
            "  'value': os.environ.get('DEEPSEEK_API_KEY', ''),\n"
            "  'phase': os.environ.get('AUTO_HARNESS_CHILD_PHASE', ''),\n"
            "}), encoding='utf-8')\n"
            "time.sleep(5)\n",
            encoding="utf-8",
        )
        command = [sys.executable, str(script)]
        allowed = [Path(sys.executable).name]
        result = RunnerModule().run(
            repo,
            {"run_candidates": [{"cmd": command, "expected_port": 0}]},
            execute=True,
            wait_seconds=0.2,
            allowed_commands=allowed,
        )
        try:
            for _ in range(20):
                if output.exists():
                    break
                time.sleep(0.05)
            observed = json.loads(output.read_text(encoding="utf-8"))
            assert observed == {
                "has_deepseek": False,
                "value": "",
                "phase": "service",
            }
        finally:
            pid = int(result.data.get("pid") or 0)
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def test_runner_allows_project_venv_console_script_and_reuses_install_home(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "workspace" / "repo"
    command = repo / ".venv" / "bin" / "demo-cli"
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    command.chmod(0o755)
    captured = {}

    class FakeProcess:
        pid = 321

        @staticmethod
        def poll():
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr("auto_harness.modules.runner.subprocess.Popen", fake_popen)

    result = RunnerModule().run(
        repo,
        {"run_candidates": [{"cmd": [str(command)], "expected_port": 0}]},
        execute=True,
        wait_seconds=0,
        allowed_commands=["python3"],
    )

    assert result.status == "passed"
    assert captured["cmd"] == [str(command)]
    assert captured["env"]["HOME"] == str(repo.parent / "install_home")


def test_env_deploy_retains_host_pip_cache_with_task_scoped_home(
    tmp_path, monkeypatch,
):
    repo = tmp_path / "workspace" / "repo"
    repo.mkdir(parents=True)
    pip_cache = tmp_path / "pip-cache"
    pip_cache.mkdir()
    captured = {}

    class Result:
        cmd = ["python3", "-m", "pip", "--version"]
        exit_code = 0
        stdout = "pip"
        stderr = ""
        timed_out = False

    def fake_runner(cmd, cwd, timeout_seconds, env):
        captured["env"] = env
        return Result()

    monkeypatch.setenv("PIP_CACHE_DIR", str(pip_cache))
    monkeypatch.setattr(
        "auto_harness.modules.env_deploy.run_command", fake_runner,
    )
    result = EnvDeployModule().deploy(
        repo,
        {"install_plan": [["python3", "-m", "pip", "--version"]]},
        execute=True,
        allowed_commands=["python3"],
    )

    assert result.status == "passed"
    assert captured["env"]["HOME"] == str(repo.parent / "install_home")
    assert captured["env"]["PIP_CACHE_DIR"] == str(pip_cache)
