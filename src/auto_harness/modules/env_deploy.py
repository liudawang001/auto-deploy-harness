from pathlib import Path
from typing import Dict, List

from auto_harness.models.result import StageResult
from auto_harness.diagnostics import LogClassifier
from auto_harness.runtime import DockerSandboxBackend
from auto_harness.utils.commands import is_allowed_command
from auto_harness.utils.shell import run_command


class EnvDeployModule:
    def __init__(self, log_classifier: LogClassifier = None) -> None:
        self.log_classifier = log_classifier or LogClassifier()

    def deploy(
        self,
        repo_dir: Path,
        analysis: Dict,
        execute: bool = False,
        timeout_seconds: int = 900,
        allowed_commands=None,
        execution_backend: str = "local",
        docker_image: str = "python:3.10-slim",
        docker_network: str = "bridge",
        docker_gpus: str = "none",
        docker_model_cache_dir: str = "",
        docker_security_options: Dict = None,
    ) -> StageResult:
        plan: List[List[str]] = analysis.get("install_plan", [])
        env_solution = analysis.get("env_solution") if isinstance(analysis.get("env_solution"), dict) else {}
        conda_plan = env_solution.get("conda") if isinstance(env_solution.get("conda"), dict) else {}
        if env_solution.get("backend") in ("conda", "mamba") and conda_plan.get("commands"):
            plan = [list(cmd) for cmd in conda_plan.get("commands") or []]
        if not plan:
            return StageResult("env_deploy", "uncertain", "no install plan detected", {"commands": []})
        effective_plan, sandbox = self._effective_plan(
            repo_dir,
            plan,
            execution_backend,
            docker_image,
            docker_network,
            docker_gpus,
            docker_model_cache_dir,
            docker_security_options,
        )
        if not execute:
            return StageResult(
                "env_deploy",
                "passed",
                "dry-run install plan generated",
                {
                    "commands": plan,
                    "effective_commands": effective_plan,
                    "execution_backend": execution_backend,
                    "environment_backend": env_solution.get("backend", "venv"),
                    "environment_prefix": env_solution.get("environment_prefix", ""),
                    "environment_python": env_solution.get("environment_python", ""),
                    "environment_solution": env_solution,
                    "sandbox": sandbox,
                    "executed": False,
                },
            )

        allowed_commands = allowed_commands or []
        command_results = []
        for original_cmd, cmd in zip(plan, effective_plan):
            if not is_allowed_command(cmd, allowed_commands):
                return StageResult(
                    "env_deploy",
                    "failed",
                    "command rejected by policy",
                    {
                        "cmd": cmd,
                        "original_cmd": original_cmd,
                        "allowed_commands": list(allowed_commands),
                        "execution_backend": execution_backend,
                        "environment_backend": env_solution.get("backend", "venv"),
                        "environment_prefix": env_solution.get("environment_prefix", ""),
                        "environment_python": env_solution.get("environment_python", ""),
                        "sandbox": sandbox,
                    },
                    error="disallowed command: %s" % (cmd[0] if cmd else ""),
                )
            result = run_command(cmd, repo_dir, timeout_seconds=timeout_seconds)
            command_results.append({
                "cmd": result.cmd,
                "original_cmd": original_cmd,
                "exit_code": result.exit_code,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
                "timed_out": result.timed_out,
            })
            if result.exit_code != 0:
                diagnosis = self.log_classifier.classify(result.stderr + "\n" + result.stdout)
                return StageResult(
                    "env_deploy",
                    "failed",
                    "dependency installation failed",
                {"commands": command_results, "diagnosis": diagnosis},
                    error=result.stderr[-2000:],
                )
        return StageResult(
            "env_deploy",
            "passed",
            "environment deployed",
            {
                "commands": command_results,
                "execution_backend": execution_backend,
                "environment_backend": env_solution.get("backend", "venv"),
                "environment_prefix": env_solution.get("environment_prefix", ""),
                "environment_python": env_solution.get("environment_python", ""),
                "environment_solution": env_solution,
                "sandbox": sandbox,
            },
        )

    def _effective_plan(
        self,
        repo_dir: Path,
        plan: List[List[str]],
        execution_backend: str,
        docker_image: str,
        docker_network: str,
        docker_gpus: str,
        docker_model_cache_dir: str,
        docker_security_options: Dict = None,
    ):
        if execution_backend != "docker":
            return [list(cmd) for cmd in plan], {"backend": "local"}
        backend = DockerSandboxBackend.for_phase(
            "install",
            image=docker_image,
            network=docker_network,
            gpus=docker_gpus,
            model_cache_dir=Path(docker_model_cache_dir) if docker_model_cache_dir else None,
            **(docker_security_options or {}),
        )
        sandbox_commands = [backend.wrap(repo_dir, cmd).to_dict() for cmd in plan]
        return [item["effective_cmd"] for item in sandbox_commands], {
            "backend": "docker",
            "image": docker_image,
            "network": docker_network,
            "gpus": docker_gpus,
            "model_cache_dir": docker_model_cache_dir,
            "commands": sandbox_commands,
        }
