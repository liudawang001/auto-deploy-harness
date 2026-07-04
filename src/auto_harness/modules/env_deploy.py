from pathlib import Path
from typing import Dict, List

from auto_harness.models.result import StageResult
from auto_harness.diagnostics import LogClassifier
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
    ) -> StageResult:
        plan: List[List[str]] = analysis.get("install_plan", [])
        if not plan:
            return StageResult("env_deploy", "uncertain", "no install plan detected", {"commands": []})
        if not execute:
            return StageResult("env_deploy", "passed", "dry-run install plan generated", {"commands": plan, "executed": False})

        allowed_commands = allowed_commands or []
        command_results = []
        for cmd in plan:
            if not is_allowed_command(cmd, allowed_commands):
                return StageResult(
                    "env_deploy",
                    "failed",
                    "command rejected by policy",
                    {"cmd": cmd, "allowed_commands": list(allowed_commands)},
                    error="disallowed command: %s" % (cmd[0] if cmd else ""),
                )
            result = run_command(cmd, repo_dir, timeout_seconds=timeout_seconds)
            command_results.append({
                "cmd": result.cmd,
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
        return StageResult("env_deploy", "passed", "environment deployed", {"commands": command_results})
