import subprocess
import time
from pathlib import Path
from typing import Dict, List

from auto_harness.models.result import StageResult
from auto_harness.diagnostics import LogClassifier
from auto_harness.utils.commands import is_allowed_command
from auto_harness.utils.ports import is_port_open


class RunnerModule:
    def __init__(self, log_classifier: LogClassifier = None) -> None:
        self.log_classifier = log_classifier or LogClassifier()

    def run(
        self,
        repo_dir: Path,
        analysis: Dict,
        execute: bool = False,
        wait_seconds: int = 10,
        allowed_commands=None,
    ) -> StageResult:
        candidates: List[Dict] = analysis.get("run_candidates", [])
        if not candidates:
            return StageResult("runner", "uncertain", "no run candidate detected", {"run_candidates": []})
        candidate = candidates[0]
        if not execute:
            return StageResult("runner", "passed", "dry-run run candidate selected", {"candidate": candidate, "executed": False})
        allowed_commands = allowed_commands or []
        if not is_allowed_command(candidate["cmd"], allowed_commands):
            return StageResult(
                "runner",
                "failed",
                "command rejected by policy",
                {"cmd": candidate["cmd"], "allowed_commands": list(allowed_commands)},
                error="disallowed command: %s" % candidate["cmd"][0],
            )

        logs_dir = repo_dir.parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "runner.log"
        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            candidate["cmd"],
            cwd=str(repo_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.close()
        time.sleep(wait_seconds)
        port = int(candidate.get("expected_port") or 0)
        ready = bool(port and is_port_open("127.0.0.1", port))
        status = "passed" if proc.poll() is None else "failed"
        data = {
            "pid": proc.pid,
            "cmd": candidate["cmd"],
            "expected_port": port,
            "service_ready": ready,
            "log_path": str(log_path),
        }
        if status == "failed":
            try:
                data["diagnosis"] = self.log_classifier.classify(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
            except OSError:
                data["diagnosis"] = self.log_classifier.classify("")
        return StageResult("runner", status, "service process started" if status == "passed" else "service process exited", data)
