import subprocess
import time
from pathlib import Path
from typing import Dict, List

from auto_harness.models.result import StageResult
from auto_harness.env import CondaBackend
from auto_harness.diagnostics import LogClassifier
from auto_harness.runtime import DockerSandboxBackend
from auto_harness.utils.commands import is_allowed_command
from auto_harness.utils.ports import is_port_open
from auto_harness.utils.files import short_hash


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
        execution_backend: str = "local",
        docker_image: str = "python:3.10-slim",
        docker_network: str = "bridge",
        docker_gpus: str = "none",
        docker_model_cache_dir: str = "",
        docker_security_options: Dict = None,
        stage_hints: Dict = None,
    ) -> StageResult:
        candidates: List[Dict] = analysis.get("run_candidates", [])
        if not candidates:
            return StageResult("runner", "uncertain", "no run candidate detected", {"run_candidates": []})
        # Apply plan hints: prefer_entrypoint_patterns for candidate ordering
        hints = stage_hints or {}
        prefer_patterns = hints.get("prefer_entrypoint_patterns", [])
        if prefer_patterns:
            candidates = self._reorder_candidates_by_hints(candidates, prefer_patterns)
        candidate = dict(candidates[0])
        candidate["env_solution"] = analysis.get("env_solution") if isinstance(analysis.get("env_solution"), dict) else {}
        effective_candidate, sandbox = self._effective_candidate(
            repo_dir,
            candidate,
            execution_backend,
            docker_image,
            docker_network,
            docker_gpus,
            docker_model_cache_dir,
            docker_security_options,
        )
        if not execute:
            return StageResult(
                "runner",
                "passed",
                "dry-run run candidate selected",
                {
                    "candidate": candidate,
                    "candidate_selection": self._candidate_selection(candidate),
                    "effective_candidate": effective_candidate,
                    "execution_backend": execution_backend,
                    "sandbox": sandbox,
                    "executed": False,
                },
            )
        allowed_commands = allowed_commands or []
        if not is_allowed_command(effective_candidate["cmd"], allowed_commands):
            return StageResult(
                "runner",
                "failed",
                "command rejected by policy",
            {
                "cmd": effective_candidate["cmd"],
                "original_cmd": candidate["cmd"],
                "candidate_selection": self._candidate_selection(candidate),
                "allowed_commands": list(allowed_commands),
                "execution_backend": execution_backend,
                "sandbox": sandbox,
                },
                error="disallowed command: %s" % effective_candidate["cmd"][0],
            )

        logs_dir = repo_dir.parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "runner.log"
        log_file = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            effective_candidate["cmd"],
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
            "cmd": effective_candidate["cmd"],
            "original_cmd": candidate["cmd"],
            "candidate_selection": self._candidate_selection(candidate),
            "expected_port": port,
            "service_ready": ready,
            "log_path": str(log_path),
            "execution_backend": execution_backend,
            "sandbox": sandbox,
        }
        if status == "failed":
            try:
                data["diagnosis"] = self.log_classifier.classify(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
            except OSError:
                data["diagnosis"] = self.log_classifier.classify("")
        return StageResult("runner", status, "service process started" if status == "passed" else "service process exited", data)

    def _reorder_candidates_by_hints(self, candidates: List[Dict], prefer_patterns: List[str]) -> List[Dict]:
        """Reorder candidates based on plan hints.

        Candidates matching prefer_entrypoint_patterns get boosted to the front.
        """
        if not prefer_patterns:
            return candidates

        matched = []
        unmatched = []
        for c in candidates:
            cmd = " ".join(c.get("cmd", []))
            if any(pattern in cmd for pattern in prefer_patterns):
                matched.append(c)
            else:
                unmatched.append(c)
        return matched + unmatched

    def _candidate_selection(self, candidate: Dict) -> Dict:
        return {
            "cmd": candidate.get("cmd", []),
            "score": float(candidate.get("score") or candidate.get("confidence") or 0),
            "score_reasons": list(candidate.get("score_reasons") or []),
            "selected_by": candidate.get("selected_by") or candidate.get("preferred_by") or "deterministic",
        }

    def _effective_candidate(
        self,
        repo_dir: Path,
        candidate: Dict,
        execution_backend: str,
        docker_image: str,
        docker_network: str,
        docker_gpus: str,
        docker_model_cache_dir: str,
        docker_security_options: Dict = None,
    ):
        if execution_backend != "docker":
            effective = dict(candidate)
            env_solution = candidate.get("env_solution") or {}
            if env_solution.get("backend") in ("conda", "mamba") and env_solution.get("environment_prefix"):
                spec = CondaBackend(backend=env_solution["backend"]).build_spec(repo_dir, env_solution, conda_file=(env_solution.get("conda_file") or {}))
                spec.prefix = env_solution["environment_prefix"]
                effective["cmd"] = CondaBackend(backend=env_solution["backend"]).run_cmd(spec, candidate.get("cmd", []))
            return effective, {"backend": "local", "environment_backend": env_solution.get("backend", "venv")}
        port = int(candidate.get("expected_port") or 0)
        container_name = "auto-harness-%s-%s" % (short_hash(str(repo_dir), 8), port or "svc")
        sandbox_command = DockerSandboxBackend.for_phase(
            "runtime",
            image=docker_image,
            network=docker_network,
            gpus=docker_gpus,
            model_cache_dir=Path(docker_model_cache_dir) if docker_model_cache_dir else None,
            **(docker_security_options or {}),
        ).wrap(
            repo_dir,
            candidate.get("cmd", []),
            ports=[port] if port else [],
            container_name=container_name,
        )
        effective = dict(candidate)
        effective["cmd"] = sandbox_command.effective_cmd
        return effective, sandbox_command.to_dict()
