import subprocess
import time
import json
import socket
from pathlib import Path
from typing import Dict, List

from auto_harness.models.result import StageResult
from auto_harness.env import CondaBackend
from auto_harness.diagnostics import LogClassifier
from auto_harness.runtime import (
    ChildEnvironmentPolicy,
    DockerSandboxBackend,
    local_docker_environment,
)
from auto_harness.utils.commands import is_allowed_command
from auto_harness.utils.ports import is_port_open
from auto_harness.utils.files import short_hash
from auto_harness.command_auth import CommandAuthorizationEngine, CommandRegistry
from auto_harness.command_auth.schemas import canonical_hash, sandbox_policy_fingerprint


class RunnerModule:
    def __init__(self, log_classifier: LogClassifier = None, child_environment_policy=None) -> None:
        self.log_classifier = log_classifier or LogClassifier()
        self.child_environment_policy = child_environment_policy or ChildEnvironmentPolicy()

    def run(
        self,
        repo_dir: Path,
        analysis: Dict,
        execute: bool = False,
        wait_seconds: int = 30,
        allowed_commands=None,
        execution_backend: str = "local",
        docker_image: str = "python:3.13-slim",
        docker_network: str = "bridge",
        docker_gpus: str = "none",
        docker_model_cache_dir: str = "",
        docker_security_options: Dict = None,
        stage_hints: Dict = None,
        run_dir: Path = None,
        max_candidate_attempts: int = 3,
    ) -> StageResult:
        candidates: List[Dict] = analysis.get("run_candidates", [])
        if not candidates:
            return StageResult("runner", "uncertain", "no run candidate detected", {"run_candidates": []})
        # Apply plan hints: prefer_entrypoint_patterns for candidate ordering
        hints = stage_hints or {}
        prefer_patterns = hints.get("prefer_entrypoint_patterns", [])
        if prefer_patterns:
            candidates = self._reorder_candidates_by_hints(candidates, prefer_patterns)
        registry_scoped = bool(
            isinstance(analysis.get("command_registry"), dict)
            and analysis.get("command_registry")
        )
        authorization_network = "none" if registry_scoped else docker_network
        candidate, authorization_attempts = self._select_authorized_candidate(
            repo_dir,
            candidates,
            analysis,
            execution_backend,
            require_executable=execute,
            allow_approval_preview=not execute,
            max_candidate_attempts=max_candidate_attempts,
            run_dir=run_dir,
            sandbox_fingerprint=sandbox_policy_fingerprint(
                phase="runtime", image=docker_image, network=authorization_network,
                gpus=docker_gpus, model_cache_dir=docker_model_cache_dir,
                security_options=docker_security_options,
            ),
        )
        self._write_authorization_attempts(run_dir, authorization_attempts)
        if candidate is None:
            return StageResult(
                "runner",
                "failed" if execute else "uncertain",
                "no authorized run candidate",
                {"authorization_attempts": authorization_attempts},
                error="no_safe_command_candidate" if execute else "",
            )
        if execute:
            self._write_candidate_attempt(run_dir, analysis, candidate, "authorized")
        candidate["env_solution"] = analysis.get("env_solution") if isinstance(analysis.get("env_solution"), dict) else {}
        effective_backend = candidate.get("required_backend") or execution_backend
        effective_network = (
            "none" if candidate.get("network_profile") == "none"
            else docker_network
        )
        effective_candidate, sandbox = self._effective_candidate(
            repo_dir,
            candidate,
            effective_backend,
            docker_image,
            effective_network,
            docker_gpus,
            docker_model_cache_dir,
            docker_security_options,
            remap_occupied_port=execute,
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
                    "execution_backend": effective_backend,
                    "sandbox": sandbox,
                    "authorization_attempts": authorization_attempts,
                    "executed": False,
                },
            )
        allowed_commands = allowed_commands or []
        registry_authorized = bool(candidate.get("command_candidate_id"))
        allowed = registry_authorized or is_allowed_command(effective_candidate["cmd"], allowed_commands)
        command_path = Path(effective_candidate["cmd"][0]) if effective_candidate.get("cmd") else Path()
        if not command_path.is_absolute():
            command_path = repo_dir / command_path
        if not allowed and command_path.is_file():
            try:
                command_path.resolve().relative_to(repo_dir.resolve())
                # A console script created by the accepted install plan lives
                # under the Harness-owned project virtualenv.  Treat it like
                # the small built-in project entrypoint allowlist while still
                # rejecting arbitrary repository executables.
                in_project_venv = False
                try:
                    command_path.resolve().relative_to(
                        (repo_dir / ".venv" / "bin").resolve()
                    )
                    in_project_venv = command_path.name not in {
                        "bash", "sh", "zsh", "fish", "csh", "tcsh",
                    }
                except ValueError:
                    pass
                allowed = in_project_venv or command_path.name in {
                    "python", "python3", "uvicorn", "streamlit",
                }
            except ValueError:
                allowed = False
        if not allowed:
            return StageResult(
                "runner",
                "failed",
                "command rejected by policy",
            {
                "cmd": effective_candidate["cmd"],
                "original_cmd": candidate["cmd"],
                "candidate_selection": self._candidate_selection(candidate),
                "allowed_commands": list(allowed_commands),
                "execution_backend": effective_backend,
                "sandbox": sandbox,
                "authorization_attempts": authorization_attempts,
                },
                error="disallowed command: %s" % effective_candidate["cmd"][0],
            )

        if candidate.get("_requires_command_approval"):
            consumed = self._consume_command_approval(run_dir, candidate)
            if not consumed:
                return self._fallback_after_failure(
                    repo_dir, analysis, candidates, candidate,
                    "approval_already_consumed", authorization_attempts,
                    execute, wait_seconds, allowed_commands, execution_backend,
                    docker_image, docker_network, docker_gpus,
                    docker_model_cache_dir, docker_security_options, hints,
                    run_dir, max_candidate_attempts,
                )

        logs_dir = repo_dir.parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "runner.log"
        # Install-time initialization and runtime must share the same
        # task-scoped HOME.  Otherwise commands such as `qwenpaw init` write
        # configuration that the deployed process cannot see.  Provider
        # credentials are still filtered by ChildEnvironmentPolicy.
        child_env = self.child_environment_policy.build_for_service(
            home_dir=repo_dir.parent / "install_home",
            extra=(
                local_docker_environment()
                if effective_backend == "docker" else None
            ),
        )
        log_file = log_path.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                effective_candidate["cmd"],
                cwd=str(repo_dir),
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                # A deployed service must outlive the short-lived Harness CLI
                # process and its controlling terminal/session.
                start_new_session=True,
            )
        except OSError as exc:
            log_file.close()
            return self._fallback_after_failure(
                repo_dir, analysis, candidates, candidate,
                "process_start_failed:%s" % type(exc).__name__, authorization_attempts,
                execute, wait_seconds, allowed_commands, execution_backend,
                docker_image, docker_network, docker_gpus,
                docker_model_cache_dir, docker_security_options, hints,
                run_dir, max_candidate_attempts,
            )
        log_file.close()
        port = int(effective_candidate.get("expected_port") or 0)
        deadline = time.monotonic() + max(0, float(wait_seconds))
        ready = bool(port and is_port_open("127.0.0.1", port))
        while proc.poll() is None and port and not ready and time.monotonic() < deadline:
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            ready = is_port_open("127.0.0.1", port)
        # A pre-existing listener can make the first port probe look ready
        # before the child finishes binding. Docker's published-port proxy can
        # also accept connections before the application inside the container
        # has finished starting. Require a survival window after readiness so
        # neither case is reported as a successful deployment prematurely.
        if proc.poll() is None and ready:
            stability_seconds = 10.0 if effective_backend == "docker" else 1.0
            stability_deadline = time.monotonic() + min(
                stability_seconds,
                max(1.0, float(wait_seconds)),
            )
            while proc.poll() is None and time.monotonic() < stability_deadline:
                time.sleep(min(0.1, stability_deadline - time.monotonic()))
            ready = proc.poll() is None and is_port_open("127.0.0.1", port)
        status = "passed" if proc.poll() is None else "failed"
        data = {
            "pid": proc.pid,
            "cmd": effective_candidate["cmd"],
            "original_cmd": candidate["cmd"],
            "candidate_selection": self._candidate_selection(candidate),
            "expected_port": port,
            "service_ready": ready,
            "log_path": str(log_path),
            "execution_backend": effective_backend,
            "sandbox": sandbox,
            "authorization_attempts": authorization_attempts,
        }
        if status == "failed":
            try:
                data["diagnosis"] = self.log_classifier.classify(log_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
            except OSError:
                data["diagnosis"] = self.log_classifier.classify("")
            return self._fallback_after_failure(
                repo_dir, analysis, candidates, candidate,
                "service_process_exited", authorization_attempts,
                execute, wait_seconds, allowed_commands, execution_backend,
                docker_image, docker_network, docker_gpus,
                docker_model_cache_dir, docker_security_options, hints,
                run_dir, max_candidate_attempts,
                terminal_data=data,
            )
        return StageResult("runner", status, "service process started" if status == "passed" else "service process exited", data)

    def run_model_runtime(
        self,
        *,
        run_dir,
        task_id,
        runtime_plan,
        bundle,
        execute=False,
        command_runner=None,
        docker_network="bridge",
        docker_security_options=None,
        operation_id="",
    ) -> StageResult:
        """Start or plan the managed vLLM inference container (Document B).

        Builds the deterministic Docker command from the runtime plan, attaches
        ownership labels, executes ``docker run -d``, captures the container id,
        and verifies the labels/config match. It does not wait for readiness —
        that is the Startup Readiness Gate (Document B Phase B5).
        """
        from auto_harness.runtime import DockerSandboxBackend

        security = dict(docker_security_options or {})
        labels = {
            "auto-harness.task-id": str(task_id),
            "auto-harness.operation-id": str(operation_id or ""),
            "auto-harness.plan-hash": runtime_plan.plan_hash,
            "auto-harness.model-hash": runtime_plan.resolved_model_hash,
        }
        gpu_index = (list(runtime_plan.gpu_indexes) or [0])[0]
        backend = DockerSandboxBackend.for_model_runtime(
            image=runtime_plan.image,
            gpu_index=gpu_index,
            network=docker_network,
            memory=security.get("memory", "32g"),
            cpus=security.get("cpus", 8.0),
            pids_limit=security.get("pids_limit", 1024),
            tmpfs_size=security.get("tmpfs_size", "1g"),
            read_only_rootfs=True,
            user=security.get("user", ""),
        )
        sandbox = backend.wrap_model_runtime(
            model_host_dir=runtime_plan.model_host_path,
            host_port=runtime_plan.expected_port,
            command=runtime_plan.command,
            container_name=runtime_plan.container_name,
            labels=labels,
            shm_size=security.get("shm_size", "8g"),
        )

        data = {
            "container_name": runtime_plan.container_name,
            "runtime_plan_hash": runtime_plan.plan_hash,
            "model_identity": runtime_plan.model_identity,
            "image_digest": runtime_plan.image_digest,
            "gpu_indexes": list(runtime_plan.gpu_indexes),
            "expected_port": runtime_plan.expected_port,
            "sandbox": sandbox.to_dict(),
            "executed": False,
        }
        if not execute:
            return StageResult(
                "runner",
                "passed",
                "dry-run managed model runtime plan",
                data,
            )

        runner = command_runner or self._subprocess_command_runner()
        started = runner(sandbox.effective_cmd)
        if started.get("exit_code") != 0:
            return StageResult(
                "runner",
                "failed",
                "docker run failed",
                {**data, "executed": True, "stderr_tail": (started.get("stderr") or "")[-500:]},
                error="docker_run_failed",
            )
        container_id = (started.get("stdout") or "").strip().splitlines()
        container_id = container_id[-1].strip() if container_id else ""
        if not container_id:
            return StageResult(
                "runner", "failed", "docker run returned no container id", data,
                error="container_id_missing",
            )

        verified = self._verify_model_runtime_container(
            runner, container_id, labels, runtime_plan
        )
        if not verified:
            return StageResult(
                "runner", "failed", "container labels/config mismatch after start",
                {**data, "executed": True, "container_id": container_id},
                error="container_verification_failed",
            )

        data["executed"] = True
        data["container_id"] = container_id
        data["ready"] = False  # readiness is decided by the Startup Readiness Gate
        return StageResult(
            "runner", "passed", "managed model runtime container started", data,
        )

    @staticmethod
    def _subprocess_command_runner():
        def _run(cmd):
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
            }
        return _run

    @staticmethod
    def _verify_model_runtime_container(runner, container_id, labels, runtime_plan) -> bool:
        inspected = runner(["docker", "inspect", container_id])
        if inspected.get("exit_code") != 0:
            return False
        try:
            data = json.loads(inspected.get("stdout") or "")[0]
        except (ValueError, IndexError, TypeError):
            return False
        actual_labels = data.get("Config", {}).get("Labels", {}) or {}
        for key, value in labels.items():
            if actual_labels.get(key) != value:
                return False
        return True

    def _select_authorized_candidate(
        self, repo_dir, candidates, analysis, execution_backend, require_executable,
        max_candidate_attempts=3, sandbox_fingerprint="", run_dir=None,
        allow_approval_preview=False,
    ):
        registry_data = analysis.get("command_registry")
        scope = str(analysis.get("command_registry_scope") or "")
        if not isinstance(registry_data, dict) or not registry_data:
            fallback = self._first_unattempted_candidate(
                candidates, analysis, execution_backend, sandbox_fingerprint,
            )
            return (dict(fallback), []) if fallback is not None else (None, [])
        registry = CommandRegistry.from_dict(registry_data)
        # A registry compiled from an explicit deployment contract is
        # authoritative: undeclared commands are rejected.  A discovery-mode
        # registry is additive evidence; undeclared candidates keep the
        # pre-existing deterministic path instead of being rejected.
        discovery_mode = scope == "discovery"
        by_id = {item.candidate_id: item for item in registry.candidates}
        engine = CommandAuthorizationEngine()
        attempts = []
        attempted_keys = {
            str(item) for item in (analysis.get("_attempted_command_keys") or [])
        }
        undeclared_fallback = None
        pending_approval = False
        for raw in candidates[:max(1, int(max_candidate_attempts))]:
            candidate = dict(raw)
            key = self._attempt_key(
                analysis, candidate, execution_backend, sandbox_fingerprint,
            )
            if key in attempted_keys:
                attempts.append({
                    "candidate_id": candidate.get("command_candidate_id", ""),
                    "verdict": "candidate_rejected",
                    "reason_code": "duplicate_attempt_key_skipped",
                })
                continue
            declared = by_id.get(candidate.get("command_candidate_id", ""))
            if declared is None:
                declared = registry.candidate_for_argv(candidate.get("cmd", []))
            if declared is None:
                attempts.append({
                    "candidate_id": candidate.get("id", ""),
                    "verdict": "candidate_rejected",
                    "reason_code": "repository_command_not_declared",
                })
                if discovery_mode and undeclared_fallback is None:
                    undeclared_fallback = candidate
                continue
            approval = analysis.get("command_approval") or None
            if (
                declared.source_kind in {
                    "make_target", "repository_script", "python_entrypoint", "manifest_command",
                }
                and self._command_approval_consumed(
                    run_dir, declared, registry.repository_fingerprint,
                )
                and isinstance(approval, dict)
            ):
                approval = {**approval, "execution_count": 1}
            decision = engine.authorize(
                declared,
                registry,
                repo_dir=Path(repo_dir),
                execution_backend=execution_backend,
                sandbox_policy_fingerprint=sandbox_fingerprint,
                approval=approval,
                require_executable=require_executable,
                environment_ownership_marker=(
                    Path(run_dir) / "environment" / "venv_owner.json"
                    if run_dir else None
                ),
            )
            attempts.append(decision.to_dict())
            if decision.verdict == "auto_allowed" or (
                allow_approval_preview and decision.verdict == "approval_required"
            ):
                candidate["cmd"] = list(decision.normalized_argv)
                candidate["command_candidate_id"] = declared.candidate_id
                candidate["required_backend"] = decision.effective_backend
                candidate["network_profile"] = declared.network_profile
                candidate["filesystem_profile"] = declared.filesystem_profile
                candidate["_requires_command_approval"] = decision.required_approval
                candidate["_authorization_operation_id"] = decision.operation_id
                return candidate, attempts
            if decision.verdict == "approval_required":
                pending_approval = True
        if discovery_mode and undeclared_fallback is not None and not pending_approval:
            return undeclared_fallback, attempts
        return None, attempts

    @staticmethod
    def _attempt_key(analysis, candidate, execution_backend, sandbox_fingerprint):
        return canonical_hash({
            "repository_fingerprint": str(analysis.get("repository_fingerprint") or ""),
            "argv": list(candidate.get("cmd") or []),
            "backend": str(candidate.get("required_backend") or execution_backend),
            "sandbox": str(sandbox_fingerprint),
        })

    @staticmethod
    def _first_unattempted_candidate(candidates, analysis, execution_backend, sandbox_fingerprint):
        attempted_keys = {
            str(item) for item in (analysis.get("_attempted_command_keys") or [])
        }
        for raw in candidates:
            key = canonical_hash({
                "repository_fingerprint": str(analysis.get("repository_fingerprint") or ""),
                "argv": list(raw.get("cmd") or []),
                "backend": str(raw.get("required_backend") or execution_backend),
                "sandbox": str(sandbox_fingerprint),
            })
            if key not in attempted_keys:
                return raw
        return None

    @staticmethod
    def _approval_consumption_path(run_dir, operation_id):
        return Path(run_dir) / "approvals" / ("consumed_%s.json" % operation_id) if run_dir and operation_id else None

    def _command_approval_consumed(self, run_dir, declared, repository_fingerprint):
        from auto_harness.command_auth.approval import command_operation_id
        path = self._approval_consumption_path(
            run_dir, command_operation_id(declared, repository_fingerprint)
        )
        return bool(path and path.exists())

    def _consume_command_approval(self, run_dir, candidate):
        operation_id = candidate.get("_authorization_operation_id", "")
        path = self._approval_consumption_path(run_dir, operation_id)
        if path is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump({
                    "operation_id": operation_id,
                    "candidate_id": candidate.get("command_candidate_id", ""),
                    "consumed_at": time.time(),
                }, handle, ensure_ascii=False, sort_keys=True)
            return True
        except FileExistsError:
            return False

    def _fallback_after_failure(
        self, repo_dir, analysis, candidates, candidate, reason, attempts,
        execute, wait_seconds, allowed_commands, execution_backend,
        docker_image, docker_network, docker_gpus, docker_model_cache_dir,
        docker_security_options, hints, run_dir, max_candidate_attempts,
        terminal_data=None,
    ):
        selected_index = next((
            index for index, item in enumerate(candidates)
            if item.get("id") == candidate.get("id")
            and item.get("cmd") == candidate.get("cmd")
        ), len(candidates) - 1)
        remaining = candidates[selected_index + 1:]
        remaining_budget = max(0, int(max_candidate_attempts) - len(attempts))
        diagnosis = (terminal_data or {}).get("diagnosis") or {}
        failure_signature = "%s:%s" % (
            reason,
            str(diagnosis.get("category") or diagnosis.get("error_class") or ""),
        )
        registry_scoped = bool(
            isinstance(analysis.get("command_registry"), dict)
            and analysis.get("command_registry")
        )
        authorization_network = "none" if registry_scoped else docker_network
        attempt_key = self._attempt_key(
            analysis, candidate, execution_backend, sandbox_policy_fingerprint(
                phase="runtime", image=docker_image, network=authorization_network,
                gpus=docker_gpus, model_cache_dir=docker_model_cache_dir,
                security_options=docker_security_options,
            ),
        )
        fallback_record = {
            "candidate_id": candidate.get("command_candidate_id") or candidate.get("id", ""),
            "reason": reason,
            "failure_signature": failure_signature,
            "attempt_key": attempt_key,
        }
        self._write_fallback(run_dir, fallback_record)
        self._write_candidate_attempt(
            run_dir, analysis, candidate,
            "fallback_after_start_failure" if terminal_data is not None else "fallback_before_start",
            failure_signature=failure_signature,
        )
        if remaining and remaining_budget:
            next_analysis = dict(analysis)
            next_analysis["run_candidates"] = remaining
            next_analysis["_attempted_command_keys"] = list(
                analysis.get("_attempted_command_keys") or []
            ) + [attempt_key]
            result = self.run(
                repo_dir, next_analysis, execute=execute,
                wait_seconds=wait_seconds, allowed_commands=allowed_commands,
                execution_backend=execution_backend, docker_image=docker_image,
                docker_network=docker_network, docker_gpus=docker_gpus,
                docker_model_cache_dir=docker_model_cache_dir,
                docker_security_options=docker_security_options,
                stage_hints=hints, run_dir=run_dir,
                max_candidate_attempts=remaining_budget,
            )
            result.data["authorization_attempts"] = attempts + list(
                result.data.get("authorization_attempts", [])
            )
            result.data["fallbacks"] = [fallback_record] + list(
                result.data.get("fallbacks", [])
            )
            return result
        data = dict(terminal_data or {})
        data.setdefault("authorization_attempts", attempts)
        data["fallbacks"] = [fallback_record]
        if terminal_data is not None:
            return StageResult(
                "runner", "failed", "service process exited", data,
            )
        return StageResult(
            "runner", "failed", "all authorized run candidates failed",
            data, error=reason,
        )

    @staticmethod
    def _write_authorization_attempts(run_dir, attempts):
        if not run_dir or not attempts:
            return
        path = Path(run_dir) / "reports" / "command_attempts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for attempt in attempts:
                handle.write(json.dumps(attempt, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_fallback(run_dir, fallback):
        if not run_dir:
            return
        path = Path(run_dir) / "reports" / "command_fallbacks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(fallback, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_candidate_attempt(run_dir, analysis, candidate, outcome, failure_signature=""):
        """Phase B4: deployment-candidate level attempt journal."""
        if not run_dir:
            return
        registry_data = analysis.get("command_registry") or {}
        source_kind = str(candidate.get("source_kind") or "")
        if not source_kind and isinstance(registry_data, dict):
            wanted = str(candidate.get("command_candidate_id") or "")
            for item in registry_data.get("candidates", []):
                if isinstance(item, dict) and item.get("candidate_id") == wanted:
                    source_kind = str(item.get("source_kind") or "")
                    break
        record = {
            "outcome": outcome,
            "candidate_id": candidate.get("command_candidate_id") or candidate.get("id", ""),
            "argv": list(candidate.get("cmd") or []),
            "source_kind": source_kind,
            "expected_port": int(candidate.get("expected_port") or 0),
            "required_backend": str(candidate.get("required_backend") or ""),
            "repository_fingerprint": str(analysis.get("repository_fingerprint") or ""),
            "failure_signature": failure_signature,
        }
        path = Path(run_dir) / "reports" / "candidate_attempts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

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
        remap_occupied_port: bool = True,
    ):
        if execution_backend != "docker":
            effective = dict(candidate)
            env_solution = candidate.get("env_solution") or {}
            if env_solution.get("backend") in ("conda", "mamba") and env_solution.get("environment_prefix"):
                raw_cmd = candidate.get("cmd", [])
                executable = repo_dir / raw_cmd[0] if raw_cmd else None
                if executable is None or not executable.is_file():
                    spec = CondaBackend(backend=env_solution["backend"]).build_spec(repo_dir, env_solution, conda_file=(env_solution.get("conda_file") or {}))
                    spec.prefix = env_solution["environment_prefix"]
                    effective["cmd"] = CondaBackend(backend=env_solution["backend"]).run_cmd(spec, raw_cmd)
            return effective, {"backend": "local", "environment_backend": env_solution.get("backend", "venv")}
        port = int(candidate.get("expected_port") or 0)
        host_port, remapped = (
            self._select_host_port(port)
            if remap_occupied_port
            else (port, False)
        )
        container_name = "auto-harness-%s-%s" % (
            short_hash(str(repo_dir), 8), host_port or "svc",
        )
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
            ports=[host_port] if host_port else [],
            port_mappings=[(host_port, port)] if host_port and port else [],
            container_name=container_name,
        )
        effective = dict(candidate)
        effective["cmd"] = sandbox_command.effective_cmd
        effective["expected_port"] = host_port
        if remapped:
            effective["container_port"] = port
            effective["port_remapped"] = True
        return effective, sandbox_command.to_dict()

    @staticmethod
    def _select_host_port(container_port: int):
        if not container_port or not is_port_open("127.0.0.1", container_port):
            return container_port, False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1]), True
