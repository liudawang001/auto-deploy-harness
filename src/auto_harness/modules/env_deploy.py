import sys
import os
from pathlib import Path
from typing import Dict, List

from auto_harness.env.ownership import EnvironmentOwnership
from auto_harness.env.postcheck import EnvironmentPostchecker
from auto_harness.models.result import StageResult
from auto_harness.models.base import write_json
from auto_harness.diagnostics import LogClassifier
from auto_harness.preflight.policy import EnvironmentPreflightPolicy
from auto_harness.recovery.dependency import DependencyReconciler
from auto_harness.recovery.journal import OperationJournal
from auto_harness.recovery.schemas import canonical_json, compute_operation_id
from auto_harness.runtime import (
    ChildEnvironmentPolicy,
    DockerSandboxBackend,
    local_docker_environment,
)
from auto_harness.utils.commands import is_allowed_command
from auto_harness.utils.shell import run_command
from auto_harness.command_auth import CommandAuthorizationEngine, CommandRegistry


UV_DOCKER_IMAGE = "ghcr.io/astral-sh/uv:python3.13-bookworm-slim"
NODE_DOCKER_IMAGE = "node:22-bookworm-slim"


class EnvDeployModule:
    def __init__(
        self,
        log_classifier: LogClassifier = None,
        command_runner=None,
        postchecker=None,
        ownership=None,
        environment_policy=None,
        child_environment_policy=None,
    ) -> None:
        self.log_classifier = log_classifier or LogClassifier()
        self.command_runner = command_runner or run_command
        self.postchecker = postchecker or EnvironmentPostchecker()
        self.ownership = ownership or EnvironmentOwnership()
        self.environment_policy = environment_policy or EnvironmentPreflightPolicy()
        self.child_environment_policy = child_environment_policy or ChildEnvironmentPolicy()
        self._uses_default_command_runner = command_runner is None

    def deploy(
        self,
        repo_dir: Path,
        analysis: Dict,
        execute: bool = False,
        timeout_seconds: int = 900,
        allowed_commands=None,
        execution_backend: str = "local",
        docker_image: str = "python:3.13-slim",
        docker_network: str = "bridge",
        docker_gpus: str = "none",
        docker_model_cache_dir: str = "",
        docker_security_options: Dict = None,
        config=None,
        run_dir: Path = None,
        task_id: str = "",
        operation_id: str = "",
        operation_prepared: bool = False,
    ) -> StageResult:
        plan: List[List[str]] = analysis.get("install_plan", [])
        env_solution = analysis.get("env_solution") if isinstance(analysis.get("env_solution"), dict) else {}
        conda_plan = env_solution.get("conda") if isinstance(env_solution.get("conda"), dict) else {}
        backend = env_solution.get("backend", "venv")
        decision = env_solution.get("compatibility_decision") or {}
        action = conda_plan.get("action") or decision.get("action") or "create"
        if backend in ("conda", "mamba", "micromamba"):
            plan = [list(cmd) for cmd in conda_plan.get("commands") or []]
        if not plan and action != "reuse":
            return StageResult("env_deploy", "uncertain", "no install plan detected", {"commands": []})
        registry_data = analysis.get("command_registry")
        command_registry = (
            CommandRegistry.from_dict(registry_data)
            if isinstance(registry_data, dict) and registry_data else None
        )
        repository_commands = {
            tuple(candidate.argv): candidate
            for candidate in (command_registry.candidates if command_registry else [])
        }
        effective_backend = execution_backend
        if any(
            repository_commands.get(tuple(command), None)
            and repository_commands[tuple(command)].required_backend == "docker"
            for command in plan
        ) or any(
            isinstance(candidate, dict)
            and candidate.get("required_backend") == "docker"
            for candidate in analysis.get("run_candidates", [])
        ):
            effective_backend = "docker"
        effective_plan, sandbox = self._effective_plan(
            repo_dir,
            plan,
            effective_backend,
            docker_image,
            docker_network,
            docker_gpus,
            docker_model_cache_dir,
            docker_security_options,
            command_network_profiles={
                command: candidate.network_profile
                for command, candidate in repository_commands.items()
            },
        )
        if not execute:
            return StageResult(
                "env_deploy",
                "passed",
                "dry-run install plan generated",
                {
                    "commands": plan,
                    "effective_commands": effective_plan,
                    "execution_backend": effective_backend,
                    "environment_backend": env_solution.get("backend", "venv"),
                    "environment_prefix": env_solution.get("environment_prefix", ""),
                    "environment_python": env_solution.get("environment_python", ""),
                    "environment_solution": env_solution,
                    "environment_action": action,
                    "sandbox": sandbox,
                    "executed": False,
                },
            )

        allowed_commands = allowed_commands or []
        preflight_policy = env_solution.get("preflight_policy") or {}
        if (
            backend in ("conda", "mamba", "micromamba")
            and decision
            and action != "reuse"
            and not preflight_policy.get("mutation_authorized")
        ):
            return StageResult(
                "env_deploy",
                "failed",
                "environment mutation was not authorized by preflight",
                {
                    "environment_backend": backend,
                    "environment_prefix": decision.get("target_prefix", ""),
                    "preflight_policy": preflight_policy,
                },
                error="preflight mutation authorization missing",
            )
        operation = {}
        journal = None
        owns_operation = False
        if backend in ("conda", "mamba", "micromamba") and decision and run_dir and task_id:
            operation = self._build_environment_operation(
                task_id, run_dir, repo_dir, env_solution, conda_plan, decision,
            )
            if operation_id and operation_id != operation["operation_id"]:
                return StageResult(
                    "env_deploy",
                    "failed",
                    "environment operation identity mismatch",
                    {
                        "expected_operation_id": operation["operation_id"],
                        "provided_operation_id": operation_id,
                    },
                    error="environment operation identity mismatch",
                )
            operation_id = operation["operation_id"]
            journal = OperationJournal(Path(run_dir))
            prepared = self._prepare_environment_operation(
                journal, operation, operation_prepared=operation_prepared,
            )
            if prepared["decision"] == "reuse":
                action = "reuse"
                plan = []
                effective_plan = []
            elif prepared["decision"] != "execute":
                return StageResult(
                    "env_deploy",
                    "failed",
                    "environment operation requires manual recovery",
                    {"operation_id": operation_id, "recovery": prepared},
                    error=prepared.get("reason") or "environment recovery blocked",
                )
            owns_operation = not operation_prepared
        command_results = []
        authorization_attempts = []
        for original_cmd, cmd in zip(plan, effective_plan):
            policy_result = {"allowed": True, "reason": "legacy command allowlist"}
            declared_command = repository_commands.get(tuple(original_cmd))
            if declared_command is not None and command_registry is not None:
                command_decision = CommandAuthorizationEngine().authorize(
                    declared_command,
                    command_registry,
                    repo_dir=repo_dir,
                    execution_backend=execution_backend,
                    sandbox_policy_fingerprint=analysis.get("sandbox_policy_fingerprint", ""),
                    approval=analysis.get("command_approval") or None,
                    require_executable=True,
                )
                authorization_attempts.append(command_decision.to_dict())
                policy_result = {
                    "allowed": command_decision.verdict == "auto_allowed",
                    "reason": command_decision.reason_code,
                }
            elif backend in ("conda", "mamba", "micromamba") and decision:
                policy_result = self.environment_policy.validate_mutation_command(
                    original_cmd,
                    decision,
                    repo_dir,
                    config,
                )
            elif not is_allowed_command(original_cmd, allowed_commands):
                # The Docker argv is generated by Harness after policy
                # evaluation. Authorize the repository-facing command, not
                # the trusted sandbox wrapper executable.
                command_path = Path(original_cmd[0]) if original_cmd else Path()
                if not command_path.is_absolute():
                    command_path = repo_dir / command_path
                project_cli_allowed = False
                if command_path.is_file():
                    try:
                        command_path.resolve().relative_to((repo_dir / ".venv" / "bin").resolve())
                        project_cli_allowed = command_path.name not in {
                            "bash", "sh", "zsh", "fish", "csh", "tcsh",
                        }
                    except ValueError:
                        project_cli_allowed = False
                if command_registry is not None and original_cmd:
                    root = str(original_cmd[0])
                    if root.startswith(".venv/bin/") and Path(root).name not in {"python", "python3", "pip"}:
                        project_cli_allowed = False
                policy_result = {
                    "allowed": project_cli_allowed,
                    "reason": (
                        "project-local generated console script"
                        if project_cli_allowed
                        else "global command allowlist failed"
                    ),
                }
            if not policy_result.get("allowed"):
                self._fail_owned_operation(
                    journal,
                    operation_id,
                    owns_operation,
                    policy_result.get("reason") or "command policy rejected",
                )
                return StageResult(
                    "env_deploy",
                    "failed",
                    "command rejected by policy",
                    {
                        "cmd": cmd,
                        "original_cmd": original_cmd,
                        "allowed_commands": list(allowed_commands),
                        "execution_backend": effective_backend,
                        "environment_backend": env_solution.get("backend", "venv"),
                        "environment_prefix": env_solution.get("environment_prefix", ""),
                        "environment_python": env_solution.get("environment_python", ""),
                        "sandbox": sandbox,
                        "environment_policy": policy_result,
                        "authorization_attempts": authorization_attempts,
                    },
                    error="disallowed command: %s" % (
                        policy_result.get("reason") or (cmd[0] if cmd else ""),
                    ),
                )
            if self._uses_default_command_runner:
                child_env = self.child_environment_policy.build_for_install(
                    home_dir=repo_dir.parent / "install_home",
                    extra=(
                        local_docker_environment()
                        if effective_backend == "docker" else None
                    ),
                )
                # Retain the host's reusable pip download/build cache even
                # though the target process gets a task-scoped HOME.  This is
                # operational state, not a credential, and substantially
                # reduces safe retries after an interrupted install.
                cache_root = os.environ.get("PIP_CACHE_DIR", "").strip()
                if not cache_root:
                    cache_root = str(Path.home() / "Library" / "Caches" / "pip")
                if Path(cache_root).is_dir():
                    child_env["PIP_CACHE_DIR"] = cache_root
                result = self.command_runner(
                    cmd,
                    repo_dir,
                    timeout_seconds=timeout_seconds,
                    env=child_env,
                )
            else:
                result = self.command_runner(cmd, repo_dir, timeout_seconds=timeout_seconds)
            command_results.append({
                "cmd": result.cmd,
                "original_cmd": original_cmd,
                "exit_code": result.exit_code,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
                "timed_out": result.timed_out,
            })
            if result.exit_code != 0:
                self._fail_owned_operation(
                    journal,
                    operation_id,
                    owns_operation,
                    result.stderr[-2000:] or "dependency installation failed",
                )
                diagnosis = self.log_classifier.classify(result.stderr + "\n" + result.stdout)
                return StageResult(
                    "env_deploy",
                    "failed",
                    "dependency installation failed",
                {"commands": command_results, "diagnosis": diagnosis},
                    error=result.stderr[-2000:],
                )
            if (
                len(original_cmd) >= 4
                and original_cmd[0] in {"python", "python3"}
                and original_cmd[1:3] == ["-m", "venv"]
                and original_cmd[3] == ".venv"
                and command_registry is not None
            ):
                if not run_dir:
                    continue
                write_json(Path(run_dir) / "environment" / "venv_owner.json", {
                    "schema_version": 1,
                    "task_id": str(task_id),
                    "repository_fingerprint": command_registry.repository_fingerprint,
                    "environment_path": str((repo_dir / ".venv").resolve()),
                })
        postcheck = {}
        evidence = []
        if backend in ("conda", "mamba", "micromamba") and decision:
            spec = conda_plan.get("spec") or {}
            prefix = Path(decision.get("target_prefix") or conda_plan.get("environment_prefix") or "")
            package_specs = list(spec.get("conda_dependencies") or []) + list(spec.get("pip_dependencies") or [])
            postcheck = self.postchecker.check(
                decision.get("tool") or conda_plan.get("tool") or backend,
                prefix,
                str(decision.get("python") or spec.get("python") or ""),
                package_specs,
                bool(env_solution.get("gpu_required")),
                str(decision.get("spec_hash") or spec.get("spec_hash") or ""),
            )
            if run_dir:
                path = Path(run_dir) / "environment" / "environment_postcheck.json"
                write_json(path, postcheck)
                evidence.append(str(path))
            if postcheck.get("status") != "passed":
                self._fail_owned_operation(
                    journal,
                    operation_id,
                    owns_operation,
                    "; ".join(postcheck.get("errors") or ["environment postcheck failed"]),
                )
                return StageResult(
                    "env_deploy",
                    "failed",
                    "environment postcheck failed",
                    {
                        "commands": command_results,
                        "environment_solution": env_solution,
                        "environment_postcheck": postcheck,
                    },
                    evidence=evidence,
                    error="; ".join(postcheck.get("errors") or ["environment postcheck failed"]),
                )
            marker = self.ownership.write(
                prefix,
                str(decision.get("project_id") or ""),
                str(decision.get("repo_fingerprint") or ""),
                operation_id,
                str(decision.get("spec_hash") or ""),
                str(decision.get("python") or ""),
            )
            marker_path = self.ownership.marker_path(prefix)
            evidence.append(str(marker_path))
            if run_dir:
                spec_path = Path(run_dir) / "environment" / "environment_spec.json"
                write_json(spec_path, {
                    "operation_id": operation_id,
                    "decision": decision,
                    "spec": spec,
                    "ownership": marker,
                })
                evidence.append(str(spec_path))
            if owns_operation and journal:
                record = journal.load(operation_id)
                if record and record.get("status") == "running":
                    journal.transition(
                        operation_id,
                        "committed",
                        result_artifacts=list(evidence),
                    )
        return StageResult(
            "env_deploy",
            "passed",
            "environment reused and verified" if action == "reuse" else "environment deployed",
            {
                "commands": command_results,
                "execution_backend": effective_backend,
                "environment_backend": env_solution.get("backend", "venv"),
                "environment_prefix": env_solution.get("environment_prefix", ""),
                "environment_python": env_solution.get("environment_python", ""),
                "environment_solution": env_solution,
                "environment_action": action,
                "environment_postcheck": postcheck,
                "operation_id": operation_id,
                "sandbox": sandbox,
                "authorization_attempts": authorization_attempts,
            },
            evidence=evidence,
        )

    def _build_environment_operation(
        self, task_id, run_dir, repo_dir, env_solution, conda_plan, decision,
    ):
        spec = conda_plan.get("spec") or {}
        normalized_input = {
            "backend": env_solution.get("backend", "auto"),
            "action": conda_plan.get("action") or decision.get("action", ""),
            "package_specs": (
                list(spec.get("conda_dependencies") or [])
                + list(spec.get("pip_dependencies") or [])
            ),
            "python_version": str(spec.get("python") or env_solution.get("python") or ""),
            "spec_hash": str(spec.get("spec_hash") or decision.get("spec_hash", "")),
        }
        resource_identity = {
            "environment_path": str(
                decision.get("target_prefix")
                or conda_plan.get("environment_prefix")
                or env_solution.get("environment_prefix")
                or (Path(run_dir) / "workspace")
            ),
            "backend": env_solution.get("backend", "venv"),
            "tool": decision.get("tool") or conda_plan.get("tool", ""),
            "python_version": str(decision.get("python") or env_solution.get("python") or ""),
            "project_id": decision.get("project_id", ""),
            "repo_fingerprint": decision.get("repo_fingerprint", ""),
            "spec_hash": decision.get("spec_hash", ""),
            "gpu_required": bool(env_solution.get("gpu_required")),
            "repo_path": str(repo_dir),
        }
        operation_id = compute_operation_id(
            task_id=task_id,
            stage="env_deploy",
            action="install_dependencies",
            normalized_input=normalized_input,
            resource_identity=resource_identity,
        )
        return {
            "operation_id": operation_id,
            "idempotency_key": operation_id,
            "task_id": task_id,
            "run_dir": str(run_dir),
            "stage": "env_deploy",
            "action": "install_dependencies",
            "resource_type": "dependency_install",
            "normalized_input": normalized_input,
            "normalized_input_hash": canonical_json(normalized_input),
            "resource_identity": resource_identity,
            "status": "planned",
        }

    def _prepare_environment_operation(
        self, journal, operation, operation_prepared=False,
    ):
        existing = journal.load(operation["operation_id"])
        if operation_prepared:
            if not existing or existing.get("status") != "running":
                return {
                    "decision": "manual",
                    "reason": "outer recovery operation is not running",
                }
            return {"decision": "execute", "reason": "outer recovery prepared operation"}
        if existing is None:
            journal.begin(operation)
            return {"decision": "execute", "reason": "new operation"}
        status = existing.get("status")
        if status == "committed":
            return {"decision": "reuse", "reason": "operation already committed"}
        if status == "running":
            existing = journal.recover_running(operation["operation_id"])
            status = existing.get("status")
        if status == "unknown":
            result = DependencyReconciler(
                ownership=self.ownership,
                postchecker=self.postchecker,
            ).reconcile(existing)
            decision = result.get("decision", "manual")
            if decision == "reuse":
                journal.transition(
                    operation["operation_id"],
                    "committed",
                    reconcile_result=result,
                )
                return {"decision": "reuse", "reason": result.get("reason", "")}
            if decision == "retry":
                journal.transition(
                    operation["operation_id"],
                    "retryable",
                    reconcile_result=result,
                )
                journal.begin(operation)
                return {"decision": "execute", "reason": result.get("reason", "")}
            target_status = "conflict" if decision == "conflict" else "manual"
            journal.transition(
                operation["operation_id"],
                target_status,
                reconcile_result=result,
            )
            return {"decision": "manual", "reason": result.get("reason", "")}
        if status == "retryable":
            journal.begin(operation)
            return {"decision": "execute", "reason": "retryable operation"}
        return {
            "decision": "manual",
            "reason": "operation status requires approval: %s" % status,
        }

    @staticmethod
    def _fail_owned_operation(journal, operation_id, owns_operation, error):
        if not journal or not operation_id or not owns_operation:
            return
        record = journal.load(operation_id)
        if record and record.get("status") == "running":
            journal.transition(operation_id, "failed", error=str(error)[:2000])

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
        command_network_profiles: Dict = None,
    ):
        if execution_backend != "docker":
            effective_plan = []
            for command in plan:
                effective = list(command)
                if len(effective) >= 3 and effective[:3] == ["python3", "-m", "venv"]:
                    # ChildEnvironmentPolicy deliberately uses a minimal PATH,
                    # which can resolve python3 to macOS' legacy system Python.
                    # Build venvs with the already validated Harness runtime so
                    # the effective interpreter matches host preflight.
                    effective[0] = sys.executable
                effective_plan.append(effective)
            return effective_plan, {
                "backend": "local",
                "venv_bootstrap_python": sys.executable,
            }
        profiles = command_network_profiles or {}
        sandbox_commands = []
        for cmd in plan:
            network = "none" if profiles.get(tuple(cmd)) == "none" else docker_network
            executable = Path(str(cmd[0])).name if cmd else ""
            if executable == "uv":
                command_image = UV_DOCKER_IMAGE
            elif executable in {"npm", "npx", "node", "corepack", "pnpm", "yarn"}:
                command_image = NODE_DOCKER_IMAGE
            else:
                command_image = docker_image
            backend = DockerSandboxBackend.for_phase(
                "install",
                image=command_image,
                network=network,
                gpus=docker_gpus,
                model_cache_dir=Path(docker_model_cache_dir) if docker_model_cache_dir else None,
                **(docker_security_options or {}),
            )
            sandbox_commands.append(backend.wrap(repo_dir, cmd).to_dict())
        return [item["effective_cmd"] for item in sandbox_commands], {
            "backend": "docker",
            "image": docker_image,
            "command_images": [item.get("image", docker_image) for item in sandbox_commands],
            "network": (
                "mixed" if len({item.get("network") for item in sandbox_commands}) > 1
                else sandbox_commands[0].get("network", docker_network)
                if sandbox_commands else docker_network
            ),
            "gpus": docker_gpus,
            "model_cache_dir": docker_model_cache_dir,
            "commands": sandbox_commands,
        }
