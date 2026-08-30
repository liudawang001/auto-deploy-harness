from pathlib import Path
from typing import Dict, List

from auto_harness.agent.repair_actions import install_package_command
from auto_harness.command_auth import CommandAuthorizationEngine
from auto_harness.models.base import write_json
from auto_harness.repair.actions import RepairActionNormalizer, RepairActionRegistry
from auto_harness.runtime import (
    ChildEnvironmentPolicy,
    DockerSandboxBackend,
    local_docker_environment,
)
from auto_harness.utils.shell import run_command


class RepairApplier:
    """Applies only non-executing repair artifacts; shell/source changes remain gated."""

    def __init__(self, command_authorization=None) -> None:
        self.normalizer = RepairActionNormalizer()
        self.registry = RepairActionRegistry()
        self.child_environment_policy = ChildEnvironmentPolicy()
        self.command_authorization = (
            command_authorization or CommandAuthorizationEngine()
        )

    def apply(
        self,
        run_dir: Path,
        plan: Dict,
        policy_result: Dict,
        execute: bool = False,
        command_runner=None,
        timeout_seconds: int = 900,
        allowed_commands: List[str] = None,
        env_context: Dict = None,
    ) -> Dict:
        if (
            command_runner is None
            and (env_context or {}).get("execution_backend") == "docker"
        ):
            command_runner = self._docker_command_runner(run_dir, env_context or {})
        repair_dir = run_dir / "repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)
        normalized_actions = self.normalizer.normalize_many(plan.get("actions", []))
        contract_decisions = [self.registry.validate(action) for action in normalized_actions]
        contract_allowed = all(item["allowed"] for item in contract_decisions)
        effective_policy = dict(policy_result)
        effective_policy["contract_decisions"] = contract_decisions
        effective_policy["allowed"] = bool(policy_result.get("allowed")) and contract_allowed
        result = {
            "status": "applied" if effective_policy["allowed"] else "rejected",
            "artifacts": [],
            "policy": effective_policy,
            "executed": False,
            "executed_action_count": 0,
            "action_results": [],
            "command_authorization": [],
        }
        safe_plan = self._sanitize(plan)
        write_json(repair_dir / "repair_plan.json", safe_plan)
        result["artifacts"].append(str(repair_dir / "repair_plan.json"))
        if not effective_policy["allowed"]:
            result["action_results"] = [
                {
                    "action_type": item["action_type"],
                    "executed": False,
                    "status": "rejected",
                    "reason": "; ".join(item["reasons"]),
                }
                for item in contract_decisions
                if not item["allowed"]
            ]
            write_json(repair_dir / "repair_rejected.json", result)
            result["artifacts"].append(str(repair_dir / "repair_rejected.json"))
            write_json(repair_dir / "repair_apply_result.json", result)
            result["artifacts"].append(str(repair_dir / "repair_apply_result.json"))
            return result

        install_commands: List[List[str]] = []
        required_env: List[str] = []
        verify_hints: List[Dict] = []
        for action in normalized_actions:
            action_type = action.get("type")
            payload = action.get("payload") or {}
            if action_type in ("install_package", "install_pip_package", "pin_dependency") and payload.get("package"):
                command = install_package_command(str(payload["package"]), env_context=env_context)
                if command["status"] == "ready":
                    install_commands.append(command["cmd"])
                    command_decision = self._authorize_command(
                        command["cmd"], allowed_commands,
                    )
                    result["command_authorization"].append(command_decision)
                    if execute:
                        if not command_decision["allowed"]:
                            result["action_results"].append({
                                "action_type": action_type,
                                "executed": False,
                                "status": "rejected",
                                "cmd": command["cmd"],
                                "reason": "command is not allowed by command policy",
                                "reason_code": command_decision["reason_code"],
                                "command_decision": command_decision,
                            })
                        else:
                            executed = self._execute_command(
                                run_dir, action_type, command["cmd"],
                                command_runner, timeout_seconds,
                            )
                            executed["command_decision"] = command_decision
                            result["action_results"].append(executed)
                else:
                    result["action_results"].append({
                        "action_type": action_type,
                        "executed": False,
                        "status": "rejected",
                        "reason": command["reason"],
                    })
            elif action_type == "install_conda_package" and payload.get("package"):
                command = self._install_conda_package_command(str(payload["package"]), payload, env_context or {})
                install_commands.append(command)
                command_decision = self._authorize_command(command, allowed_commands)
                result["command_authorization"].append(command_decision)
                if execute:
                    if not command_decision["allowed"]:
                        result["action_results"].append({
                            "action_type": action_type,
                            "executed": False,
                            "status": "rejected",
                            "cmd": command,
                            "reason": "command is not allowed by command policy",
                            "reason_code": command_decision["reason_code"],
                            "command_decision": command_decision,
                        })
                    else:
                        executed = self._execute_command(
                            run_dir, action_type, command,
                            command_runner, timeout_seconds,
                        )
                        executed["command_decision"] = command_decision
                        result["action_results"].append(executed)
            elif action_type == "set_env_var_name_only":
                required_env.extend(payload.get("env_vars") or [])
            elif action_type == "update_verify_hint":
                verify_hints.append(payload)
            elif action_type == "rerun_from_stage":
                result["action_results"].append({
                    "action_type": action_type,
                    "executed": False,
                    "status": "metadata_only",
                    "rerun_from": payload.get("stage") or plan.get("rerun_from"),
                })

        if install_commands:
            path = repair_dir / "repair_install_plan.json"
            write_json(path, {"commands": install_commands, "executed": bool(execute), "results": result["action_results"]})
            result["artifacts"].append(str(path))
        if required_env:
            path = repair_dir / "required_env_vars.json"
            write_json(path, {"env_vars": sorted(set(required_env)), "values_recorded": False})
            result["artifacts"].append(str(path))
        if verify_hints:
            path = repair_dir / "repair_verify_hints.json"
            write_json(path, {"verify_hints": verify_hints})
            result["artifacts"].append(str(path))
        executed_results = [item for item in result["action_results"] if item.get("executed")]
        metadata_results = [item for item in result["action_results"] if item.get("status") == "metadata_only"]
        result["executed"] = bool(executed_results)
        result["executed_action_count"] = len(executed_results)
        result["metadata_action_count"] = len(metadata_results)
        result["repair_applied"] = result["status"] == "applied"
        result["repair_executed"] = bool(executed_results)
        result["repair_effective"] = False
        result["repair_verified"] = False
        result["effectiveness_note"] = "repair is effective only after a rerun and final trace-based verify pass"
        write_json(repair_dir / "repair_apply_result.json", result)
        result["artifacts"].append(str(repair_dir / "repair_apply_result.json"))
        return result

    def _install_conda_package_command(self, package: str, payload: Dict, env_context: Dict) -> List[str]:
        backend = str(env_context.get("backend") or env_context.get("environment_backend") or "conda")
        tool = "mamba" if backend == "mamba" else "conda"
        prefix = str(env_context.get("conda_prefix") or env_context.get("environment_prefix") or ".conda/envs/auto-harness")
        channels = []
        for channel in payload.get("channels") or []:
            channels.extend(["-c", str(channel)])
        return [tool, "install", "-y", "-p", prefix] + channels + [package]

    def _authorize_command(
        self,
        cmd: List[str],
        allowed_commands: List[str] = None,
    ) -> Dict:
        return self.command_authorization.authorize_argv(
            cmd,
            allowed_commands=allowed_commands or (),
            section="repair.apply",
            strict_allowlist=allowed_commands is not None,
        )

    def _execute_command(self, run_dir: Path, action_type: str, cmd: List[str], command_runner, timeout_seconds: int) -> Dict:
        if command_runner:
            raw = command_runner(cmd, run_dir / "workspace" / "repo", timeout_seconds)
            if isinstance(raw, dict):
                return {
                    "action_type": action_type,
                    "executed": True,
                    "cmd": cmd,
                    "exit_code": int(raw.get("exit_code") or 0),
                    "stdout_tail": str(raw.get("stdout") or "")[-4000:],
                    "stderr_tail": str(raw.get("stderr") or "")[-4000:],
                    "timed_out": bool(raw.get("timed_out")),
                }
        result = run_command(
            cmd,
            run_dir / "workspace" / "repo",
            timeout_seconds=timeout_seconds,
            env=self.child_environment_policy.build_for_install(
                home_dir=run_dir / "workspace" / "install_home",
            ),
        )
        return {
            "action_type": action_type,
            "executed": True,
            "cmd": result.cmd,
            "exit_code": result.exit_code,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "timed_out": result.timed_out,
        }

    def _docker_command_runner(self, run_dir: Path, env_context: Dict):
        repo_dir = run_dir / "workspace" / "repo"

        def execute(cmd, cwd, timeout_seconds):
            effective_cmd = DockerSandboxBackend.for_phase(
                "install",
                image=str(env_context.get("docker_image") or "python:3.13-slim"),
                network=str(env_context.get("docker_network") or "bridge"),
                gpus=str(env_context.get("docker_gpus") or "none"),
            ).wrap(repo_dir, cmd).effective_cmd
            result = run_command(
                effective_cmd,
                cwd,
                timeout_seconds=timeout_seconds,
                env=self.child_environment_policy.build_for_install(
                    home_dir=run_dir / "workspace" / "install_home",
                    extra=local_docker_environment(),
                ),
            )
            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            }

        return execute

    def _sanitize(self, value):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in ("secret", "token_value", "api_key", "password")):
                    cleaned[key] = "[REDACTED]"
                else:
                    cleaned[key] = self._sanitize(item)
            return cleaned
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, str) and (("hf_" in value.lower() and len(value) > 20) or "bearer " in value.lower()):
            return "[REDACTED]"
        return value
