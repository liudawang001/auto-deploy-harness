import os
from pathlib import Path
from typing import Dict, List

from auto_harness.agent.repair_actions import install_package_command
from auto_harness.models.base import write_json
from auto_harness.utils.shell import run_command


class RepairApplier:
    """Applies only non-executing repair artifacts; shell/source changes remain gated."""

    def apply(
        self,
        run_dir: Path,
        plan: Dict,
        policy_result: Dict,
        execute: bool = False,
        command_runner=None,
        timeout_seconds: int = 900,
        allowed_commands: List[str] = None,
    ) -> Dict:
        repair_dir = run_dir / "repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "applied" if policy_result.get("allowed") else "rejected",
            "artifacts": [],
            "policy": policy_result,
            "executed": False,
            "executed_action_count": 0,
            "action_results": [],
        }
        safe_plan = self._sanitize(plan)
        write_json(repair_dir / "repair_plan.json", safe_plan)
        result["artifacts"].append(str(repair_dir / "repair_plan.json"))
        if not policy_result.get("allowed"):
            write_json(repair_dir / "repair_rejected.json", result)
            result["artifacts"].append(str(repair_dir / "repair_rejected.json"))
            write_json(repair_dir / "repair_apply_result.json", result)
            result["artifacts"].append(str(repair_dir / "repair_apply_result.json"))
            return result

        install_commands: List[List[str]] = []
        required_env: List[str] = []
        verify_hints: List[Dict] = []
        for action in plan.get("actions", []):
            action_type = action.get("type")
            payload = action.get("payload") or {}
            if action_type == "install_package" and payload.get("package"):
                command = install_package_command(str(payload["package"]))
                if command["status"] == "ready":
                    install_commands.append(command["cmd"])
                    if execute:
                        command_reject = self._command_policy_reject(command["cmd"], allowed_commands)
                        if command_reject:
                            result["action_results"].append({
                                "action_type": action_type,
                                "executed": False,
                                "status": "rejected",
                                "cmd": command["cmd"],
                                "reason": command_reject,
                            })
                        else:
                            result["action_results"].append(self._execute_command(run_dir, action_type, command["cmd"], command_runner, timeout_seconds))
                else:
                    result["action_results"].append({
                        "action_type": action_type,
                        "executed": False,
                        "status": "rejected",
                        "reason": command["reason"],
                    })
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
        result["executed"] = bool(executed_results)
        result["executed_action_count"] = len(executed_results)
        write_json(repair_dir / "repair_apply_result.json", result)
        result["artifacts"].append(str(repair_dir / "repair_apply_result.json"))
        return result

    def _command_policy_reject(self, cmd: List[str], allowed_commands: List[str] = None) -> str:
        if allowed_commands is None:
            return ""
        executable = os.path.basename(str(cmd[0] or "")) if cmd else ""
        allowed = {os.path.basename(str(item)) for item in allowed_commands}
        if executable not in allowed:
            return "command is not allowed by command policy"
        return ""

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
        result = run_command(cmd, run_dir / "workspace" / "repo", timeout_seconds=timeout_seconds)
        return {
            "action_type": action_type,
            "executed": True,
            "cmd": result.cmd,
            "exit_code": result.exit_code,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "timed_out": result.timed_out,
        }

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
