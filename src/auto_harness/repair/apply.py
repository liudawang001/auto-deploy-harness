from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import write_json


class RepairApplier:
    """Applies only non-executing repair artifacts; shell/source changes remain gated."""

    def apply(self, run_dir: Path, plan: Dict, policy_result: Dict) -> Dict:
        repair_dir = run_dir / "repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "applied" if policy_result.get("allowed") else "rejected",
            "artifacts": [],
            "policy": policy_result,
        }
        write_json(repair_dir / "repair_plan.json", plan)
        result["artifacts"].append(str(repair_dir / "repair_plan.json"))
        if not policy_result.get("allowed"):
            write_json(repair_dir / "repair_rejected.json", result)
            result["artifacts"].append(str(repair_dir / "repair_rejected.json"))
            return result

        install_commands: List[List[str]] = []
        required_env: List[str] = []
        verify_hints: List[Dict] = []
        for action in plan.get("actions", []):
            action_type = action.get("type")
            payload = action.get("payload") or {}
            if action_type == "install_package" and payload.get("package"):
                install_commands.append([".venv/bin/python", "-m", "pip", "install", payload["package"]])
            elif action_type == "set_env_var_name_only":
                required_env.extend(payload.get("env_vars") or [])
            elif action_type == "update_verify_hint":
                verify_hints.append(payload)

        if install_commands:
            path = repair_dir / "repair_install_plan.json"
            write_json(path, {"commands": install_commands, "executed": False})
            result["artifacts"].append(str(path))
        if required_env:
            path = repair_dir / "required_env_vars.json"
            write_json(path, {"env_vars": sorted(set(required_env)), "values_recorded": False})
            result["artifacts"].append(str(path))
        if verify_hints:
            path = repair_dir / "repair_verify_hints.json"
            write_json(path, {"verify_hints": verify_hints})
            result["artifacts"].append(str(path))
        write_json(repair_dir / "repair_apply_result.json", result)
        result["artifacts"].append(str(repair_dir / "repair_apply_result.json"))
        return result
