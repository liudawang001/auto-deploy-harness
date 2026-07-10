from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json


class RepairOverlay:
    """Loads non-executing repair artifacts and merges them into rerun inputs."""

    def load(self, run_dir: Path) -> Dict:
        repair_dir = run_dir / "repairs"
        if not repair_dir.exists():
            return {"active": False, "install_commands": [], "verify_hints": []}

        apply_result = self._read_optional(repair_dir / "repair_apply_result.json")
        if apply_result and apply_result.get("status") != "applied":
            return {
                "active": False,
                "reason": "latest repair apply result is not applied",
                "policy": apply_result.get("policy"),
                "install_commands": [],
                "verify_hints": [],
            }
        if not apply_result and (repair_dir / "repair_rejected.json").exists():
            rejected = self._read_optional(repair_dir / "repair_rejected.json")
            return {
                "active": False,
                "reason": "repair was rejected by policy",
                "policy": rejected.get("policy") if isinstance(rejected, dict) else None,
                "install_commands": [],
                "verify_hints": [],
            }

        install_plan = self._read_optional(repair_dir / "repair_install_plan.json")
        verify_plan = self._read_optional(repair_dir / "repair_verify_hints.json")
        install_commands = install_plan.get("commands", []) if isinstance(install_plan, dict) else []
        verify_hints = verify_plan.get("verify_hints", []) if isinstance(verify_plan, dict) else []
        return {
            "active": bool(install_commands or verify_hints),
            "install_commands": self._valid_commands(install_commands),
            "verify_hints": [hint for hint in verify_hints if isinstance(hint, dict)],
            "source_dir": str(repair_dir),
            "policy": apply_result.get("policy") if isinstance(apply_result, dict) else None,
        }

    def merge_analysis(self, analysis: Dict, overlay: Dict) -> Dict:
        merged = deepcopy(analysis)
        if not overlay.get("active"):
            return merged
        install_commands = overlay.get("install_commands") or []
        if install_commands:
            existing = merged.get("install_plan") or []
            merged["install_plan"] = existing + [
                command for command in install_commands
                if command not in existing
            ]
        verify_hint = self._merge_verify_hints(merged.get("verify_hint") or {}, overlay.get("verify_hints") or [])
        if verify_hint:
            merged["verify_hint"] = verify_hint
        merged["repair_overlay"] = {
            "active": True,
            "install_command_count": len(install_commands),
            "verify_hint_count": len(overlay.get("verify_hints") or []),
            "source_dir": overlay.get("source_dir"),
        }
        return merged

    def _merge_verify_hints(self, base_hint: Dict, hints: List[Dict]) -> Dict:
        merged = deepcopy(base_hint)
        for hint in hints:
            # Unwrap verify_hint key if present (from normalizer)
            actual = hint.get("verify_hint") if isinstance(hint.get("verify_hint"), dict) else hint
            if "request" in actual and isinstance(actual["request"], dict):
                request = merged.get("request") if isinstance(merged.get("request"), dict) else {}
                request.update(actual["request"])
                merged["request"] = request
            for key in ("endpoint", "service_type"):
                if actual.get(key):
                    merged[key] = actual[key]
        return merged

    def _valid_commands(self, commands: List) -> List[List[str]]:
        valid = []
        for command in commands:
            if isinstance(command, list) and command and all(isinstance(item, str) for item in command):
                valid.append(command)
        return valid

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
