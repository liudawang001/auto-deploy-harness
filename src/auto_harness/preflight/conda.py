"""Conda runtime and environment inventory probes."""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from auto_harness.env.ownership import EnvironmentOwnership
from auto_harness.preflight.schemas import CondaEnvironmentInventory


def _completed(value):
    if isinstance(value, dict):
        return (
            int(value.get("exit_code", value.get("returncode", 1))),
            str(value.get("stdout", "")),
            str(value.get("stderr", "")),
        )
    return (
        int(getattr(value, "returncode", 1)),
        str(getattr(value, "stdout", "") or ""),
        str(getattr(value, "stderr", "") or ""),
    )


class CondaRuntimeProbe:
    TOOLS = ("conda", "mamba", "micromamba")

    def __init__(self, command_runner=None, which=None, timeout_seconds: int = 10) -> None:
        self.command_runner = command_runner or subprocess.run
        self.which = which or shutil.which
        self.timeout_seconds = timeout_seconds

    def probe(self) -> Dict[str, Dict]:
        return {tool: self._probe_tool(tool) for tool in self.TOOLS}

    def _probe_tool(self, tool: str) -> Dict:
        path = self.which(tool) or ""
        if not path:
            return {"available": False, "path": "", "version": "", "root_prefix": "", "error": "not_found"}
        try:
            version_result = self.command_runner(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            info_result = self.command_runner(
                [path, "info", "--json"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"available": False, "path": path, "version": "", "root_prefix": "", "error": "timeout"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"available": False, "path": path, "version": "", "root_prefix": "", "error": str(exc)[:500]}
        version_code, version_out, version_err = _completed(version_result)
        info_code, info_out, info_err = _completed(info_result)
        if version_code != 0 or info_code != 0:
            return {
                "available": False,
                "path": path,
                "version": "",
                "root_prefix": "",
                "error": (info_err or version_err)[-500:],
            }
        try:
            info = json.loads(info_out or "{}")
        except ValueError:
            return {"available": False, "path": path, "version": "", "root_prefix": "", "error": "invalid_info_json"}
        version = (version_out or version_err).strip().split()[-1] if (version_out or version_err).strip() else ""
        return {
            "available": True,
            "path": str(Path(path).resolve()),
            "version": version,
            "root_prefix": str(info.get("root_prefix") or info.get("base environment") or ""),
            "active_prefix": str(info.get("active_prefix") or ""),
            "error": "",
        }


class CondaInventoryProbe:
    def __init__(self, command_runner=None, timeout_seconds: int = 30, max_envs: int = 50) -> None:
        self.command_runner = command_runner or subprocess.run
        self.timeout_seconds = timeout_seconds
        self.max_envs = max_envs

    def probe(self, runtime: Dict, project_id: str = "") -> Dict:
        if not runtime.get("available") or not runtime.get("path"):
            return CondaEnvironmentInventory(errors=["runtime_unavailable"]).to_dict()
        path = runtime["path"]
        try:
            result = self.command_runner(
                [path, "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CondaEnvironmentInventory(
                tool_path=path, errors=["inventory_timeout"]
            ).to_dict()
        except (OSError, subprocess.SubprocessError) as exc:
            return CondaEnvironmentInventory(
                tool_path=path, errors=[str(exc)[:500]]
            ).to_dict()
        code, stdout, stderr = _completed(result)
        if code != 0:
            return CondaEnvironmentInventory(
                tool_path=path, errors=[stderr[-500:] or "inventory_failed"]
            ).to_dict()
        try:
            prefixes = json.loads(stdout or "{}").get("envs", [])
        except (ValueError, AttributeError):
            return CondaEnvironmentInventory(
                tool_path=path, errors=["invalid_inventory_json"]
            ).to_dict()
        environments: List[Dict] = []
        ownership = EnvironmentOwnership()
        for raw_prefix in prefixes[:self.max_envs]:
            prefix = Path(str(raw_prefix)).resolve()
            marker = ownership.read(prefix)
            environments.append({
                "prefix": str(prefix),
                "name": prefix.name,
                "exists": prefix.exists(),
                "python_version": marker.get("python_version", ""),
                "owned_by_harness": ownership.is_valid(marker),
                "owner_project_id": marker.get("project_id", ""),
                "same_project": bool(project_id and marker.get("project_id") == project_id),
                "spec_hash": marker.get("spec_hash", ""),
                "inspection_status": "marker_verified" if ownership.is_valid(marker) else "external",
            })
        return CondaEnvironmentInventory(
            tool=Path(path).name,
            tool_path=path,
            root_prefix=runtime.get("root_prefix", ""),
            active_prefix=runtime.get("active_prefix", ""),
            environments=environments,
        ).to_dict()
