import configparser
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class GitSubmoduleDetector:
    def __init__(self, available: Optional[bool] = None) -> None:
        self.available = available

    def detect(self, repo_dir: Path) -> Dict:
        submodules = self._read_gitmodules(repo_dir)
        required = bool(submodules)
        available = self._available()
        plan = {
            "required": required,
            "available": available,
            "submodules": submodules,
            "submodule_count": len(submodules),
            "prepare_commands": [
                ["git", "submodule", "sync", "--recursive"],
                ["git", "submodule", "update", "--init", "--recursive"],
            ] if required else [],
        }
        if required and not available:
            plan["diagnosis"] = {
                "category": "git_missing",
                "signal": ".gitmodules detected but git is not available",
                "suggested_fix": "install git and run git submodule update --init --recursive before model_prepare",
                "confidence": 0.9,
            }
        return plan

    def _available(self) -> bool:
        if self.available is not None:
            return self.available
        return shutil.which("git") is not None

    def _read_gitmodules(self, repo_dir: Path) -> List[Dict]:
        path = repo_dir / ".gitmodules"
        if not path.exists():
            return []
        parser = configparser.ConfigParser()
        try:
            parser.read_string(path.read_text(encoding="utf-8", errors="ignore"))
        except configparser.Error:
            return []
        submodules = []
        for section in parser.sections():
            if not section.startswith("submodule "):
                continue
            name = section.split(" ", 1)[1].strip().strip('"')
            module_path = parser.get(section, "path", fallback="").strip()
            url = parser.get(section, "url", fallback="").strip()
            branch = parser.get(section, "branch", fallback="").strip()
            if not module_path and not url:
                continue
            record = {
                "name": name,
                "path": module_path,
                "url": url,
                "branch": branch,
                "initialized": self._is_initialized(repo_dir, module_path),
            }
            submodules.append(record)
        return sorted(submodules, key=lambda item: item.get("path") or item.get("name") or "")

    def _is_initialized(self, repo_dir: Path, module_path: str) -> bool:
        if not module_path:
            return False
        target = repo_dir / module_path
        try:
            return target.exists() and any(target.iterdir())
        except OSError:
            return False
