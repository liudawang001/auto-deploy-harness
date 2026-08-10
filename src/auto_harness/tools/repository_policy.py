"""Fail-closed policy for planner repository observations."""
import fnmatch
from pathlib import Path
from typing import Any, Dict

from auto_harness.context.repository import safe_repo_path


_DENIED_PARTS = {
    ".git", ".ssh", ".aws", ".azure", "gcloud", "node_modules",
    ".venv", "venv",
}
_DENIED_NAMES = {
    ".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519",
    "credentials", "credentials.json",
}
_DENIED_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx", ".bin", ".pth", ".pt", ".onnx",
    ".safetensors", ".gguf", ".ckpt", ".zip", ".tar", ".gz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".mp3", ".mp4", ".wav",
}


class RepositoryReadPolicy:
    """Validate and normalize read-only repository tool inputs."""

    ALLOWED_TOOLS = {
        "inspect_repo_tree", "search_repo", "read_selected_files",
        "parse_dependency_files",
    }

    def __init__(self, config: Any = None):
        self.config = config

    def validate_and_normalize(
        self, tool_name: str, tool_input: Dict, repo_dir: Path
    ) -> Dict:
        if tool_name not in self.ALLOWED_TOOLS:
            return self._reject("tool is not an allowed repository observation")
        if not isinstance(tool_input, dict):
            return self._reject("tool input must be an object")
        try:
            if tool_name == "inspect_repo_tree":
                normalized = self._tree_input(tool_input, repo_dir)
            elif tool_name == "search_repo":
                normalized = self._search_input(tool_input)
            elif tool_name == "read_selected_files":
                normalized = self._read_input(tool_input, repo_dir)
            else:
                normalized = self._dependency_input(tool_input, repo_dir)
        except (TypeError, ValueError) as exc:
            return self._reject(str(exc))
        return {"allowed": True, "reason": "repository read allowed", "normalized_input": normalized}

    def _tree_input(self, value: Dict, repo_dir: Path) -> Dict:
        path = self._relative_path(value.get("path", "."), repo_dir, allow_directory=True)
        depth = self._bounded_int(value.get("max_depth", 3), 1, 8, "max_depth")
        maximum = self._bounded_int(
            value.get("max_entries", 300), 1,
            self._cfg("agent_repo_tree_max_entries", 5000), "max_entries",
        )
        pattern = str(value.get("path_glob", "**/*") or "**/*")[:200]
        return {"path": path, "max_depth": depth, "max_entries": maximum, "path_glob": pattern}

    def _search_input(self, value: Dict) -> Dict:
        query = str(value.get("query", ""))
        if not query or len(query) > 500:
            raise ValueError("query must contain 1-500 characters")
        if "\x00" in query:
            raise ValueError("query contains NUL")
        maximum = self._bounded_int(
            value.get("max_results", 30), 1,
            self._cfg("agent_repo_search_max_results", 30), "max_results",
        )
        return {
            "query": query,
            "path_glob": str(value.get("path_glob", "**/*") or "**/*")[:200],
            "case_sensitive": bool(value.get("case_sensitive", False)),
            "max_results": maximum,
        }

    def _read_input(self, value: Dict, repo_dir: Path) -> Dict:
        files = value.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("files must be a non-empty list")
        max_requests = self._cfg("agent_repo_max_requests_per_round", 4)
        if len(files) > max_requests:
            raise ValueError("too many files in one read request")
        max_lines = self._cfg("agent_repo_max_lines_per_read", 400)
        normalized = []
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("each files item must be an object")
            path = self._relative_path(item.get("path", ""), repo_dir)
            start = self._bounded_int(item.get("start_line", 1), 1, 10_000_000, "start_line")
            end = self._bounded_int(item.get("end_line", start + max_lines - 1), start, 10_000_000, "end_line")
            if end - start + 1 > max_lines:
                end = start + max_lines - 1
            normalized.append({"path": path, "start_line": start, "end_line": end})
        return {"files": normalized}

    def _dependency_input(self, value: Dict, repo_dir: Path) -> Dict:
        paths = value.get("paths", [])
        if paths and not isinstance(paths, list):
            raise ValueError("paths must be a list")
        return {"paths": [self._relative_path(item, repo_dir) for item in paths[:16]]}

    def _relative_path(self, raw: Any, repo_dir: Path, allow_directory: bool = False) -> str:
        value = str(raw or "")
        if not value or "\x00" in value:
            raise ValueError("repository path is empty or invalid")
        path_value = Path(value)
        if path_value.is_absolute():
            raise ValueError("absolute repository paths are not allowed")
        normalized = value.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized or "."
        parts = [part.lower() for part in Path(normalized).parts]
        name = Path(normalized).name.lower()
        if any(part in _DENIED_PARTS for part in parts):
            raise ValueError("repository path is denied")
        if name in _DENIED_NAMES or name.startswith(".env.") or name.startswith("credentials"):
            raise ValueError("sensitive repository file is denied")
        if Path(name).suffix.lower() in _DENIED_SUFFIXES:
            raise ValueError("binary or sensitive repository file is denied")
        target = safe_repo_path(Path(repo_dir), normalized)
        if not target.exists():
            raise ValueError("repository path does not exist")
        if allow_directory and not target.is_dir():
            raise ValueError("repository tree path must be a directory")
        if not allow_directory and not target.is_file():
            raise ValueError("repository read path must be a regular file")
        return normalized

    @staticmethod
    def path_allowed(relative_path: str) -> bool:
        try:
            parts = [part.lower() for part in Path(relative_path).parts]
            name = Path(relative_path).name.lower()
            return not (
                any(part in _DENIED_PARTS for part in parts)
                or name in _DENIED_NAMES
                or name.startswith(".env.")
                or name.startswith("credentials")
                or Path(name).suffix.lower() in _DENIED_SUFFIXES
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def glob_matches(path: str, pattern: str) -> bool:
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)

    def _cfg(self, name: str, default: int) -> int:
        if isinstance(self.config, dict):
            return int(self.config.get(name, default))
        return int(getattr(self.config, name, default))

    @staticmethod
    def _bounded_int(value: Any, minimum: int, maximum: int, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError("%s must be an integer" % name)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be an integer" % name) from exc
        if parsed < minimum:
            raise ValueError("%s is below minimum" % name)
        return min(parsed, maximum)

    @staticmethod
    def _reject(reason: str) -> Dict:
        return {"allowed": False, "reason": reason[:300], "normalized_input": None}
