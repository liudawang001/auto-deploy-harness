"""Resolve project executables without searching untrusted global PATH entries."""

import json
from pathlib import Path

from auto_harness.command_auth.schemas import CommandCandidate


OWNED_SHELLS = {"bash", "sh", "zsh", "fish", "csh", "tcsh"}

# Harness-owned artifact output paths (Phase B3).  A repository build may
# only run from these controlled locations, and only when the hash-bound
# artifact evidence still matches the file on disk.
HARNESS_OWNED_ARTIFACT_PREFIXES = (
    ".harness/bin/",
    "target/release/",
    "target/debug/",
)


class ExecutableResolver:
    def resolve(
        self, repo_dir: Path, candidate: CommandCandidate, require_exists=True,
        repository_fingerprint: str = "",
        ownership_marker_path: Path = None,
    ):
        if not candidate.argv:
            return {"resolved": False, "reason_code": "command_empty", "path": ""}
        raw = candidate.argv[0]
        if raw.startswith(".venv/bin/"):
            root = (Path(repo_dir) / ".venv" / "bin").resolve()
            path = (Path(repo_dir) / raw).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                return {"resolved": False, "reason_code": "path_escape_hard_denied", "path": str(path)}
            if path.name in OWNED_SHELLS:
                return {"resolved": False, "reason_code": "shell_wrapper_hard_denied", "path": str(path)}
            if require_exists and not path.is_file():
                return {"resolved": False, "reason_code": "owned_executable_missing", "path": str(path)}
            if require_exists:
                marker_path = Path(ownership_marker_path) if ownership_marker_path else None
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path else {}
                except (OSError, TypeError, ValueError):
                    marker = {}
                if not marker:
                    return {"resolved": False, "reason_code": "owned_environment_marker_missing", "path": str(path)}
                if (
                    repository_fingerprint
                    and marker.get("repository_fingerprint") != repository_fingerprint
                ):
                    return {"resolved": False, "reason_code": "owned_environment_marker_mismatch", "path": str(path)}
                if marker.get("environment_path") != str((Path(repo_dir) / ".venv").resolve()):
                    return {"resolved": False, "reason_code": "owned_environment_marker_mismatch", "path": str(path)}
            return {"resolved": True, "reason_code": "declared_cli_bound_to_owned_env", "path": str(path)}
        for prefix in HARNESS_OWNED_ARTIFACT_PREFIXES:
            if raw.startswith("./" + prefix) or raw.startswith(prefix):
                normalized = raw[2:] if raw.startswith("./") else raw
                path = Path(repo_dir) / normalized
                if require_exists and not path.is_file():
                    return {
                        "resolved": False,
                        "reason_code": "build_artifact_missing",
                        "path": str(path),
                    }
                return {
                    "resolved": True,
                    "reason_code": "harness_owned_build_artifact",
                    "path": str(path),
                }
        if "/" in raw and not raw.startswith("/bin/") and not raw.startswith("/usr/bin/"):
            return {"resolved": False, "reason_code": "unbound_executable_path", "path": raw}
        return {"resolved": True, "reason_code": "controlled_system_tool", "path": raw}
