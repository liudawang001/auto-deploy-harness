"""Tamper-evident metadata helpers for local release gates."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from auto_harness.utils.time import utc_now_iso


SCHEMA_VERSION = "1.0"


def git_identity(project_root: Path) -> Dict[str, Any]:
    root = Path(project_root)
    sha = _git(root, ["rev-parse", "HEAD"]) or "unknown"
    status = _git(root, ["status", "--porcelain", "--untracked-files=normal"])
    return {"commit_sha": sha, "dirty_worktree": bool(status)}


def environment_metadata() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def evidence_hash(payload: Dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("evidence_sha256", None)
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_evidence(
    project_root: Path,
    command: Iterable[str] | str,
    status: str,
    passed: int,
    failed: int,
    skipped: int = 0,
    **details: Any,
) -> Dict[str, Any]:
    identity = git_identity(project_root)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "generated_at": utc_now_iso(),
        "command": list(command) if not isinstance(command, str) else command,
        "status": status,
        "passed": int(passed),
        "failed": int(failed),
        "skipped": int(skipped),
        "environment": environment_metadata(),
        **details,
    }
    payload["evidence_sha256"] = evidence_hash(payload)
    return payload


def validate_evidence(payload: Any, project_root: Path) -> list[str]:
    if not isinstance(payload, dict):
        return ["evidence must be a JSON object"]
    required = {
        "schema_version", "commit_sha", "dirty_worktree", "generated_at",
        "command", "status", "passed", "failed", "skipped",
        "environment", "evidence_sha256",
    }
    errors = ["missing field: %s" % key for key in sorted(required - set(payload))]
    identity = git_identity(project_root)
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("commit_sha") != identity["commit_sha"]:
        errors.append("commit_sha does not match current HEAD")
    if payload.get("dirty_worktree") is not False:
        errors.append("evidence was generated from a dirty worktree")
    if identity["dirty_worktree"]:
        errors.append("current worktree is dirty")
    if payload.get("status") != "passed":
        errors.append("status is not passed")
    if payload.get("failed") != 0:
        errors.append("failed count is not zero")
    if payload.get("evidence_sha256") != evidence_hash(payload):
        errors.append("evidence hash mismatch")
    if not isinstance(payload.get("environment"), dict):
        errors.append("environment must be an object")
    return errors


def _git(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
