"""Evidence manifest and exporter.

Generates evidence packages with commit SHA, artifact hashes,
and provenance information. All hashes are computed by the
exporter, never trusted from input.

Key invariants:
- Hashes are always computed, never trusted
- Missing required artifacts cause export failure
- Path traversal and external symlinks are rejected
- Secret values are never included in the archive
- Commit SHA comes from git, not from input
"""
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.utils.time import utc_now_iso


# Required artifacts that must exist for a valid evidence package
REQUIRED_ARTIFACTS = [
    "task.json",
    "reports/controller_result.json",
    "reports/project_snapshot.json",
]
CONTRIBUTION_ARTIFACTS = [
    "reports/llm_contribution_evidence.json",
    "reports/agent_contribution.json",
]

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|password|"
    r"credential|private[_-]?key|client[_-]?secret|hf[_-]?token)",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = [
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|"
        r"password|client[_-]?secret)\b(\s*[:=]\s*)([\"']?)[^,\s}\"']+"
    ),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_git_commit_sha(project_root: Path) -> str:
    """Get the current git commit SHA.

    Args:
        project_root: Path to the project root.

    Returns:
        Commit SHA string, or "unknown" if git is not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _is_dirty_worktree(project_root: Path) -> bool:
    """Check if the git worktree has uncommitted changes.

    Args:
        project_root: Path to the project root.

    Returns:
        True if there are uncommitted changes, False otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return True  # Assume dirty if we can't check


def _validate_artifact_path(path: Path, evidence_root: Path) -> bool:
    """Validate that an artifact path is safe and within the evidence root.

    Rejects:
    - Path traversal (..)
    - Absolute external paths
    - Non-existent files
    - Directories
    - Symlinks pointing outside the root

    Args:
        path: Path to validate.
        evidence_root: Root directory that artifacts must be within.

    Returns:
        True if the path is valid, False otherwise.
    """
    try:
        resolved = path.resolve()
        root_resolved = evidence_root.resolve()
        # Must be within evidence root
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            return False
        # Must be a file (not directory, not broken symlink)
        if not resolved.is_file():
            return False
        # Must not be a symlink pointing outside root
        if path.is_symlink():
            target = path.resolve()
            try:
                target.relative_to(root_resolved)
            except ValueError:
                return False
        return True
    except (OSError, ValueError):
        return False


def _redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_KEY_RE.search(str(key)) else _redact_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in _SECRET_VALUE_PATTERNS:
        if "BEGIN [A-Z ]*PRIVATE KEY" in pattern.pattern:
            redacted = pattern.sub("[REDACTED PRIVATE KEY]", redacted)
        elif pattern.pattern.startswith("(?i)\\b(api"):
            redacted = pattern.sub(lambda match: "%s%s[REDACTED]" % (match.group(1), match.group(2)), redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_artifact_bytes(path: Path, content: bytes) -> bytes:
    """Return archive-safe content without mutating the source artifact."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
            return (json.dumps(_redact_json(value), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        except ValueError:
            pass
    return _redact_text(text).encode("utf-8")


class EvidenceExporter:
    """Export evidence packages with manifest, hashes, and provenance."""

    def __init__(self, project_root: Path = None) -> None:
        self.project_root = Path(project_root or os.getcwd())

    def export(
        self,
        run_dir: Path,
        task_id: str,
        evidence_root: Path = None,
        redact_secrets: bool = True,
        output_path: Path = None,
    ) -> Dict[str, Any]:
        """Export an evidence package from a run directory.

        Args:
            run_dir: Path to the run directory.
            task_id: Task identifier.
            evidence_root: Root directory for evidence (defaults to run_dir).
            redact_secrets: Whether to redact secret values.

        Returns:
            Manifest dict. Status will be "failed" if required
            artifacts are missing.
        """
        run_dir = Path(run_dir)
        evidence_root = Path(evidence_root or run_dir)

        # Get git information
        commit_sha = _get_git_commit_sha(self.project_root)
        dirty = _is_dirty_worktree(self.project_root)

        # Collect artifacts
        artifacts = []
        archive_payloads = {}
        missing_required = []

        for required in REQUIRED_ARTIFACTS:
            artifact_path = run_dir / required
            if not _validate_artifact_path(artifact_path, evidence_root):
                missing_required.append(required)
                continue
            entry, payload = self._build_artifact_entry(artifact_path, run_dir, redact_secrets)
            artifacts.append(entry)
            archive_payloads[entry["path"]] = payload

        contribution_path = next(
            (
                run_dir / relative
                for relative in CONTRIBUTION_ARTIFACTS
                if _validate_artifact_path(run_dir / relative, evidence_root)
            ),
            None,
        )
        if contribution_path is None:
            missing_required.append("reports/{llm_contribution_evidence.json|agent_contribution.json}")
        else:
            entry, payload = self._build_artifact_entry(contribution_path, run_dir, redact_secrets)
            artifacts.append(entry)
            archive_payloads[entry["path"]] = payload

        # Collect additional artifacts
        for pattern in ("evidence/*.json", "repairs/*.json"):
            for path in run_dir.glob(pattern):
                if _validate_artifact_path(path, evidence_root):
                    rel = str(path.relative_to(run_dir))
                    if not any(a["path"] == rel for a in artifacts):
                        entry, payload = self._build_artifact_entry(path, run_dir, redact_secrets)
                        artifacts.append(entry)
                        archive_payloads[entry["path"]] = payload

        # Build manifest
        manifest = {
            "schema_version": 1,
            "evidence_id": "ev_%s_%s" % (task_id, commit_sha[:8] if commit_sha != "unknown" else "unknown"),
            "project_commit_sha": commit_sha,
            "dirty_worktree": dirty,
            "created_at": utc_now_iso(),
            "task_id": task_id,
            "controller": "langgraph",
            "final_status": "",
            "verify_status": "",
            "trace_id": "",
            "provider": {
                "name": "",
                "model": "",
                "protocol": "json_action",
            },
            "environment": {
                "os": platform.system(),
                "python": "%s.%s.%s" % sys.version_info[:3],
                "docker": "",
                "gpu": "",
                "cuda": "",
            },
            "repair_count": 0,
            "resume_count": 0,
            "artifacts": artifacts,
            "external_gates": [],
        }

        # Read controller result for status info
        cr_path = run_dir / "reports" / "controller_result.json"
        if cr_path.is_file():
            try:
                cr = json.loads(cr_path.read_text(encoding="utf-8"))
                manifest["final_status"] = cr.get("status") or cr.get("final_status", "")
                verify = cr.get("verify") if isinstance(cr.get("verify"), dict) else {}
                manifest["verify_status"] = cr.get("verify_status") or verify.get("status", "")
                manifest["controller"] = cr.get("controller", "langgraph")
            except (OSError, ValueError):
                pass

        if missing_required:
            manifest["status"] = "failed"
            manifest["missing_artifacts"] = missing_required
        else:
            manifest["status"] = "complete"
            output_path = Path(output_path or run_dir / "reports" / ("%s-evidence.tar.gz" % task_id))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            manifest["archive_path"] = str(output_path)
            self._write_archive(output_path, archive_payloads, manifest)
            manifest["archive_sha256"] = sha256_file(output_path)
            manifest["archive_size_bytes"] = output_path.stat().st_size

        return manifest

    def _build_artifact_entry(
        self,
        path: Path,
        run_dir: Path,
        redact_secrets: bool,
    ):
        """Build an artifact entry with computed hash.

        Args:
            path: Path to the artifact file.
            run_dir: Run directory for relative path computation.
            redact_secrets: Whether to redact secret values.

        Returns:
            Artifact entry dict with path, sha256, size_bytes.
        """
        source = path.read_bytes()
        payload = redact_artifact_bytes(path, source) if redact_secrets else source
        entry = {
            "path": str(path.relative_to(run_dir)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "redacted": payload != source,
        }
        return entry, payload

    def _write_archive(
        self,
        output_path: Path,
        payloads: Dict[str, bytes],
        manifest: Dict[str, Any],
    ) -> None:
        with tarfile.open(str(output_path), "w:gz") as archive:
            for relative, payload in sorted(payloads.items()):
                info = tarfile.TarInfo(relative)
                info.size = len(payload)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
            manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(manifest_bytes))
