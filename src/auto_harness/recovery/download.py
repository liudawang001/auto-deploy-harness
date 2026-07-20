"""Download reconciler: detects external state of model downloads.

Reconciles model download operations against the filesystem:
- Complete file with valid size/hash → reuse
- Partial file with matching metadata → continue (with offset)
- Partial file with mismatched metadata → manual
- No file → retry
- Checksum failure → conflict (cannot trust the file)
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.utils.atomic import atomic_write_text


# Keys used to identify a partial download's revision/source
PARTIAL_IDENTITY_KEYS = (
    "source", "repo_id", "revision", "relative_path",
    "expected_size", "etag", "sha256",
)


def reconcile_result(decision, reason, evidence_paths=None, **observed):
    """Build a ReconcileResult dict."""
    return {
        "decision": decision,
        "observed_state": observed,
        "reason": reason,
        "evidence_paths": list(evidence_paths or []),
    }


def sha256_file(path, chunk_size=1024 * 1024):
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def partial_metadata_path(part):
    """Get the sidecar metadata path for a .part file."""
    return Path(str(part) + ".auto_harness_meta.json")


def write_partial_metadata(part, identity):
    """Write partial download identity metadata atomically.

    Called before the first network request so that if the process
    crashes, the partial file's identity can be verified on resume.
    """
    safe = {key: identity.get(key) for key in PARTIAL_IDENTITY_KEYS}
    atomic_write_text(
        partial_metadata_path(part),
        json.dumps(safe, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
    )


def read_partial_metadata(part):
    """Read partial download identity metadata.

    Returns None if the metadata file doesn't exist or is invalid.
    """
    path = partial_metadata_path(part)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def meta_matches(meta, identity):
    """Check if partial metadata matches the requested identity."""
    if not meta:
        return False
    return all(meta.get(key) == identity.get(key) for key in PARTIAL_IDENTITY_KEYS)


class DownloadReconciler:
    """Reconciler for model download operations.

    Checks the filesystem for existing complete or partial downloads
    and determines whether to reuse, continue, retry, or escalate.
    """
    resource_type = "model_download"

    def reconcile(self, operation):
        """Reconcile a model download operation against the filesystem.

        Decision logic:
        1. Target exists and is valid → reuse
        2. Target exists but invalid → conflict (integrity failure)
        3. .part exists with matching metadata → continue (with offset)
        4. .part exists with mismatched metadata → manual
        5. No file → retry
        """
        identity = operation["resource_identity"]
        target = Path(identity["target_path"])
        part = Path(str(target) + ".part")
        expected = int(identity.get("expected_size") or 0)
        expected_hash = identity.get("sha256", "")

        # 1. Complete target file exists
        if target.exists() and target.is_file():
            valid_size = not expected or target.stat().st_size == expected
            valid_hash = not expected_hash or sha256_file(target) == expected_hash
            if valid_size and valid_hash:
                return reconcile_result(
                    "reuse", "verified target exists",
                    target=str(target), size=target.stat().st_size,
                )
            return reconcile_result(
                "conflict", "target exists but integrity check failed",
                target=str(target), size=target.stat().st_size,
                expected_size=expected,
            )

        # 2. Partial file exists
        if part.exists():
            meta = read_partial_metadata(part)
            if meta_matches(meta, identity):
                return reconcile_result(
                    "continue", "matching partial file exists",
                    offset=part.stat().st_size,
                    part_path=str(part),
                )
            # No sidecar or mismatched metadata
            if meta is None:
                return reconcile_result(
                    "manual",
                    "partial file exists but has no identity metadata; "
                    "cannot determine revision",
                    part_path=str(part), part_size=part.stat().st_size,
                )
            return reconcile_result(
                "manual",
                "partial metadata does not match requested revision",
                part_path=str(part),
                meta_source=meta.get("source", ""),
                meta_revision=meta.get("revision", ""),
            )

        # 3. No file at all
        return reconcile_result("retry", "no cached or partial file exists")
