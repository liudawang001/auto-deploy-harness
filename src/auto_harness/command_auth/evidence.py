"""Repository evidence hashing and boundary checks."""

import hashlib
from pathlib import Path

from auto_harness.command_auth.schemas import CommandEvidence, canonical_hash


def safe_repository_file(repo_dir: Path, relative: str) -> Path:
    root = Path(repo_dir).resolve()
    raw = Path(str(relative or ""))
    if raw.is_absolute() or not str(relative or "") or "\x00" in str(relative):
        raise ValueError("repository evidence path is invalid")
    candidate = root / raw
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("repository evidence must be a regular non-symlink file")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("repository evidence escapes workspace") from exc
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_evidence(
    repo_dir: Path,
    source_type: str,
    relative: str,
    repository_fingerprint: str,
    *,
    line_start: int = 0,
    line_end: int = 0,
    declaration_key: str = "",
    declared_value: str = "",
) -> CommandEvidence:
    path = safe_repository_file(repo_dir, relative)
    sha256 = file_sha256(path)
    identity = {
        "source_type": source_type,
        "path": str(relative).replace("\\", "/"),
        "sha256": sha256,
        "line_start": int(line_start or 0),
        "line_end": int(line_end or 0),
        "declaration_key": declaration_key,
        "declared_value": declared_value,
        "repository_fingerprint": repository_fingerprint,
    }
    return CommandEvidence(
        evidence_id="ev_%s" % canonical_hash(identity)[:20],
        **identity,
    )


def revalidate_evidence(repo_dir: Path, evidence: CommandEvidence) -> bool:
    try:
        return file_sha256(safe_repository_file(repo_dir, evidence.path)) == evidence.sha256
    except (OSError, ValueError):
        return False
