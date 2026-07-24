"""Project-level evidence archive used for review and interview artifacts."""
import io
import tarfile
from pathlib import Path
from typing import List

from auto_harness.evidence.exporter import redact_artifact_bytes


_INCLUDE_PATTERNS = [
    "docs/evidence/real-model-deployment/**",
    "docs/evidence/memory-skill-evolution-smoke/**",
    "memory/skill_candidates/*.json",
    "memory/skill_outcomes.jsonl",
    "docs/memory-skill-threat-model.md",
]
_INCLUDE_DIRS = ["runs/evals"]
_EXCLUDED_PARTS = {
    ".git",
    ".conda",
    "venv",
    "__pycache__",
    "model_cache",
}
_MAX_FILE_SIZE = 10 * 1024 * 1024


def _should_include(path: Path) -> bool:
    if any(part in _EXCLUDED_PARTS for part in path.parts):
        return False
    lowered = path.name.lower()
    if lowered.endswith((".pyc", ".tar.gz", ".whl", ".bin", ".safetensors")):
        return False
    if any(marker in lowered for marker in ("token", "credential", "private_key")):
        return False
    try:
        return path.is_file() and path.stat().st_size <= _MAX_FILE_SIZE
    except OSError:
        return False


def collect_evidence_files(project_root: Path) -> List[Path]:
    root = Path(project_root).resolve()
    files = []
    for pattern in _INCLUDE_PATTERNS:
        files.extend(path for path in sorted(root.glob(pattern)) if _should_include(path))
    for relative in _INCLUDE_DIRS:
        directory = root / relative
        if directory.is_dir():
            files.extend(path for path in sorted(directory.rglob("*")) if _should_include(path))
    return list(dict.fromkeys(files))


def create_evidence_package(
    project_root: Path,
    output_path: Path,
    extra_files: List[Path] = None,
) -> dict:
    root = Path(project_root).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = collect_evidence_files(root)
    for candidate in extra_files or []:
        candidate = Path(candidate).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if _should_include(candidate):
            files.append(candidate)

    total_size = 0
    file_count = 0
    with tarfile.open(str(output_path), "w:gz") as archive:
        for path in dict.fromkeys(files):
            try:
                payload = redact_artifact_bytes(path, path.read_bytes())
                relative = str(path.relative_to(root))
                info = tarfile.TarInfo(relative)
                info.size = len(payload)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
                total_size += len(payload)
                file_count += 1
            except (OSError, tarfile.TarError, ValueError):
                continue

    if not output_path.is_file():
        return {"status": "failed", "error": "output file was not created"}
    return {
        "status": "ok",
        "file_count": file_count,
        "total_size": total_size,
        "archive_size": output_path.stat().st_size,
        "output_path": str(output_path),
        "redaction_applied": True,
    }
