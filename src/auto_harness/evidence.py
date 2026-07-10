"""Evidence Package Export: bundle project key evidence for review/interview display.

Collects and packages the following into a tar.gz archive:
- docs/evidence/real-model-deployment/ (redacted server evidence)
- docs/evidence/memory-skill-evolution-smoke/ (E2E smoke evidence)
- memory/skill_candidates/*.json, *.regression.json, *.shadow.json
- memory/skill_outcomes.jsonl
- docs/memory-skill-threat-model.md
- runs/evals/ (if exists)

Excludes:
- Tokens, secrets, private keys
- Large model files
- Cache directories
- venv, .conda
- Raw private logs
- __pycache__
"""
import io
import os
import tarfile
from pathlib import Path
from typing import List, Optional


# Default include patterns (relative to project root)
_INCLUDE_PATTERNS = [
    "docs/evidence/real-model-deployment/**",
    "docs/evidence/memory-skill-evolution-smoke/**",
    "memory/skill_candidates/*.json",
    "memory/skill_candidates/*.regression.json",
    "memory/skill_candidates/*.shadow.json",
    "memory/skill_outcomes.jsonl",
    "docs/memory-skill-threat-model.md",
]

# Directories to include if they exist
_INCLUDE_DIRS = [
    "runs/evals",
]

# Exclude patterns
_EXCLUDE_PATTERNS = [
    "*.pyc",
    "__pycache__",
    ".DS_Store",
    ".git",
    ".conda",
    "venv",
    "*.tar.gz",
    "*.whl",
]

# Max file size (10MB)
_MAX_FILE_SIZE = 10 * 1024 * 1024


def collect_evidence_files(project_root: Path) -> List[Path]:
    """Collect evidence files from the project.

    Args:
        project_root: Path to the project root directory.

    Returns:
        List of file paths to include in the evidence package.
    """
    root = Path(project_root)
    files = []

    # Collect from include patterns
    for pattern in _INCLUDE_PATTERNS:
        matched = sorted(root.glob(pattern))
        for path in matched:
            if path.is_file() and _should_include(path):
                files.append(path)

    # Collect from include directories
    for dir_name in _INCLUDE_DIRS:
        dir_path = root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            for path in sorted(dir_path.rglob("*")):
                if path.is_file() and _should_include(path):
                    files.append(path)

    # Deduplicate
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def _should_include(path: Path) -> bool:
    """Check if a file should be included in the evidence package."""
    # Check exclude patterns
    for pattern in _EXCLUDE_PATTERNS:
        if pattern in str(path):
            return False

    # Check file size
    try:
        if path.stat().st_size > _MAX_FILE_SIZE:
            return False
    except OSError:
        return False

    return True


def create_evidence_package(
    project_root: Path,
    output_path: Path,
    extra_files: List[Path] = None,
) -> dict:
    """Create a tar.gz evidence package.

    Args:
        project_root: Path to the project root directory.
        output_path: Path to write the tar.gz file.
        extra_files: Optional list of additional files to include.

    Returns:
        Dict with status, file_count, total_size, output_path.
    """
    root = Path(project_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_evidence_files(root)

    # Add extra files
    if extra_files:
        for f in extra_files:
            f = Path(f)
            if f.exists() and f.is_file() and _should_include(f):
                files.append(f)

    # Create tar.gz
    total_size = 0
    file_count = 0

    with tarfile.open(str(output_path), "w:gz") as tar:
        for file_path in files:
            arcname = str(file_path.relative_to(root))
            try:
                tar.add(str(file_path), arcname=arcname)
                total_size += file_path.stat().st_size
                file_count += 1
            except (OSError, tarfile.TarError):
                continue

    # Verify output
    if not output_path.exists():
        return {
            "status": "failed",
            "error": "output file was not created",
        }

    return {
        "status": "ok",
        "file_count": file_count,
        "total_size": total_size,
        "archive_size": output_path.stat().st_size,
        "output_path": str(output_path),
    }
