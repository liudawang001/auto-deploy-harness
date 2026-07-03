import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    allowed = []
    for ch in value.lower():
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        elif ch in (" ", ".", "/"):
            allowed.append("-")
    result = "".join(allowed).strip("-")
    return result or "project"


def short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def snapshot_files(root: Path) -> Dict[str, float]:
    if not root.exists():
        return {}
    result: Dict[str, float] = {}
    for base, _, files in os.walk(str(root)):
        for name in files:
            path = Path(base) / name
            try:
                result[str(path.relative_to(root))] = path.stat().st_mtime
            except OSError:
                continue
    return result


def diff_snapshot(before: Dict[str, float], after: Dict[str, float]) -> List[str]:
    changed: List[str] = []
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            changed.append(path)
    return sorted(changed)


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return None

