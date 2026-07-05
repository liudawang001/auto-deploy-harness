import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class GitLFSDetector:
    POINTER_HEADER = "version https://git-lfs.github.com/spec/v1"
    OID_RE = re.compile(r"oid sha256:([0-9a-fA-F]{64})")
    SIZE_RE = re.compile(r"size (\d+)")

    def __init__(self, available: Optional[bool] = None) -> None:
        self.available = available

    def detect(self, repo_dir: Path) -> Dict:
        patterns = self._patterns(repo_dir)
        pointers = self._pointers(repo_dir)
        required = bool(patterns or pointers)
        available = self._available()
        plan = {
            "required": required,
            "available": available,
            "patterns": patterns,
            "pointers": pointers,
            "pointer_count": len(pointers),
            "total_pointer_size_bytes": sum(item.get("size_bytes") or 0 for item in pointers),
            "prepare_commands": [["git", "lfs", "install"], ["git", "lfs", "pull"]] if required else [],
        }
        if required and not available:
            plan["diagnosis"] = {
                "category": "git_lfs_missing",
                "signal": "Git LFS pointers or attributes detected but git-lfs is not available",
                "suggested_fix": "install git-lfs and run git lfs pull before model_prepare",
                "confidence": 0.9,
            }
        return plan

    def _available(self) -> bool:
        if self.available is not None:
            return self.available
        return shutil.which("git-lfs") is not None

    def _patterns(self, repo_dir: Path) -> List[str]:
        path = repo_dir / ".gitattributes"
        if not path.exists():
            return []
        patterns = []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "filter=lfs" not in stripped:
                continue
            patterns.append(stripped.split()[0])
        return sorted(set(patterns))

    def _pointers(self, repo_dir: Path) -> List[Dict]:
        pointers = []
        for path in sorted(repo_dir.rglob("*")):
            if path.is_dir() or ".git" in path.parts:
                continue
            try:
                if path.stat().st_size > 4096:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.startswith(self.POINTER_HEADER):
                continue
            oid = self._match(self.OID_RE, text)
            size = self._match(self.SIZE_RE, text)
            pointers.append({
                "path": str(path.relative_to(repo_dir)),
                "oid": oid,
                "size_bytes": int(size) if size else None,
            })
        return pointers

    def _match(self, pattern, text: str):
        match = pattern.search(text)
        return match.group(1) if match else None


class GitLFSProgressParser:
    """Parses common git-lfs progress lines into structured stage progress."""

    OBJECTS_RE = re.compile(r"Downloading LFS objects:\s*(\d+)%\s*\((\d+)/(\d+)\)", re.IGNORECASE)
    FILE_BYTES_RE = re.compile(r"\((\d+)\s+of\s+(\d+)\s+files?\)\s+([0-9.]+)\s*([KMGT]?B)\s*/\s*([0-9.]+)\s*([KMGT]?B)", re.IGNORECASE)

    def parse(self, text: str) -> Dict:
        progress: Dict = {}
        if not text:
            return progress
        for line in text.splitlines():
            objects_match = self.OBJECTS_RE.search(line)
            if objects_match:
                progress.update({
                    "percent": int(objects_match.group(1)),
                    "files_done": int(objects_match.group(2)),
                    "files_total": int(objects_match.group(3)),
                    "status": "git_lfs_downloading",
                })
            bytes_match = self.FILE_BYTES_RE.search(line)
            if bytes_match:
                downloaded = self._to_bytes(float(bytes_match.group(3)), bytes_match.group(4))
                total = self._to_bytes(float(bytes_match.group(5)), bytes_match.group(6))
                progress.update({
                    "files_done": int(bytes_match.group(1)),
                    "files_total": int(bytes_match.group(2)),
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                    "percent": int(downloaded * 100 / total) if total else progress.get("percent"),
                    "status": "git_lfs_downloading",
                })
        return progress

    def _to_bytes(self, value: float, unit: str) -> int:
        factors = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
            "TB": 1024 ** 4,
        }
        return int(value * factors.get(unit.upper(), 1))
