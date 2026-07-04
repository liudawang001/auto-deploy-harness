import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.assets.manifest import ModelAsset
from auto_harness.utils.files import ensure_dir, safe_name, short_hash


class ModelCache:
    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)

    def reserve(self, asset: ModelAsset) -> ModelAsset:
        cache_key = self.cache_key(asset)
        source_dir = ensure_dir(self.root / asset.source)
        cache_path = source_dir / cache_key
        asset.cache_key = cache_key
        asset.cache_path = str(cache_path)
        return asset

    def cache_key(self, asset: ModelAsset) -> str:
        base = "%s:%s:%s:%s" % (asset.source, asset.repo_id, asset.revision, asset.origin)
        readable = safe_name(asset.repo_id.replace("/", "-") or asset.asset_id)
        return "%s_%s" % (readable[:80], short_hash(base, 12))

    def summary(self) -> Dict:
        return {
            "root": str(self.root),
            "exists": self.root.exists(),
        }

    def entries(self) -> List[Dict]:
        entries = []
        if not self.root.exists():
            return entries
        for source_dir in self.root.iterdir():
            if not source_dir.is_dir():
                continue
            for cache_dir in source_dir.iterdir():
                if not cache_dir.is_dir():
                    continue
                size = self._dir_size(cache_dir)
                entries.append({
                    "source": source_dir.name,
                    "cache_key": cache_dir.name,
                    "path": str(cache_dir),
                    "size_bytes": size,
                    "mtime": cache_dir.stat().st_mtime,
                })
        return sorted(entries, key=lambda item: item["mtime"])

    def cleanup(self, max_total_bytes: Optional[int] = None, older_than_days: Optional[float] = None, dry_run: bool = True) -> Dict:
        entries = self.entries()
        now = time.time()
        candidates = []
        if older_than_days is not None:
            cutoff = now - older_than_days * 86400
            candidates.extend([entry for entry in entries if entry["mtime"] < cutoff])
        if max_total_bytes is not None:
            total = sum(entry["size_bytes"] for entry in entries)
            for entry in entries:
                if total <= max_total_bytes:
                    break
                if entry not in candidates:
                    candidates.append(entry)
                total -= entry["size_bytes"]
        deleted = []
        errors = []
        for entry in candidates:
            if dry_run:
                continue
            try:
                shutil.rmtree(entry["path"])
                deleted.append(entry)
            except OSError as exc:
                errors.append({"path": entry["path"], "error": str(exc)})
        return {
            "dry_run": dry_run,
            "root": str(self.root),
            "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
            "candidate_count": len(candidates),
            "candidate_size_bytes": sum(entry["size_bytes"] for entry in candidates),
            "candidates": candidates,
            "deleted": deleted,
            "errors": errors,
        }

    def _dir_size(self, path: Path) -> int:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total
