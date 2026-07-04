import shutil
import time
import json
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
        ensure_dir(cache_path)
        self._write_asset_meta(cache_path, asset)
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
                meta = self._read_asset_meta(cache_dir)
                size = self._dir_size(cache_dir)
                entries.append({
                    "source": source_dir.name,
                    "cache_key": cache_dir.name,
                    "repo_id": meta.get("repo_id", ""),
                    "revision": meta.get("revision", ""),
                    "origin": meta.get("origin", ""),
                    "path": str(cache_dir),
                    "size_bytes": size,
                    "mtime": cache_dir.stat().st_mtime,
                })
        return sorted(entries, key=lambda item: item["mtime"])

    def cleanup(
        self,
        max_total_bytes: Optional[int] = None,
        older_than_days: Optional[float] = None,
        dry_run: bool = True,
        source: Optional[str] = None,
        repo_id: Optional[str] = None,
        keep_cache_keys: Optional[List[str]] = None,
        keep_repo_ids: Optional[List[str]] = None,
    ) -> Dict:
        entries = self.entries()
        scoped_entries = self._filter_entries(entries, source=source, repo_id=repo_id)
        keep_cache_keys = set(keep_cache_keys or [])
        keep_repo_ids = set(keep_repo_ids or [])
        now = time.time()
        candidates = []
        if older_than_days is not None:
            cutoff = now - older_than_days * 86400
            candidates.extend([entry for entry in scoped_entries if entry["mtime"] < cutoff and not self._kept(entry, keep_cache_keys, keep_repo_ids)])
        if max_total_bytes is not None:
            total = sum(entry["size_bytes"] for entry in scoped_entries)
            for entry in scoped_entries:
                if total <= max_total_bytes:
                    break
                if self._kept(entry, keep_cache_keys, keep_repo_ids):
                    continue
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
            "filters": {
                "source": source,
                "repo_id": repo_id,
                "keep_cache_keys": sorted(keep_cache_keys),
                "keep_repo_ids": sorted(keep_repo_ids),
            },
            "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
            "scoped_size_bytes": sum(entry["size_bytes"] for entry in scoped_entries),
            "scoped_count": len(scoped_entries),
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

    def _filter_entries(self, entries: List[Dict], source: Optional[str] = None, repo_id: Optional[str] = None) -> List[Dict]:
        filtered = []
        for entry in entries:
            if source and entry.get("source") != source:
                continue
            if repo_id and not self._repo_matches(entry, repo_id):
                continue
            filtered.append(entry)
        return filtered

    def _repo_matches(self, entry: Dict, repo_id: str) -> bool:
        fallback = safe_name(repo_id.replace("/", "-"))
        cache_key = entry.get("cache_key", "")
        return entry.get("repo_id") == repo_id or repo_id in cache_key or fallback in cache_key

    def _kept(self, entry: Dict, keep_cache_keys: set, keep_repo_ids: set) -> bool:
        if entry.get("cache_key") in keep_cache_keys or entry.get("repo_id") in keep_repo_ids:
            return True
        return any(self._repo_matches(entry, repo_id) for repo_id in keep_repo_ids)

    def _asset_meta_path(self, cache_dir: Path) -> Path:
        return cache_dir / ".auto_harness_asset.json"

    def _write_asset_meta(self, cache_dir: Path, asset: ModelAsset) -> None:
        meta = {
            "source": asset.source,
            "repo_id": asset.repo_id,
            "revision": asset.revision,
            "origin": asset.origin,
            "asset_id": asset.asset_id,
            "cache_key": asset.cache_key,
        }
        self._asset_meta_path(cache_dir).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _read_asset_meta(self, cache_dir: Path) -> Dict:
        path = self._asset_meta_path(cache_dir)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
