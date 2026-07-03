from pathlib import Path
from typing import Dict

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
