from pathlib import Path
from typing import Dict

from auto_harness.assets import ModelCache
from auto_harness.assets.manifest import AssetManifest, ModelAsset
from auto_harness.models.base import write_json
from auto_harness.models.result import StageResult


class ModelPrepareModule:
    def __init__(self, cache: ModelCache) -> None:
        self.cache = cache

    def prepare(self, run_dir: Path, resource_plan: Dict, execute: bool = False) -> StageResult:
        raw_assets = resource_plan.get("model_assets") or []
        assets = []
        for raw in raw_assets:
            asset = ModelAsset(**{key: value for key, value in raw.items() if key in ModelAsset.__dataclass_fields__})
            self.cache.reserve(asset)
            if execute:
                asset.status = "planned"
                asset.last_error = "download execution is not implemented yet"
            assets.append(asset)

        manifest = AssetManifest(
            assets=assets,
            total_expected_size_bytes=self._total_size(assets),
            cache_root=str(self.cache.root),
            status="planned" if assets else "empty",
        )
        manifest_path = run_dir / "reports" / "model_assets_manifest.json"
        write_json(manifest_path, manifest)
        if not assets:
            return StageResult(
                "model_prepare",
                "passed",
                "no external model assets detected",
                {"manifest_path": str(manifest_path), "assets": [], "cache": self.cache.summary()},
                evidence=[str(manifest_path)],
            )
        status = "uncertain" if execute else "passed"
        summary = "model assets planned; download not executed" if not execute else "model download execution not implemented"
        return StageResult(
            "model_prepare",
            status,
            summary,
            {
                "manifest_path": str(manifest_path),
                "assets": [asset.__dict__ for asset in assets],
                "cache": self.cache.summary(),
                "executed": execute,
            },
            evidence=[str(manifest_path)],
            error="download execution is not implemented yet" if execute else None,
        )

    def _total_size(self, assets):
        sizes = [asset.expected_size_bytes for asset in assets if asset.expected_size_bytes]
        if not sizes:
            return None
        return sum(sizes)
