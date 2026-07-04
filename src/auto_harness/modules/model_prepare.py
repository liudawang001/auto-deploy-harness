from pathlib import Path
from typing import Dict

from typing import Callable, Optional

from auto_harness.assets import HuggingFaceDownloader, ModelCache, ModelScopeDownloader
from auto_harness.assets.manifest import AssetManifest, ModelAsset
from auto_harness.models.base import write_json
from auto_harness.models.result import StageResult


class ModelPrepareModule:
    def __init__(
        self,
        cache: ModelCache,
        huggingface_downloader: HuggingFaceDownloader = None,
        modelscope_downloader: ModelScopeDownloader = None,
    ) -> None:
        self.cache = cache
        self.huggingface_downloader = huggingface_downloader or HuggingFaceDownloader()
        self.modelscope_downloader = modelscope_downloader or ModelScopeDownloader()

    def prepare(
        self,
        run_dir: Path,
        resource_plan: Dict,
        execute: bool = False,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> StageResult:
        raw_assets = resource_plan.get("model_assets") or []
        assets = []
        progress = {
            "status": "planned" if raw_assets else "empty",
            "downloaded_bytes": 0,
            "total_bytes": self._resource_total(raw_assets),
            "current_file": "",
        }

        def update_progress(update: Dict) -> None:
            progress.update({key: value for key, value in update.items() if value is not None})
            if progress_callback:
                progress_callback(dict(progress))

        for raw in raw_assets:
            asset = ModelAsset(**{key: value for key, value in raw.items() if key in ModelAsset.__dataclass_fields__})
            self.cache.reserve(asset)
            if execute:
                if asset.source == "huggingface":
                    update_progress({"status": "downloading", "asset_id": asset.asset_id})
                    asset = self.huggingface_downloader.download(asset, update_progress)
                elif asset.source == "modelscope":
                    update_progress({"status": "downloading", "asset_id": asset.asset_id})
                    asset = self.modelscope_downloader.download(asset, update_progress)
                else:
                    asset.status = "unsupported"
                    asset.last_error = "model source is not supported for download yet"
            assets.append(asset)
            progress["downloaded_bytes"] = sum(asset.downloaded_bytes for asset in assets)
            if progress_callback:
                progress_callback(dict(progress))

        manifest = AssetManifest(
            assets=assets,
            total_expected_size_bytes=self._total_size(assets),
            cache_root=str(self.cache.root),
            status=self._manifest_status(assets),
        )
        manifest_path = run_dir / "reports" / "model_assets_manifest.json"
        write_json(manifest_path, manifest)
        if not assets:
            return StageResult(
                "model_prepare",
                "passed",
                "no external model assets detected",
                {"manifest_path": str(manifest_path), "assets": [], "cache": self.cache.summary(), "progress": progress},
                evidence=[str(manifest_path)],
            )
        if execute and any(asset.status == "failed" for asset in assets):
            status = "failed"
        elif execute and any(asset.status == "unsupported" for asset in assets):
            status = "uncertain"
        else:
            status = "passed"
        summary = "model assets prepared" if execute and status == "passed" else "model assets planned; download not executed"
        if execute and status != "passed":
            summary = "model asset preparation incomplete"
        return StageResult(
            "model_prepare",
            status,
            summary,
            {
                "manifest_path": str(manifest_path),
                "assets": [asset.__dict__ for asset in assets],
                "cache": self.cache.summary(),
                "executed": execute,
                "progress": progress,
            },
            evidence=[str(manifest_path)],
            error=self._first_error(assets) if status != "passed" else None,
        )

    def _total_size(self, assets):
        sizes = [asset.expected_size_bytes for asset in assets if asset.expected_size_bytes]
        if not sizes:
            return None
        return sum(sizes)

    def _resource_total(self, raw_assets):
        sizes = [raw.get("expected_size_bytes") for raw in raw_assets if raw.get("expected_size_bytes")]
        if not sizes:
            return None
        return sum(sizes)

    def _manifest_status(self, assets):
        if not assets:
            return "empty"
        if any(asset.status == "failed" for asset in assets):
            return "failed"
        if any(asset.status == "unsupported" for asset in assets):
            return "partial"
        if all(asset.status in ("downloaded", "cached") for asset in assets):
            return "ready"
        return "planned"

    def _first_error(self, assets):
        for asset in assets:
            if asset.last_error:
                return asset.last_error
        return None
