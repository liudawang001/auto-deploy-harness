import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.assets.base import ResumableDownloadMixin
from auto_harness.assets.manifest import ModelAsset
from auto_harness.assets.selection import ModelFileSelector


class ModelScopeDownloader(ResumableDownloadMixin):
    def __init__(
        self,
        urlopen=None,
        token: Optional[str] = None,
        api_base: Optional[str] = None,
        download_base: Optional[str] = None,
        chunk_size: int = 1024 * 1024,
        selector: ModelFileSelector = None,
        max_workers: int = 1,
        retry_count: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.urlopen = urlopen or urllib.request.urlopen
        self.token = token if token is not None else os.environ.get("MODELSCOPE_TOKEN")
        self.api_base = (api_base or os.environ.get("MODELSCOPE_API_BASE") or "https://www.modelscope.cn/api/v1/models").rstrip("/")
        self.download_base = (download_base or os.environ.get("MODELSCOPE_DOWNLOAD_BASE") or "https://www.modelscope.cn/models").rstrip("/")
        self.chunk_size = chunk_size
        self.selector = selector or ModelFileSelector()
        self.max_workers = max(1, int(max_workers or 1))
        self.retry_count = max(0, int(retry_count or 0))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds or 0.0))

    def download(self, asset: ModelAsset, progress_callback: Optional[Callable[[Dict], None]] = None) -> ModelAsset:
        if asset.source != "modelscope":
            asset.status = "unsupported"
            asset.last_error = "only modelscope assets are supported by this downloader"
            return asset
        cache_path = Path(asset.cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            files = self._list_files(asset)
            asset.files = []
            for record in self._download_files(asset, files, cache_path, progress_callback):
                asset.files.append(record)
                asset.downloaded_bytes = sum(int(file.get("downloaded_bytes") or 0) for file in asset.files)
            asset.status = "downloaded"
            asset.last_error = None
        except Exception as exc:  # noqa: BLE001 - persisted into manifest
            asset.status = "failed"
            asset.last_error = str(exc)
        return asset

    def _list_files(self, asset: ModelAsset) -> List[Dict]:
        repo = urllib.parse.quote(asset.repo_id, safe="/")
        revision = urllib.parse.quote(asset.revision or "master", safe="")
        url = "%s/%s/repo/files?Revision=%s&Recursive=true" % (self.api_base, repo, revision)
        req = urllib.request.Request(url, method="GET")
        self._add_auth(req)
        def fetch_body():
            with self.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")

        body = self._call_with_retries(fetch_body)
        payload = json.loads(body)
        items = payload.get("Data") or payload.get("data") or payload.get("files") or payload
        if isinstance(items, dict):
            items = items.get("Files") or items.get("files") or []
        files = []
        for item in items:
            path = item.get("Path") or item.get("path") or item.get("Name") or item.get("name")
            if not path or not self._should_download(path):
                continue
            files.append({
                "path": path,
                "size_bytes": item.get("Size") or item.get("size"),
                "etag": item.get("Sha256") or item.get("sha256") or item.get("Revision") or item.get("revision"),
                "sha256": item.get("Sha256") or item.get("sha256"),
                "status": "planned",
            })
        if not files:
            raise RuntimeError("no downloadable model files discovered for %s" % asset.repo_id)
        return files

    def _download_files(self, asset: ModelAsset, files: List[Dict], cache_path: Path, progress_callback) -> List[Dict]:
        if self.max_workers <= 1 or len(files) <= 1:
            return [self._download_file(asset, dict(item), cache_path, progress_callback) for item in files]
        results = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as executor:
            future_map = {
                executor.submit(self._download_file, asset, dict(item), cache_path, progress_callback): index
                for index, item in enumerate(files)
            }
            for future in as_completed(future_map):
                results[future_map[future]] = future.result()
        return [results[index] for index in sorted(results)]

    def _download_file(self, asset: ModelAsset, item: Dict, cache_path: Path, progress_callback) -> Dict:
        rel_path = item["path"]
        def download_once():
            req = urllib.request.Request(self._resolve_url(asset, rel_path), method="GET")
            self._add_auth(req)
            return self._download_file_to_cache(
                req,
                rel_path,
                cache_path,
                item,
                progress_callback,
                lambda cb, path, downloaded, total, status: self._emit(cb, asset, path, downloaded, total, status),
            )

        return self._call_with_retries(download_once)

    def _resolve_url(self, asset: ModelAsset, rel_path: str) -> str:
        repo = urllib.parse.quote(asset.repo_id, safe="/")
        revision = urllib.parse.quote(asset.revision or "master", safe="")
        path = urllib.parse.quote(rel_path, safe="/")
        return "%s/%s/resolve/%s/%s" % (self.download_base, repo, revision, path)

    def _should_download(self, path: str) -> bool:
        return self.selector.should_download(path)

    def _add_auth(self, req) -> None:
        if self.token:
            req.add_header("Authorization", "Bearer %s" % self.token)

    def _read_chunk(self, resp):
        try:
            return resp.read(self.chunk_size)
        except TypeError:
            return resp.read()

    def _emit(self, callback, asset: ModelAsset, file_path: str, downloaded, total, status: str) -> None:
        if not callback:
            return
        callback({
            "asset_id": asset.asset_id,
            "repo_id": asset.repo_id,
            "current_file": file_path,
            "downloaded_bytes": downloaded or 0,
            "total_bytes": total,
            "status": status,
        })
