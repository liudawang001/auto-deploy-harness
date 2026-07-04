import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.assets.manifest import ModelAsset


class HuggingFaceDownloader:
    """Small stdlib downloader with resumable file writes for Hugging Face assets."""

    WEIGHT_SUFFIXES = (
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
        ".model",
    )
    CONFIG_SUFFIXES = (
        ".json",
        ".txt",
        ".tiktoken",
        ".py",
        ".md",
    )

    def __init__(self, urlopen=None, token: Optional[str] = None, chunk_size: int = 1024 * 1024) -> None:
        self.urlopen = urlopen or urllib.request.urlopen
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.chunk_size = chunk_size

    def download(self, asset: ModelAsset, progress_callback: Optional[Callable[[Dict], None]] = None) -> ModelAsset:
        if asset.source != "huggingface":
            asset.status = "unsupported"
            asset.last_error = "only huggingface assets are supported by this downloader"
            return asset
        cache_path = Path(asset.cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            files = self._list_files(asset)
            asset.files = []
            for item in files:
                record = self._download_file(asset, item, cache_path, progress_callback)
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
        revision = urllib.parse.quote(asset.revision or "main", safe="")
        url = "https://huggingface.co/api/models/%s/tree/%s?recursive=true" % (repo, revision)
        req = urllib.request.Request(url, method="GET")
        self._add_auth(req)
        with self.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        files = []
        for item in data:
            if item.get("type") not in (None, "file"):
                continue
            path = item.get("path") or item.get("rfilename")
            if not path or not self._should_download(path):
                continue
            files.append({
                "path": path,
                "size_bytes": item.get("size"),
                "status": "planned",
            })
        if not files:
            raise RuntimeError("no downloadable model files discovered for %s" % asset.repo_id)
        return files

    def _download_file(self, asset: ModelAsset, item: Dict, cache_path: Path, progress_callback) -> Dict:
        rel_path = item["path"]
        target = cache_path / rel_path
        part = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("size_bytes")
        if target.exists() and expected and target.stat().st_size == expected:
            item.update({"status": "cached", "downloaded_bytes": expected, "local_path": str(target)})
            self._emit(progress_callback, asset, rel_path, expected, expected, "cached")
            return item

        existing = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(self._resolve_url(asset, rel_path), method="GET")
        self._add_auth(req)
        if existing:
            req.add_header("Range", "bytes=%d-" % existing)
        with self.urlopen(req, timeout=300) as resp:
            status = getattr(resp, "status", None) or getattr(resp, "code", None)
            if existing and status == 200:
                existing = 0
                part.write_bytes(b"")
            mode = "ab" if existing else "wb"
            downloaded = existing
            with part.open(mode) as f:
                while True:
                    chunk = self._read_chunk(resp)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    self._emit(progress_callback, asset, rel_path, downloaded, expected, "downloading")
        if expected and downloaded != expected:
            item.update({"status": "partial", "downloaded_bytes": downloaded, "local_path": str(part)})
            raise RuntimeError("downloaded size mismatch for %s: %s != %s" % (rel_path, downloaded, expected))
        part.replace(target)
        item.update({"status": "downloaded", "downloaded_bytes": downloaded, "local_path": str(target)})
        self._emit(progress_callback, asset, rel_path, downloaded, expected, "downloaded")
        return item

    def _resolve_url(self, asset: ModelAsset, rel_path: str) -> str:
        repo = urllib.parse.quote(asset.repo_id, safe="/")
        revision = urllib.parse.quote(asset.revision or "main", safe="")
        path = urllib.parse.quote(rel_path, safe="/")
        return "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, path)

    def _should_download(self, path: str) -> bool:
        lower = path.lower()
        return lower.endswith(self.WEIGHT_SUFFIXES) or lower.endswith(self.CONFIG_SUFFIXES)

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
