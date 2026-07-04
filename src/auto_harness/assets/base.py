import hashlib
from pathlib import Path
from typing import Dict, Optional


class ResumableDownloadMixin:
    def _download_file_to_cache(self, req, rel_path: str, cache_path: Path, item: Dict, progress_callback, emit_fn) -> Dict:
        target = cache_path / rel_path
        part = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("size_bytes")
        sha256 = item.get("sha256")
        etag = item.get("etag")
        if target.exists() and self._target_valid(target, expected, sha256):
            item.update({
                "status": "cached",
                "downloaded_bytes": target.stat().st_size,
                "local_path": str(target),
                "verified": bool(sha256),
            })
            if etag:
                item["etag"] = etag
            emit_fn(progress_callback, rel_path, target.stat().st_size, expected, "cached")
            return item

        existing = part.stat().st_size if part.exists() else 0
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
                    emit_fn(progress_callback, rel_path, downloaded, expected, "downloading")
        if expected and downloaded != expected:
            item.update({"status": "partial", "downloaded_bytes": downloaded, "local_path": str(part)})
            raise RuntimeError("downloaded size mismatch for %s: %s != %s" % (rel_path, downloaded, expected))
        if sha256 and self._sha256(part) != sha256:
            item.update({"status": "checksum_failed", "downloaded_bytes": downloaded, "local_path": str(part)})
            raise RuntimeError("sha256 mismatch for %s" % rel_path)
        part.replace(target)
        item.update({
            "status": "downloaded",
            "downloaded_bytes": downloaded,
            "local_path": str(target),
            "verified": bool(sha256),
        })
        if etag:
            item["etag"] = etag
        emit_fn(progress_callback, rel_path, downloaded, expected, "downloaded")
        return item

    def _target_valid(self, path: Path, expected: Optional[int], sha256: Optional[str]) -> bool:
        if expected and path.stat().st_size != expected:
            return False
        if sha256 and self._sha256(path) != sha256:
            return False
        return True

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
