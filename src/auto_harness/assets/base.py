import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional


class ResumableDownloadMixin:
    def _call_with_retries(self, fn):
        attempts = max(1, int(getattr(self, "retry_count", 0) or 0) + 1)
        last_error = None
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - persisted by callers
                if not self._should_retry(exc) or attempt == attempts - 1:
                    raise
                last_error = exc
                backoff = float(getattr(self, "retry_backoff_seconds", 0.0) or 0.0)
                if backoff > 0:
                    time.sleep(backoff * (attempt + 1))
        if last_error:
            raise last_error

    def _should_retry(self, exc: Exception) -> bool:
        message = str(exc).lower()
        permanent_markers = (
            "sha256 mismatch",
            "checksum",
            "no downloadable model files",
            "only huggingface assets",
            "only modelscope assets",
        )
        return not any(marker in message for marker in permanent_markers)

    def _download_file_to_cache(self, req, rel_path: str, cache_path: Path, item: Dict, progress_callback, emit_fn) -> Dict:
        target = cache_path / rel_path
        part = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("size_bytes")
        sha256 = item.get("sha256")
        etag = item.get("etag")
        if target.exists() and self._target_valid(target, expected, sha256, etag):
            item.update({
                "status": "cached",
                "downloaded_bytes": target.stat().st_size,
                "local_path": str(target),
                "verified": bool(sha256),
                "etag_verified": bool(etag),
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
        self._write_meta(target, expected, sha256, etag)
        item.update({
            "status": "downloaded",
            "downloaded_bytes": downloaded,
            "local_path": str(target),
            "verified": bool(sha256),
            "etag_verified": bool(etag),
        })
        if etag:
            item["etag"] = etag
        emit_fn(progress_callback, rel_path, downloaded, expected, "downloaded")
        return item

    def _target_valid(self, path: Path, expected: Optional[int], sha256: Optional[str], etag: Optional[str] = None) -> bool:
        if expected and path.stat().st_size != expected:
            return False
        if sha256 and self._sha256(path) != sha256:
            return False
        if etag and not self._etag_matches(path, etag):
            return False
        return True

    def _etag_matches(self, path: Path, etag: str) -> bool:
        meta = self._read_meta(path)
        return meta.get("etag") == etag

    def _meta_path(self, path: Path) -> Path:
        return path.with_name(path.name + ".auto_harness_meta.json")

    def _read_meta(self, path: Path) -> Dict:
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_meta(self, path: Path, expected: Optional[int], sha256: Optional[str], etag: Optional[str]) -> None:
        meta = {
            "size_bytes": path.stat().st_size,
            "expected_size_bytes": expected,
            "sha256": sha256,
            "etag": etag,
        }
        self._meta_path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Frozen ModelFilePlan download (Document A Phase A5).
    # ------------------------------------------------------------------

    def _precheck_disk(self, cache_path: Path, files: List[Dict], disk_safety_ratio: float) -> Dict:
        """Compute remaining bytes and fail before any large request if disk is short."""
        remaining = 0
        reusable = 0
        for item in files:
            rel_path = item["path"]
            target = cache_path / rel_path
            expected = item.get("size_bytes")
            sha256 = item.get("sha256")
            etag = item.get("etag")
            if target.exists() and self._target_valid(target, expected, sha256, etag):
                continue
            part = target.with_name(target.name + ".part")
            part_size = part.stat().st_size if part.exists() else 0
            reusable += part_size
            remaining += max(0, int(expected or 0) - part_size)
        actual_peak = remaining
        required_free = int(actual_peak * float(disk_safety_ratio or 1.0))
        free = shutil.disk_usage(str(cache_path)).free
        return {
            "remaining_download_bytes": remaining,
            "partial_bytes_reusable": reusable,
            "temporary_peak_bytes": actual_peak,
            "required_free_bytes": required_free,
            "available_free_bytes": free,
            "sufficient": free >= required_free,
        }

    def _finalize_plan(self, plan, cache_path: Path, progress_callback=None) -> Dict:
        """Re-verify every required file and atomically write the complete marker."""
        from auto_harness.model_runtime.schemas import CacheCompleteMarker
        from auto_harness.models.base import write_json
        from auto_harness.utils.time import utc_now_iso

        required = [f for f in plan.files if f.get("required")]
        marker_files: List[Dict] = []
        failed: List[str] = []
        for item in required:
            rel_path = item["path"]
            target = cache_path / rel_path
            sha = item.get("sha256")
            size = item.get("size_bytes")
            if not target.exists():
                failed.append(rel_path)
                continue
            if isinstance(size, int) and target.stat().st_size != size:
                failed.append(rel_path)
                continue
            if sha and self._sha256(target) != sha:
                failed.append(rel_path)
                continue
            marker_files.append({
                "path": rel_path,
                "size_bytes": target.stat().st_size,
                "sha256": sha,
            })
        if failed:
            return {
                "status": "integrity_failed",
                "failed_files": failed,
                "complete_marker_path": "",
            }
        marker = CacheCompleteMarker(
            status="complete",
            model_identity=plan.model_identity,
            file_plan_hash=plan.plan_hash,
            files=marker_files,
            verified_at=utc_now_iso(),
        )
        marker.marker_hash = marker.compute_marker_hash()
        marker_path = cache_path / ".auto_harness_complete.json"
        write_json(marker_path, marker)
        if progress_callback:
            progress_callback({"status": "complete_marker_written", "current_file": ""})
        return {
            "status": "complete",
            "complete_marker_path": str(marker_path),
            "marker_hash": marker.marker_hash,
            "verified_files": len(marker_files),
        }
