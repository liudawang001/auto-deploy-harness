"""Phase A5 tests: revision-bound cache identity and frozen-plan download."""
import hashlib
import json
from pathlib import Path

import pytest

from auto_harness.assets.cache import ModelCache, revision_cache_key
from auto_harness.assets.huggingface import HuggingFaceDownloader
from auto_harness.model_runtime.schemas import CacheCompleteMarker, ModelFilePlan

SHA = "a" * 40
SHA2 = "b" * 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._body)
        chunk = self._body[:size]
        self._body = self._body[size:]
        return chunk


def _plan(repo_id="org/model", revision=SHA, identity=None):
    content = b"model-data"
    files = [
        {"path": "config.json", "role": "config", "size_bytes": len(content), "sha256": _sha(content), "etag": None, "required": True},
        {"path": "model.safetensors", "role": "weight_shard", "size_bytes": len(content), "sha256": _sha(content), "etag": None, "required": True},
    ]
    plan = ModelFilePlan(
        model_identity=identity or "huggingface:%s@%s" % (repo_id, revision),
        files=files,
        total_size_bytes=sum(f["size_bytes"] for f in files),
    )
    plan.plan_hash = plan.compute_plan_hash()
    return plan


class TestRevisionCacheKey:
    def test_deterministic(self):
        a = revision_cache_key("huggingface", "org/model", SHA, "ph")
        b = revision_cache_key("huggingface", "org/model", SHA, "ph")
        assert a == b

    def test_revision_change_isolates(self):
        a = revision_cache_key("huggingface", "org/model", SHA, "ph")
        b = revision_cache_key("huggingface", "org/model", SHA2, "ph")
        assert a != b

    def test_file_plan_change_isolates(self):
        a = revision_cache_key("huggingface", "org/model", SHA, "ph1")
        b = revision_cache_key("huggingface", "org/model", SHA, "ph2")
        assert a != b

    def test_source_isolates(self):
        a = revision_cache_key("huggingface", "org/model", SHA, "ph")
        b = revision_cache_key("modelscope", "org/model", SHA, "ph")
        assert a != b

    def test_revision_cache_path_shape(self, tmp_path):
        cache = ModelCache(tmp_path / "model_cache")
        path = cache.revision_cache_path("huggingface", "org/model", SHA, "ph")
        assert path == tmp_path / "model_cache" / "huggingface" / path.name
        assert path.name.startswith("org-model_")


class TestCompleteMarker:
    def test_round_trip(self, tmp_path):
        cache = ModelCache(tmp_path / "model_cache")
        cache_dir = cache.revision_cache_path("huggingface", "org/model", SHA, "ph")
        cache_dir.mkdir(parents=True)
        marker = CacheCompleteMarker(
            status="complete",
            model_identity="huggingface:org/model@%s" % SHA,
            file_plan_hash="ph",
            files=[{"path": "model.safetensors", "size_bytes": 3, "sha256": _sha(b"abc")}],
            verified_at="now",
        )
        marker.marker_hash = marker.compute_marker_hash()
        marker_path = cache.complete_marker_path(cache_dir)
        marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
        read = cache.read_complete_marker(cache_dir)
        assert read is not None
        assert read["status"] == "complete"
        assert read["marker_hash"] == marker.marker_hash

    def test_missing_marker_returns_none(self, tmp_path):
        cache = ModelCache(tmp_path / "model_cache")
        cache_dir = cache.revision_cache_path("huggingface", "org/model", SHA, "ph")
        cache_dir.mkdir(parents=True)
        assert cache.read_complete_marker(cache_dir) is None


class TestPlanDownload:
    def test_download_plan_complete(self, tmp_path):
        downloader = HuggingFaceDownloader(urlopen=self._urlopen({"config.json": b"model-data", "model.safetensors": b"model-data"}), token="")
        plan = _plan()
        cache_dir = tmp_path / "cache"
        result = downloader.download_plan("org/model", SHA, plan, cache_dir)
        assert result["status"] == "complete"
        assert (cache_dir / ".auto_harness_complete.json").exists()
        assert (cache_dir / "model.safetensors").read_bytes() == b"model-data"

    def test_download_plan_resume_206(self, tmp_path):
        content = b"abcdef"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors.part").write_bytes(b"abc")
        calls = []

        def fake(req, timeout):
            calls.append(req)
            return FakeResponse(b"def", status=206)

        downloader = HuggingFaceDownloader(urlopen=fake, token="", chunk_size=2)
        files = [
            {"path": "model.safetensors", "role": "weight_shard", "size_bytes": len(content), "sha256": _sha(content), "etag": None, "required": True},
        ]
        plan = ModelFilePlan(model_identity="huggingface:org/model@%s" % SHA, files=files, total_size_bytes=len(content))
        plan.plan_hash = plan.compute_plan_hash()
        result = downloader.download_plan("org/model", SHA, plan, cache_dir)
        assert result["status"] == "complete"
        assert calls[0].headers.get("Range") == "bytes=3-"
        assert (cache_dir / "model.safetensors").read_bytes() == content

    def test_download_plan_server_ignores_range_200(self, tmp_path):
        content = b"XYZ"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors.part").write_bytes(b"stale-old-part-data")
        calls = []

        def fake(req, timeout):
            calls.append(req)
            return FakeResponse(content, status=200)

        downloader = HuggingFaceDownloader(urlopen=fake, token="", chunk_size=8)
        files = [
            {"path": "model.safetensors", "role": "weight_shard", "size_bytes": len(content), "sha256": _sha(content), "etag": None, "required": True},
        ]
        plan = ModelFilePlan(model_identity="huggingface:org/model@%s" % SHA, files=files, total_size_bytes=len(content))
        plan.plan_hash = plan.compute_plan_hash()
        result = downloader.download_plan("org/model", SHA, plan, cache_dir)
        assert result["status"] == "complete"
        # stale .part was reset, final content is exactly the full body
        assert (cache_dir / "model.safetensors").read_bytes() == content

    def test_download_plan_insufficient_disk_no_request(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        download_requests = []

        def fake(req, timeout):
            download_requests.append(req)
            return FakeResponse(b"", status=200)

        downloader = HuggingFaceDownloader(urlopen=fake, token="")
        # 10 PB plan size -> required free exceeds any real disk
        files = [
            {"path": "model.safetensors", "role": "weight_shard", "size_bytes": 10 * 1024 ** 5, "sha256": None, "etag": None, "required": True},
        ]
        plan = ModelFilePlan(model_identity="huggingface:org/model@%s" % SHA, files=files, total_size_bytes=10 * 1024 ** 5)
        plan.plan_hash = plan.compute_plan_hash()
        result = downloader.download_plan("org/model", SHA, plan, cache_dir)
        assert result["status"] == "insufficient_disk"
        assert download_requests == []

    def test_finalize_plan_integrity_failed(self, tmp_path):
        downloader = HuggingFaceDownloader(token="")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"wrong-content")
        plan = _plan()  # expects "model-data" with its sha256
        result = downloader._finalize_plan(plan, cache_dir)
        assert result["status"] == "integrity_failed"
        assert "model.safetensors" in result["failed_files"]

    @staticmethod
    def _urlopen(routes):
        def fake(req, timeout):
            path = req.full_url.split("/")[-1]
            return FakeResponse(routes.get(path, b""), status=200)
        return fake
