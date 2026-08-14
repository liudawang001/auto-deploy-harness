"""Phase A8 tests: full offline model preparation pipeline."""
import hashlib
import json
from pathlib import Path

import pytest

from auto_harness.assets.cache import ModelCache
from auto_harness.assets.huggingface import HuggingFaceDownloader
from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.preparation import ModelPreparationOrchestrator

SHA = "a" * 40
GB = 1024 ** 3


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


class FakeSourceClient:
    def __init__(self, config, files, revision=SHA, index_content=None):
        self._config = config
        self._files = files
        self._revision = revision
        self._index_content = index_content

    def resolve_revision(self, repo_id, requested_revision):
        return {"resolved_revision": self._revision, "gated": False, "license": "apache-2.0", "private": False}

    def fetch_model_config(self, repo_id, resolved_revision):
        return self._config

    def fetch_file(self, repo_id, resolved_revision, path):
        if self._index_content is not None:
            return json.dumps(self._index_content).encode()
        return b"{}"

    def list_files(self, repo_id, resolved_revision):
        return self._files


CONFIG_7B = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 32768,
    "torch_dtype": "float16",
}


def _contents(files):
    contents = {}
    for f in files:
        data = b"content-" + f["path"].encode() if f["path"] != "model.safetensors" else b"weight"
        f["size_bytes"] = len(data)
        f["sha256"] = _sha(data)
        contents[f["path"]] = data
    return contents


def _single_files():
    return [
        {"path": "config.json", "size_bytes": 0, "sha256": "", "etag": None},
        {"path": "tokenizer.json", "size_bytes": 0, "sha256": "", "etag": None},
        {"path": "tokenizer_config.json", "size_bytes": 0, "sha256": "", "etag": None},
        {"path": "model.safetensors", "size_bytes": 0, "sha256": "", "etag": None},
    ]


def _host(**overrides):
    defaults = dict(
        gpu_indexes=[0],
        gpu_memory_total_bytes=24 * GB,
        gpu_memory_free_bytes=23 * GB,
        ram_total_bytes=128 * GB,
        ram_available_bytes=96 * GB,
        disk_total_bytes=512 * GB,
        disk_free_bytes=200 * GB,
    )
    defaults.update(overrides)
    return defaults


def _repo(tmp_path, text="model_id: org/model\n"):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "config.yaml").write_text(text, encoding="utf-8")
    return repo


class TestPreparationPipeline:
    def test_full_chain_offline(self, tmp_path):
        repo = _repo(tmp_path)
        files = _single_files()
        contents = _contents(files)
        download_log = []

        def fake_urlopen(req, timeout):
            path = req.full_url.rstrip("/").split("/")[-1]
            if path in contents:
                download_log.append(path)
                return FakeResponse(contents[path])
            return FakeResponse(b"", status=404)

        config = HarnessConfig(
            model_inference_enabled=True,
            model_cache_dir=str(tmp_path / "model_cache"),
        )
        cache = ModelCache(tmp_path / "model_cache")
        client = FakeSourceClient(CONFIG_7B, files)
        downloader = HuggingFaceDownloader(urlopen=fake_urlopen, token="")
        orchestrator = ModelPreparationOrchestrator()

        bundle = orchestrator.run(
            repo, tmp_path / "run", config, client, _host(), cache,
            downloader=downloader, execute=True,
        )
        assert bundle.status == "prepared"
        assert bundle.spec.status == "resolved"
        assert bundle.decision.status == "allowed"
        assert bundle.prepare_result["status"] == "complete"

        run_dir = tmp_path / "run"
        model_dir = run_dir / "reports" / "model"
        assert (model_dir / "resolved_model.json").exists()
        assert (model_dir / "model_file_plan.json").exists()
        assert (model_dir / "resource_decision.json").exists()
        assert (model_dir / "preparation_checkpoint.json").exists()
        assert (Path(bundle.cache_dir) / ".auto_harness_complete.json").exists()

    def test_multiple_candidates_needs_human_input(self, tmp_path):
        repo = _repo(tmp_path, "model_id: org/model-a\nmodel_name: org/model-b\n")
        config = HarnessConfig(model_inference_enabled=True, model_cache_dir=str(tmp_path / "cache"))
        cache = ModelCache(tmp_path / "cache")
        client = FakeSourceClient(CONFIG_7B, _single_files())
        orchestrator = ModelPreparationOrchestrator()
        bundle = orchestrator.run(repo, tmp_path / "run", config, client, _host(), cache)
        assert bundle.status == "needs_human_input"
        assert bundle.prepare_result == {}
        # no download happened
        assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").rglob("*.safetensors"))

    def test_resource_blocked_no_download(self, tmp_path):
        repo = _repo(tmp_path)
        download_log = []
        # 14B model: ~28GB weights exceed the 24GB GPU.
        big_config = dict(CONFIG_7B, hidden_size=5120, num_hidden_layers=40, num_attention_heads=40, num_key_value_heads=8)
        files = [
            {"path": "config.json", "size_bytes": 100, "sha256": _sha(b"c"), "etag": None},
            {"path": "tokenizer.json", "size_bytes": 100, "sha256": _sha(b"t"), "etag": None},
            {"path": "tokenizer_config.json", "size_bytes": 100, "sha256": _sha(b"tc"), "etag": None},
            {"path": "model.safetensors", "size_bytes": 28 * GB, "sha256": "e" * 64, "etag": None},
        ]

        def fake_urlopen(req, timeout):
            download_log.append(req.full_url)
            return FakeResponse(b"", status=200)

        config = HarnessConfig(model_inference_enabled=True, model_cache_dir=str(tmp_path / "cache"))
        cache = ModelCache(tmp_path / "cache")
        client = FakeSourceClient(big_config, files)
        downloader = HuggingFaceDownloader(urlopen=fake_urlopen, token="")
        orchestrator = ModelPreparationOrchestrator()
        bundle = orchestrator.run(repo, tmp_path / "run", config, client, _host(), cache, downloader=downloader, execute=True)
        assert bundle.decision.status == "insufficient_gpu_memory"
        assert bundle.prepare_result == {}
        assert download_log == []

    def test_checkpoint_records_hashes(self, tmp_path):
        repo = _repo(tmp_path)
        files = _single_files()
        contents = _contents(files)

        def fake_urlopen(req, timeout):
            path = req.full_url.rstrip("/").split("/")[-1]
            return FakeResponse(contents.get(path, b""), status=200 if path in contents else 404)

        config = HarnessConfig(model_inference_enabled=True, model_cache_dir=str(tmp_path / "cache"))
        cache = ModelCache(tmp_path / "cache")
        client = FakeSourceClient(CONFIG_7B, files)
        downloader = HuggingFaceDownloader(urlopen=fake_urlopen, token="")
        orchestrator = ModelPreparationOrchestrator()
        bundle = orchestrator.run(repo, tmp_path / "run", config, client, _host(), cache, downloader=downloader, execute=True)

        checkpoint = json.loads((Path(bundle.checkpoint_path)).read_text(encoding="utf-8"))
        assert checkpoint["model_identity"] == bundle.spec.model_identity
        assert checkpoint["resolved_revision"] == SHA
        assert checkpoint["file_plan_hash"] == bundle.plan.plan_hash
        assert checkpoint["resource_decision_hash"] == bundle.decision.decision_hash
        assert checkpoint["complete_marker_hash"]
        assert checkpoint["cache_identity"] == bundle.cache_identity
