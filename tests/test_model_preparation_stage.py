"""Mainline integration tests: model_prepare stage -> managed vLLM chain."""
import json
import subprocess
from pathlib import Path

from auto_harness.assets.cache import ModelCache
from auto_harness.assets.huggingface import HuggingFaceDownloader
from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.controller import ModelRuntimeController
from auto_harness.model_runtime.mainline import ModelPreparationStageRunner

SHA = "a" * 40
GB = 1024 ** 3

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


def _sha(data: bytes) -> str:
    import hashlib
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
    def __init__(self, config, files, revision=SHA):
        self._config = config
        self._files = files
        self._revision = revision

    def resolve_revision(self, repo_id, requested_revision):
        return {"resolved_revision": self._revision, "gated": False, "license": "apache-2.0", "private": False}

    def fetch_model_config(self, repo_id, resolved_revision):
        return self._config

    def fetch_file(self, repo_id, resolved_revision, path):
        return b"{}"

    def list_files(self, repo_id, resolved_revision):
        return self._files


def _files():
    files = [
        {"path": "config.json", "size_bytes": 0, "sha256": "", "etag": None},
        {"path": "tokenizer.json", "size_bytes": 0, "sha256": "", "etag": None},
        {"path": "tokenizer_config.json", "size_bytes": 0, "sha256": "", "etag": None},
        {"path": "model.safetensors", "size_bytes": 0, "sha256": "", "etag": None},
    ]
    for item in files:
        data = b"weight" if item["path"] == "model.safetensors" else b"content-" + item["path"].encode()
        item["size_bytes"] = len(data)
        item["sha256"] = _sha(data)
    return files


def _download_urlopen(contents):
    def fake_urlopen(req, timeout):
        path = req.full_url.rstrip("/").split("/")[-1]
        return FakeResponse(contents.get(path, b""), status=200 if path in contents else 404)
    return fake_urlopen


def _gpu_runner(cmd, text=True, capture_output=True, timeout=None):
    if len(cmd) > 2:
        stdout = "0, GPU-x, NVIDIA GeForce RTX 4090D, 550.54, 24564, 23800"
    else:
        stdout = "CUDA Version: 12.4"
    return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")


def _no_gpu_runner(cmd, text=True, capture_output=True, timeout=None):
    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="nvidia-smi: not found")


def _stage_config(tmp_path, **overrides):
    defaults = dict(
        model_inference_enabled=True,
        model_id_override="huggingface:org/model",
        model_cache_dir=str(tmp_path / "model_cache"),
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


def _run_stage(tmp_path, config, *, execute=False, gpu=True, contents=None):
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    contents = contents if contents is not None else {
        item["path"]: (b"weight" if item["path"] == "model.safetensors" else b"content-" + item["path"].encode())
        for item in _files()
    }
    runner = ModelPreparationStageRunner(
        probe_command_runner=_gpu_runner if gpu else _no_gpu_runner,
        source_client_factory=lambda source, transport=None, token=None, api_base=None: FakeSourceClient(CONFIG_7B, _files()),
        downloader_factory=lambda source: HuggingFaceDownloader(
            urlopen=_download_urlopen(contents), token="",
        ),
    )
    return runner.run(
        run_dir=tmp_path / "run",
        task_id="stage-task",
        repo_dir=tmp_path / "repo",
        config=config,
        execute=execute,
    )


def test_dry_run_writes_artifacts_without_download(tmp_path):
    result = _run_stage(tmp_path, _stage_config(tmp_path), execute=False)

    assert result.status == "passed"
    assert "resource decision allowed" in result.summary
    model_dir = tmp_path / "run" / "reports" / "model"
    assert (model_dir / "resolved_model.json").exists()
    assert (model_dir / "model_file_plan.json").exists()
    assert (model_dir / "resource_decision.json").exists()
    assert (model_dir / "preparation_checkpoint.json").exists()
    assert result.data["decision_status"] == "allowed"
    assert result.data["host_facts"]["gpu_memory_free_bytes"] == 23800 * 1024 * 1024
    # dry-run never downloads: no complete marker anywhere in the cache
    cache_root = tmp_path / "model_cache"
    assert not cache_root.exists() or not any(cache_root.rglob(".auto_harness_complete.json"))


def test_execute_downloads_weights_and_prepares(tmp_path):
    result = _run_stage(tmp_path, _stage_config(tmp_path), execute=True)

    assert result.status == "passed"
    assert result.data["bundle_status"] == "prepared"
    assert result.data["prepare_result_status"] == "complete"
    marker = list((tmp_path / "model_cache").rglob(".auto_harness_complete.json"))
    assert len(marker) == 1


def test_no_gpu_blocks_fail_closed_without_download(tmp_path):
    downloads = []

    def recording_downloader(source):
        class _Recorder(HuggingFaceDownloader):
            def download_plan(self, *args, **kwargs):
                downloads.append(args)
                return super().download_plan(*args, **kwargs)
        return _Recorder(urlopen=_download_urlopen({}), token="")

    result = _run_stage(
        tmp_path, _stage_config(tmp_path), execute=True, gpu=False,
    )

    assert result.status == "uncertain"
    assert result.error == "resource_decision_gpu_busy"
    assert result.data["decision_status"] == "gpu_busy"
    assert result.data["decision_reasons"]
    assert result.data["host_facts"]["gpu_memory_free_bytes"] == 0
    assert downloads == []


def test_missing_model_reference_fails_closed(tmp_path):
    (tmp_path / "repo").mkdir(parents=True)
    config = HarnessConfig(
        model_inference_enabled=True,
        model_cache_dir=str(tmp_path / "model_cache"),
    )

    runner = ModelPreparationStageRunner(probe_command_runner=_gpu_runner)
    result = runner.run(
        run_dir=tmp_path / "run",
        task_id="stage-task",
        repo_dir=tmp_path / "repo",
        config=config,
        execute=False,
    )

    assert result.status == "failed"
    assert result.error == "model_reference_unresolved"
    assert result.data["errors"]


def test_invalid_override_fails_closed(tmp_path):
    (tmp_path / "repo").mkdir(parents=True)
    config = _stage_config(tmp_path, model_id_override="not a valid repo id")

    result = ModelPreparationStageRunner(probe_command_runner=_gpu_runner).run(
        run_dir=tmp_path / "run",
        task_id="stage-task",
        repo_dir=tmp_path / "repo",
        config=config,
        execute=False,
    )

    assert result.status == "failed"
    assert result.error == "model_reference_unresolved"


def test_prepared_stage_artifacts_feed_the_managed_runtime_chain(tmp_path):
    result = _run_stage(tmp_path, _stage_config(tmp_path, model_cache_dir=str(tmp_path / "model_cache")), execute=True)
    assert result.status == "passed"

    # Resume verification from the on-disk artifacts: the preparation gate
    # must accept what the mainline stage wrote, without any test-side
    # synthesis of bundles.
    config = _stage_config(
        tmp_path,
        model_cache_dir=str(tmp_path / "model_cache"),
        model_runtime_image="vllm/vllm-openai:v0.6.1@" + "sha256:" + "d" * 64,
    )

    class FakeHTTPResponse:
        def __init__(self, body: bytes, status: int = 200):
            self._body = body
            self.status = status
            self.code = status

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size=-1):
            return self._body

    class FakeStreamResponse:
        def __init__(self, status, lines):
            self.status = status
            self.code = status
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self._lines)

    served_name = "org/model"

    def fake_urlopen(req, timeout=5):
        url = req.full_url
        if url.endswith("/v1/models"):
            body = json.dumps({"data": [{"id": served_name}]}).encode()
            return FakeHTTPResponse(body)
        if url.endswith("/v1/chat/completions"):
            request_body = json.loads(req.data.decode("utf-8")) if req.data else {}
            trace = request_body["messages"][1]["content"].rsplit(" ", 1)[-1]
            usage = {"prompt_tokens": 18, "completion_tokens": 6}
            if request_body.get("stream"):
                lines = [
                    "data: " + json.dumps({"choices": [{"delta": {"content": "token "}}]}),
                    "",
                    "data: " + json.dumps({"choices": [{"delta": {"content": trace}}]}),
                    "",
                    "data: " + json.dumps({"choices": [], "usage": usage}),
                    "",
                    "data: [DONE]",
                    "",
                ]
                return FakeStreamResponse(200, lines)
            body = json.dumps({
                "model": served_name,
                "choices": [{"message": {"content": "token %s" % trace}}],
                "usage": usage,
            }).encode()
            return FakeHTTPResponse(body)
        return FakeHTTPResponse(b"", status=404)

    class FakeDocker:
        def __init__(self):
            self.labels = {}
            self.commands = []

        def __call__(self, cmd):
            self.commands.append(cmd)
            if cmd[:2] == ["docker", "run"]:
                for flag, value in zip(cmd, cmd[1:]):
                    if flag == "--label":
                        key, _, val = value.partition("=")
                        self.labels[key] = val
                return {"exit_code": 0, "stdout": "cid123\n", "stderr": ""}
            if cmd[:2] == ["docker", "inspect"]:
                return {
                    "exit_code": 0,
                    "stdout": json.dumps([{"State": {"Running": True}, "Config": {"Labels": self.labels}}]),
                    "stderr": "",
                }
            if cmd[:2] == ["docker", "logs"]:
                return {"exit_code": 0, "stdout": "", "stderr": ""}
            return {"exit_code": 1, "stdout": "", "stderr": "unexpected command"}

    docker = FakeDocker()
    phase = ModelRuntimeController().run_runtime_phase(
        run_dir=tmp_path / "run",
        task_id="stage-task",
        config=config,
        cache_root=tmp_path / "model_cache",
        execute=True,
        allow_start=True,
        command_runner=docker,
        urlopen=fake_urlopen,
    )
    assert phase.status == "passed", phase.errors
    assert phase.container_id == "cid123"

    verify = ModelRuntimeController().verify_phase(
        run_dir=tmp_path / "run",
        task_id="stage-task",
        runtime_plan=phase.plan,
        startup_evidence=phase.startup_evidence,
        urlopen=fake_urlopen,
    )
    assert verify.status == "passed"
    model_dir = tmp_path / "run" / "reports" / "model"
    assert (model_dir / "runtime_plan.json").exists()
    assert (model_dir / "startup_evidence.json").exists()
