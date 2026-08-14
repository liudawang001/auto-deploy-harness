"""Phase B3 tests: managed model-runtime Docker security profile.

Covers the exact read-only model mount, token-env filtering, GPU device
selection, loopback port binding, detached/no-``--rm`` lifecycle, complete
ownership labels, and hard rejection of host network / privileged / socket.
"""
from pathlib import Path

import pytest

from auto_harness.runtime.sandbox import (
    MODEL_RUNTIME_CONTAINER_MODEL_PATH,
    MODEL_RUNTIME_CONTAINER_PORT,
    DockerSandboxBackend,
)


def _labels():
    return {
        "auto-harness.task-id": "task-1",
        "auto-harness.operation-id": "op-1",
        "auto-harness.plan-hash": "sha256:plan",
        "auto-harness.model-hash": "sha256:model",
    }


def _wrap(tmp_path, **overrides):
    backend = DockerSandboxBackend.for_model_runtime(
        image="vllm/vllm-openai:v0.6.1@sha256:" + "d" * 64,
        gpu_index=0,
        **overrides,
    )
    return backend.wrap_model_runtime(
        model_host_dir=str(tmp_path / "model_cache" / "huggingface" / "key"),
        host_port=8000,
        command=["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/current"],
        container_name="auto-harness-abc-vllm",
        labels=_labels(),
    )


def test_exact_read_only_model_mount(tmp_path):
    cmd = _wrap(tmp_path).effective_cmd
    host = str((tmp_path / "model_cache" / "huggingface" / "key").resolve())
    mount = "%s:%s:ro" % (host, MODEL_RUNTIME_CONTAINER_MODEL_PATH)
    idx = cmd.index("-v")
    assert cmd[idx + 1] == mount
    # No full model_cache mount and no home/SSH mount.
    joined = " ".join(cmd)
    assert "/workspace/model_cache" not in joined
    assert ".ssh" not in joined
    assert "/home" not in joined


def test_token_env_filtered(tmp_path):
    joined = " ".join(_wrap(tmp_path).effective_cmd)
    for secret in ("HF_TOKEN", "MODELSCOPE_API_TOKEN", "XUNFEI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        assert secret not in joined


def test_gpu_device_index(tmp_path):
    cmd = _wrap(tmp_path).effective_cmd
    idx = cmd.index("--gpus")
    assert cmd[idx + 1] == "device=0"


def test_loopback_port(tmp_path):
    cmd = _wrap(tmp_path, ).effective_cmd
    idx = cmd.index("-p")
    assert cmd[idx + 1] == "127.0.0.1:8000:%d" % MODEL_RUNTIME_CONTAINER_PORT


def test_detached_no_rm(tmp_path):
    cmd = _wrap(tmp_path).effective_cmd
    assert "-d" in cmd
    assert "--rm" not in cmd


def test_labels_complete(tmp_path):
    cmd = _wrap(tmp_path).effective_cmd
    for key, value in _labels().items():
        assert "--label" in cmd
        assert "%s=%s" % (key, value) in cmd


def test_host_network_rejected(tmp_path):
    with pytest.raises(ValueError):
        _wrap(tmp_path, network="host")


def test_no_privileged_or_socket(tmp_path):
    joined = " ".join(_wrap(tmp_path).effective_cmd)
    assert "--privileged" not in joined
    assert "docker.sock" not in joined
    assert "/var/run/docker.sock" not in joined


def test_security_profile_in_options(tmp_path):
    sandbox = _wrap(tmp_path)
    assert sandbox.security_options["security_profile"] == "model_runtime_v1"
    assert sandbox.security_options["model_mount"]["mount_mode"] == "ro"
    assert sandbox.security_options["model_mount"]["container_path"] == "/models/current"


def test_invalid_gpu_index(tmp_path):
    with pytest.raises(ValueError):
        DockerSandboxBackend.for_model_runtime(
            image="vllm/vllm-openai@sha256:" + "d" * 64, gpu_index=-1,
        )
