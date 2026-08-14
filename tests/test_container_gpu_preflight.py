"""Phase A7 tests: Docker GPU container probe and host storage probe."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

from auto_harness.preflight.container import DockerGpuProbe
from auto_harness.preflight.storage import StorageProbe


def completed(code=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


class TestDockerGpuProbe:
    def test_no_docker_client(self):
        def runner(cmd, **kwargs):
            raise FileNotFoundError("docker")
        result = DockerGpuProbe(command_runner=runner).probe(0)
        assert result["status"] == "no_docker_client"

    def test_daemon_permission_denied(self):
        def runner(cmd, **kwargs):
            if cmd[1] == "--version":
                return completed(0, "Docker version 24.0.0")
            return completed(1, "", "permission denied while trying to connect to the Docker daemon socket")
        result = DockerGpuProbe(command_runner=runner).probe(0)
        assert result["status"] == "daemon_denied"

    def test_container_toolkit_missing(self):
        def runner(cmd, **kwargs):
            if cmd[1] == "--version":
                return completed(0, "Docker version 24.0.0")
            if cmd[1] == "info":
                return completed(0, "Server Version: 24.0.0\nRuntimes: nvidia runc")
            return completed(1, "", "could not select device driver")
        result = DockerGpuProbe(command_runner=runner).probe(0)
        assert result["status"] == "container_toolkit_missing"

    def test_container_probe_detected(self):
        def runner(cmd, **kwargs):
            if cmd[1] == "--version":
                return completed(0, "Docker version 24.0.0")
            if cmd[1] == "info":
                return completed(0, "Server Version: 24.0.0\nRuntimes: nvidia runc")
            return completed(0, "0, GPU-uuid-1, NVIDIA A100, 535.54, 81920, 70000\n")
        result = DockerGpuProbe(command_runner=runner).probe(0)
        assert result["status"] == "detected"
        assert result["daemon_accessible"] is True
        assert result["nvidia_container_toolkit"] is True
        assert result["container_gpus"][0]["uuid"] == "GPU-uuid-1"
        assert result["container_gpus"][0]["memory_free_mb"] == 70000

    def test_probe_timeout_not_no_gpu(self):
        def runner(cmd, **kwargs):
            if cmd[1] == "--version":
                return completed(0, "Docker version 24.0.0")
            if cmd[1] == "info":
                return completed(0, "Server Version: 24.0.0\nRuntimes: nvidia runc")
            raise subprocess.TimeoutExpired(cmd, 30)
        result = DockerGpuProbe(command_runner=runner).probe(0)
        assert result["status"] == "timeout"

    def test_parse_error_not_no_gpu(self):
        def runner(cmd, **kwargs):
            if cmd[1] == "--version":
                return completed(0, "Docker version 24.0.0")
            if cmd[1] == "info":
                return completed(0, "Server Version: 24.0.0\nRuntimes: nvidia runc")
            return completed(0, "garbage output\n")
        result = DockerGpuProbe(command_runner=runner).probe(0)
        assert result["status"] == "parse_error"


class TestStorageProbe:
    def test_probe_disk(self, tmp_path):
        result = StorageProbe().probe(tmp_path)
        assert result["disk_total_bytes"] > 0
        assert result["disk_free_bytes"] >= 0
        assert result["ram_total_bytes"] >= 0
        assert result["cache_root"] == str(tmp_path)

    def test_probe_creates_missing_cache_root(self, tmp_path):
        missing = tmp_path / "nested" / "cache"
        result = StorageProbe().probe(missing)
        assert missing.exists()
        assert result["disk_free_bytes"] >= 0
