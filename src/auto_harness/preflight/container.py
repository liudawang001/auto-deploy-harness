"""Docker GPU container preflight probe.

Host ``nvidia-smi`` success is NOT proof that Docker can expose the GPU. This
probe runs a controlled ``docker run --rm --gpus device=<index>`` container
with a fixed image and parses the container-visible GPU list.

All commands are fixed ``list[str]``; the command runner is injectable; every
probe has a timeout and bounded output; nothing is run as privileged, on the
host network, or with a Docker socket mount.
"""
import os
import subprocess
from typing import Dict, List, Optional

DEFAULT_PROBE_IMAGE = "nvidia/cuda:12.1.0-base-ubuntu22.04"
MAX_OUTPUT_CHARS = 8000


def _parse_nvidia_smi_csv(output: str) -> List[Dict]:
    gpus: List[Dict] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in (4, 6):
            continue
        try:
            index = int(parts[0])
            if len(parts) == 6:
                uuid, name, driver = parts[1], parts[2], parts[3]
                memory_total_mb = int(float(parts[4]))
                memory_free_mb = int(float(parts[5]))
            else:
                uuid, name, driver = "", parts[1], ""
                memory_total_mb = int(float(parts[2]))
                memory_free_mb = int(float(parts[3]))
        except ValueError:
            continue
        gpus.append({
            "index": index,
            "uuid": uuid,
            "name": name,
            "driver_version": driver,
            "memory_total_mb": memory_total_mb,
            "memory_free_mb": memory_free_mb,
        })
    return gpus


class DockerGpuProbe:
    """Probe the Docker daemon and a container-visible NVIDIA GPU."""

    def __init__(
        self,
        command_runner=None,
        probe_image: Optional[str] = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.command_runner = command_runner or subprocess.run
        self.probe_image = (
            probe_image
            or os.environ.get("AUTO_HARNESS_DOCKER_GPU_PROBE_IMAGE")
            or DEFAULT_PROBE_IMAGE
        )
        self.timeout_seconds = timeout_seconds

    def probe(self, gpu_index: int = 0) -> Dict:
        errors: List[str] = []
        client_version, client_error = self._client_version()
        if client_error:
            return {
                "schema_version": 1,
                "status": "no_docker_client",
                "client_version": "",
                "server_version": "",
                "daemon_accessible": False,
                "nvidia_container_toolkit": False,
                "container_gpus": [],
                "probe_image": self.probe_image,
                "errors": [client_error],
            }

        info = self._docker_info()
        if info["status"] != "ok":
            return {
                "schema_version": 1,
                "status": info["status"],
                "client_version": client_version,
                "server_version": "",
                "daemon_accessible": False,
                "nvidia_container_toolkit": False,
                "container_gpus": [],
                "probe_image": self.probe_image,
                "errors": info["errors"],
            }

        container_result = self._container_probe(gpu_index)
        return {
            "schema_version": 1,
            "status": container_result["status"],
            "client_version": client_version,
            "server_version": info["server_version"],
            "daemon_accessible": True,
            "nvidia_container_toolkit": info["nvidia_runtime"],
            "container_gpus": container_result["gpus"],
            "probe_image": self.probe_image,
            "errors": container_result["errors"],
        }

    def _client_version(self):
        try:
            result = self.command_runner(
                ["docker", "--version"],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError:
            return "", "docker client not found"
        except subprocess.TimeoutExpired:
            return "", "docker --version timed out"
        except (OSError, subprocess.SubprocessError) as exc:
            return "", str(exc)
        if getattr(result, "returncode", 1) != 0:
            return "", "docker --version failed"
        return (result.stdout or "").strip()[:200], None

    def _docker_info(self) -> Dict:
        try:
            result = self.command_runner(
                ["docker", "info"],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"status": "daemon_unavailable", "errors": ["docker info timed out"]}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "daemon_unavailable", "errors": [str(exc)]}
        if getattr(result, "returncode", 1) != 0:
            stderr = (result.stderr or "")[:MAX_OUTPUT_CHARS]
            lowered = stderr.lower()
            if "permission" in lowered or "denied" in lowered:
                return {"status": "daemon_denied", "errors": [stderr]}
            return {"status": "daemon_unavailable", "errors": [stderr]}
        stdout = (result.stdout or "")[:MAX_OUTPUT_CHARS]
        lowered = stdout.lower()
        return {
            "status": "ok",
            "server_version": self._parse_server_version(stdout),
            "nvidia_runtime": "nvidia" in lowered and ("runtime" in lowered or "runtimes" in lowered),
            "errors": [],
        }

    def _container_probe(self, gpu_index: int) -> Dict:
        cmd = [
            "docker", "run", "--rm", "--gpus", "device=%d" % gpu_index,
            self.probe_image,
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = self.command_runner(
                cmd,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "gpus": [], "errors": ["container GPU probe timed out"]}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "container_probe_failed", "gpus": [], "errors": [str(exc)]}
        if getattr(result, "returncode", 1) != 0:
            stderr = (result.stderr or "")[:MAX_OUTPUT_CHARS]
            lowered = stderr.lower()
            if "could not select device driver" in lowered or "nvidia" in lowered and "not" in lowered:
                return {"status": "container_toolkit_missing", "gpus": [], "errors": [stderr]}
            return {"status": "container_probe_failed", "gpus": [], "errors": [stderr]}
        gpus = _parse_nvidia_smi_csv(result.stdout or "")
        if not gpus:
            return {"status": "parse_error", "gpus": [], "errors": ["container nvidia-smi output unparseable"]}
        return {"status": "detected", "gpus": gpus, "errors": []}

    @staticmethod
    def _parse_server_version(stdout: str) -> str:
        for line in stdout.splitlines():
            if "Server Version" in line and ":" in line:
                return line.split(":", 1)[1].strip()
        return ""
