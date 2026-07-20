from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SandboxCommand:
    backend: str
    image: str
    original_cmd: List[str]
    effective_cmd: List[str]
    workdir: str
    ports: List[int]
    network: str
    gpus: str = "none"
    model_cache_mount: Dict = None
    container_name: str = ""
    log_command: List[str] = None
    cleanup_command: List[str] = None

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "image": self.image,
            "original_cmd": self.original_cmd,
            "effective_cmd": self.effective_cmd,
            "workdir": self.workdir,
            "ports": self.ports,
            "network": self.network,
            "gpus": self.gpus,
            "model_cache_mount": self.model_cache_mount or {},
            "container_name": self.container_name,
            "log_command": self.log_command or [],
            "cleanup_command": self.cleanup_command or [],
        }


class DockerSandboxBackend:
    """Builds Docker command wrappers for isolated dependency install and service startup."""

    def __init__(
        self,
        image: str = "python:3.10-slim",
        network: str = "bridge",
        gpus: str = "none",
        model_cache_dir: Optional[Path] = None,
    ) -> None:
        self.image = image
        self.network = network
        self.gpus = gpus or "none"
        self.model_cache_dir = Path(model_cache_dir).resolve() if model_cache_dir else None

    def wrap(self, repo_dir: Path, cmd: List[str], ports: Optional[List[int]] = None, container_name: str = "", auto_remove: bool = True, detached: bool = False, labels: Optional[Dict[str, str]] = None) -> SandboxCommand:
        ports = [int(port) for port in (ports or []) if int(port) > 0]
        repo_dir = Path(repo_dir).resolve()
        effective = [
            "docker",
            "run",
        ]
        if auto_remove:
            effective.append("--rm")
        if detached:
            effective.append("-d")
        for key, value in sorted((labels or {}).items()):
            effective.extend(["--label", "%s=%s" % (key, value)])
        effective.extend([
            "-v",
            "%s:/workspace/repo" % repo_dir,
            "-w",
            "/workspace/repo",
        ])
        if container_name:
            effective.extend(["--name", container_name])
        if self.network:
            effective.extend(["--network", self.network])
        if self.gpus and self.gpus != "none":
            effective.extend(["--gpus", self.gpus])
        model_cache_mount = {}
        if self.model_cache_dir:
            effective.extend(["-v", "%s:/workspace/model_cache" % self.model_cache_dir])
            effective.extend(["-e", "AUTO_HARNESS_MODEL_CACHE=/workspace/model_cache"])
            model_cache_mount = {
                "host_path": str(self.model_cache_dir),
                "container_path": "/workspace/model_cache",
                "env": "AUTO_HARNESS_MODEL_CACHE",
            }
        for port in ports:
            effective.extend(["-p", "127.0.0.1:%s:%s" % (port, port)])
        effective.append(self.image)
        effective.extend(cmd)
        log_command = ["docker", "logs", container_name] if container_name else []
        cleanup_command = ["docker", "rm", "-f", container_name] if container_name else []
        return SandboxCommand(
            backend="docker",
            image=self.image,
            original_cmd=list(cmd),
            effective_cmd=effective,
            workdir="/workspace/repo",
            ports=ports,
            network=self.network,
            gpus=self.gpus,
            model_cache_mount=model_cache_mount,
            container_name=container_name,
            log_command=log_command,
            cleanup_command=cleanup_command,
        )
