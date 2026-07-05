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

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "image": self.image,
            "original_cmd": self.original_cmd,
            "effective_cmd": self.effective_cmd,
            "workdir": self.workdir,
            "ports": self.ports,
            "network": self.network,
        }


class DockerSandboxBackend:
    """Builds Docker command wrappers for isolated dependency install and service startup."""

    def __init__(self, image: str = "python:3.10-slim", network: str = "bridge") -> None:
        self.image = image
        self.network = network

    def wrap(self, repo_dir: Path, cmd: List[str], ports: Optional[List[int]] = None) -> SandboxCommand:
        ports = [int(port) for port in (ports or []) if int(port) > 0]
        repo_dir = Path(repo_dir).resolve()
        effective = [
            "docker",
            "run",
            "--rm",
            "-v",
            "%s:/workspace/repo" % repo_dir,
            "-w",
            "/workspace/repo",
        ]
        if self.network:
            effective.extend(["--network", self.network])
        for port in ports:
            effective.extend(["-p", "127.0.0.1:%s:%s" % (port, port)])
        effective.append(self.image)
        effective.extend(cmd)
        return SandboxCommand(
            backend="docker",
            image=self.image,
            original_cmd=list(cmd),
            effective_cmd=effective,
            workdir="/workspace/repo",
            ports=ports,
            network=self.network,
        )
