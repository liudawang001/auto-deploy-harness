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
    security_options: Dict = None

    def to_dict(self) -> Dict:
        result = {
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
        if self.security_options:
            result["security_options"] = self.security_options
        return result


# Secret environment variables that must NEVER be auto-inherited by Docker
DockerSandboxBackend_FORBIDDEN_SECRETS = frozenset({
    "XUNFEI_API_KEY",
    "XUNFEI_API_SECRET",
    "XUNFEI_APP_ID",
    "HF_TOKEN",
    "MODELSCOPE_API_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
})


class DockerSandboxBackend:
    """Builds Docker command wrappers for isolated dependency install and service startup.

    Security features:
    - Capabilities dropped (--cap-drop ALL)
    - No new privileges (--security-opt no-new-privileges)
    - Memory, CPU, and PID limits
    - /tmp as tmpfs (noexec, nosuid)
    - Host network is rejected
    - Repo mount mode configurable (rw by default, risk recorded)
    - Forbidden secrets never auto-inherited
    """

    def __init__(
        self,
        image: str = "python:3.10-slim",
        network: str = "bridge",
        gpus: str = "none",
        model_cache_dir: Optional[Path] = None,
        cap_drop_all: bool = True,
        no_new_privileges: bool = True,
        memory: str = "8g",
        cpus: float = 4.0,
        pids_limit: int = 512,
        tmpfs_size: str = "1g",
        repo_mount_mode: str = "rw",
        read_only_rootfs: bool = False,
        user: str = "",
    ) -> None:
        # Validate
        if network == "host":
            raise ValueError("host network is not allowed")
        if not memory:
            raise ValueError("memory must be non-empty")
        if cpus <= 0:
            raise ValueError("cpus must be positive, got: %s" % cpus)
        if pids_limit <= 0:
            raise ValueError("pids_limit must be positive, got: %s" % pids_limit)
        if repo_mount_mode not in ("ro", "rw"):
            raise ValueError("repo_mount_mode must be 'ro' or 'rw', got: %s" % repo_mount_mode)

        self.image = image
        self.network = network
        self.gpus = gpus or "none"
        self.model_cache_dir = Path(model_cache_dir).resolve() if model_cache_dir else None
        self.cap_drop_all = cap_drop_all
        self.no_new_privileges = no_new_privileges
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.tmpfs_size = tmpfs_size
        self.repo_mount_mode = repo_mount_mode
        self.read_only_rootfs = read_only_rootfs
        self.user = user

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

        # Security: drop all capabilities
        if self.cap_drop_all:
            effective.extend(["--cap-drop", "ALL"])

        # Security: no new privileges
        if self.no_new_privileges:
            effective.extend([
                "--security-opt",
                "no-new-privileges",
            ])

        # Resource limits
        effective.extend(["--memory", self.memory])
        effective.extend(["--cpus", str(self.cpus)])
        effective.extend(["--pids-limit", str(self.pids_limit)])

        # /tmp as tmpfs
        effective.extend([
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=%s" % self.tmpfs_size,
        ])

        # User
        if self.user:
            effective.extend(["--user", self.user])

        # Read-only root filesystem
        if self.read_only_rootfs:
            effective.append("--read-only")

        effective.extend([
            "-v",
            "%s:/workspace/repo:%s" % (repo_dir, self.repo_mount_mode),
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

        security_options = {
            "cap_drop_all": self.cap_drop_all,
            "no_new_privileges": self.no_new_privileges,
            "memory": self.memory,
            "cpus": self.cpus,
            "pids_limit": self.pids_limit,
            "tmpfs_size": self.tmpfs_size,
            "repo_mount_mode": self.repo_mount_mode,
            "read_only_rootfs": self.read_only_rootfs,
        }

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
            security_options=security_options,
        )
