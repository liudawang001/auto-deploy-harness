import shutil
import subprocess
from typing import Callable, Dict, List, Optional


class DockerSmokeChecker:
    """Plans or optionally probes Docker/GPU runtime readiness."""

    def __init__(self, command_runner: Optional[Callable[[List[str], int], Dict]] = None) -> None:
        self.command_runner = command_runner or self._run_command

    def check(self, probe: bool = False, image: str = "python:3.10-slim", require_gpu: bool = False) -> Dict:
        checks = self._planned_checks(image=image, require_gpu=require_gpu)
        if probe:
            for item in checks:
                if item["optional"] and not require_gpu:
                    item["status"] = "skipped"
                    item["reason"] = "optional check skipped because require_gpu=false"
                    continue
                result = self.command_runner(item["command"], item["timeout_seconds"])
                item["status"] = "passed" if result.get("exit_code") == 0 else "failed"
                item["exit_code"] = result.get("exit_code")
                item["stdout_tail"] = str(result.get("stdout", ""))[-2000:]
                item["stderr_tail"] = str(result.get("stderr", ""))[-2000:]
                item["reason"] = "probe command completed" if item["status"] == "passed" else "probe command failed"
        else:
            for item in checks:
                item["status"] = "planned"
                item["reason"] = "probe=false; command was not executed"

        if not probe:
            status = "planned"
        elif all(item["status"] in ("passed", "skipped") for item in checks):
            status = "passed"
        else:
            status = "failed"
        return {
            "status": status,
            "probe": probe,
            "image": image,
            "require_gpu": require_gpu,
            "docker_cli": shutil.which("docker") or "",
            "checks": checks,
            "notes": [
                "Use --probe only on machines where Docker commands are allowed.",
                "GPU checks require NVIDIA container runtime and a visible GPU.",
                "This command never records secrets.",
            ],
        }

    def _planned_checks(self, image: str, require_gpu: bool) -> List[Dict]:
        return [
            {
                "id": "docker_version",
                "purpose": "Verify Docker CLI can reach the Docker daemon.",
                "command": ["docker", "version", "--format", "{{json .}}"],
                "timeout_seconds": 10,
                "optional": False,
            },
            {
                "id": "docker_info",
                "purpose": "Verify Docker runtime metadata is readable for audit.",
                "command": ["docker", "info", "--format", "{{json .}}"],
                "timeout_seconds": 10,
                "optional": False,
            },
            {
                "id": "docker_image_python",
                "purpose": "Verify the configured Python image can run a trivial command.",
                "command": ["docker", "run", "--rm", image, "python", "-c", "print('auto-harness-docker-ok')"],
                "timeout_seconds": 60,
                "optional": False,
            },
            {
                "id": "docker_gpu_runtime",
                "purpose": "Verify --gpus all is accepted and GPU is visible in the container.",
                "command": ["docker", "run", "--rm", "--gpus", "all", image, "python", "-c", "print('auto-harness-gpu-probe')"],
                "timeout_seconds": 60,
                "optional": not require_gpu,
            },
        ]

    def _run_command(self, cmd: List[str], timeout_seconds: int) -> Dict:
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
            return {
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "exit_code": 127,
                "stdout": "",
                "stderr": str(exc),
            }
