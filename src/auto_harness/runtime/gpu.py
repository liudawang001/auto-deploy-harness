import os
import subprocess
from typing import Dict, List, Optional


class GpuResourceProbe:
    """Detects available GPU slots without making GPU a hard dependency."""

    def __init__(
        self,
        command_runner=None,
        environ: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 5,
        allow_slot_override: bool = True,
    ) -> None:
        self.command_runner = command_runner or subprocess.run
        self.environ = environ if environ is not None else os.environ
        self.timeout_seconds = timeout_seconds
        self.allow_slot_override = allow_slot_override

    def probe(self) -> Dict:
        override = self.environ.get("AUTO_HARNESS_GPU_SLOTS") if self.allow_slot_override else None
        if override not in (None, ""):
            try:
                slots = max(0, int(override))
            except ValueError:
                return {
                    "status": "invalid_override",
                    "source": "env",
                    "available_slots": 0,
                    "gpus": [],
                    "error": "AUTO_HARNESS_GPU_SLOTS must be an integer",
                }
            return {
                "status": "detected",
                "source": "env",
                "available_slots": slots,
                "gpus": [{"index": index, "source": "env"} for index in range(slots)],
            }
        try:
            result = self.command_runner(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "source": "nvidia-smi",
                "available_slots": 0,
                "gpus": [],
                "error": "nvidia-smi timed out",
            }
        except FileNotFoundError:
            return {
                "status": "not_found",
                "source": "nvidia-smi",
                "available_slots": 0,
                "gpus": [],
                "error": "nvidia-smi not found",
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "status": "permission_denied" if isinstance(exc, PermissionError) else "probe_error",
                "source": "nvidia-smi",
                "available_slots": 0,
                "gpus": [],
                "error": str(exc),
            }
        if result.returncode != 0:
            error = (result.stderr or "")[-1000:]
            lowered = error.lower()
            return {
                "status": "permission_denied" if "permission" in lowered else "probe_error",
                "source": "nvidia-smi",
                "available_slots": 0,
                "gpus": [],
                "error": error,
            }
        gpus = self._parse_nvidia_smi(result.stdout or "")
        driver_cuda_version = self._probe_driver_cuda_version() if gpus else ""
        return {
            "status": "detected" if gpus else "parse_error",
            "source": "nvidia-smi",
            "available_slots": len(gpus),
            "gpus": gpus,
            "driver_version": gpus[0].get("driver_version", "") if gpus else "",
            "driver_cuda_version": driver_cuda_version,
        }

    def _probe_driver_cuda_version(self) -> str:
        try:
            result = self.command_runner(
                ["nvidia-smi"],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if getattr(result, "returncode", 1) != 0:
            return ""
        import re
        match = re.search(
            r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)",
            getattr(result, "stdout", "") or "",
        )
        return match.group(1) if match else ""

    def _parse_nvidia_smi(self, output: str) -> List[Dict]:
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
