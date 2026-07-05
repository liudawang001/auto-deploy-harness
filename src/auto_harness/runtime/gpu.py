import os
import subprocess
from typing import Dict, List, Optional


class GpuResourceProbe:
    """Detects available GPU slots without making GPU a hard dependency."""

    def __init__(self, command_runner=None, environ: Optional[Dict[str, str]] = None) -> None:
        self.command_runner = command_runner or subprocess.run
        self.environ = environ if environ is not None else os.environ

    def probe(self) -> Dict:
        override = self.environ.get("AUTO_HARNESS_GPU_SLOTS")
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
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "status": "unavailable",
                "source": "nvidia-smi",
                "available_slots": 0,
                "gpus": [],
                "error": str(exc),
            }
        if result.returncode != 0:
            return {
                "status": "unavailable",
                "source": "nvidia-smi",
                "available_slots": 0,
                "gpus": [],
                "error": (result.stderr or "")[-1000:],
            }
        gpus = self._parse_nvidia_smi(result.stdout or "")
        return {
            "status": "detected" if gpus else "unavailable",
            "source": "nvidia-smi",
            "available_slots": len(gpus),
            "gpus": gpus,
        }

    def _parse_nvidia_smi(self, output: str) -> List[Dict]:
        gpus: List[Dict] = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                index = int(parts[0])
                memory_total_mb = int(float(parts[2]))
                memory_free_mb = int(float(parts[3]))
            except ValueError:
                continue
            gpus.append({
                "index": index,
                "name": parts[1],
                "memory_total_mb": memory_total_mb,
                "memory_free_mb": memory_free_mb,
            })
        return gpus
