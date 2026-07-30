"""NVIDIA GPU preflight probe with explicit failure semantics."""
import subprocess
from typing import Dict

from auto_harness.runtime.gpu import GpuResourceProbe


class NvidiaGpuProbe:
    def __init__(self, command_runner=None, timeout_seconds: int = 5) -> None:
        self.command_runner = command_runner
        self.timeout_seconds = timeout_seconds

    def probe(self) -> Dict:
        probe = GpuResourceProbe(
            command_runner=self.command_runner,
            timeout_seconds=self.timeout_seconds,
            allow_slot_override=False,
        )
        result = probe.probe()
        status = result.get("status", "")
        if status == "unavailable":
            status = "not_found"
        return {
            "status": status,
            "vendor": "nvidia" if result.get("gpus") else "",
            "driver_version": result.get("driver_version", ""),
            "driver_cuda_version": result.get("driver_cuda_version", ""),
            "devices": list(result.get("gpus") or []),
            "errors": [result.get("error")] if result.get("error") else [],
            "source": result.get("source", "nvidia-smi"),
        }
