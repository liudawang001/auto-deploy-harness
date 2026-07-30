"""Pipeline adapter for shared host preflight."""
from pathlib import Path
from typing import Dict

from auto_harness.models.result import StageResult
from auto_harness.preflight import HostPreflightService
from auto_harness.preflight.conda import CondaInventoryProbe, CondaRuntimeProbe
from auto_harness.preflight.gpu import NvidiaGpuProbe


class HostPreflightModule:
    def __init__(self, config, service=None) -> None:
        self.config = config
        self.service = service or HostPreflightService(
            gpu_probe=NvidiaGpuProbe(
                timeout_seconds=getattr(config, "gpu_probe_timeout_seconds", 5),
            ),
            runtime_probe=CondaRuntimeProbe(
                timeout_seconds=getattr(config, "conda_probe_timeout_seconds", 10),
            ),
            inventory_probe=CondaInventoryProbe(
                timeout_seconds=getattr(config, "conda_inventory_timeout_seconds", 30),
                max_envs=getattr(config, "conda_inventory_max_envs", 50),
            ),
        )

    def run(
        self,
        repo_dir: Path,
        analysis: Dict,
        resource_plan: Dict,
        run_dir: Path = None,
        allow_mutation: bool = False,
    ) -> StageResult:
        if not getattr(self.config, "preflight_enabled", True):
            return StageResult(
                "host_preflight", "passed", "host preflight disabled",
                {"disabled": True},
            )
        effective_resource_plan = dict(resource_plan)
        if getattr(self.config, "preflight_require_gpu", False):
            effective_resource_plan["gpu_required"] = True
        if getattr(self.config, "min_gpu_memory_mb", 0):
            effective_resource_plan["min_gpu_memory_mb"] = self.config.min_gpu_memory_mb
        data = self.service.run(
            repo_dir,
            analysis,
            effective_resource_plan,
            self.config,
            run_dir=run_dir,
            allow_mutation=allow_mutation,
        )
        decision = data.get("compatibility_decision") or {}
        policy = data.get("policy") or {}
        if not policy.get("allowed") or decision.get("status") == "blocked":
            status = "failed"
        elif decision.get("status") == "uncertain":
            status = "uncertain"
        else:
            status = "passed"
        return StageResult(
            "host_preflight",
            status,
            "host preflight %s" % status,
            data,
            evidence=list((data.get("evidence_paths") or {}).values()),
            error="; ".join(policy.get("reasons") or []) if status != "passed" else None,
        )
