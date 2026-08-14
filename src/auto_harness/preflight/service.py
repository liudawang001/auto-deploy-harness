"""Shared GPU/Conda preflight service used by all controllers."""
import platform
from pathlib import Path
from typing import Dict

from auto_harness.env import CondaEnvironmentParser
from auto_harness.preflight.compatibility import EnvironmentCompatibilityResolver
from auto_harness.preflight.conda import CondaInventoryProbe, CondaRuntimeProbe
from auto_harness.preflight.container import DockerGpuProbe
from auto_harness.preflight.evidence import PreflightEvidenceWriter
from auto_harness.preflight.gpu import NvidiaGpuProbe
from auto_harness.preflight.policy import EnvironmentPreflightPolicy
from auto_harness.preflight.schemas import HostCapabilitySnapshot
from auto_harness.preflight.storage import StorageProbe
from auto_harness.utils.time import utc_now_iso


class HostPreflightService:
    def __init__(
        self,
        gpu_probe=None,
        runtime_probe=None,
        inventory_probe=None,
        resolver=None,
        policy=None,
        storage_probe=None,
        docker_gpu_probe=None,
    ) -> None:
        self.gpu_probe = gpu_probe or NvidiaGpuProbe()
        self.runtime_probe = runtime_probe or CondaRuntimeProbe()
        self.inventory_probe = inventory_probe or CondaInventoryProbe()
        self.resolver = resolver or EnvironmentCompatibilityResolver()
        self.policy = policy or EnvironmentPreflightPolicy()
        self.storage_probe = storage_probe or StorageProbe()
        self.docker_gpu_probe = docker_gpu_probe

    def run(
        self,
        repo_dir: Path,
        analysis: Dict,
        resource_plan: Dict,
        config,
        run_dir: Path = None,
        allow_mutation: bool = False,
    ) -> Dict:
        repo_dir = Path(repo_dir).resolve()
        conda_file = CondaEnvironmentParser().parse_repo(
            repo_dir,
            default_python=getattr(config, "conda_python_default", "3.10"),
        )
        gpu = self.gpu_probe.probe()
        runtimes = self.runtime_probe.probe()
        model_inference = bool(getattr(config, "model_inference_enabled", False))
        host_resources: Dict = {}
        docker_gpu: Dict = {}
        if model_inference:
            host_resources = self.storage_probe.probe(
                getattr(config, "model_cache_path", None) or Path("model_cache")
            )
            docker_gpu = self._probe_docker_gpu(config, gpu)
        capabilities = HostCapabilitySnapshot(
            collected_at=utc_now_iso(),
            host={
                "platform": platform.system().lower(),
                "machine": platform.machine().lower(),
            },
            gpu=gpu,
            environment_runtimes=runtimes,
        ).to_dict()
        preliminary = self.resolver.resolve(
            repo_dir, analysis, resource_plan, conda_file,
            capabilities, {"environments": []}, config,
        )
        runtime = runtimes.get(preliminary.get("backend")) or {}
        inventory = self.inventory_probe.probe(
            runtime,
            project_id=preliminary.get("project_id", ""),
        ) if runtime.get("available") else {
            "schema_version": 1,
            "tool": "",
            "tool_path": "",
            "root_prefix": "",
            "active_prefix": "",
            "environments": [],
            "errors": [],
        }
        decision = self.resolver.resolve(
            repo_dir, analysis, resource_plan, conda_file,
            capabilities, inventory, config,
        )
        policy = self.policy.evaluate(
            decision, repo_dir, config, allow_mutation=allow_mutation,
        )
        paths = {}
        if run_dir:
            writer = PreflightEvidenceWriter(run_dir)
            paths = {
                "host_capabilities": writer.write("host_capabilities", capabilities),
                "gpu_probe": writer.write("gpu_probe", gpu),
                "conda_runtime_probe": writer.write("conda_runtime_probe", runtimes),
                "conda_environment_inventory": writer.write("conda_environment_inventory", inventory),
                "compatibility_decision": writer.write("compatibility_decision", decision),
                "policy_decision": writer.write("policy_decision", policy),
            }
            if model_inference:
                paths["host_resources"] = writer.write("host_resources", host_resources)
                paths["docker_gpu"] = writer.write("docker_gpu", docker_gpu)
        return {
            "capabilities": capabilities,
            "conda_file": conda_file,
            "conda_inventory": inventory,
            "compatibility_decision": decision,
            "policy": policy,
            "host_resources": host_resources,
            "docker_gpu": docker_gpu,
            "evidence_paths": paths,
        }

    def _probe_docker_gpu(self, config, gpu: Dict) -> Dict:
        probe_image = getattr(config, "docker_gpu_probe_image", "") or None
        if self.docker_gpu_probe is None:
            devices = gpu.get("devices") or []
            selected = int(devices[0].get("index", 0)) if devices else 0
            self.docker_gpu_probe = DockerGpuProbe(probe_image=probe_image)
            return self.docker_gpu_probe.probe(selected)
        return self.docker_gpu_probe.probe(0)
