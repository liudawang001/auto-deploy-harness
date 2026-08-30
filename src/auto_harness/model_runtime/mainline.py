"""Mainline bridge between the LangGraph model_prepare stage and the
Document A preparation chain.

The stage executor routes here when ``config.model_inference_enabled`` is
true. The runner resolves the model reference (operator override, or offline
repository discovery), builds a real source client, probes host facts, and
delegates to ``ModelPreparationOrchestrator``. Status mapping is fail-closed:
only a verified download (``prepared``) or a dry-run ``allowed`` resource
decision maps to a passed stage; everything else surfaces structured errors
and never fabricates GPU capability the host does not have.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.assets.cache import ModelCache
from auto_harness.models.result import StageResult
from auto_harness.model_runtime.preparation import (
    ModelPreparationOrchestrator,
    downloader_for,
)
from auto_harness.model_runtime.resolver import ModelReferenceResolver
from auto_harness.model_runtime.source_clients import (
    UrlopenTransport,
    source_client_for,
)
from auto_harness.runtime.gpu import GpuResourceProbe

_TOKEN_ENV_BY_SOURCE = {
    "huggingface": "HF_TOKEN",
    "modelscope": "MODELSCOPE_TOKEN",
}

# Standard mirror-endpoint overrides (huggingface_hub convention). GPU rental
# platforms and CN networks typically reach hf-mirror.com instead of
# huggingface.co; the URL templates are mirror-compatible. Setting the env
# var is the explicit operator opt-in for that exact host.
_ENDPOINT_ENV_BY_SOURCE = {
    "huggingface": "HF_ENDPOINT",
    "modelscope": "MODELSCOPE_ENDPOINT",
}

_MB_TO_BYTES = 1024 * 1024


def build_host_facts(
    command_runner=None,
    cache_dir: Optional[Path] = None,
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Probe host facts for the resource solver; every failure degrades to zero.

    The v1 runtime boundary is single-GPU (tensor_parallel_size=1), so the
    GPU pool is the largest single card — summing multiple cards would
    over-approve a model that no single card can hold.
    """
    facts: Dict[str, Any] = {
        "gpu_indexes": [],
        "gpu_memory_total_bytes": 0,
        "gpu_memory_free_bytes": 0,
        "ram_total_bytes": 0,
        "ram_available_bytes": 0,
        "disk_total_bytes": 0,
        "disk_free_bytes": 0,
    }
    probe = GpuResourceProbe(
        command_runner=command_runner,
        environ=environ,
        allow_slot_override=False,
    ).probe()
    gpus = probe.get("gpus") or []
    if gpus:
        best = max(gpus, key=lambda g: int(g.get("memory_free_mb") or 0))
        facts["gpu_indexes"] = [int(best.get("index") or 0)]
        facts["gpu_memory_total_bytes"] = int(best.get("memory_total_mb") or 0) * _MB_TO_BYTES
        facts["gpu_memory_free_bytes"] = int(best.get("memory_free_mb") or 0) * _MB_TO_BYTES
    total, available = _ram_bytes()
    facts["ram_total_bytes"] = total
    facts["ram_available_bytes"] = available
    disk_total, disk_free = _disk_bytes(cache_dir)
    facts["disk_total_bytes"] = disk_total
    facts["disk_free_bytes"] = disk_free
    return facts


def _ram_bytes():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
    except (ValueError, OSError, AttributeError):
        return 0, 0
    available = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        available = total
    return total, available


def _disk_bytes(cache_dir: Optional[Path]):
    target = Path(cache_dir) if cache_dir else Path.cwd()
    try:
        usage = shutil.disk_usage(target)
        return int(usage.total), int(usage.free)
    except OSError:
        return 0, 0


class ModelPreparationStageRunner:
    """Run the Document A preparation chain as a mainline pipeline stage."""

    def __init__(
        self,
        orchestrator: Optional[ModelPreparationOrchestrator] = None,
        probe_command_runner=None,
        environ: Optional[Dict[str, str]] = None,
        source_client_factory=None,
        downloader_factory=None,
    ) -> None:
        self.orchestrator = orchestrator or ModelPreparationOrchestrator()
        self.probe_command_runner = probe_command_runner
        self.environ = environ if environ is not None else dict(os.environ)
        self.source_client_factory = source_client_factory or source_client_for
        self.downloader_factory = downloader_factory or downloader_for

    def run(
        self,
        *,
        run_dir,
        task_id: str,
        repo_dir,
        config,
        execute: bool = False,
    ) -> StageResult:
        source, reference_error = self._resolve_source(repo_dir, config)
        if reference_error:
            return StageResult(
                "model_prepare",
                "failed",
                "model reference could not be resolved",
                {"errors": [reference_error], "model_inference": True},
                error="model_reference_unresolved",
            )

        source_client = self.source_client_factory(
            source,
            transport=UrlopenTransport(),
            token=(self.environ.get(_TOKEN_ENV_BY_SOURCE.get(source, "")) or ""),
            api_base=(self.environ.get(_ENDPOINT_ENV_BY_SOURCE.get(source, "")) or "") or None,
        )
        cache = ModelCache(Path(str(getattr(config, "model_cache_dir", "model_cache"))))
        host_facts = build_host_facts(
            command_runner=self.probe_command_runner,
            cache_dir=cache.root,
            environ=self.environ,
        )
        downloader = self.downloader_factory(source) if execute else None
        try:
            bundle = self.orchestrator.run(
                repo_dir,
                run_dir,
                config,
                source_client,
                host_facts,
                cache,
                downloader=downloader,
                execute=execute,
            )
        except ValueError as exc:
            return StageResult(
                "model_prepare",
                "failed",
                "model preparation chain failed",
                {"errors": [str(exc)], "model_inference": True},
                error="model_preparation_failed",
            )

        decision = bundle.decision
        data = self._stage_data(bundle, source, host_facts)
        if bundle.status in ("prepared", "allowed"):
            return StageResult(
                "model_prepare",
                "passed",
                "model prepared for managed inference"
                if bundle.status == "prepared"
                else "dry-run resource decision allowed; weights not downloaded",
                data,
            )
        if decision is not None and decision.status not in ("allowed",):
            return StageResult(
                "model_prepare",
                "uncertain",
                "resource decision did not allow managed inference",
                data,
                error="resource_decision_%s" % (decision.status or "unknown"),
            )
        return StageResult(
            "model_prepare",
            "failed",
            "model preparation chain blocked",
            data,
            error="preparation_%s" % (bundle.status or "blocked"),
        )

    def _resolve_source(self, repo_dir, config):
        override = str(getattr(config, "model_id_override", "") or "").strip()
        try:
            if override:
                result = ModelReferenceResolver().select_primary(
                    [], operator_override=override,
                    revision_override=getattr(config, "model_revision_override", "") or None,
                )
            else:
                # Offline repository discovery; no network until resolve_model.
                result = ModelReferenceResolver().resolve_reference(repo_dir)
        except ValueError as exc:
            return "", str(exc)
        selected = result.get("selected")
        if result.get("status") != "resolved" or selected is None:
            reasons = "; ".join(str(reason) for reason in result.get("reasons") or [])
            return "", "model reference not resolved: %s (%s)" % (
                result.get("status"), reasons or "no candidate",
            )
        return str(selected.source), ""

    def _stage_data(self, bundle, source: str, host_facts: Dict[str, Any]) -> Dict[str, Any]:
        spec = bundle.spec
        plan = bundle.plan
        decision = bundle.decision
        data: Dict[str, Any] = {
            "model_inference": True,
            "source": source,
            "bundle_status": bundle.status,
            "errors": list(bundle.errors or []),
            "prepare_result_status": (bundle.prepare_result or {}).get("status", ""),
            "cache_identity": bundle.cache_identity,
            "host_facts": host_facts,
        }
        if spec is not None:
            data["model_identity"] = spec.model_identity
            data["resolved_revision"] = spec.resolved_revision
            data["spec_status"] = spec.status
        if plan is not None:
            data["file_plan_status"] = plan.status
            data["file_total_size_bytes"] = int(getattr(plan, "total_size_bytes", 0) or 0)
            data["required_file_count"] = sum(
                1 for item in (plan.files or []) if item.get("required", True)
            )
        if decision is not None:
            data["decision_status"] = decision.status
            data["decision_reasons"] = list(decision.reasons or [])
            data["required_vram_bytes"] = int(getattr(decision, "required_vram_bytes", 0) or 0)
        return data
