"""Model preparation orchestrator (Document A Phase A8).

Ties discovery -> resolution -> file closure -> resource decision -> download
into one deterministic, offline-testable chain that produces the four frozen
Document A artifacts. It never starts a runtime or inference container (that
is Document B).

The source client, host facts, downloader, and cache are all injected so the
whole chain runs offline in tests with no network, GPU, or Docker daemon.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.assets.huggingface import HuggingFaceDownloader
from auto_harness.assets.modelscope import ModelScopeDownloader
from auto_harness.model_runtime.evidence import ModelArtifactWriter
from auto_harness.model_runtime.file_closure import ModelFileClosure
from auto_harness.model_runtime.resolver import ModelReferenceResolver
from auto_harness.model_runtime.resource_solver import ModelResourceSolver
from auto_harness.model_runtime.schemas import (
    InferenceResourceDecision,
    ModelFilePlan,
    ResolvedModelSpec,
)
from auto_harness.models.base import write_json


@dataclass
class PreparationBundle:
    """Result of a full model-preparation chain."""
    status: str = ""
    spec: Optional[ResolvedModelSpec] = None
    plan: Optional[ModelFilePlan] = None
    decision: Optional[InferenceResourceDecision] = None
    errors: List[str] = field(default_factory=list)
    prepare_result: Dict[str, Any] = field(default_factory=dict)
    cache_dir: str = ""
    cache_identity: str = ""
    checkpoint_path: str = ""


class ModelPreparationOrchestrator:
    """Deterministic preparation chain bound to a repo and host facts."""

    def __init__(
        self,
        resolver: Optional[ModelReferenceResolver] = None,
        closure: Optional[ModelFileClosure] = None,
        solver: Optional[ModelResourceSolver] = None,
    ) -> None:
        self.resolver = resolver or ModelReferenceResolver()
        self.closure = closure or ModelFileClosure()
        self.solver = solver or ModelResourceSolver()

    # ---- stages ----

    def resolve_spec(
        self,
        repo_dir,
        source_client,
        operator_override: Optional[str] = None,
        revision_override: Optional[str] = None,
    ) -> ResolvedModelSpec:
        result = self.resolver.resolve_reference(repo_dir, operator_override, revision_override)
        grounding_hash = result.get("grounding_hash", "")
        if result["status"] != "resolved":
            return ResolvedModelSpec(status=result["status"], grounding_hash=grounding_hash)
        candidate = result["selected"]
        spec = self.resolver.resolve_model(candidate, source_client)
        spec.grounding_hash = grounding_hash
        return spec

    def build_plan(self, spec: ResolvedModelSpec, source_client, require_strong: bool = True):
        files = source_client.list_files(spec.repo_id, spec.resolved_revision)
        index_content = None
        for item in files:
            if str(item.get("path", "")).endswith(".safetensors.index.json"):
                try:
                    raw = source_client.fetch_file(spec.repo_id, spec.resolved_revision, item["path"])
                    index_content = json.loads(raw)
                except (ValueError, OSError):
                    index_content = None
                break
        plan, errors = self.closure.build(spec, files, index_content, require_strong)
        if errors:
            plan.status = "blocked"
        return plan, errors

    def decide(self, spec, plan, config, host_facts: Dict) -> InferenceResourceDecision:
        return self.solver.solve(spec, plan, config, host_facts)

    def prepare(self, plan, decision, downloader, cache_dir, disk_safety_ratio: float = 1.2) -> Dict:
        if decision.status != "allowed":
            return {"status": decision.status, "complete_marker_path": ""}
        repo_id, revision = split_model_identity(plan.model_identity)
        return downloader.download_plan(
            repo_id, revision, plan, Path(cache_dir),
            disk_safety_ratio=disk_safety_ratio,
        )

    # ---- full chain ----

    def run(
        self,
        repo_dir,
        run_dir,
        config,
        source_client,
        host_facts: Dict,
        cache,
        downloader=None,
        execute: bool = False,
    ) -> PreparationBundle:
        writer = ModelArtifactWriter(run_dir)
        spec = self.resolve_spec(
            repo_dir, source_client,
            operator_override=getattr(config, "model_id_override", "") or None,
            revision_override=getattr(config, "model_revision_override", "") or None,
        )
        writer.write_resolved_model(spec)
        if spec.status != "resolved":
            return PreparationBundle(status=spec.status, spec=spec)

        plan, errors = self.build_plan(
            spec, source_client,
            require_strong=bool(getattr(config, "model_runtime_require_strong_weight_integrity", True)),
        )
        if errors or plan.status == "blocked":
            writer.write_file_plan(plan)
            return PreparationBundle(status="blocked", spec=spec, plan=plan, errors=errors)

        plan.plan_hash = plan.compute_plan_hash()
        writer.write_file_plan(plan)

        decision = self.decide(spec, plan, config, host_facts)
        writer.write_resource_decision(decision)

        cache_dir = ""
        cache_identity = ""
        prepare_result: Dict[str, Any] = {}
        if execute and decision.status == "allowed":
            cache_dir = str(cache.revision_cache_path(
                spec.source, spec.repo_id, spec.resolved_revision, plan.plan_hash
            ))
            cache_identity = cache.revision_cache_path(
                spec.source, spec.repo_id, spec.resolved_revision, plan.plan_hash
            ).name
            downloader = downloader or downloader_for(spec.source)
            prepare_result = self.prepare(
                plan, decision, downloader, cache_dir,
                disk_safety_ratio=float(getattr(config, "model_runtime_disk_safety_ratio", 1.2)),
            )
            if prepare_result.get("status") == "complete":
                plan.status = "verified"
                plan.remaining_download_bytes = 0
                writer.write_file_plan(plan)

        checkpoint_path = write_preparation_checkpoint(
            run_dir, spec, plan, decision, cache_identity, prepare_result
        )
        status = "prepared" if prepare_result.get("status") == "complete" else decision.status
        return PreparationBundle(
            status=status,
            spec=spec,
            plan=plan,
            decision=decision,
            errors=errors,
            prepare_result=prepare_result,
            cache_dir=cache_dir,
            cache_identity=cache_identity,
            checkpoint_path=checkpoint_path,
        )


def split_model_identity(model_identity: str):
    """Split ``source:org/model@commit`` into (org/model, commit)."""
    rest = model_identity.split(":", 1)[1]
    if "@" in rest:
        return rest.rsplit("@", 1)
    return rest, ""


def downloader_for(source: str):
    if source == "huggingface":
        return HuggingFaceDownloader()
    if source == "modelscope":
        return ModelScopeDownloader()
    raise ValueError("unsupported model source: %s" % source)


def write_preparation_checkpoint(
    run_dir,
    spec: ResolvedModelSpec,
    plan: ModelFilePlan,
    decision: InferenceResourceDecision,
    cache_identity: str,
    prepare_result: Dict,
) -> str:
    """Persist the frozen Document A checkpoint fields (no tokens)."""
    payload = {
        "schema_version": 1,
        "model_identity": spec.model_identity,
        "resolved_revision": spec.resolved_revision,
        "grounding_hash": spec.grounding_hash,
        "resolved_model_hash": spec.source_metadata_hash,
        "file_plan_hash": plan.plan_hash,
        "resource_decision_hash": decision.decision_hash,
        "cache_identity": cache_identity,
        "complete_marker_hash": prepare_result.get("marker_hash", "") if isinstance(prepare_result, dict) else "",
        "pending_download_operation_id": "",
    }
    path = Path(run_dir) / "reports" / "model" / "preparation_checkpoint.json"
    write_json(path, payload)
    return str(path)
