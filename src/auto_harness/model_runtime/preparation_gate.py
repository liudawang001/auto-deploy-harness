"""Preparation Artifact Gate (Document B Phase B0).

Re-validates the four frozen Document A artifacts before any runtime work:

    runs/<task-id>/reports/model/resolved_model.json
    runs/<task-id>/reports/model/model_file_plan.json
    runs/<task-id>/reports/model/resource_decision.json
    model_cache/<source>/<cache-key>/.auto_harness_complete.json

The gate is read-only. It never patches, re-resolves, or repairs an artifact;
on any failure it returns a structured ``PreparationBundle`` whose ``status``
is one of the Document B failure codes so the caller can route back to the
corresponding Document A phase.

Checks performed (in order):
1.  Locate the four artifacts without accepting arbitrary external paths.
2.  Validate schema version and status for each artifact.
3.  Recompute plan / decision / marker hashes and compare against stored.
4.  Verify the model identity is consistent across all four artifacts.
5.  Verify the cache directory is inside the Harness-owned model_cache root.
6.  Re-read every required file and verify size / sha256.
7.  Verify the resource decision is not stale (GPU index present, free VRAM
    still above the allowed threshold) via an injected host-facts provider.
8.  Scan every artifact for secret-like fields / values.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from auto_harness.assets.cache import COMPLETE_MARKER_NAME, revision_cache_key
from auto_harness.model_runtime.evidence import scan_forbidden_fields
from auto_harness.model_runtime.schemas import (
    CacheCompleteMarker,
    InferenceResourceDecision,
    ModelFilePlan,
    ResolvedModelSpec,
)
from auto_harness.models.base import read_json
from auto_harness.utils.redaction import check_redaction

_RESOLVED_MODEL = "resolved_model.json"
_MODEL_FILE_PLAN = "model_file_plan.json"
_RESOURCE_DECISION = "resource_decision.json"

# Failure statuses (Document B B5).
FAILURE_PREPARATION_ARTIFACT_MISSING = "preparation_artifact_missing"
FAILURE_SCHEMA_UNSUPPORTED = "preparation_schema_unsupported"
FAILURE_HASH_MISMATCH = "preparation_hash_mismatch"
FAILURE_IDENTITY_MISMATCH = "model_identity_mismatch"
FAILURE_CACHE_PATH_ESCAPE = "cache_path_escape"
FAILURE_CACHE_FILE_MISSING = "cache_file_missing"
FAILURE_CACHE_INTEGRITY_FAILED = "cache_integrity_failed"
FAILURE_RESOURCE_STALE = "resource_decision_stale"
FAILURE_RESOURCE_NO_LONGER_AVAILABLE = "resource_no_longer_available"


@dataclass
class PreparationBundle:
    """Read-only, validated runtime preparation bundle handed to the adapter."""

    status: str = "ready"
    spec: Optional[ResolvedModelSpec] = None
    plan: Optional[ModelFilePlan] = None
    decision: Optional[InferenceResourceDecision] = None
    marker: Optional[CacheCompleteMarker] = None
    resolved_model_hash: str = ""
    file_plan_hash: str = ""
    cache_marker_hash: str = ""
    resource_decision_hash: str = ""
    model_host_path: str = ""
    model_container_path: str = "/models/current"
    cache_root: str = ""
    gpu_indexes: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ready"


class PreparationArtifactGate:
    """Re-validate the frozen Document A artifacts before runtime work."""

    def __init__(
        self,
        run_dir,
        cache_root=None,
        host_facts_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.cache_root = Path(cache_root) if cache_root else Path("model_cache")
        self.host_facts_provider = host_facts_provider

    # -- public ---------------------------------------------------------

    def validate(self) -> PreparationBundle:
        model_dir = self.run_dir / "reports" / "model"
        resolved_path = model_dir / _RESOLVED_MODEL
        plan_path = model_dir / _MODEL_FILE_PLAN
        decision_path = model_dir / _RESOURCE_DECISION

        missing = [p.name for p in (resolved_path, plan_path, decision_path) if not p.exists()]
        if missing:
            return self._fail(FAILURE_PREPARATION_ARTIFACT_MISSING, missing)

        spec, err = self._read(resolved_path, ResolvedModelSpec)
        if err:
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, [err])
        plan, err = self._read(plan_path, ModelFilePlan)
        if err:
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, [err])
        decision, err = self._read(decision_path, InferenceResourceDecision)
        if err:
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, [err])

        # Status gate.
        if spec.status != "resolved":
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, ["resolved_model.status != resolved"])
        if decision.status != "allowed":
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, ["resource_decision.status != allowed"])
        if not self._is_commit_sha(spec.resolved_revision):
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, ["resolved_revision is not an immutable commit SHA"])

        # Identity consistency across artifacts.
        identities = {
            spec.model_identity,
            plan.model_identity,
            decision.model_identity,
        }
        if len(identities) != 1 or not spec.model_identity:
            return self._fail(FAILURE_IDENTITY_MISMATCH, sorted(identities))

        # Hash recomputation.
        if plan.compute_plan_hash() != plan.plan_hash:
            return self._fail(FAILURE_HASH_MISMATCH, ["model_file_plan.plan_hash"])
        if decision.compute_decision_hash() != decision.decision_hash:
            return self._fail(FAILURE_HASH_MISMATCH, ["resource_decision.decision_hash"])

        # Cache directory derivation and containment.
        cache_dir, err = self._cache_dir(spec, plan)
        if err:
            return self._fail(FAILURE_CACHE_PATH_ESCAPE, [err])

        marker_path = cache_dir / COMPLETE_MARKER_NAME
        if not marker_path.exists():
            return self._fail(FAILURE_CACHE_FILE_MISSING, [str(marker_path)])
        marker, err = self._read(marker_path, CacheCompleteMarker)
        if err:
            return self._fail(FAILURE_SCHEMA_UNSUPPORTED, [err])

        if marker.model_identity != spec.model_identity:
            return self._fail(FAILURE_IDENTITY_MISMATCH, [marker.model_identity, spec.model_identity])
        if marker.file_plan_hash != plan.plan_hash:
            return self._fail(FAILURE_HASH_MISMATCH, ["complete marker.file_plan_hash"])
        if marker.compute_marker_hash() != marker.marker_hash:
            return self._fail(FAILURE_HASH_MISMATCH, ["complete marker.marker_hash"])

        # Required file integrity.
        for item in plan.files:
            if not item.get("required", True):
                continue
            problem = self._check_file(cache_dir, item)
            if problem:
                code, detail = problem
                return self._fail(code, [detail])

        # Secret scan on all four artifacts.
        for path in (resolved_path, plan_path, decision_path, marker_path):
            problem = self._check_secrets(path)
            if problem:
                return self._fail(problem[0], [problem[1]])

        # Resource decision freshness (host facts injected; skip when absent).
        stale = self._check_freshness(decision)
        if stale:
            return self._fail(*stale)

        return PreparationBundle(
            status="ready",
            spec=spec,
            plan=plan,
            decision=decision,
            marker=marker,
            resolved_model_hash=spec.source_metadata_hash,
            file_plan_hash=plan.plan_hash,
            cache_marker_hash=marker.marker_hash,
            resource_decision_hash=decision.decision_hash,
            model_host_path=str(cache_dir),
            model_container_path="/models/current",
            cache_root=str(self.cache_root),
            gpu_indexes=list(decision.gpu_indexes),
        )

    # -- helpers --------------------------------------------------------

    def _fail(self, status: str, errors: List[str]) -> PreparationBundle:
        return PreparationBundle(status=status, errors=list(errors))

    @staticmethod
    def _read(path: Path, cls):
        try:
            data = read_json(path)
        except (OSError, ValueError) as exc:
            return None, "cannot read %s: %s" % (path.name, exc)
        try:
            return cls.from_dict(data), ""
        except (ValueError, TypeError) as exc:
            return None, "%s invalid: %s" % (cls.__name__, exc)

    @staticmethod
    def _is_commit_sha(value: str) -> bool:
        return bool(value) and len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value)

    def _cache_dir(self, spec: ResolvedModelSpec, plan: ModelFilePlan):
        source = spec.source
        repo_id = spec.repo_id
        revision = spec.resolved_revision
        if not repo_id and ":" in spec.model_identity:
            rest = spec.model_identity.split(":", 1)[1]
            if "@" in rest:
                repo_id, revision = rest.rsplit("@", 1)
        if not source or not repo_id or not revision:
            return None, "cannot derive cache identity from resolved model"
        key = revision_cache_key(source, repo_id, revision, plan.plan_hash)
        cache_dir = (self.cache_root / source / key).resolve()
        root = self.cache_root.resolve()
        try:
            cache_dir.relative_to(root)
        except ValueError:
            return None, "cache dir escapes cache root: %s" % cache_dir
        return cache_dir, ""

    def _check_file(self, cache_dir: Path, item: Dict[str, Any]):
        rel = str(item.get("path", ""))
        if not rel or rel.startswith(("/", "\\")) or ".." in rel.replace("\\", "/").split("/"):
            return FAILURE_CACHE_PATH_ESCAPE, "unsafe file path in plan: %s" % rel
        path = (cache_dir / rel).resolve()
        try:
            path.relative_to(cache_dir)
        except ValueError:
            return FAILURE_CACHE_PATH_ESCAPE, "file escapes cache dir: %s" % rel
        if not path.is_file():
            return FAILURE_CACHE_FILE_MISSING, "required file missing: %s" % rel
        size = item.get("size_bytes")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            actual = path.stat().st_size
            if actual != size:
                return FAILURE_CACHE_INTEGRITY_FAILED, "size mismatch for %s: expected %d got %d" % (rel, size, actual)
        sha = item.get("sha256")
        if sha:
            import hashlib

            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha = digest.hexdigest()
            if actual_sha != sha:
                return FAILURE_CACHE_INTEGRITY_FAILED, "sha256 mismatch for %s" % rel
        return None

    def _check_secrets(self, path: Path):
        try:
            text = path.read_text(encoding="utf-8")
            data = read_json(path)
        except (OSError, ValueError):
            return None
        forbidden = scan_forbidden_fields(data)
        if forbidden:
            return FAILURE_SCHEMA_UNSUPPORTED, "secret-like fields: %s" % ", ".join(forbidden)
        redacted = check_redaction(text)
        if redacted:
            return FAILURE_SCHEMA_UNSUPPORTED, "unredacted content: %s" % ", ".join(
                item["pattern_name"] for item in redacted
            )
        return None

    def _check_freshness(self, decision: InferenceResourceDecision):
        if not self.host_facts_provider:
            return None
        try:
            facts = self.host_facts_provider() or {}
        except Exception:
            return None
        present = set(int(i) for i in (facts.get("gpu_indexes") or []))
        for idx in decision.gpu_indexes:
            if int(idx) not in present:
                return FAILURE_RESOURCE_STALE, ["gpu index %s no longer present" % idx]
        free = int(facts.get("gpu_memory_free_bytes") or 0)
        if free > 0 and free < int(decision.required_vram_bytes):
            return (
                FAILURE_RESOURCE_NO_LONGER_AVAILABLE,
                ["free vram %d below required %d" % (free, decision.required_vram_bytes)],
            )
        return None
