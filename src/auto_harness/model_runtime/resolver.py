"""Deterministic model reference resolution (Document A Phase A2).

The LLM may propose model candidates and explain ambiguity, but it never
decides the local path, the GPU facts, the Docker image, or free commands.
This resolver turns grounded repository evidence into either:
  - a single primary generation model (status == resolved), or
  - a fail-closed needs_human_input / not_found outcome.

Dynamic Python values that cannot be statically resolved are NOT guessed;
they surface as ambiguity rather than triggering a network call.
"""
from typing import Dict, List, Optional

from auto_harness.assets.detector import ModelReferenceDetector, is_valid_repo_id
from auto_harness.model_runtime.schemas import (
    ModelReferenceCandidate,
    ResolvedModelSpec,
    hash_payload,
)
from auto_harness.model_runtime.source_clients import (
    SourceClientError,
    source_metadata_hash,
)
from auto_harness.utils.time import utc_now_iso

# A candidate is "high confidence" when it came from a strong discovery method.
HIGH_CONFIDENCE = 0.8
# Clear winner requires a confidence gap over the next primary candidate.
CLEAR_WINNER_GAP = 0.2


def compute_grounding_hash(candidates: List[ModelReferenceCandidate]) -> str:
    """Hash the repository evidence backing a candidate set.

    Binds candidates to the repository files they were discovered in. Includes
    file path + content hash + expression, so any change to the evidence
    changes the grounding hash.
    """
    evidence = []
    for candidate in candidates:
        for item in candidate.evidence:
            evidence.append(
                {
                    "file": item.get("file"),
                    "sha256": item.get("sha256"),
                    "expression": item.get("expression"),
                }
            )
    evidence.sort(key=lambda item: (item.get("file") or "", item.get("expression") or ""))
    return hash_payload({"schema_version": 1, "evidence": evidence})


class ModelReferenceResolver:
    """Resolve repository model references into a unique primary model."""

    def __init__(self, detector: Optional[ModelReferenceDetector] = None) -> None:
        self.detector = detector or ModelReferenceDetector()

    def discover(self, repo_dir) -> List[ModelReferenceCandidate]:
        return self.detector.detect(repo_dir)

    def select_primary(
        self,
        candidates: List[ModelReferenceCandidate],
        operator_override: Optional[str] = None,
        revision_override: Optional[str] = None,
    ) -> Dict:
        """Select a single primary generation model or fail closed.

        Returns a dict with ``status`` in
        {resolved, needs_human_input, not_found, ambiguous}, the selected
        candidate (when resolved), the full candidate list, and reasons.
        """
        candidates = list(candidates or [])
        warnings: List[str] = []

        if operator_override:
            source, repo_id = self._parse_override(operator_override)
            candidate = ModelReferenceCandidate(
                source=source,
                repo_id=repo_id,
                requested_revision=revision_override or "main",
                role="primary_generation_model",
                confidence=1.0,
                evidence=[],
                discovered_by="operator_override",
            )
            warnings.append(
                "operator override applied; selected model is not grounded in repository evidence"
            )
            return {
                "status": "resolved",
                "selected": candidate,
                "candidates": [candidate],
                "reasons": ["operator override selected the model"],
                "warnings": warnings,
            }

        primary = [c for c in candidates if c.role == "primary_generation_model"]
        if not primary:
            if candidates:
                return {
                    "status": "needs_human_input",
                    "selected": None,
                    "candidates": candidates,
                    "reasons": [
                        "only tokenizer/accessory model candidates were found; "
                        "no primary generation model could be selected"
                    ],
                    "warnings": warnings,
                }
            return {
                "status": "not_found",
                "selected": None,
                "candidates": [],
                "reasons": ["no model references were discovered in the repository"],
                "warnings": warnings,
            }

        if len(primary) == 1:
            return {
                "status": "resolved",
                "selected": primary[0],
                "candidates": candidates,
                "reasons": ["single primary generation model candidate"],
                "warnings": warnings,
            }

        ordered = sorted(primary, key=lambda c: -c.confidence)
        top, second = ordered[0], ordered[1]
        if top.confidence >= HIGH_CONFIDENCE and (top.confidence - second.confidence) >= CLEAR_WINNER_GAP:
            warnings.append(
                "multiple primary model candidates; selected the highest-confidence candidate"
            )
            return {
                "status": "resolved",
                "selected": top,
                "candidates": candidates,
                "reasons": ["multiple primary candidates with a clear confidence winner"],
                "warnings": warnings,
            }

        return {
            "status": "needs_human_input",
            "selected": None,
            "candidates": candidates,
            "reasons": [
                "multiple high-confidence primary generation model candidates; "
                "cannot select a unique model without human input"
            ],
            "warnings": warnings,
        }

    def resolve_reference(
        self,
        repo_dir,
        operator_override: Optional[str] = None,
        revision_override: Optional[str] = None,
    ) -> Dict:
        candidates = self.discover(repo_dir)
        result = self.select_primary(candidates, operator_override, revision_override)
        result["grounding_hash"] = compute_grounding_hash(candidates)
        result["candidate_count"] = len(candidates)
        return result

    def resolve_model(self, candidate: ModelReferenceCandidate, source_client) -> ResolvedModelSpec:
        """Resolve a candidate into an immutable ResolvedModelSpec.

        The source client pins the mutable revision to a commit SHA, then the
        config and file list are fetched against that same commit. A change in
        the source's commit between calls fails closed (metadata_invalid).

        Returns a ResolvedModelSpec whose ``status`` is one of the allowed
        states; ``resolved`` only when the model is uniquely pinned.
        """
        base = {
            "source": candidate.source,
            "repo_id": candidate.repo_id,
            "requested_revision": candidate.requested_revision,
            "resolved_at": utc_now_iso(),
        }
        try:
            revision = source_client.resolve_revision(
                candidate.repo_id, candidate.requested_revision
            )
            resolved_revision = str(revision.get("resolved_revision") or "")
            if not resolved_revision:
                return ResolvedModelSpec(status="metadata_invalid", **base)
            config = source_client.fetch_model_config(candidate.repo_id, resolved_revision)
            files = source_client.list_files(candidate.repo_id, resolved_revision)
        except SourceClientError as exc:
            return ResolvedModelSpec(status=exc.status, **base)

        if config.get("auto_map"):
            return ResolvedModelSpec(
                status="remote_code_required",
                resolved_revision=resolved_revision,
                requires_remote_code=True,
                **base,
            )
        quantization = self._detect_quantization(config)
        if quantization:
            return ResolvedModelSpec(
                status="unsupported_quantization",
                resolved_revision=resolved_revision,
                quantization=quantization,
                **base,
            )
        architectures = self._architectures(config)
        if architectures and not self._supported_architectures(architectures):
            return ResolvedModelSpec(
                status="unsupported_architecture",
                resolved_revision=resolved_revision,
                architectures=architectures,
                **base,
            )

        model_identity = "%s:%s@%s" % (candidate.source, candidate.repo_id, resolved_revision)
        metadata_hash = source_metadata_hash(config, files, resolved_revision)
        return ResolvedModelSpec(
            status="resolved",
            resolved_revision=resolved_revision,
            model_identity=model_identity,
            model_type=str(config.get("model_type") or ""),
            architectures=architectures,
            task="text-generation",
            dtype=self._dtype(config),
            parameter_count=self._optional_int(config.get("num_parameters")),
            hidden_size=self._optional_int(config.get("hidden_size")),
            num_hidden_layers=self._optional_int(config.get("num_hidden_layers")),
            num_attention_heads=self._optional_int(config.get("num_attention_heads")),
            num_key_value_heads=self._optional_int(config.get("num_key_value_heads")),
            head_dim=self._optional_int(config.get("head_dim")),
            max_position_embeddings=self._optional_int(config.get("max_position_embeddings")),
            quantization=None,
            requires_remote_code=False,
            gated=bool(revision.get("gated", False)),
            license=str(revision.get("license") or ""),
            source_metadata_hash=metadata_hash,
            grounding_hash="",
            **base,
        )

    @staticmethod
    def _optional_int(value) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dtype(config: Dict) -> str:
        dtype = str(config.get("torch_dtype") or "").lower()
        if dtype in ("float16", "half", "fp16"):
            return "float16"
        if dtype in ("bfloat16", "bf16"):
            return "bfloat16"
        if dtype in ("float32", "float", "fp32"):
            return "float32"
        return "float16"

    @staticmethod
    def _architectures(config: Dict) -> List[str]:
        value = config.get("architectures")
        if isinstance(value, list):
            return [str(item) for item in value if item]
        return []

    @staticmethod
    def _detect_quantization(config: Dict) -> Optional[str]:
        q = config.get("quantization_config")
        if isinstance(q, dict):
            method = str(q.get("quant_method") or q.get("method") or "").lower()
            if method in ("awq", "gptq", "fp8", "bitsandbytes", "bnb"):
                return method
            return "unknown"
        if config.get("quantization_method"):
            return str(config["quantization_method"]).lower()
        return None

    @staticmethod
    def _supported_architectures(architectures: List[str]) -> bool:
        # Text-generation decoder-only families vLLM can serve.
        supported = {
            "qwen2forcausallm", "qwen2moe", "llamaforcausallm",
            "mistralforcausallm", "gpt2lmheadmodel", "gptneoxforcausallm",
            "falconforcausallm", "gptjforcausallm", "optforcausallm",
            "bloomforcausallm", "gemmaforcausallm", "gemma2forcausallm",
            "mptforcausallm", "starcoder2forcausallm", "chatglmforconditionalgeneration",
        }
        return all(name.lower().replace("_", "") in supported for name in architectures)

    @staticmethod
    def _parse_override(override: str):
        """Parse ``[source:]org/model`` into (source, repo_id)."""
        text = (override or "").strip()
        if not text:
            raise ValueError("operator model override must be non-empty")
        source = "huggingface"
        repo_id = text
        if ":" in text:
            prefix, rest = text.split(":", 1)
            if prefix in ("huggingface", "modelscope"):
                source = prefix
                repo_id = rest
            else:
                # A colon that isn't a source prefix — treat as invalid, not a URL.
                raise ValueError("operator model override must be source:org/model")
        if "modelscope" in text.lower() and ":" not in text:
            source = "modelscope"
        if not is_valid_repo_id(repo_id):
            raise ValueError(
                "operator model override repo id %r must be a safe org/name id" % repo_id
            )
        return source, repo_id
