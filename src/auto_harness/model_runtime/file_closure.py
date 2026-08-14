"""Safetensors file closure (Document A Phase A4).

Builds a frozen ModelFilePlan for one immutable model revision: the exact
weight shards, index, config, and tokenizer files needed to serve the model —
no more (READMEs, scripts, unrelated variants) and no less (a missing shard
blocks the plan).

Security invariants:
  - every path is a safe relative path (no absolute path, ``..``, or control chars)
  - the index ``weight_map`` must be an object of safe relative ``.safetensors`` paths
  - mixed ``.bin``/``.safetensors``, multiple indices, and multiple single-file
    variants are rejected
  - a valid 64-char LFS OID may be promoted to SHA-256 for strong integrity
  - ``require_strong`` blocks a plan whose weights cannot be strongly verified
"""
from typing import Any, Dict, List, Optional, Tuple

from auto_harness.assets.selection import ModelFileSelector
from auto_harness.model_runtime.schemas import ModelFilePlan, ResolvedModelSpec

WEIGHT_INDEX_SUFFIX = ".safetensors.index.json"
SHARD_SUFFIX = ".safetensors"

# Files always required for a text-generation model.
ALWAYS_REQUIRED = {
    "config.json": "config",
    "tokenizer_config.json": "tokenizer_config",
}

# Optional-but-needed files included only when present.
OPTIONAL_FILES = {
    "generation_config.json": "generation_config",
    "special_tokens_map.json": "special_tokens_map",
    "added_tokens.json": "added_tokens",
    "chat_template.jinja": "chat_template",
    "vocab.json": "vocab",
    "merges.txt": "merges",
}


def is_lfs_oid(value: Any) -> bool:
    """True when a value is a valid 64-char hex LFS OID (a SHA-256)."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def safe_relative_path(path: str) -> bool:
    """Reject absolute paths, ``..`` segments, empty paths, and control chars."""
    if not path or not isinstance(path, str):
        return False
    if path.startswith(("/", "\\")) or ":" in path.split("/")[0]:
        return False
    if any(ord(ch) < 32 for ch in path):
        return False
    parts = path.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


class ModelFileClosure:
    """Build a deterministic file closure from a source file list + index."""

    def __init__(self, selector: Optional[ModelFileSelector] = None) -> None:
        self.selector = selector or ModelFileSelector()

    def build(
        self,
        resolved_spec: ResolvedModelSpec,
        source_files: List[Dict[str, Any]],
        index_content: Optional[Dict[str, Any]] = None,
        require_strong: bool = True,
    ) -> Tuple[ModelFilePlan, List[str]]:
        """Return ``(plan, errors)``.

        ``plan.status`` is ``planned`` on success or ``blocked`` when a
        fail-closed condition was hit (with the reasons in ``errors``).
        """
        errors: List[str] = []
        model_identity = resolved_spec.model_identity

        # 1. Validate every source path and normalize a lookup by path.
        by_path: Dict[str, Dict[str, Any]] = {}
        for item in source_files:
            path = item.get("path")
            if not safe_relative_path(path):
                errors.append("unsafe or non-relative file path: %r" % path)
                continue
            by_path[path] = item
        if errors:
            return self._blocked(model_identity), errors

        # 2. Reject unsupported formats.
        shards = [p for p in by_path if p.lower().endswith(SHARD_SUFFIX) and not p.lower().endswith(WEIGHT_INDEX_SUFFIX)]
        legacy = [p for p in by_path if self.selector.role(p) == "legacy_weight"]
        gguf = [p for p in by_path if self.selector.role(p) == "gguf_weight"]
        if gguf:
            errors.append("GGUF weights are unsupported")
        if legacy:
            errors.append(".bin/.pt/.pth/.ckpt weights are unsupported")
        if shards and legacy:
            errors.append("mixed .safetensors and legacy weight files are ambiguous")

        # 3. Find index files.
        index_paths = [p for p in by_path if p.lower().endswith(WEIGHT_INDEX_SUFFIX)]
        if len(index_paths) > 1:
            errors.append("multiple safetensors index files are ambiguous")

        weight_shard_paths: List[str] = []
        if index_paths:
            index_path = index_paths[0]
            if index_content is None:
                errors.append("safetensors index present but index content was not provided")
            else:
                weight_map = index_content.get("weight_map")
                if not isinstance(weight_map, dict):
                    errors.append("index weight_map must be an object")
                else:
                    referenced = sorted(set(weight_map.values()))
                    for shard in referenced:
                        if not safe_relative_path(shard) or not shard.lower().endswith(SHARD_SUFFIX):
                            errors.append("index references an unsafe shard path: %r" % shard)
                            continue
                        if shard not in by_path:
                            errors.append("index references a shard missing from the file list: %s" % shard)
                            continue
                        weight_shard_paths.append(shard)
        else:
            if len(shards) == 1:
                weight_shard_paths = shards
            elif len(shards) > 1:
                errors.append(
                    "multiple single-file safetensors variants without an index are ambiguous"
                )

        if errors:
            return self._blocked(model_identity), errors
        if not weight_shard_paths:
            errors.append("no safetensors weight shards resolved")
            return self._blocked(model_identity), errors

        # 4. Assemble the required file list.
        plan_files: List[Dict[str, Any]] = []

        # weight index
        if index_paths:
            plan_files.append(self._file_entry(by_path[index_paths[0]], "weight_index", required=True))

        # weight shards
        for shard in weight_shard_paths:
            plan_files.append(self._file_entry(by_path[shard], "weight_shard", required=True))

        # always-required config/tokenizer_config
        for name, role in ALWAYS_REQUIRED.items():
            if name in by_path:
                plan_files.append(self._file_entry(by_path[name], role, required=True))
            else:
                errors.append("required file missing from source list: %s" % name)

        # tokenizer (tokenizer.json preferred over tokenizer.model)
        tokenizer_files = [p for p in by_path if self.selector.role(p) == "tokenizer"]
        if "tokenizer.json" in by_path:
            plan_files.append(self._file_entry(by_path["tokenizer.json"], "tokenizer", required=True))
        elif "tokenizer.model" in by_path:
            plan_files.append(self._file_entry(by_path["tokenizer.model"], "tokenizer", required=True))
        else:
            errors.append("no tokenizer file (tokenizer.json / tokenizer.model) found")

        # optional files present in the source list
        for name, role in OPTIONAL_FILES.items():
            if name in by_path:
                plan_files.append(self._file_entry(by_path[name], role, required=False))

        for path in sorted(by_path):
            if path.lower().endswith(".tiktoken"):
                plan_files.append(self._file_entry(by_path[path], "tiktoken", required=False))

        if errors:
            return self._blocked(model_identity), errors

        # 5. Deduplicate and sort by path.
        plan_files = self._dedupe_sort(plan_files)

        # 6. Weight shard sizes must be known (needed for resource solving).
        weight_files = [f for f in plan_files if f["role"] == "weight_shard"]
        for item in weight_files:
            if not isinstance(item.get("size_bytes"), int) or item["size_bytes"] < 0:
                errors.append("weight shard %s has an unknown size" % item["path"])

        # 7. Integrity level.
        strong = all(self._effective_sha(item) for item in weight_files)
        integrity_level = "strong" if strong else "bounded"
        if require_strong and not strong:
            errors.append(
                "strong weight integrity is required but the source provides no SHA-256 for some shards"
            )

        total_size_bytes = sum(
            int(item.get("size_bytes") or 0) for item in plan_files
        )
        if errors:
            return self._blocked(model_identity), errors

        plan = ModelFilePlan(
            status="planned",
            model_identity=model_identity,
            format="safetensors",
            variant=self._variant(resolved_spec),
            files=plan_files,
            total_size_bytes=total_size_bytes,
            remaining_download_bytes=total_size_bytes,
            integrity_level=integrity_level,
        )
        plan.plan_hash = plan.compute_plan_hash()
        return plan, []

    def _file_entry(self, source: Dict[str, Any], role: str, required: bool) -> Dict[str, Any]:
        sha = source.get("sha256") or (source.get("etag") if is_lfs_oid(source.get("etag")) else None)
        return {
            "path": source.get("path"),
            "role": role,
            "size_bytes": int(source.get("size_bytes") or 0),
            "sha256": sha,
            "etag": source.get("etag"),
            "required": required,
        }

    @staticmethod
    def _effective_sha(item: Dict[str, Any]) -> Optional[str]:
        sha = item.get("sha256")
        if sha:
            return sha
        etag = item.get("etag")
        return etag if is_lfs_oid(etag) else None

    @staticmethod
    def _dedupe_sort(plan_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for item in plan_files:
            seen[item["path"]] = item
        return [seen[path] for path in sorted(seen)]

    @staticmethod
    def _variant(resolved_spec: ResolvedModelSpec) -> str:
        dtype = (resolved_spec.dtype or "float16").lower()
        if dtype in ("bfloat16", "bf16"):
            return "bf16"
        if dtype in ("float32", "float", "fp32"):
            return "fp32"
        return "fp16"

    @staticmethod
    def _blocked(model_identity: str) -> ModelFilePlan:
        return ModelFilePlan(status="blocked", model_identity=model_identity)
