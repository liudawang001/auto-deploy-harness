"""Deterministic model resource solver (Document A Phase A6).

Pure function that turns a ResolvedModelSpec + ModelFilePlan + host facts into
an InferenceResourceDecision. It never falls back to a fixed 16GB estimate and
never auto-enables quantization or CPU offload.

All ratios and floors are named constants recorded into the decision artifact;
they are not hidden tuning knobs.
"""
from typing import Dict, List, Optional

from auto_harness.model_runtime.compatibility import dtype_element_bytes, select_dtype
from auto_harness.model_runtime.schemas import (
    InferenceResourceDecision,
    ModelFilePlan,
    ResolvedModelSpec,
)

# Named constants (recorded into the decision artifact).
WEIGHT_RUNTIME_SAFETY_FACTOR = 1.05
RUNTIME_OVERHEAD_RATIO = 0.10
RUNTIME_OVERHEAD_FLOOR_BYTES = 1 * 1024 ** 3  # 1 GiB
KV_CACHE_TENSORS = 2  # key + value
RAM_FLOOR_BYTES = 2 * 1024 ** 3  # 2 GiB host-side framework floor

BYTES_PER_GIB = 1024 ** 3


class ModelResourceSolver:
    """Solve GPU/RAM/disk requirements for one model revision."""

    def __init__(
        self,
        *,
        weight_runtime_safety_factor: float = WEIGHT_RUNTIME_SAFETY_FACTOR,
        runtime_overhead_ratio: float = RUNTIME_OVERHEAD_RATIO,
        runtime_overhead_floor_bytes: int = RUNTIME_OVERHEAD_FLOOR_BYTES,
        ram_floor_bytes: int = RAM_FLOOR_BYTES,
    ) -> None:
        self.weight_runtime_safety_factor = weight_runtime_safety_factor
        self.runtime_overhead_ratio = runtime_overhead_ratio
        self.runtime_overhead_floor_bytes = runtime_overhead_floor_bytes
        self.ram_floor_bytes = ram_floor_bytes

    def solve(
        self,
        spec: ResolvedModelSpec,
        plan: ModelFilePlan,
        config,
        host_facts: Dict,
    ) -> InferenceResourceDecision:
        """Return a decision bound to the model identity and host facts.

        ``host_facts`` keys: gpu_indexes, gpu_memory_total_bytes,
        gpu_memory_free_bytes, ram_total_bytes, ram_available_bytes,
        disk_total_bytes, disk_free_bytes.
        """
        warnings: List[str] = []
        reasons: List[str] = []
        model_identity = spec.model_identity or plan.model_identity

        if spec.status != "resolved":
            return self._decision(
                status="unsupported_model",
                model_identity=model_identity,
                reasons=["model is not in a resolved state: %s" % spec.status],
            )

        weight_bytes = self._weight_bytes(plan)
        if weight_bytes is None:
            return self._decision(
                status="uncertain",
                model_identity=model_identity,
                reasons=["weight bytes could not be determined from the file plan"],
            )
        if weight_bytes < 0:
            return self._decision(
                status="uncertain",
                model_identity=model_identity,
                reasons=["negative weight bytes"],
            )

        dtype = select_dtype(spec, getattr(config, "model_runtime_dtype", "auto"))
        bytes_per_element = dtype_element_bytes(dtype)
        weight_runtime_bytes = int(weight_bytes * self.weight_runtime_safety_factor)

        max_model_len = self._max_model_len(spec, config)
        max_num_seqs = int(getattr(config, "model_runtime_max_num_seqs", 1) or 1)
        if max_num_seqs < 1:
            max_num_seqs = 1

        kv_cache_bytes, kv_cache_known = self._kv_cache(
            spec, bytes_per_element, max_model_len, max_num_seqs
        )
        if not kv_cache_known:
            warnings.append(
                "KV cache could not be computed from missing model fields; "
                "the decision is a lower-bound estimate and is treated as uncertain"
            )

        runtime_overhead_bytes = max(
            self.runtime_overhead_floor_bytes,
            int(weight_runtime_bytes * self.runtime_overhead_ratio),
        )
        required_vram_bytes = weight_runtime_bytes + kv_cache_bytes + runtime_overhead_bytes

        required_ram_bytes = int(weight_runtime_bytes * self._ram_safety_ratio(config)) + self.ram_floor_bytes
        required_disk_bytes = int(plan.total_size_bytes * self._disk_safety_ratio(config))

        gpu_memory_utilization = float(
            getattr(config, "model_runtime_gpu_memory_utilization", 0.9) or 0.9
        )
        gpu_indexes = list(host_facts.get("gpu_indexes") or [0])
        gpu_total = int(host_facts.get("gpu_memory_total_bytes") or 0)
        gpu_free = int(host_facts.get("gpu_memory_free_bytes") or 0)
        ram_available = int(host_facts.get("ram_available_bytes") or 0)
        disk_free = int(host_facts.get("disk_free_bytes") or 0)

        usable_vram_bytes = int(gpu_free * gpu_memory_utilization)

        reasons.append(
            "weight_runtime safety factor x%.3f; overhead ratio %.2f, floor %d bytes; "
            "ram floor %d bytes"
            % (
                self.weight_runtime_safety_factor,
                self.runtime_overhead_ratio,
                self.runtime_overhead_floor_bytes,
                self.ram_floor_bytes,
            )
        )

        status = self._final_status(
            required_vram_bytes=required_vram_bytes,
            usable_vram_bytes=usable_vram_bytes,
            gpu_total=gpu_total,
            required_ram_bytes=required_ram_bytes,
            ram_available=ram_available,
            required_disk_bytes=required_disk_bytes,
            disk_free=disk_free,
            kv_cache_known=kv_cache_known,
        )

        decision = self._decision(
            status=status,
            model_identity=model_identity,
            runtime="vllm",
            gpu_indexes=gpu_indexes,
            gpu_memory_total_bytes=gpu_total,
            gpu_memory_free_bytes=gpu_free,
            weight_bytes=weight_bytes,
            weight_runtime_bytes=weight_runtime_bytes,
            kv_cache_bytes=kv_cache_bytes,
            runtime_overhead_bytes=runtime_overhead_bytes,
            required_vram_bytes=required_vram_bytes,
            usable_vram_bytes=usable_vram_bytes,
            required_ram_bytes=required_ram_bytes,
            required_disk_bytes=required_disk_bytes,
            selected_dtype=dtype,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            reasons=reasons,
            warnings=warnings,
        )
        decision.decision_hash = decision.compute_decision_hash()
        return decision

    # ---- helpers ----

    def _final_status(
        self,
        *,
        required_vram_bytes: int,
        usable_vram_bytes: int,
        gpu_total: int,
        required_ram_bytes: int,
        ram_available: int,
        required_disk_bytes: int,
        disk_free: int,
        kv_cache_known: bool,
    ) -> str:
        if not kv_cache_known:
            return "uncertain"
        if gpu_total > 0 and required_vram_bytes > gpu_total:
            return "insufficient_gpu_memory"
        if required_vram_bytes > usable_vram_bytes:
            return "gpu_busy"
        if ram_available > 0 and required_ram_bytes > ram_available:
            return "insufficient_system_memory"
        if disk_free > 0 and required_disk_bytes > disk_free:
            return "insufficient_disk"
        return "allowed"

    @staticmethod
    def _weight_bytes(plan: ModelFilePlan) -> Optional[int]:
        shards = [f for f in plan.files if f.get("role") == "weight_shard"]
        if not shards:
            return None
        sizes = [int(f.get("size_bytes") or 0) for f in shards]
        if any(size < 0 for size in sizes):
            return -1
        return sum(sizes)

    @staticmethod
    def _max_model_len(spec: ResolvedModelSpec, config) -> int:
        configured = int(getattr(config, "model_runtime_max_model_len", 4096) or 4096)
        model_max = spec.max_position_embeddings or configured
        return max(1, min(configured, model_max))

    @staticmethod
    def _kv_cache(
        spec: ResolvedModelSpec, bytes_per_element: int, max_model_len: int, max_num_seqs: int
    ):
        head_dim = spec.head_dim
        if head_dim is None and spec.hidden_size and spec.num_attention_heads:
            if spec.num_attention_heads <= 0:
                return 0, False
            head_dim = spec.hidden_size // spec.num_attention_heads
        num_kv_heads = spec.num_key_value_heads or spec.num_attention_heads
        if not (head_dim and spec.num_hidden_layers and num_kv_heads):
            return 0, False
        value = (
            KV_CACHE_TENSORS
            * spec.num_hidden_layers
            * num_kv_heads
            * head_dim
            * bytes_per_element
            * max_model_len
            * max_num_seqs
        )
        return int(value), True

    @staticmethod
    def _ram_safety_ratio(config) -> float:
        return float(getattr(config, "model_runtime_ram_safety_ratio", 1.2) or 1.2)

    @staticmethod
    def _disk_safety_ratio(config) -> float:
        return float(getattr(config, "model_runtime_disk_safety_ratio", 1.2) or 1.2)

    @staticmethod
    def _decision(**kwargs) -> InferenceResourceDecision:
        return InferenceResourceDecision(**kwargs)
