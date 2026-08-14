"""Deterministic model-runtime compatibility decisions (Document A Phase A6).

Small pure helpers shared by the resource solver and later the vLLM adapter
(Document B): dtype selection and quantization/remote-code gating.
"""
from typing import Dict, List, Optional

from auto_harness.model_runtime.schemas import ResolvedModelSpec

DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "half": 2,
    "float32": 4,
    "float": 4,
}

SUPPORTED_DTYPES = ("float16", "bfloat16", "float32")


def select_dtype(spec: ResolvedModelSpec, config_dtype: Optional[str] = None) -> str:
    """Pick a deterministic runtime dtype from spec then config.

    Preference order: the model's declared dtype (float16/bfloat16/float32),
    then an explicit config dtype, then float16.
    """
    spec_dtype = (spec.dtype or "").lower()
    if spec_dtype in SUPPORTED_DTYPES:
        return spec_dtype
    configured = (config_dtype or "auto").lower()
    if configured in SUPPORTED_DTYPES:
        return configured
    return "float16"


def dtype_element_bytes(dtype: str) -> int:
    """Bytes per element for a runtime dtype; defaults to 2 (16-bit)."""
    return int(DTYPE_BYTES.get((dtype or "float16").lower(), 2))


def runtime_blockers(spec: ResolvedModelSpec) -> List[str]:
    """Return non-empty list when the model cannot be served by the runtime."""
    blockers = []
    if spec.quantization:
        blockers.append("quantized model unsupported: %s" % spec.quantization)
    if spec.requires_remote_code:
        blockers.append("model requires remote code")
    return blockers
