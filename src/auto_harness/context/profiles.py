from typing import Dict

from auto_harness.context.models import ContextProfile


_PROFILE_CAPS: Dict[str, int] = {
    "plan": 12000,
    "replan": 10000,
    "diagnose": 10000,
    "repair": 8000,
    "env_solve": 6000,
    "runner": 6000,
    "model_prepare": 6000,
    "verify": 6000,
    "memory_curate": 8000,
    "analyze": 10000,
    "default": 8000,
}


def get_context_profile(name: str, reserved_output_tokens: int = 4096) -> ContextProfile:
    normalized = str(name or "default").strip().lower()
    if normalized not in _PROFILE_CAPS:
        normalized = "default"
    required_sections = ["instructions", "task"]
    return ContextProfile(
        name=normalized,
        version="v1",
        total_input_cap_tokens=_PROFILE_CAPS[normalized],
        reserved_output_tokens=max(1, int(reserved_output_tokens)),
        required_sections=tuple(required_sections),
        fallback_behavior="fail",
    )
