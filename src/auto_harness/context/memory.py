from typing import Any, Dict, List


_ALLOWED_MEMORY_FIELDS = {
    "id",
    "stage",
    "category",
    "failure_signature",
    "symptom",
    "root_cause",
    "repair_action",
    "verification_trace_id",
    "regression_case_id",
    "verified",
    "created_at",
    "version",
}


def compact_memory_hits(
    memories: List[Dict[str, Any]],
    *,
    limit: int = 3,
    max_text_chars: int = 1000,
) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    ordered = sorted(
        memories or [],
        key=lambda item: (not bool(item.get("verified")), str(item.get("id", ""))),
    )
    for item in ordered:
        compacted = {}
        for key in _ALLOWED_MEMORY_FIELDS:
            if key not in item:
                continue
            value = item[key]
            if isinstance(value, str):
                value = value[:max_text_chars]
            compacted[key] = value
        signature = str(compacted.get("failure_signature") or compacted.get("id") or compacted)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(compacted)
        if len(result) >= limit:
            break
    return result
