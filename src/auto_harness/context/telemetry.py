from auto_harness.context.budget import context_telemetry
from auto_harness.context.tokens import normalize_usage


def safe_context_telemetry(value):
    if not isinstance(value, dict):
        return {}

    def normalize(item, depth=0):
        if depth > 6:
            return None
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, dict):
            return {
                str(key): normalized
                for key, child in item.items()
                if (normalized := normalize(child, depth + 1)) is not None
            }
        if isinstance(item, (list, tuple)):
            return [
                normalized
                for child in item
                if (normalized := normalize(child, depth + 1)) is not None
            ]
        return None

    return normalize(value) or {}


__all__ = ["context_telemetry", "normalize_usage", "safe_context_telemetry"]
