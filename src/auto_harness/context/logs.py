import json
import re
from typing import Any, Dict, List


_ERROR_PATTERN = re.compile(
    r"(traceback|error|exception|failed|fatal|out of memory|oom|killed)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9_]{12,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(
        r"(?i)((?:api_key|api_secret|password|token|secret)\s*[=:]\s*)[^\s'\"\n]+"
    ),
)


class LogCompactor:
    def compact(
        self,
        text: Any,
        *,
        max_chars: int = 4000,
        exit_code=None,
    ) -> Dict[str, Any]:
        sanitized = _redact_secrets(str(text or ""))
        lines = [_clean_line(line) for line in sanitized.splitlines()]
        lines = [line for line in lines if line]
        first_error = next((line for line in lines if _ERROR_PATTERN.search(line)), "")
        important = []
        counts: Dict[str, int] = {}
        for line in lines:
            if _ERROR_PATTERN.search(line):
                counts[line] = counts.get(line, 0) + 1
                if line not in important and len(important) < 12:
                    important.append(line)
        head = lines[:12]
        tail = lines[-30:]
        payload = {
            "exit_code": exit_code,
            "first_error_line": first_error[:1000],
            "important_lines": [line[:1000] for line in important],
            "head": [line[:1000] for line in head],
            "stack_tail": [line[:1000] for line in tail],
            "repeated_patterns": [
                {"pattern": line[:500], "count": count}
                for line, count in counts.items()
                if count > 1
            ][:10],
            "omitted_line_count": max(0, len(lines) - len(set(head + tail))),
            "truncated": False,
        }
        serialized = str(payload)
        if len(serialized) > max_chars:
            payload["head"] = payload["head"][:5]
            payload["stack_tail"] = payload["stack_tail"][-15:]
            payload["important_lines"] = payload["important_lines"][:6]
            payload["truncated"] = True
        return _fit_payload(payload, max(0, int(max_chars)))


def _clean_line(line: str) -> str:
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
    return "".join(ch for ch in line if ch.isprintable())[:4000]


def _redact_secrets(text: str) -> str:
    result = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda match: match.group(1) + "[REDACTED_SECRET]", result)
        else:
            result = pattern.sub("[REDACTED_SECRET]", result)
    return result


def _fit_payload(payload: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """Fit the serialized payload inside max_chars without losing its shape.

    The caller treats max_chars as a hard boundary.  Reducing list counts once
    is not sufficient because individual log lines can still be large.
    """
    if max_chars <= 0:
        return {}

    def size(value) -> int:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True))

    if size(payload) <= max_chars:
        return payload

    result = dict(payload)
    result["truncated"] = True
    for line_cap, head_count, tail_count, important_count, repeat_count in (
        (500, 4, 10, 5, 5),
        (240, 2, 6, 3, 3),
        (120, 1, 3, 2, 1),
        (60, 0, 2, 1, 0),
    ):
        result["first_error_line"] = str(
            result.get("first_error_line", "")
        )[:line_cap]
        result["head"] = [
            str(line)[:line_cap]
            for line in (result.get("head") or [])[:head_count]
        ]
        result["stack_tail"] = [
            str(line)[:line_cap]
            for line in (result.get("stack_tail") or [])[-tail_count:]
        ] if tail_count else []
        result["important_lines"] = [
            str(line)[:line_cap]
            for line in (result.get("important_lines") or [])[:important_count]
        ]
        result["repeated_patterns"] = [
            {
                "pattern": str(item.get("pattern", ""))[:line_cap],
                "count": item.get("count", 0),
            }
            for item in (result.get("repeated_patterns") or [])[:repeat_count]
            if isinstance(item, dict)
        ]
        if size(result) <= max_chars:
            return result

    minimal = {
        "exit_code": result.get("exit_code"),
        "first_error_line": "",
        "omitted_line_count": result.get("omitted_line_count", 0),
        "truncated": True,
    }
    available = max_chars - size(minimal)
    if available > 0:
        minimal["first_error_line"] = str(
            result.get("first_error_line", "")
        )[:available]
        while minimal["first_error_line"] and size(minimal) > max_chars:
            minimal["first_error_line"] = minimal["first_error_line"][:-1]
    if size(minimal) <= max_chars:
        return minimal
    fallback = {"truncated": True}
    return fallback if size(fallback) <= max_chars else {}
