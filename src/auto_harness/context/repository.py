import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List


_DEPENDENCY_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    "poetry.lock",
    "conda.yml",
}


class RepoEvidenceSelector:
    def select(
        self,
        selected_files: Dict[str, Any],
        failure_context: Any,
        *,
        max_files: int = 4,
        max_chars: int = 2500,
    ) -> Dict[str, Dict[str, Any]]:
        failure_text = _bounded_failure_text(failure_context)
        stack_locations = _stack_locations(failure_text)
        stack_paths = set(stack_locations)
        failure_tokens = _failure_tokens(failure_text)
        ranked = []
        for path, value in (selected_files or {}).items():
            normalized = str(path).replace("\\", "/")
            content = _content(value)
            searchable_content = _bounded_text(content, 20000).lower()
            score = 0
            reason = []
            if normalized in stack_paths or any(normalized.endswith(item) for item in stack_paths):
                score += 100
                reason.append("failure_stack")
            if Path(normalized).name in _DEPENDENCY_FILES:
                score += 60
                reason.append("dependency_file")
            if Path(normalized).name in {"app.py", "main.py", "server.py", "demo.py"}:
                score += 40
                reason.append("entrypoint")
            for token in failure_tokens:
                if token and token.lower() in searchable_content:
                    score += 2
            ranked.append((-score, normalized, content, ",".join(reason) or "priority_file"))
        ranked.sort(key=lambda item: (item[0], item[1]))

        result = {}
        for _, path, content, reason in ranked[:max_files]:
            target_line = stack_locations.get(path)
            if target_line is None:
                for stack_path, line_number in stack_locations.items():
                    if path.endswith(stack_path):
                        target_line = line_number
                        break
            if target_line is None:
                target_line = _matching_line(content, failure_tokens)
            snippet, line_start, line_end, truncated = _line_snippet(
                content, max_chars, target_line
            )
            result[path] = {
                "path": path,
                "content": snippet,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "reason": reason,
                "line_start": line_start,
                "line_end": line_end,
                "truncated": truncated,
            }
        return result


def safe_repo_path(repo_dir: Path, relative_path: str) -> Path:
    root = Path(repo_dir).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("repository path escapes root: %s" % relative_path)
    return candidate


def _content(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value or "")


def _bounded_failure_text(value: Any, max_chars: int = 20000) -> str:
    parts = []
    remaining = max_chars

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal remaining
        if remaining <= 0 or depth > 3:
            return
        if isinstance(item, str):
            text = _bounded_text(item, min(remaining, 8000))
            parts.append(text)
            remaining -= len(text)
            return
        if isinstance(item, dict):
            preferred = (
                "error",
                "error_type",
                "message",
                "summary",
                "stderr",
                "stdout",
                "traceback",
                "log",
                "logs",
                "output",
            )
            visited = set()
            for key in preferred:
                if key in item:
                    visited.add(key)
                    visit(item[key], depth + 1)
            for key, child in list(item.items())[:20]:
                if key not in visited:
                    visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in list(item)[:20]:
                visit(child, depth + 1)
            return
        text = _bounded_text(str(item or ""), min(remaining, 1000))
        parts.append(text)
        remaining -= len(text)

    visit(value)
    return "\n".join(parts)


def _bounded_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    head = max_chars // 2
    return value[:head] + "\n...\n" + value[-(max_chars - head) :]


def _failure_tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{3,}", text)
        if token.lower() not in {"error", "failed", "traceback", "exception"}
    ][:20]


def _stack_locations(text: str) -> Dict[str, int]:
    result = {}
    pattern = re.compile(r"""File\s+["']([^"']+\.py)["']\s*,\s*line\s+(\d+)""")
    for path, line_number in pattern.findall(text):
        result[path.replace("\\", "/")] = max(1, int(line_number))
    for path in re.findall(r"[\w./\\-]+\.py", text):
        result.setdefault(path.replace("\\", "/"), 1)
    return result


def _matching_line(content: str, tokens: List[str]) -> int:
    lowered_tokens = [token.lower() for token in tokens]
    for index, line in enumerate(content.splitlines(), start=1):
        lowered = line.lower()
        if any(token in lowered for token in lowered_tokens):
            return index
    return 1


def _line_snippet(text: str, max_chars: int, target_line: int):
    lines = text.splitlines()
    if not lines:
        return "", 1, 1, False
    target_index = min(max(0, int(target_line or 1) - 1), len(lines) - 1)
    start = max(0, target_index - 20)
    selected = []
    used = 0
    for line in lines[start:]:
        available = max_chars - used
        if available <= 0:
            break
        selected.append(line[:available])
        used += min(len(line), available) + 1
        if used >= max_chars:
            break
    end = start + len(selected)
    return "\n".join(selected), start + 1, max(start + 1, end), end < len(lines)
