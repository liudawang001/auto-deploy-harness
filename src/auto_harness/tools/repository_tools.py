"""Portable, bounded, read-only repository observation handlers."""
import hashlib
from pathlib import Path
from typing import Dict, List

from auto_harness.agent.safety import AgentInputSanitizer
from auto_harness.agent_runtime.schemas import ToolResult
from auto_harness.context.repository import safe_repo_path
from auto_harness.tools.repository_policy import RepositoryReadPolicy


_DEPENDENCY_NAMES = {
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "environment.yml", "environment.yaml", "poetry.lock", "Pipfile",
}


def inspect_repo_tree(tool_input: Dict, context: Dict) -> ToolResult:
    root = Path(context["repo_dir"]).resolve()
    base = safe_repo_path(root, tool_input["path"])
    entries = []
    max_depth = int(tool_input["max_depth"])
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            continue
        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
            depth = len(path.relative_to(base).parts)
        except ValueError:
            continue
        if depth > max_depth or not RepositoryReadPolicy.path_allowed(rel):
            continue
        if not RepositoryReadPolicy.glob_matches(rel, tool_input["path_glob"]):
            continue
        entries.append({"path": rel, "type": "directory" if path.is_dir() else "file"})
        if len(entries) >= int(tool_input["max_entries"]):
            break
    return _passed("inspect_repo_tree", {"entries": entries, "truncated": len(entries) >= int(tool_input["max_entries"])})


def search_repo(tool_input: Dict, context: Dict) -> ToolResult:
    root = Path(context["repo_dir"]).resolve()
    config = context.get("config")
    max_files = _cfg(config, "agent_repo_search_max_files", 5000)
    max_bytes = _cfg(config, "agent_repo_search_max_bytes", 50_000_000)
    query = tool_input["query"]
    needle = query if tool_input["case_sensitive"] else query.lower()
    results = []
    scanned_files = 0
    scanned_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            continue
        if not RepositoryReadPolicy.path_allowed(rel):
            continue
        try:
            safe_repo_path(root, rel)
        except ValueError:
            continue
        if not RepositoryReadPolicy.glob_matches(rel, tool_input["path_glob"]):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _cfg(config, "agent_repo_max_chars_per_read", 12000) * 20:
            continue
        if scanned_files >= max_files or scanned_bytes + size > max_bytes:
            break
        scanned_files += 1
        scanned_bytes += size
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if tool_input["case_sensitive"] else line.lower()
            if needle in haystack:
                scan = AgentInputSanitizer().scan_text(line[:1000])
                results.append({
                    "path": rel,
                    "line": line_number,
                    "snippet": scan["text"],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "untrusted_content_risks": scan["risks"],
                })
                if len(results) >= int(tool_input["max_results"]):
                    return _passed("search_repo", {"results": results, "scanned_files": scanned_files, "scanned_bytes": scanned_bytes, "truncated": True})
    return _passed("search_repo", {"results": results, "scanned_files": scanned_files, "scanned_bytes": scanned_bytes, "truncated": False})


def read_selected_files(tool_input: Dict, context: Dict) -> ToolResult:
    root = Path(context["repo_dir"]).resolve()
    max_chars = _cfg(context.get("config"), "agent_repo_max_chars_per_read", 12000)
    results: List[Dict] = []
    errors: List[Dict] = []
    for item in tool_input["files"]:
        try:
            path = safe_repo_path(root, item["path"])
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            confirmation = path.read_bytes()
        except OSError as exc:
            errors.append({
                "path": item["path"],
                "error": "repository read failed: %s" % str(exc)[:120],
            })
            continue
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or hashlib.sha256(raw).digest() != hashlib.sha256(confirmation).digest()
        ):
            return _failed("read_selected_files", "changed_during_read")
        if b"\x00" in raw:
            return _failed("read_selected_files", "binary repository file is denied")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return _failed("read_selected_files", "non-UTF-8 repository file is denied")
        lines = text.splitlines()
        start = min(max(1, int(item["start_line"])), max(1, len(lines)))
        end = min(int(item["end_line"]), len(lines))
        content = "\n".join(lines[start - 1:end])
        # A caller-requested line window is complete evidence for that window;
        # it is not "truncated" merely because the file has later lines, nor
        # when the requested window extends beyond EOF and all remaining lines
        # were returned. Mark only actual byte clipping.
        truncated = len(content) > max_chars
        content = content[:max_chars]
        scan = AgentInputSanitizer().scan_text(content)
        results.append({
            "path": item["path"],
            "line_start": start,
            "line_end": max(start, end),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content": scan["text"],
            "truncated": truncated,
            "redactions": scan["redactions"],
            "untrusted_content_risks": scan["risks"],
            "trust_level": "untrusted_repository",
        })
    if not results:
        detail = errors[0]["error"] if errors else "no readable repository files"
        return _failed("read_selected_files", detail)
    return _passed("read_selected_files", {"files": results, "errors": errors})


def parse_dependency_files(tool_input: Dict, context: Dict) -> ToolResult:
    root = Path(context["repo_dir"]).resolve()
    paths = tool_input.get("paths") or [
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name in _DEPENDENCY_NAMES
        and RepositoryReadPolicy.path_allowed(
            str(path.relative_to(root)).replace("\\", "/")
        )
    ][:16]
    files = [{"path": path, "start_line": 1, "end_line": 400} for path in paths]
    if not files:
        return _passed("parse_dependency_files", {"files": [], "dependency_files": []})
    result = read_selected_files({"files": files}, context)
    if result.status != "passed":
        result.tool_name = "parse_dependency_files"
        return result
    evidence = result.evidence
    evidence["dependency_files"] = [item["path"] for item in evidence["files"]]
    result.tool_name = "parse_dependency_files"
    return result


def _cfg(config, name: str, default: int) -> int:
    if isinstance(config, dict):
        return int(config.get(name, default))
    return int(getattr(config, name, default))


def _passed(name: str, evidence: Dict) -> ToolResult:
    evidence = dict(evidence)
    evidence.setdefault("trust_level", "untrusted_repository")
    return ToolResult(status="passed", tool_name=name, category="read_only", policy_allowed=True, executed=True, metadata_only=True, evidence=evidence)


def _failed(name: str, error: str) -> ToolResult:
    return ToolResult(status="failed", tool_name=name, category="read_only", policy_allowed=True, executed=True, metadata_only=True, error=error)
