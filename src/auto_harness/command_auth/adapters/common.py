"""Shared static parsing helpers."""

import re
import shlex
from pathlib import Path
from typing import Dict, List


_FORBIDDEN_INLINE = (";", "&&", "||", "|", "`", "$(", ">", "<")


def read_text(repo_dir: Path, relative: str, limit: int = 1_000_000) -> str:
    path = Path(repo_dir) / relative
    if path.is_symlink() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def readme_commands(repo_dir: Path, file_tree: List[str]) -> List[Dict]:
    commands = []
    for relative in file_tree:
        normalized = relative.replace("\\", "/")
        name = Path(normalized).name.lower()
        if not (
            name.startswith("readme")
            or normalized.lower() in {"deploy/entrypoint.sh", "docker/entrypoint.sh"}
        ):
            continue
        content = read_text(repo_dir, relative)
        for line_number, raw in enumerate(content.splitlines(), start=1):
            line = raw.strip().strip("`").strip()
            line = re.split(r"\s+#\s+", line, maxsplit=1)[0].strip()
            if not line or any(token in line for token in _FORBIDDEN_INLINE):
                continue
            try:
                argv = shlex.split(line, posix=True)
            except ValueError:
                continue
            if argv and all(isinstance(item, str) and "\x00" not in item for item in argv):
                commands.append({"argv": argv, "path": relative, "line": line_number})
    return commands
