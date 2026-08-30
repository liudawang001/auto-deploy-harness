"""Deterministic selection of compact core repository evidence."""
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple


_MANIFESTS = {
    "auto-deploy.yaml",
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
    "environment.yml", "environment.yaml", "poetry.lock", "Pipfile",
    "Makefile", "package.json",
}
_DEPLOY_FILES = {
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml",
    "compose.yaml", "Procfile",
}
_ENTRYPOINTS = {
    "app.py", "main.py", "server.py", "webui.py", "demo.py",
    "gradio_app.py", "api.py",
}


class CoreEvidenceSelector:
    """Select high-value files within a deterministic character budget."""

    def __init__(self, budget_tokens: int = 12000, max_file_chars: int = 6000):
        self.budget_tokens = max(1, int(budget_tokens))
        self.max_file_chars = max(256, int(max_file_chars))

    def select(
        self,
        repo_dir: Path,
        file_tree: Iterable[str],
        read_file: Callable[[Path], Optional[str]],
        detected_signals: Dict = None,
    ) -> Dict[str, str]:
        root = Path(repo_dir)
        entrypoints = set((detected_signals or {}).get("entrypoint_candidates") or [])
        ranked: List[Tuple[int, str]] = []
        for raw in file_tree:
            rel = str(raw).replace("\\", "/")
            name = Path(rel).name
            suffix = Path(rel).suffix.lower()
            score = 0
            if rel.lower() in {"readme.md", "readme"}:
                score = 120
            elif "/" not in rel and name in _MANIFESTS:
                score = 115
            elif rel.lower() in {"deploy/entrypoint.sh", "docker/entrypoint.sh"}:
                # Container entrypoints are authoritative evidence for
                # unattended initialization flags that READMEs may omit.
                score = 114
            elif rel.lower().startswith("scripts/") and name.lower().startswith("readme"):
                score = 110
            elif name in _MANIFESTS:
                score = 105
            elif name.lower().startswith("readme"):
                # Public launch examples are often the only authoritative
                # entrypoint for packaged applications. Keep them in the
                # compact core evidence before lower-value deploy files.
                score = 99
            elif name in _DEPLOY_FILES:
                score = 95
            elif rel in entrypoints or name in _ENTRYPOINTS:
                score = 90
            elif ".github/workflows/" in rel and suffix in {".yml", ".yaml"}:
                score = 70
            elif suffix in {".py", ".toml", ".yml", ".yaml", ".cfg", ".ini"}:
                score = 20
            if score:
                ranked.append((-score, rel))
        ranked.sort()

        # A conservative four-characters-per-token estimate. Context governance
        # performs the final provider-aware token enforcement.
        remaining_chars = self.budget_tokens * 4
        selected: Dict[str, str] = {}
        for _, rel in ranked:
            if remaining_chars <= 0:
                break
            content = read_file(root / rel)
            if content is None:
                continue
            bounded = content[: min(self.max_file_chars, remaining_chars)]
            if len(content) > len(bounded):
                bounded += "\n\n[truncated]"
            selected[rel] = bounded
            remaining_chars -= len(bounded)
        return selected
