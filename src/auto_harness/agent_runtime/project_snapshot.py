"""Project snapshot builder for LLM plan-first deployment.

Collects project file tree, selected files, detected signals, memory hits,
and performs secret redaction. The snapshot is the input to LLMDeploymentPlanner.
"""
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent.safety import AgentInputSanitizer


# Priority files to read first (in order of importance)
PRIORITY_FILES = (
    "README.md",
    "readme.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "environment.yml",
    "environment.yaml",
    "Dockerfile",
    "app.py",
    "main.py",
    "server.py",
    "webui.py",
    "demo.py",
    "gradio_app.py",
    "api.py",
)

# Skip these directories entirely
SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", ".venv", "venv"})

# Skip files with these extensions (binaries, weights, caches)
SKIP_EXTENSIONS = frozenset({
    ".bin", ".pth", ".pt", ".onnx", ".safetensors", ".gguf", ".ckpt",
    ".so", ".dylib", ".dll", ".exe", ".o", ".a",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pyc", ".pyo", ".egg", ".whl",
})


class ProjectSnapshotBuilder:
    """Builds a redacted project snapshot for LLM consumption."""

    def __init__(
        self,
        max_files: int = 80,
        max_file_chars: int = 6000,
        max_tree_entries: int = 20000,
    ) -> None:
        self.max_files = max_files
        self.max_file_chars = max_file_chars
        self.max_tree_entries = max_tree_entries
        self._last_total_file_count = 0

    def build(
        self,
        repo_dir: Path,
        task_id: str = "",
        memory_hits: Optional[List[Dict]] = None,
        selected_skills: Optional[List[Dict]] = None,
        skill_context: Optional[Dict] = None,
    ) -> Dict:
        """Build a project snapshot dict.

        Args:
            repo_dir: Repository directory.
            task_id: Task identifier.
            memory_hits: Optional memory hits for context.
            selected_skills: Optional list of selected skill dicts.
            skill_context: Optional skill context from SkillContextBuilder.
        """
        repo_dir = Path(repo_dir)
        memory_hits = memory_hits or []
        selected_skills = selected_skills or []
        skill_context = skill_context or {}

        # 1. Collect file tree
        file_tree = self._collect_file_tree(repo_dir)

        # 2. Select and read priority files
        selected_files = self._select_files(repo_dir, file_tree)

        # 3. Detect signals
        detected_signals = self._detect_signals(repo_dir, file_tree, selected_files)

        # 4. Redact secrets
        sanitizer = AgentInputSanitizer()
        sanitized_files = sanitizer.sanitize_selected_files(selected_files)

        # 5. Compute sha256 for each file
        file_digests = {}
        for name, content in selected_files.items():
            file_digests[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Build selected_files with metadata
        selected_with_meta = {}
        for name in sanitized_files:
            selected_with_meta[name] = {
                "path": name,
                "content": sanitized_files[name],
                "sha256": file_digests.get(name, ""),
            }

        return {
            "task_id": task_id,
            "repo_dir": str(repo_dir),
            "file_tree": file_tree,
            "file_tree_summary": {
                "total_file_count": self._last_total_file_count,
                "omitted_file_count": max(
                    0, self._last_total_file_count - len(file_tree)
                ),
                "truncated": self._last_total_file_count > len(file_tree),
            },
            "selected_files": selected_with_meta,
            "detected_signals": detected_signals,
            "memory_hits": memory_hits,
            "selected_skills": selected_skills,
            "skill_context": skill_context,
            "redactions": sanitizer.redactions,
            "untrusted_content_risks": sanitizer.risks,
        }

    def _collect_file_tree(self, repo_dir: Path) -> List[str]:
        """Collect the full file tree, skipping .git and binary dirs."""
        result: List[str] = []
        seen = set()
        self._last_total_file_count = 0
        for name in PRIORITY_FILES:
            path = repo_dir / name
            if path.is_file() and path.suffix.lower() not in SKIP_EXTENSIONS:
                result.append(name)
                seen.add(name)
        for path in sorted(repo_dir.rglob("*")):
            if path.is_dir():
                continue
            # Skip files inside skipped directories
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            # Skip binary/weight files by extension
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue
            try:
                rel = str(path.relative_to(repo_dir))
            except ValueError:
                continue
            if rel in seen:
                continue
            self._last_total_file_count += 1
            if len(result) < self.max_tree_entries:
                result.append(rel)
                seen.add(rel)
        self._last_total_file_count += len(
            [name for name in result if name in PRIORITY_FILES]
        )
        return result

    def _select_files(self, repo_dir: Path, file_tree: List[str]) -> Dict[str, str]:
        """Select priority files and read their content."""
        selected: Dict[str, str] = {}
        file_set = set(file_tree)

        # Read priority files first
        for name in PRIORITY_FILES:
            if (name in file_set or (repo_dir / name).is_file()) and len(selected) < self.max_files:
                path = repo_dir / name
                content = self._read_file(path)
                if content is not None:
                    selected[name] = content

        # Then read remaining .py, .yml, .yaml, .toml, .txt, .md files
        for rel in file_tree:
            if len(selected) >= self.max_files:
                break
            if rel in selected:
                continue
            if Path(rel).suffix.lower() not in (
                ".py", ".yml", ".yaml", ".toml", ".txt", ".md", ".cfg", ".ini", ".json",
            ):
                continue
            # Skip files in subdirectories if we already have enough
            if "/" in rel and len(selected) >= self.max_files // 2:
                continue
            path = repo_dir / rel
            content = self._read_file(path)
            if content is not None:
                selected[rel] = content

        return selected

    def _read_file(self, path: Path) -> Optional[str]:
        """Read a single file, truncated to max_file_chars."""
        try:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        if len(text) > self.max_file_chars:
            text = text[: self.max_file_chars] + "\n\n[truncated]"
        return text

    def _detect_signals(
        self,
        repo_dir: Path,
        file_tree: List[str],
        selected_files: Dict[str, str],
    ) -> Dict:
        """Detect frameworks, entrypoint candidates, dependency files, model mentions, ports."""
        file_set = set(file_tree)
        all_text = "\n".join(selected_files.values()).lower()

        # Framework detection (reuse analyzer patterns)
        frameworks: List[str] = []
        for key in ("gradio", "streamlit", "fastapi", "flask", "torch", "transformers", "vllm"):
            if key in all_text:
                frameworks.append(key)
        if "httpserver" in all_text or "basehttprequesthandler" in all_text or "from http.server" in all_text:
            frameworks.append("http.server")
        if "openai-compatible" in all_text or "openai compatible" in all_text:
            frameworks.append("openai_compatible")

        # Entrypoint candidates
        entrypoint_candidates = [
            name for name in ("app.py", "main.py", "server.py", "webui.py", "demo.py", "gradio_app.py", "api.py")
            if name in file_set
        ]

        # Dependency files
        dependency_files = [
            name for name in ("requirements.txt", "pyproject.toml", "setup.py", "environment.yml", "environment.yaml")
            if name in file_set
        ]

        # Model mentions (HuggingFace / ModelScope references)
        model_mentions: List[str] = []
        model_pattern = re.compile(
            r'["\']([A-Za-z0-9_\-/]+(?:/[A-Za-z0-9_\-]+)+)["\']',
        )
        for line in all_text.splitlines():
            if "from_pretrained" in line or "modelscope" in line:
                for match in model_pattern.finditer(line):
                    candidate = match.group(1)
                    if "/" in candidate and not candidate.startswith(".") and len(candidate) < 200:
                        model_mentions.append(candidate)

        # Port detection - look for common port patterns in source
        ports: List[int] = []
        # Match: HTTPServer(('host', PORT)), .run(port=PORT), port=PORT, :PORT
        port_patterns = [
            re.compile(r"HTTPServer\(\s*\(\s*['\"][^'\"]*['\"]\s*,\s*(\d{2,5})\s*\)", re.IGNORECASE),
            re.compile(r"(?:port\s*[=:]\s*)(\d{2,5})", re.IGNORECASE),
            re.compile(r"uvicorn\.run\([^)]*port\s*=\s*(\d{2,5})", re.IGNORECASE),
        ]
        for pattern in port_patterns:
            for match in pattern.finditer(all_text):
                try:
                    port = int(match.group(1))
                    if 1024 <= port <= 65535 and port not in ports:
                        ports.append(port)
                except (ValueError, TypeError):
                    continue

        # Other signals
        has_dockerfile = "Dockerfile" in file_set
        has_environment_yml = "environment.yml" in file_set or "environment.yaml" in file_set

        return {
            "frameworks": sorted(set(frameworks)),
            "entrypoint_candidates": entrypoint_candidates,
            "dependency_files": dependency_files,
            "model_mentions": model_mentions[:10],
            "ports": ports[:5],
            "has_dockerfile": has_dockerfile,
            "has_environment_yml": has_environment_yml,
        }
