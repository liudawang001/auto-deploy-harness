import ast
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from auto_harness.assets.manifest import ModelAsset
from auto_harness.model_runtime.schemas import ModelReferenceCandidate


class ModelAssetDetector:
    HF_REPO = re.compile(r"(?:huggingface\.co/|hf\.co/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
    FROM_PRETRAINED = re.compile(r"from_pretrained\(\s*['\"]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]")
    HF_SNAPSHOT = re.compile(r"snapshot_download\(\s*repo_id\s*=\s*['\"]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]")
    MODELSCOPE = re.compile(r"(?:modelscope\.cn/models/|model_id\s*=\s*['\"])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
    SIZE_HINT = re.compile(r"(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)", re.IGNORECASE)

    def detect(self, repo_dir: Path, analysis: Dict = None) -> List[ModelAsset]:
        seen: Set[str] = set()
        assets: List[ModelAsset] = []
        for rel_path, text in self._candidate_texts(repo_dir):
            for repo_id in self._find_huggingface_ids(text):
                key = "huggingface:%s" % repo_id
                if key in seen:
                    continue
                seen.add(key)
                assets.append(
                    ModelAsset(
                        asset_id=key,
                        source="huggingface",
                        repo_id=repo_id,
                        origin=rel_path,
                        expected_size_bytes=self._size_hint(text),
                    )
                )
            for repo_id in self._find_modelscope_ids(text):
                key = "modelscope:%s" % repo_id
                if key in seen:
                    continue
                seen.add(key)
                assets.append(
                    ModelAsset(
                        asset_id=key,
                        source="modelscope",
                        repo_id=repo_id,
                        origin=rel_path,
                        expected_size_bytes=self._size_hint(text),
                    )
                )
        return assets

    def _candidate_texts(self, repo_dir: Path) -> Iterable:
        names = {"README.md", "readme.md", "app.py", "main.py", "server.py", "webui.py", "demo.py", "config.json"}
        for path in sorted(repo_dir.rglob("*")):
            if path.is_dir() or ".git" in path.parts:
                continue
            if path.name not in names and path.suffix not in (".py", ".md", ".json", ".yaml", ".yml"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            yield str(path.relative_to(repo_dir)), text

    def _find_huggingface_ids(self, text: str) -> List[str]:
        found = []
        for pattern in (self.HF_REPO, self.FROM_PRETRAINED, self.HF_SNAPSHOT):
            found.extend(match.group(1).strip("/") for match in pattern.finditer(text))
        return self._dedupe(found)

    def _find_modelscope_ids(self, text: str) -> List[str]:
        return self._dedupe(match.group(1).strip("/") for match in self.MODELSCOPE.finditer(text))

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            value = value.strip().strip(".,);]")
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _size_hint(self, text: str):
        match = self.SIZE_HINT.search(text)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = 1024 * 1024 * 1024 if unit in ("gb", "gib") else 1024 * 1024
        return int(value * multiplier)


# ---------------------------------------------------------------------------
# Model reference discovery (Document A Phase A2).
#
# Produces grounded ModelReferenceCandidate objects from repository evidence:
#   - structured config files (JSON/YAML/TOML) key/value pairs
#   - Python AST literals (from_pretrained / snapshot_download / argparse)
#   - README startup commands and model links
#
# It never executes Python, never evaluates function calls, and never reads
# credentials (.env), git internals, binary weights, or files outside the repo.
# ---------------------------------------------------------------------------

# Two-segment source id, e.g. "org/name". Digits, letters, dots, dashes, underscore.
VALID_REPO_ID = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")

# URL forms
HF_URL = re.compile(
    r"(?:https?://)?(?:huggingface\.co|hf\.co)/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)
MODELSCOPE_URL = re.compile(
    r"(?:https?://)?(?:www\.)?modelscope\.cn/(?:api/v1/)?models/"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)

# config key = value patterns (JSON/YAML/TOML tolerant)
CONFIG_KEY = re.compile(
    r"[\"']?(model_id|model_name|model_name_or_path|pretrained_model_name_or_path|"
    r"checkpoint|checkpoint_path|base_model|base_model_name_or_path|repo_id)"
    r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']?"
)
CONFIG_ACCESSORY_KEY = re.compile(
    r"[\"']?(tokenizer_name|tokenizer_name_or_path|embedding_model|embed_model|"
    r"reranker_model|vae_model|encoder_model|vision_model)"
    r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']?"
)

# Python call sites
FROM_PRETRAINED_CALL = re.compile(
    r"from_pretrained\(\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']"
)
SNAPSHOT_DOWNLOAD_CALL = re.compile(
    r"snapshot_download\(\s*repo_id\s*=\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']"
)
MODELSCOPE_SNAPSHOT_CALL = re.compile(
    r"snapshot_download\(\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']"
)

# CLI --model literal / argparse default
CLI_MODEL = re.compile(
    r"--model(?:-name|-name-or-path)?\s+[\"']?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']?"
)

# Files we refuse to read.
_FORBIDDEN_FILENAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    ".gitignore", ".npmrc",
}
_CONFIG_SUFFIXES = (".json", ".yaml", ".yml", ".toml")
_PYTHON_SUFFIXES = (".py",)
_README_NAMES = {"README.md", "readme.md", "README.rst", "README"}


class ModelReferenceDetector:
    """Discover grounded model reference candidates from a repository."""

    def detect(self, repo_dir: Path) -> List[ModelReferenceCandidate]:
        repo_dir = Path(repo_dir)
        by_key: Dict[str, ModelReferenceCandidate] = {}

        for rel_path, text in self._candidate_texts(repo_dir):
            file_sha = self._sha256_text(text)
            line_lookup = self._line_lookup(text)
            for raw in self._scan_file(rel_path, text):
                if not is_valid_repo_id(raw["repo_id"]):
                    continue
                key = "%s:%s" % (raw["source"], raw["repo_id"])
                existing = by_key.get(key)
                if existing is not None:
                    self._merge(existing, raw)
                else:
                    candidate = ModelReferenceCandidate(
                        source=raw["source"],
                        repo_id=raw["repo_id"],
                        requested_revision="main",
                        role=raw["role"],
                        confidence=raw["confidence"],
                        discovered_by=raw["discovered_by"],
                    )
                    by_key[key] = candidate
                candidate = by_key[key]
                evidence = self._evidence(
                    rel_path, text, raw["span"], file_sha, line_lookup,
                    raw.get("expression", ""),
                )
                if evidence not in candidate.evidence:
                    candidate.evidence.append(evidence)

        return sorted(
            by_key.values(),
            key=lambda c: (-c.confidence, c.source, c.repo_id),
        )

    def _scan_file(self, rel_path: str, text: str) -> List[Dict]:
        name = rel_path.lower()
        if name.endswith(_CONFIG_SUFFIXES):
            return self._scan_config(rel_path, text)
        if name.endswith(_PYTHON_SUFFIXES):
            return self._scan_python(rel_path, text)
        if Path(rel_path).name in _README_NAMES:
            return self._scan_readme(rel_path, text)
        return []

    def _scan_config(self, rel_path: str, text: str) -> List[Dict]:
        results: List[Dict] = []
        for match in CONFIG_ACCESSORY_KEY.finditer(text):
            results.append(self._raw(
                self._source_from_text(match.group(0), "huggingface"),
                match.group(2), "model_accessory", 0.3,
                "deterministic_config_parser", match, match.group(0).strip(),
            ))
        for match in CONFIG_KEY.finditer(text):
            results.append(self._raw(
                self._source_from_text(match.group(0), "huggingface"),
                match.group(2), "primary_generation_model", 0.95,
                "deterministic_config_parser", match, match.group(0).strip(),
            ))
        # URLs inside config (e.g. model_id pointing to a HF URL)
        for match in HF_URL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.95,
                "deterministic_config_parser", match, match.group(0).strip(),
            ))
        for match in MODELSCOPE_URL.finditer(text):
            results.append(self._raw(
                "modelscope", match.group(1), "primary_generation_model", 0.95,
                "deterministic_config_parser", match, match.group(0).strip(),
            ))
        return results

    def _scan_python(self, rel_path: str, text: str) -> List[Dict]:
        results: List[Dict] = []
        constants = self._module_string_constants(text)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("from_pretrained", "snapshot_download"):
                        repo_id = self._first_str_arg(node.args) or self._kwarg(node.keywords, "repo_id") or self._kwarg(node.keywords, "model_id")
                        if not repo_id and node.args:
                            repo_id = self._resolve_constant(node.args[0], constants)
                        if repo_id and VALID_REPO_ID.match(repo_id):
                            source = "huggingface"
                            if node.func.attr == "snapshot_download" and "modelscope" in text.lower():
                                source = "modelscope"
                            results.append(self._raw(
                                source, repo_id, "primary_generation_model", 0.85,
                                "ast_from_pretrained", node, self._ast_expr(node),
                            ))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                    model_flag = self._add_argument_model_flag(node)
                    if model_flag:
                        default = self._add_argument_default(node, constants)
                        if default and VALID_REPO_ID.match(default):
                            results.append(self._raw(
                                "huggingface", default, "primary_generation_model", 0.9,
                                "argparse_default", node, self._ast_expr(node),
                            ))
        # Regex fallback for files that fail to parse.
        for match in FROM_PRETRAINED_CALL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.85,
                "ast_from_pretrained", match, match.group(0).strip(),
            ))
        for match in SNAPSHOT_DOWNLOAD_CALL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.85,
                "ast_from_pretrained", match, match.group(0).strip(),
            ))
        return results

    def _scan_readme(self, rel_path: str, text: str) -> List[Dict]:
        results: List[Dict] = []
        for match in FROM_PRETRAINED_CALL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.8,
                "readme_from_pretrained", match, match.group(0).strip(),
            ))
        for match in SNAPSHOT_DOWNLOAD_CALL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.8,
                "readme_from_pretrained", match, match.group(0).strip(),
            ))
        for match in CLI_MODEL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.8,
                "readme_startup_command", match, match.group(0).strip(),
            ))
        for match in HF_URL.finditer(text):
            results.append(self._raw(
                "huggingface", match.group(1), "primary_generation_model", 0.6,
                "readme_link", match, match.group(0).strip(),
            ))
        for match in MODELSCOPE_URL.finditer(text):
            results.append(self._raw(
                "modelscope", match.group(1), "primary_generation_model", 0.6,
                "readme_link", match, match.group(0).strip(),
            ))
        return results

    # ---- helpers ----

    def _raw(self, source, repo_id, role, confidence, discovered_by, span, expression):
        return {
            "source": source,
            "repo_id": repo_id,
            "role": role,
            "confidence": confidence,
            "discovered_by": discovered_by,
            "span": span,
            "expression": expression,
        }

    def _merge(self, existing: ModelReferenceCandidate, raw: Dict) -> None:
        if raw["confidence"] > existing.confidence:
            existing.confidence = raw["confidence"]
            existing.discovered_by = raw["discovered_by"]
        if raw["role"] == "primary_generation_model" and existing.role != "primary_generation_model":
            existing.role = "primary_generation_model"
            existing.confidence = raw["confidence"]

    def _source_from_text(self, text: str, default: str) -> str:
        lowered = text.lower()
        if "modelscope" in lowered:
            return "modelscope"
        if "huggingface" in lowered or "hf.co" in lowered:
            return "huggingface"
        return default

    def _sha256_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _line_lookup(self, text: str):
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def _evidence(self, rel_path, text, span, file_sha, line_lookup, expression):
        start = getattr(span, "start", None)
        if isinstance(start, int):
            pos = start
        else:
            pos = 0
        line_start = self._line_of(line_lookup, pos)
        line_end = line_start
        return {
            "file": rel_path,
            "line_start": line_start,
            "line_end": line_end,
            "sha256": file_sha,
            "observation_id": "",
            "expression": expression,
        }

    def _line_of(self, line_lookup, pos: int) -> int:
        for index in range(len(line_lookup) - 1):
            if line_lookup[index] <= pos < line_lookup[index + 1]:
                return index + 1
        return max(1, len(line_lookup) - 1)

    def _candidate_texts(self, repo_dir: Path):
        for path in sorted(repo_dir.rglob("*")):
            if path.is_dir() or ".git" in path.parts:
                continue
            if path.name in _FORBIDDEN_FILENAMES:
                continue
            rel = path.relative_to(repo_dir)
            lower = str(rel).lower()
            if not (lower.endswith(_CONFIG_SUFFIXES) or lower.endswith(_PYTHON_SUFFIXES) or path.name in _README_NAMES):
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            yield str(rel), text

    # ---- Python AST helpers ----

    def _module_string_constants(self, text: str) -> Dict[str, str]:
        constants: Dict[str, str] = {}
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return constants
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
        return constants

    def _first_str_arg(self, args) -> Optional[str]:
        if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
            return args[0].value
        return None

    def _kwarg(self, keywords, name) -> Optional[str]:
        for kw in keywords:
            if kw.arg == name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
        return None

    def _resolve_constant(self, node, constants) -> Optional[str]:
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _add_argument_model_flag(self, node) -> bool:
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.lstrip("-") == "model":
                return True
        return False

    def _add_argument_default(self, node, constants) -> Optional[str]:
        for kw in node.keywords:
            if kw.arg == "default":
                return self._resolve_constant(kw.value, constants)
        # argparse positional form: add_argument("--model", "org/model")
        for arg in node.args:
            resolved = self._resolve_constant(arg, constants)
            if resolved and VALID_REPO_ID.match(resolved):
                return resolved
        return None

    def _ast_expr(self, node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return ""


def is_valid_repo_id(repo_id: str) -> bool:
    """True when repo_id is a safe two-segment ``org/name`` id.

    Rejects URLs, absolute paths, and any ``..`` path-traversal segment.
    """
    if not repo_id:
        return False
    if ".." in repo_id or repo_id.startswith(("/", "\\")):
        return False
    return bool(VALID_REPO_ID.match(repo_id))
