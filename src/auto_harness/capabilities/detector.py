"""Evidence-backed capability detection for supported project metadata."""

import ast
from pathlib import Path
from typing import Dict, List, Tuple

from auto_harness.capabilities.dependency_parser import DependencyManifestParser
from auto_harness.capabilities.evidence import build_capability_evidence
from auto_harness.capabilities.schemas import DependencyManifest, ProjectCapabilities


PACKAGE_CAPABILITIES: Dict[str, Tuple[str, str]] = {
    "gradio": ("ui_frameworks", "gradio"),
    "streamlit": ("ui_frameworks", "streamlit"),
    "fastapi": ("service_frameworks", "fastapi"),
    "flask": ("service_frameworks", "flask"),
    "django": ("service_frameworks", "django"),
    "torch": ("ml_libraries", "torch"),
    "pytorch": ("ml_libraries", "torch"),
    "transformers": ("ml_libraries", "transformers"),
    "vllm": ("inference_runtimes", "vllm"),
}

IMPORT_CAPABILITIES = {
    **PACKAGE_CAPABILITIES,
    "http.server": ("service_frameworks", "http.server"),
}


class CapabilityDetector:
    def __init__(self, dependency_parser=None) -> None:
        self.dependency_parser = dependency_parser or DependencyManifestParser()

    def detect(self, repo_dir: Path, files: List[str]):
        repo_dir = Path(repo_dir)
        capabilities = ProjectCapabilities()
        manifests = self.dependency_parser.parse_all(repo_dir, files)
        if any(name.endswith(".py") for name in files) or any(
            item.path in {"requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml"}
            for item in manifests
        ):
            self._add(capabilities, repo_dir, "languages", "python", "file", self._python_evidence_file(files), 0.8, "Python project file detected")
        if "package.json" in files:
            self._add(capabilities, repo_dir, "languages", "node", "manifest", "package.json", 0.95, "package.json detected")

        # Phase B3: language ecosystems and build systems detected from build
        # metadata.  Detection alone never grants execution; the native build
        # adapters need wrapper/lockfile evidence on top.
        for relative in files:
            name = Path(relative).name
            if name == "pom.xml":
                self._add(capabilities, repo_dir, "languages", "java", "manifest", relative, 0.9, "pom.xml detected")
                self._add(capabilities, repo_dir, "build_systems", "maven", "manifest", relative, 0.9, "pom.xml detected")
            elif name in ("build.gradle", "build.gradle.kts"):
                self._add(capabilities, repo_dir, "languages", "java", "manifest", relative, 0.9, "%s detected" % name)
                self._add(capabilities, repo_dir, "build_systems", "gradle", "manifest", relative, 0.9, "%s detected" % name)
            elif name == "go.mod":
                self._add(capabilities, repo_dir, "languages", "go", "manifest", relative, 0.95, "go.mod detected")
                self._add(capabilities, repo_dir, "build_systems", "go", "manifest", relative, 0.95, "go.mod detected")
            elif name == "Cargo.toml":
                self._add(capabilities, repo_dir, "languages", "rust", "manifest", relative, 0.95, "Cargo.toml detected")
                self._add(capabilities, repo_dir, "build_systems", "cargo", "manifest", relative, 0.95, "Cargo.toml detected")

        for manifest in manifests:
            if manifest.status != "parsed":
                continue
            ecosystem = manifest.ecosystem
            self._add(capabilities, repo_dir, "package_ecosystems", ecosystem, "dependency", manifest.path, 0.95, "%s dependency manifest" % ecosystem)
            for name in manifest.dependency_names:
                mapping = PACKAGE_CAPABILITIES.get(name)
                if mapping:
                    self._add(capabilities, repo_dir, mapping[0], mapping[1], "dependency", manifest.path, 0.95, "declared dependency %s" % name)

        for relative in files:
            if not relative.endswith(".py"):
                continue
            for imported in self._python_imports(repo_dir / relative):
                mapping = IMPORT_CAPABILITIES.get(imported)
                if not mapping and imported.startswith("http.server"):
                    mapping = IMPORT_CAPABILITIES["http.server"]
                if mapping:
                    self._add(capabilities, repo_dir, mapping[0], mapping[1], "import", relative, 0.9, "Python import %s" % imported)

        if "http.server" in capabilities.service_frameworks:
            capabilities.protocols.append("http")
            capabilities.workload_types.append("service")
        if capabilities.service_frameworks or capabilities.ui_frameworks or capabilities.inference_runtimes:
            capabilities.workload_types.append("service")
            capabilities.protocols.append("http")
        if "fastapi" in capabilities.service_frameworks:
            capabilities.protocols.append("openapi")
        if "vllm" in capabilities.inference_runtimes:
            capabilities.protocols.append("openai_compatible")

        readme = next((name for name in ("README.md", "readme.md") if name in files), "")
        if readme:
            text = (repo_dir / readme).read_text(encoding="utf-8", errors="ignore").lower()
            if "openai-compatible" in text or "openai compatible" in text or "/v1/chat/completions" in text:
                self._add(capabilities, repo_dir, "protocols", "openai_compatible", "readme", readme, 0.7, "documented OpenAI-compatible endpoint")

        return capabilities.normalize(), manifests

    @staticmethod
    def _python_evidence_file(files: List[str]) -> str:
        for preferred in (
            "pyproject.toml", "requirements.txt", "environment.yml",
            "environment.yaml", "setup.py", "app.py", "main.py",
        ):
            if preferred in files:
                return preferred
        return next((name for name in files if name.endswith(".py")), "")

    @staticmethod
    def _python_imports(path: Path):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError, ValueError):
            return set()
        result = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                result.add(node.module)
        return result

    @staticmethod
    def _add(capabilities, repo_dir, field, value, source, relative, confidence, reason):
        values = getattr(capabilities, field)
        values.append(value)
        try:
            evidence = build_capability_evidence(
                repo_dir,
                capability_type=field,
                capability_value=value,
                source_type=source,
                relative=relative,
                confidence=confidence,
                reason=reason,
            )
        except (OSError, ValueError):
            return
        capabilities.evidence.append(evidence)
