"""Deterministic parsing for supported dependency manifests."""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List

from auto_harness.capabilities.schemas import DependencyManifest
from auto_harness.command_auth.evidence import file_sha256, safe_repository_file

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility path
    tomllib = None


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def normalize_dependency_name(requirement: str) -> str:
    text = str(requirement or "").strip()
    if not text or text.startswith(("-", "#", ".", "/")):
        return ""
    text = text.split(";", 1)[0].strip()
    match = _NAME.match(text)
    if not match:
        return ""
    return match.group(0).lower().replace("_", "-")


class DependencyManifestParser:
    SUPPORTED = (
        "requirements.txt", "pyproject.toml", "environment.yml",
        "environment.yaml", "package.json",
    )

    def parse_all(self, repo_dir: Path, files: List[str]) -> List[DependencyManifest]:
        file_set = set(files)
        result = []
        for relative in self.SUPPORTED:
            if relative not in file_set:
                continue
            result.append(self.parse(repo_dir, relative))
        return result

    def parse(self, repo_dir: Path, relative: str) -> DependencyManifest:
        ecosystem = self._ecosystem(relative)
        try:
            path = safe_repository_file(repo_dir, relative)
            text = path.read_text(encoding="utf-8", errors="ignore")
            dependencies = self._dependencies(relative, text)
        except (OSError, TypeError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
            sha256 = ""
            try:
                sha256 = file_sha256(repo_dir / relative)
            except OSError:
                pass
            return DependencyManifest(
                path=relative,
                ecosystem=ecosystem,
                status="parse_failed",
                sha256=sha256,
                reason_code="invalid_%s:%s" % (self._format(relative), type(exc).__name__),
            )
        names = sorted({name for item in dependencies if (name := normalize_dependency_name(item))})
        return DependencyManifest(
            path=relative,
            ecosystem=ecosystem,
            dependencies=dependencies,
            dependency_names=names,
            sha256=file_sha256(path),
        )

    @staticmethod
    def _ecosystem(relative: str) -> str:
        if relative == "package.json":
            return "npm"
        if relative.startswith("environment."):
            return "conda"
        return "pip"

    @staticmethod
    def _format(relative: str) -> str:
        return Path(relative).suffix.lstrip(".") or "manifest"

    def _dependencies(self, relative: str, text: str) -> List[str]:
        if relative == "requirements.txt":
            return self._requirements(text)
        if relative == "pyproject.toml":
            return self._pyproject(text)
        if relative.startswith("environment."):
            return self._environment(text)
        if relative == "package.json":
            return self._package_json(text)
        return []

    @staticmethod
    def _requirements(text: str) -> List[str]:
        result = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line and not line.startswith(("-r", "--requirement", "--index-url")):
                result.append(line)
        return result

    def _pyproject(self, text: str) -> List[str]:
        if tomllib is not None:
            data = tomllib.loads(text)
            project = data.get("project") if isinstance(data, dict) else {}
            dependencies = list(project.get("dependencies") or []) if isinstance(project, dict) else []
            optional = project.get("optional-dependencies") if isinstance(project, dict) else {}
            if isinstance(optional, dict):
                for values in optional.values():
                    if isinstance(values, list):
                        dependencies.extend(str(item) for item in values)
            poetry = ((data.get("tool") or {}).get("poetry") or {}) if isinstance(data, dict) else {}
            poetry_deps = poetry.get("dependencies") if isinstance(poetry, dict) else {}
            if isinstance(poetry_deps, dict):
                dependencies.extend(
                    str(name) for name in poetry_deps
                    if str(name).lower() != "python"
                )
            return [str(item) for item in dependencies]
        return self._fallback_pyproject(text)

    @staticmethod
    def _fallback_pyproject(text: str) -> List[str]:
        section = ""
        buffer = ""
        result = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith("[") and line.endswith("]"):
                section = line
                continue
            if section == "[project]" and (buffer or line.startswith("dependencies")):
                buffer += line.split("=", 1)[1].strip() if not buffer and "=" in line else line
                if buffer.count("[") == buffer.count("]"):
                    value = ast.literal_eval(buffer)
                    result.extend(str(item) for item in value)
                    buffer = ""
            elif section == "[tool.poetry.dependencies]" and "=" in line:
                name = line.split("=", 1)[0].strip().strip("\"'")
                if name.lower() != "python":
                    result.append(name)
        return result

    @staticmethod
    def _environment(text: str) -> List[str]:
        # A deliberately narrow YAML subset: dependency list items and nested
        # pip list items. It does not construct arbitrary YAML objects.
        result = []
        in_dependencies = False
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped == "dependencies:":
                in_dependencies = True
                continue
            if in_dependencies and stripped and not raw.startswith((" ", "\t")):
                in_dependencies = False
            if in_dependencies and stripped.startswith("-"):
                value = stripped[1:].strip()
                if value and value != "pip:":
                    result.append(value)
        return result

    @staticmethod
    def _package_json(text: str) -> List[str]:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("package.json must contain an object")
        result = []
        for key in ("dependencies", "optionalDependencies", "peerDependencies"):
            values = data.get(key)
            if isinstance(values, dict):
                result.extend(str(name) for name in values)
        return result
