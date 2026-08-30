"""Repository entrypoint evidence discovery for unknown frameworks.

Phase B1: discover install/run candidates from machine declarations and
public repository documents without invoking the LLM. Everything produced
here is evidence-bound and still has to pass the unified command
authorization engine, so none of these sources can bypass policy.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from auto_harness.command_auth.adapters.common import read_text
from auto_harness.command_auth.evidence import build_evidence
from auto_harness.command_auth.schemas import CommandCandidate


# Source kinds contributed by this module. The analyzer merges these into
# the deterministic run candidate list; README-documented node scripts and
# other pre-existing kinds stay plan-first only.
ENTRYPOINT_SOURCE_KINDS = frozenset({
    "django_manage",
    "asgi_wsgi_entrypoint",
    "procfile_web",
    "pep621_script",
    "poetry_script",
    "node_run_script",
})

_SHELL_OPERATORS = (";", "&&", "||", "|", "&", "`", "$(", ">", "<", "\n", "\r")
_EXCLUDED_DIRS = {"docs", "examples", "tests", "test", "node_modules", ".venv", "build", "dist"}
_MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_APPLICATION_SYMBOL = re.compile(r"(?m)^\s*application\s*=|[^A-Za-z_]application\s*=\s*get_")
_DJANGO_DEFAULT_PORT = 8000
_PROCFILE_ROOTS = {"python", "python3", "node", "npm", "pnpm", "yarn", "gunicorn", "uvicorn", "make"}
_NODE_LISTEN_PORT = re.compile(r"listen\s*\(\s*['\"]?(\d{2,5})['\"]?")


def _selectable(relative: str) -> bool:
    parts = Path(relative).parts
    return not any(part in _EXCLUDED_DIRS for part in parts[:-1]) and len(parts) <= 4


def declared_python_dependencies(repo_dir: Path, file_tree: List[str]) -> Dict[str, str]:
    """Bounded dependency name -> declaration file map for Python manifests."""
    declared: Dict[str, str] = {}
    for relative in file_tree:
        name = Path(relative).name
        if name not in {"requirements.txt", "pyproject.toml", "environment.yml", "environment.yaml"}:
            continue
        if not _selectable(relative):
            continue
        text = read_text(repo_dir, relative)
        if not text:
            continue
        if name == "requirements.txt":
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                package = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip()
                if package:
                    declared.setdefault(_normalize_name(package), relative)
        elif name == "pyproject.toml":
            try:
                import tomllib

                data = tomllib.loads(text)
            except Exception:  # noqa: BLE001 - invalid manifests must not break discovery
                continue
            for raw in (data.get("project", {}) or {}).get("dependencies", []) or []:
                if isinstance(raw, str) and raw.strip():
                    declared.setdefault(
                        _normalize_name(re.split(r"[<>=!~\[; ]", raw.strip(), maxsplit=1)[0]), relative,
                    )
        else:
            for line in text.splitlines():
                match = re.match(r"\s*-?\s*([A-Za-z0-9_.-]+)\s*(=.+)?$", line)
                if match and not match.group(1).startswith("-"):
                    declared.setdefault(_normalize_name(match.group(1)), relative)
    return declared


def _normalize_name(package: str) -> str:
    return re.sub(r"[-_.]+", "-", package.strip().lower())


def _module_name(relative: str) -> str:
    parts = Path(relative).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module = ".".join(parts)
    return module if module and _MODULE_NAME.fullmatch(module) else ""


def _readme_port(readme, tokens) -> int:
    joined = " ".join(tokens)
    for documented in readme:
        text = " ".join(documented["argv"])
        if all(token in text for token in tokens[:2]) or (tokens and tokens[0] in text and len(tokens) > 1 and tokens[1] in text):
            match = re.search(r"(?:--bind|0\.0\.0\.0:|127\.0\.0\.1:|localhost:|--port\s+|:)(\d{2,5})", text)
            if not match:
                match = re.search(r":(\d{2,5})", text)
            if match:
                return int(match.group(1))
    return 0


def _dependency_evidence(repo_dir, repository_fingerprint, package, declared):
    relative = declared.get(_normalize_name(package))
    if not relative:
        return None
    return build_evidence(
        repo_dir,
        "python_dependency",
        relative,
        repository_fingerprint,
        declaration_key=_normalize_name(package),
        declared_value=package,
    )


def discover_python_services(repo_dir, file_tree, readme, repository_fingerprint):
    """Discover Django/ASGI/WSGI/Procfile/Dockerfile run evidence."""
    repo_dir = Path(repo_dir)
    file_set = set(file_tree)
    evidence: List = []
    candidates: List = []
    declared = declared_python_dependencies(repo_dir, file_tree)

    # --- Django manage.py -------------------------------------------------
    for manage_relative in sorted(
        item for item in file_set
        if Path(item).name == "manage.py" and _selectable(item)
    ):
        if "django" not in declared:
            continue
        manage_dir = str(Path(manage_relative).parent).replace("\\", "/")
        cwd = "." if manage_dir == "." else manage_dir
        python = ".venv/bin/python" if cwd == "." else "../.venv/bin/python"
        manage_evidence = build_evidence(
            repo_dir, "django_manage", manage_relative, repository_fingerprint,
            declaration_key="manage_py", declared_value="runserver",
        )
        django_evidence = _dependency_evidence(repo_dir, repository_fingerprint, "django", declared)
        run_readme = [
            item for item in readme
            if len(item["argv"]) >= 2 and "manage.py" in item["argv"][:2]
        ]
        port = _DJANGO_DEFAULT_PORT
        reasons = ["django dependency declaration", "manage.py entrypoint"]
        evidence_items = [manage_evidence, django_evidence]
        if run_readme:
            documented = run_readme[0]
            port_match = re.search(r"(?:0\.0\.0\.0:|127\.0\.0\.1:|localhost:|--port\s+|:)(\d{2,5})", " ".join(documented["argv"]))
            if port_match:
                port = int(port_match.group(1))
            reference = build_evidence(
                repo_dir, "readme_reference", documented["path"], repository_fingerprint,
                line_start=documented["line"], line_end=documented["line"],
                declaration_key="manage.py", declared_value=" ".join(documented["argv"]),
            )
            evidence_items.append(reference)
            reasons.append("README command reference")
        evidence.extend(evidence_items)
        candidates.append(CommandCandidate.build(
            phase="run",
            argv=[python, manage_relative, "runserver", "0.0.0.0:%d" % port],
            cwd=cwd,
            source_kind="django_manage",
            expected_port=port,
            evidence_ids=[item.evidence_id for item in evidence_items],
            declared_executable="manage.py",
            environment_binding={"kind": "owned_python_env", "relative_prefix": ".venv"},
            required_backend="docker",
            network_profile="none",
            filesystem_profile="runtime_read_only",
            risk_level="medium",
            score=0.9,
            score_reasons=reasons,
            fallback_group="run",
        ))

    # --- ASGI / WSGI modules ---------------------------------------------
    for kind, server, argv_builder in (
        ("wsgi.py", "gunicorn", lambda module, port: [
            ".venv/bin/gunicorn", "%s:application" % module, "--bind", "0.0.0.0:%d" % port,
        ]),
        ("asgi.py", "uvicorn", lambda module, port: [
            ".venv/bin/uvicorn", "%s:application" % module, "--host", "0.0.0.0", "--port", "%d" % port,
        ]),
    ):
        if _normalize_name(server) not in declared:
            continue
        for module_relative in sorted(
            item for item in file_set
            if Path(item).name == kind and _selectable(item)
        ):
            module = _module_name(module_relative)
            if not module:
                continue
            source = read_text(repo_dir, module_relative)
            if not source:
                continue
            if not _APPLICATION_SYMBOL.search(source):
                continue
            module_evidence = build_evidence(
                repo_dir, "asgi_wsgi_module", module_relative, repository_fingerprint,
                declaration_key="application_callable", declared_value="%s:application" % module,
            )
            server_evidence = _dependency_evidence(repo_dir, repository_fingerprint, server, declared)
            tokens = [server, module]
            port = _readme_port(readme, tokens) or _DJANGO_DEFAULT_PORT
            evidence_items = [module_evidence, server_evidence]
            reasons = [
                "%s dependency declaration" % server,
                "%s module %s" % (kind, module),
            ]
            for documented in readme:
                if server in documented["argv"] and module.split(".")[-1] in " ".join(documented["argv"]):
                    reference = build_evidence(
                        repo_dir, "readme_reference", documented["path"], repository_fingerprint,
                        line_start=documented["line"], line_end=documented["line"],
                        declaration_key=module, declared_value=" ".join(documented["argv"]),
                    )
                    evidence_items.append(reference)
                    reasons.append("README command reference")
                    break
            evidence.extend(evidence_items)
            candidates.append(CommandCandidate.build(
                phase="run",
                argv=argv_builder(module, port),
                source_kind="asgi_wsgi_entrypoint",
                expected_port=port,
                evidence_ids=[item.evidence_id for item in evidence_items],
                declared_executable=server,
                environment_binding={"kind": "owned_python_env", "relative_prefix": ".venv"},
                required_backend="docker",
                network_profile="none",
                filesystem_profile="runtime_read_only",
                risk_level="medium",
                score=0.88,
                score_reasons=reasons,
                fallback_group="run",
            ))

    # --- Procfile web process --------------------------------------------
    procfile_candidates, procfile_evidence, procfile_rejections = _procfile_discovery(
        repo_dir, file_set, readme, repository_fingerprint, declared,
    )
    evidence.extend(procfile_evidence)
    candidates.extend(procfile_candidates)

    # --- Dockerfile CMD/ENTRYPOINT JSON evidence --------------------------
    dockerfile_evidence, dockerfile_rejections = _dockerfile_discovery(
        repo_dir, file_set, repository_fingerprint,
    )
    evidence.extend(dockerfile_evidence)

    rejections = procfile_rejections + dockerfile_rejections
    return evidence, candidates, rejections


def _procfile_discovery(repo_dir, file_set, readme, repository_fingerprint, declared):
    evidence: List = []
    candidates: List = []
    rejections: List[Dict] = []
    procfile_relative = "Procfile" if "Procfile" in file_set else ""
    if not procfile_relative:
        return candidates, evidence, rejections
    text = read_text(repo_dir, procfile_relative)
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.+)$", line)
        if not match or match.group(1) != "web":
            continue
        declared_value = match.group(2).strip()
        declaration = build_evidence(
            repo_dir, "procfile_web", procfile_relative, repository_fingerprint,
            line_start=line_number, line_end=line_number,
            declaration_key="web", declared_value=declared_value,
        )
        if any(operator in declared_value for operator in _SHELL_OPERATORS):
            rejections.append({
                "source_type": "procfile_web",
                "path": procfile_relative,
                "reason_code": "procfile_shell_operator_rejected",
                "declared_value": declared_value,
                "line": line_number,
            })
            continue
        try:
            argv = shlex.split(declared_value, posix=True)
        except ValueError:
            rejections.append({
                "source_type": "procfile_web",
                "path": procfile_relative,
                "reason_code": "procfile_argv_unparsable",
                "declared_value": declared_value,
                "line": line_number,
            })
            continue
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            rejections.append({
                "source_type": "procfile_web",
                "path": procfile_relative,
                "reason_code": "procfile_argv_invalid",
                "declared_value": declared_value,
                "line": line_number,
            })
            continue
        corroboration = _procfile_corroboration(
            repo_dir, file_set, argv, declared, repository_fingerprint,
        )
        if corroboration is None:
            rejections.append({
                "source_type": "procfile_web",
                "path": procfile_relative,
                "reason_code": "procfile_uncorroborated_command",
                "declared_value": declared_value,
                "line": line_number,
            })
            continue
        reference = None
        for documented in readme:
            if documented["argv"] == argv:
                reference = build_evidence(
                    repo_dir, "readme_reference", documented["path"], repository_fingerprint,
                    line_start=documented["line"], line_end=documented["line"],
                    declaration_key="web", declared_value=" ".join(argv),
                )
                break
        evidence_items = [declaration, corroboration]
        if reference is not None:
            evidence_items.append(reference)
        evidence.extend(evidence_items)
        candidates.append(CommandCandidate.build(
            phase="run",
            argv=argv,
            source_kind="procfile_web",
            expected_port=_procfile_port(readme, argv),
            evidence_ids=[item.evidence_id for item in evidence_items],
            declared_executable=Path(argv[0]).name,
            environment_binding={"kind": "repository_process_declaration"},
            required_backend="docker",
            network_profile="none",
            filesystem_profile="runtime_read_only",
            risk_level="medium",
            score=0.86,
            score_reasons=["Procfile web process declaration", "repository command corroboration"],
            fallback_group="run",
        ))
    return candidates, evidence, rejections


def _procfile_corroboration(repo_dir, file_set, argv, declared, repository_fingerprint):
    """Procfile commands must cross-check repository facts before running."""
    root = Path(repo_dir)
    for token in argv:
        if token.startswith("-") or "=" in token or ":" in token:
            continue
        candidate_path = root / token
        if token in file_set and candidate_path.is_file() and not candidate_path.is_symlink():
            return build_evidence(
                repo_dir, "repository_file", token, repository_fingerprint,
                declaration_key="procfile_web", declared_value=token,
            )
    first = Path(argv[0]).name
    if first in declared or first in _PROCFILE_ROOTS:
        for token in argv[1:]:
            if token in file_set and (root / token).is_file():
                return build_evidence(
                    repo_dir, "repository_file", token, repository_fingerprint,
                    declaration_key="procfile_web", declared_value=token,
                )
    return None


def _procfile_port(readme, argv) -> int:
    for documented in readme:
        text = " ".join(documented["argv"])
        if argv and argv[-1] in text:
            match = re.search(r"(?:--port\s+|--bind\s+\S*?:|0\.0\.0\.0:|127\.0\.0\.1:)(\d{2,5})", text)
            if match:
                return int(match.group(1))
    return 0


def _dockerfile_discovery(repo_dir, file_set, repository_fingerprint):
    """JSON-form CMD/ENTRYPOINT and EXPOSE lines are low-privilege evidence."""
    evidence: List = []
    rejections: List[Dict] = []
    dockerfile_relative = "Dockerfile" if "Dockerfile" in file_set else ""
    if not dockerfile_relative:
        return evidence, rejections
    text = read_text(repo_dir, dockerfile_relative)
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        entrypoint = re.match(r"^(CMD|ENTRYPOINT)\s+(.+)$", line)
        if entrypoint:
            instruction, payload = entrypoint.group(1), entrypoint.group(2).strip()
            if payload.startswith("["):
                try:
                    argv = json.loads(payload)
                except ValueError:
                    argv = None
                if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv):
                    evidence.append(build_evidence(
                        repo_dir, "dockerfile_entrypoint", dockerfile_relative,
                        repository_fingerprint, line_start=line_number, line_end=line_number,
                        declaration_key=instruction, declared_value=json.dumps(argv),
                    ))
                    continue
            rejections.append({
                "source_type": "dockerfile_entrypoint",
                "path": dockerfile_relative,
                "reason_code": "dockerfile_shell_form_rejected",
                "declared_value": payload[:200],
                "line": line_number,
            })
            continue
        expose = re.match(r"^EXPOSE\s+(\d{2,5})", line)
        if expose:
            evidence.append(build_evidence(
                repo_dir, "dockerfile_expose", dockerfile_relative, repository_fingerprint,
                line_start=line_number, line_end=line_number,
                declaration_key="EXPOSE", declared_value=expose.group(1),
            ))
    return evidence, rejections


def lockfile_path_for(manager: str, directory: str, file_set) -> str:
    """Resolve the lockfile binding for a package directory.

    Directory-local lockfiles are preferred.  pnpm and yarn workspaces keep
    a single lockfile at the repository root, so the root lockfile is an
    accepted binding for those managers when the directory has none.
    """
    lock_name = {"npm": "package-lock.json", "pnpm": "pnpm-lock.yaml", "yarn": "yarn.lock"}[manager]
    local = str(Path(directory) / lock_name).replace("\\", "/")
    if local.startswith("./"):
        local = local[2:]
    if local in file_set:
        return local
    if directory not in {".", ""} and manager in {"pnpm", "yarn"} and lock_name in file_set:
        return lock_name
    return ""


def declared_node_run_scripts(repo_dir, file_tree):
    """Declared start/serve scripts bound to a matching lockfile.

    ``dev`` scripts are intentionally excluded: they are not a production
    first choice unless a manifest, README, or the operator says otherwise.
    """
    repo_dir = Path(repo_dir)
    file_set = set(file_tree)
    results = []
    package_files = [
        item for item in file_tree
        if Path(item).name == "package.json" and "node_modules" not in Path(item).parts
    ]
    package_files.sort(key=lambda item: (len(Path(item).parts), item))
    for package_path in package_files[:50]:
        try:
            package = json.loads(read_text(repo_dir, package_path))
        except (TypeError, ValueError):
            continue
        scripts = package.get("scripts") if isinstance(package, dict) else {}
        if not isinstance(scripts, dict):
            continue
        directory = str(Path(package_path).parent).replace("\\", "/")
        directory = "." if directory == "." else directory
        for manager in ("npm", "pnpm", "yarn"):
            lock_path = lockfile_path_for(manager, directory, file_set)
            if not lock_path:
                continue
            for script in ("start", "serve"):
                if script not in scripts or not str(scripts[script]).strip():
                    continue
                if manager == "yarn":
                    argv = ["yarn", "--cwd", directory, "run", script]
                else:
                    argv = [manager, "--dir" if manager == "pnpm" else "--prefix", directory, "run", script]
                results.append({
                    "manager": manager,
                    "script": script,
                    "directory": directory,
                    "package_path": package_path,
                    "lock_path": lock_path,
                    "argv": argv,
                })
    return results


def source_listen_port(repo_dir, file_tree, script_value: str) -> int:
    """Best-effort port from the entry source file referenced by a script."""
    repo_dir = Path(repo_dir)
    for token in re.split(r"[\s\"']+", str(script_value or "")):
        if not token.endswith(".js") or token.startswith("-"):
            continue
        for relative in file_tree:
            if Path(relative).name == token and _selectable(relative):
                match = _NODE_LISTEN_PORT.search(read_text(repo_dir, relative))
                if match:
                    return int(match.group(1))
    return 0


def dockerfile_entrypoint_port(analysis_evidence: List[Dict]) -> int:
    for item in analysis_evidence or []:
        if item.get("source_type") == "dockerfile_expose":
            try:
                return int(item.get("declared_value") or 0)
            except (TypeError, ValueError):
                return 0
    return 0
