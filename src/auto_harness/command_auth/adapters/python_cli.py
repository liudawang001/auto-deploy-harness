"""PEP 621 and Poetry console-script discovery."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from auto_harness.command_auth.adapters.common import read_text
from auto_harness.command_auth.evidence import build_evidence
from auto_harness.command_auth.schemas import CommandCandidate


def _script_sections(text: str) -> List[Tuple[str, str, str]]:
    section = ""
    result = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if section not in {"[project.scripts]", "[tool.poetry.scripts]"}:
            continue
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, target = (part.strip() for part in line.split("=", 1))
        name = name.strip("\"'")
        target = target.split("#", 1)[0].strip().strip("\"'")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            continue
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*", target
        ):
            continue
        result.append((name, target, "pep621_script" if section == "[project.scripts]" else "poetry_script"))
    return result


def _source_frontend_variant(
    repo_dir: Path,
    file_tree: List[str],
    target: str,
    base_argv: List[str],
    repository_fingerprint: str,
):
    """Return a source-checkout CLI variant that serves built frontend assets."""
    module = target.split(":", 1)[0]
    module_suffix = module.replace(".", "/") + ".py"
    source_path = next(
        (path for path in file_tree if path.endswith(module_suffix)),
        "",
    )
    required = {
        "src/frontend/package.json",
        "src/frontend/package-lock.json",
    }
    if not source_path or not required.issubset(set(file_tree)):
        return None
    source = read_text(repo_dir, source_path)
    if "frontend_path" not in source or "typer.Option" not in source:
        imported_modules = re.findall(
            r"from\s+([A-Za-z_][A-Za-z0-9_.]+)\s+import",
            source,
        )
        imported_paths = {
            imported.replace(".", "/") + ".py"
            for imported in imported_modules
        }
        delegated_path = next(
            (
                path for path in file_tree
                if any(path.endswith(candidate) for candidate in imported_paths)
            ),
            "",
        )
        if not delegated_path:
            return None
        delegated_source = read_text(repo_dir, delegated_path)
        if "frontend_path" not in delegated_source or "typer.Option" not in delegated_source:
            return None
        source_path = delegated_path
    vite_path = next(
        (
            path for path in file_tree
            if path.startswith("src/frontend/vite.config.")
        ),
        "",
    )
    if not vite_path:
        return None
    vite_config = read_text(repo_dir, vite_path)
    output_match = re.search(r"outDir\s*:\s*['\"]([^'\"]+)['\"]", vite_config)
    if not output_match:
        return None
    output_dir = "src/frontend/%s" % output_match.group(1).strip("/")
    cli_evidence = build_evidence(
        repo_dir,
        "python_cli_option",
        source_path,
        repository_fingerprint,
        declaration_key="frontend_path",
        declared_value="--frontend-path",
    )
    output_evidence = build_evidence(
        repo_dir,
        "frontend_build_output",
        vite_path,
        repository_fingerprint,
        declaration_key="build.outDir",
        declared_value=output_match.group(1),
    )
    return (
        base_argv + ["--frontend-path", output_dir],
        [cli_evidence, output_evidence],
    )


def discover_python_cli(
    repo_dir: Path,
    file_tree: List[str],
    readme: List[Dict],
    repository_fingerprint: str,
):
    pyproject_paths = [
        path
        for path in file_tree
        if Path(path).name == "pyproject.toml"
        and not any(
            part in {"docs", "examples", "tests", "test"}
            for part in Path(path).parts[:-1]
        )
    ][:50]
    if not pyproject_paths:
        return [], []
    evidence = []
    candidates = []
    for pyproject_path in pyproject_paths:
        declarations = _script_sections(read_text(repo_dir, pyproject_path))
        for name, target, source_type in declarations:
            declaration = build_evidence(
                repo_dir,
                source_type,
                pyproject_path,
                repository_fingerprint,
                declaration_key=name,
                declared_value=target,
            )
            evidence.append(declaration)
            for documented in readme:
                argv = documented["argv"]
                if argv and argv[0] == name:
                    command_args = argv[1:]
                elif len(argv) >= 3 and argv[:2] == ["uv", "run"] and argv[2] == name:
                    command_args = argv[3:]
                else:
                    continue
                reference = build_evidence(
                    repo_dir,
                    "readme_reference",
                    documented["path"],
                    repository_fingerprint,
                    line_start=documented["line"],
                    line_end=documented["line"],
                    declaration_key=name,
                    declared_value=" ".join(argv),
                )
                evidence.append(reference)
                verb = command_args[0].lower() if command_args else ""
                phase = "setup" if verb in {"init", "setup"} else "run"
                candidates.append(CommandCandidate.build(
                    phase=phase,
                    argv=[".venv/bin/%s" % name] + command_args,
                    source_kind=source_type,
                    evidence_ids=[declaration.evidence_id, reference.evidence_id],
                    declared_executable=name,
                    environment_binding={"kind": "owned_python_env", "relative_prefix": ".venv"},
                    required_backend="docker",
                    network_profile="none",
                    filesystem_profile="install_workspace" if phase == "setup" else "runtime_read_only",
                    risk_level="medium",
                    score=0.95 if phase == "run" else 0.85,
                    score_reasons=["project metadata declaration", "README command reference"],
                    fallback_group=phase,
                ))
                if phase == "run":
                    variant = _source_frontend_variant(
                        repo_dir,
                        file_tree,
                        target,
                        [".venv/bin/%s" % name] + command_args,
                        repository_fingerprint,
                    )
                    if variant:
                        variant_argv, variant_evidence = variant
                        evidence.extend(variant_evidence)
                        candidates.append(CommandCandidate.build(
                            phase="run",
                            argv=variant_argv,
                            source_kind=source_type,
                            evidence_ids=[
                                declaration.evidence_id,
                                reference.evidence_id,
                                *(item.evidence_id for item in variant_evidence),
                            ],
                            declared_executable=name,
                            environment_binding={
                                "kind": "owned_python_env",
                                "relative_prefix": ".venv",
                            },
                            required_backend="docker",
                            network_profile="none",
                            filesystem_profile="runtime_read_only",
                            risk_level="medium",
                            score=0.98,
                            score_reasons=[
                                "project metadata declaration",
                                "README command reference",
                                "source CLI frontend option",
                                "declared frontend build output",
                            ],
                            fallback_group=phase,
                        ))
    return evidence, candidates
