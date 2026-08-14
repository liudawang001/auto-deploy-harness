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


def discover_python_cli(
    repo_dir: Path,
    file_tree: List[str],
    readme: List[Dict],
    repository_fingerprint: str,
):
    if "pyproject.toml" not in set(file_tree):
        return [], []
    declarations = _script_sections(read_text(repo_dir, "pyproject.toml"))
    evidence = []
    candidates = []
    for name, target, source_type in declarations:
        declaration = build_evidence(
            repo_dir,
            source_type,
            "pyproject.toml",
            repository_fingerprint,
            declaration_key=name,
            declared_value=target,
        )
        evidence.append(declaration)
        for documented in readme:
            argv = documented["argv"]
            if not argv or argv[0] != name:
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
            verb = argv[1].lower() if len(argv) > 1 else ""
            phase = "setup" if verb in {"init", "setup"} else "run"
            candidates.append(CommandCandidate.build(
                phase=phase,
                argv=[".venv/bin/%s" % name] + argv[1:],
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
    return evidence, candidates
