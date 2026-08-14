"""README-referenced repository script discovery."""

from pathlib import Path

from auto_harness.command_auth.evidence import build_evidence, safe_repository_file
from auto_harness.command_auth.schemas import CommandCandidate


INTERPRETERS = {"python", "python3", "sh", "bash"}
COMMON_PYTHON_ENTRYPOINTS = {
    "app.py", "main.py", "server.py", "webui.py", "demo.py",
    "gradio_app.py", "api.py",
}


def _script_reference(argv):
    if not argv:
        return "", "", []
    if argv[0] in INTERPRETERS and len(argv) >= 2:
        return argv[0], argv[1], argv[2:]
    if argv[0].startswith("./"):
        suffix = Path(argv[0]).suffix.lower()
        interpreter = "python3" if suffix == ".py" else "sh" if suffix in {".sh", ".bash"} else ""
        return interpreter, argv[0][2:], argv[1:]
    if "/" in argv[0] and not argv[0].startswith("/"):
        return "", argv[0], argv[1:]
    return "", "", []


def discover_repository_scripts(repo_dir, file_tree, readme, repository_fingerprint):
    file_set = set(file_tree)
    evidence = []
    candidates = []
    for documented in readme:
        interpreter, relative, args = _script_reference(documented["argv"])
        relative = relative.replace("\\", "/")
        if relative not in file_set:
            continue
        try:
            script_path = safe_repository_file(repo_dir, relative)
        except ValueError:
            continue
        if not interpreter:
            try:
                first_line = script_path.read_text(
                    encoding="utf-8", errors="ignore",
                ).splitlines()[:1]
            except OSError:
                continue
            shebang = first_line[0].lower() if first_line else ""
            if "python" in shebang:
                interpreter = "python3"
            elif shebang.startswith("#!") and any(
                shell in shebang for shell in ("/sh", "/bash", "/zsh")
            ):
                interpreter = "sh"
            else:
                continue
        script = build_evidence(
            repo_dir, "repository_script", relative, repository_fingerprint,
            declaration_key=relative,
        )
        reference = build_evidence(
            repo_dir, "readme_reference", documented["path"],
            repository_fingerprint, line_start=documented["line"],
            line_end=documented["line"], declaration_key=relative,
            declared_value=" ".join(documented["argv"]),
        )
        evidence.extend([script, reference])
        python = interpreter in {"python", "python3"}
        effective_interpreter = ".venv/bin/python" if python else "/bin/sh"
        candidates.append(CommandCandidate.build(
            phase="run", argv=[effective_interpreter, relative] + args,
            source_kind="repository_script",
            evidence_ids=[script.evidence_id, reference.evidence_id],
            declared_executable=interpreter,
            environment_binding={
                "kind": "owned_python_env" if python else "fixed_interpreter",
                "script_path": relative,
            },
            required_backend="docker", network_profile="none",
            filesystem_profile="runtime_read_only", risk_level="high",
            score=0.7, score_reasons=["repository script hash", "README command reference"],
            fallback_group="run",
        ))
    documented_paths = {
        item.environment_binding.get("script_path")
        for item in candidates
        if isinstance(item.environment_binding, dict)
    }
    for relative in sorted(COMMON_PYTHON_ENTRYPOINTS.intersection(file_set)):
        if relative in documented_paths:
            continue
        try:
            safe_repository_file(repo_dir, relative)
        except ValueError:
            continue
        script = build_evidence(
            repo_dir, "repository_script", relative, repository_fingerprint,
            declaration_key=relative,
        )
        evidence.append(script)
        candidates.append(CommandCandidate.build(
            phase="run", argv=[".venv/bin/python", relative],
            source_kind="python_entrypoint",
            evidence_ids=[script.evidence_id],
            declared_executable="python",
            environment_binding={"kind": "owned_python_env", "script_path": relative},
            required_backend="docker", network_profile="none",
            filesystem_profile="runtime_read_only", risk_level="high",
            score=0.55,
            score_reasons=["common Python entrypoint path", "repository file hash"],
            fallback_group="run",
        ))
    return evidence, candidates
