"""Make target discovery. Recipes remain untrusted and approval-gated."""

import re

from auto_harness.command_auth.adapters.common import read_text
from auto_harness.command_auth.evidence import build_evidence
from auto_harness.command_auth.schemas import CommandCandidate


def discover_make(repo_dir, file_tree, readme, repository_fingerprint):
    if "Makefile" not in set(file_tree):
        return [], []
    text = read_text(repo_dir, "Makefile")
    targets = {
        match.group(1)
        for match in re.finditer(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", text, re.MULTILINE)
        if not match.group(1).startswith(".")
    }
    evidence = []
    candidates = []
    for documented in readme:
        argv = documented["argv"]
        if len(argv) != 2 or argv[0] != "make" or argv[1] not in targets:
            continue
        target = argv[1]
        declaration = build_evidence(
            repo_dir, "make_target", "Makefile", repository_fingerprint,
            declaration_key=target,
        )
        reference = build_evidence(
            repo_dir, "readme_reference", documented["path"],
            repository_fingerprint, line_start=documented["line"],
            line_end=documented["line"], declaration_key=target,
            declared_value=" ".join(argv),
        )
        evidence.extend([declaration, reference])
        candidates.append(CommandCandidate.build(
            phase="run", argv=["make", "-f", "Makefile", target],
            source_kind="make_target",
            evidence_ids=[declaration.evidence_id, reference.evidence_id],
            declared_executable="make",
            environment_binding={"kind": "system_tool", "makefile": "Makefile"},
            required_backend="docker", network_profile="none",
            filesystem_profile="runtime_read_only", risk_level="high",
            score=0.75, score_reasons=["declared Make target", "README command reference"],
            fallback_group="run",
        ))
    return evidence, candidates
