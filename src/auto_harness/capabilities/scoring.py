"""Evidence-based candidate scoring for repository entrypoint discovery.

Implements the Phase B1 scoring table. Scores only order candidates; the
command authorization verdict stays independent and can always override
ranking (a rejected or hard-denied candidate never becomes executable).
"""

from typing import Dict, List

# Evidence kind -> score points (0-100 scale).
SCORE_WEIGHTS: Dict[str, int] = {
    "operator_override": 100,
    "valid_manifest": 90,
    "machine_readable_entry": 75,
    "readme_exact_reference": 20,
    "lockfile_match": 15,
    "high_confidence_adapter": 15,
    "source_port_match": 10,
    "dockerfile_entrypoint_corroboration": 10,
    "common_entrypoint_filename": 5,
    "port_conflict": -20,
    "dependency_declaration_missing": -20,
    "readme_conflict": -30,
}

# Candidates contributed by deterministic discovery declare their evidence
# kind via the command registry source kind.
_SOURCE_KIND_POINTS: Dict[str, int] = {
    "django_manage": "machine_readable_entry",
    "asgi_wsgi_entrypoint": "machine_readable_entry",
    "procfile_web": "machine_readable_entry",
    "pep621_script": "machine_readable_entry",
    "poetry_script": "machine_readable_entry",
    "node_run_script": "machine_readable_entry",
}

_COMMON_ENTRYPOINT_NAMES = frozenset({
    "app.py", "main.py", "server.py", "webui.py", "demo.py", "manage.py",
    "wsgi.py", "asgi.py", "start", "serve",
})

_ENTRY_KIND_POINTS = 25


def score_run_candidates(
    run_candidates: List[Dict],
    *,
    detections=None,
    registry=None,
) -> None:
    """Attach ``evidence_score`` (0-100) and reasons to each candidate.

    Existing ``score``/``confidence`` fields are preserved so legacy golden
    behaviour is unchanged; the new field only drives ordering.
    """
    high_confidence_adapter = any(
        getattr(item, "matched", False) and getattr(item, "confidence", 0) >= 0.85
        for item in (detections or [])
    )
    registry_evidence = {}
    dockerfile_entrypoints = []
    if registry is not None:
        registry_evidence = registry.evidence_by_id()
        for item in registry.evidence:
            if item.source_type == "dockerfile_entrypoint":
                dockerfile_entrypoints.append(item.declared_value)

    for index, candidate in enumerate(run_candidates):
        reasons: List[str] = []
        points = 0
        source_kind = str(candidate.get("source_kind") or "")
        evidence_kind = _SOURCE_KIND_POINTS.get(source_kind)
        if evidence_kind:
            points += SCORE_WEIGHTS[evidence_kind]
            reasons.append("%s +%d" % (evidence_kind, SCORE_WEIGHTS[evidence_kind]))
        else:
            # Framework/entrypoint convention tier sits below declarations.
            points += _ENTRY_KIND_POINTS
            reasons.append("entrypoint_convention +%d" % _ENTRY_KIND_POINTS)
            if high_confidence_adapter:
                points += SCORE_WEIGHTS["high_confidence_adapter"]
                reasons.append(
                    "high_confidence_adapter +%d" % SCORE_WEIGHTS["high_confidence_adapter"]
                )
        cmd = list(candidate.get("cmd") or [])
        tail_names = {str(token) for token in cmd[1:] if not str(token).startswith("-")}
        if tail_names & _COMMON_ENTRYPOINT_NAMES or (
            cmd and Path_name(cmd[0]) in _COMMON_ENTRYPOINT_NAMES
        ):
            points += SCORE_WEIGHTS["common_entrypoint_filename"]
            reasons.append(
                "common_entrypoint_filename +%d" % SCORE_WEIGHTS["common_entrypoint_filename"]
            )
        if source_kind == "node_run_script" or candidate.get("command_candidate_id"):
            registry_candidate = None
            for item in getattr(registry, "candidates", []) or []:
                if item.candidate_id == candidate.get("command_candidate_id"):
                    registry_candidate = item
                    break
            if registry_candidate is None and cmd:
                registry_candidate = registry.candidate_for_argv(cmd) if registry is not None else None
            if registry_candidate is not None:
                evidence_types = {
                    registry_evidence[evidence_id].source_type
                    for evidence_id in registry_candidate.evidence_ids
                    if evidence_id in registry_evidence
                }
                if "lockfile" in evidence_types:
                    points += SCORE_WEIGHTS["lockfile_match"]
                    reasons.append("lockfile_match +%d" % SCORE_WEIGHTS["lockfile_match"])
                if "readme_reference" in evidence_types:
                    points += SCORE_WEIGHTS["readme_exact_reference"]
                    reasons.append(
                        "readme_exact_reference +%d" % SCORE_WEIGHTS["readme_exact_reference"]
                    )
                if _dockerfile_corroborates(dockerfile_entrypoints, registry_candidate.argv):
                    points += SCORE_WEIGHTS["dockerfile_entrypoint_corroboration"]
                    reasons.append(
                        "dockerfile_entrypoint_corroboration +%d"
                        % SCORE_WEIGHTS["dockerfile_entrypoint_corroboration"]
                    )
                if any(
                    reason.startswith("README command reference")
                    for reason in registry_candidate.score_reasons
                ) and not evidence_types & {"pep621_script", "poetry_script", "package_json_script"}:
                    points += SCORE_WEIGHTS["readme_exact_reference"]
                    reasons.append(
                        "readme_exact_reference +%d" % SCORE_WEIGHTS["readme_exact_reference"]
                    )
        if candidate.get("expected_port"):
            points += SCORE_WEIGHTS["source_port_match"]
            reasons.append("source_port_match +%d" % SCORE_WEIGHTS["source_port_match"])
        if candidate.get("expected_port") == 0 and "port_conflict" not in reasons:
            points += SCORE_WEIGHTS["dependency_declaration_missing"]
            reasons.append(
                "unexplained_port_penalty %d" % SCORE_WEIGHTS["dependency_declaration_missing"]
            )
        candidate["evidence_score"] = max(0, min(100, points))
        candidate.setdefault("score_reasons", [])
        candidate["score_reasons"] = list(dict.fromkeys(
            list(candidate.get("score_reasons") or []) + reasons
        ))


def _dockerfile_corroborates(dockerfile_entrypoints, argv) -> bool:
    import json

    for payload in dockerfile_entrypoints:
        try:
            declared = json.loads(payload)
        except ValueError:
            continue
        if list(declared or []) == list(argv or []):
            return True
        if declared and argv and Path_name(declared[0]) == Path_name(argv[0]) and len(declared) > 1:
            return True
    return False


def order_run_candidates(run_candidates: List[Dict]) -> List[Dict]:
    """Stable order: evidence_score desc, original order for ties."""
    return sorted(
        run_candidates,
        key=lambda candidate: -float(candidate.get("evidence_score") or 0),
    )


def Path_name(value: str) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
