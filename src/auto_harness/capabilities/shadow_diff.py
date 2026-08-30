"""Shadow comparison between the legacy chain and the capability chain.

Phase B5: in shadow mode the new chain produces suggestions while the old
chain still decides execution.  The diff is auditable and its verdict gates
the enforce rollout: any ``new_less_safe`` classification blocks enforce.
"""

from typing import Dict, List


DIFF_CLASSIFICATIONS = (
    "equivalent",
    "new_more_complete",
    "new_safer",
    "new_less_complete",
    "new_less_safe",
    "incomparable",
)


def _argv_set(candidates: List[Dict]) -> set:
    return {tuple(item.get("cmd") or []) for item in candidates or []}


def _argv_verdicts(analysis: Dict) -> Dict[tuple, str]:
    """Map candidate argv -> authorization verdict recorded by the runner."""
    verdicts = {}
    for attempt in analysis.get("authorization_attempts") or []:
        argv = tuple(attempt.get("normalized_argv") or [])
        if argv:
            verdicts.setdefault(argv, str(attempt.get("verdict") or ""))
    return verdicts


def compute_shadow_diff(baseline: Dict, candidate: Dict) -> Dict:
    """Compare a baseline analysis against the capability-chain analysis.

    Both inputs are analyzer ``data`` dicts.  The comparison covers the
    dimensions required by the rollout plan: classification, install plan,
    run candidate set, top1 candidate, expected port and final status.
    """
    baseline_run = baseline.get("run_candidates") or []
    candidate_run = candidate.get("run_candidates") or []
    baseline_argvs = _argv_set(baseline_run)
    candidate_argvs = _argv_set(candidate_run)
    baseline_top = dict(baseline_run[0]) if baseline_run else {}
    candidate_top = dict(candidate_run[0]) if candidate_run else {}

    baseline_deployability = baseline.get("deployability") or {}
    candidate_deployability = candidate.get("deployability") or {}
    baseline_missing = set(baseline_deployability.get("missing_capabilities") or [])
    candidate_missing = set(candidate_deployability.get("missing_capabilities") or [])

    baseline_verdicts = _argv_verdicts(baseline)
    candidate_verdicts = _argv_verdicts(candidate)

    gained = sorted(candidate_argvs - baseline_argvs)
    lost = sorted(baseline_argvs - candidate_argvs)

    less_safe = []
    for argv in baseline_argvs & candidate_argvs:
        baseline_verdict = baseline_verdicts.get(argv, "")
        candidate_verdict = candidate_verdicts.get(argv, "")
        if (
            baseline_verdict == "auto_allowed"
            and candidate_verdict in {"candidate_rejected", "hard_denied"}
        ):
            less_safe.append(list(argv))
    # Losing an authorized runnable candidate is also a safety regression:
    # the operator loses a proven path without gaining anything.
    if not lost and not gained:
        pass
    for argv in lost:
        if baseline_verdicts.get(argv) == "auto_allowed" and not candidate_verdicts:
            less_safe.append(list(argv))

    if less_safe:
        classification = "new_less_safe"
    elif lost:
        classification = "new_less_complete"
    elif gained:
        classification = "new_more_complete"
    elif (
        baseline_top.get("cmd") == candidate_top.get("cmd")
        and baseline_deployability.get("status") == candidate_deployability.get("status")
        and list(baseline.get("frameworks") or []) == list(candidate.get("frameworks") or [])
        and (baseline.get("install_plan") or []) == (candidate.get("install_plan") or [])
        and str((baseline.get("verify_hint") or {}).get("service_type") or "")
        == str((candidate.get("verify_hint") or {}).get("service_type") or "")
    ):
        classification = "equivalent"
    else:
        classification = "incomparable"

    return {
        "classification": classification,
        "framework_classification": {
            "baseline": list(baseline.get("frameworks") or []),
            "candidate": list(candidate.get("frameworks") or []),
            "equal": list(baseline.get("frameworks") or [])
            == list(candidate.get("frameworks") or []),
        },
        "install_plan": {
            "baseline": baseline.get("install_plan") or [],
            "candidate": candidate.get("install_plan") or [],
            "equal": (baseline.get("install_plan") or [])
            == (candidate.get("install_plan") or []),
        },
        "run_candidate_set": {
            "baseline": [list(item) for item in sorted(baseline_argvs)],
            "candidate": [list(item) for item in sorted(candidate_argvs)],
            "gained": [list(item) for item in gained],
            "lost": [list(item) for item in lost],
        },
        "top1_candidate": {
            "baseline": baseline_top.get("cmd") or [],
            "candidate": candidate_top.get("cmd") or [],
            "equal": baseline_top.get("cmd") == candidate_top.get("cmd"),
        },
        "expected_port": {
            "baseline": int(baseline_top.get("expected_port") or 0),
            "candidate": int(candidate_top.get("expected_port") or 0),
        },
        "verify_protocol": {
            "baseline": str(
                (baseline.get("verify_hint") or {}).get("service_type") or ""
            ),
            "candidate": str(
                (candidate.get("verify_hint") or {}).get("service_type") or ""
            ),
        },
        "final_status": {
            "baseline": str(baseline_deployability.get("status") or ""),
            "candidate": str(candidate_deployability.get("status") or ""),
        },
        "missing_capabilities": {
            "baseline": sorted(baseline_missing),
            "candidate": sorted(candidate_missing),
        },
        "new_less_safe_commands": less_safe,
    }


def enforce_blockers(diff: Dict) -> List[str]:
    """Fail-closed blockers for switching the default chain to enforce."""
    blockers = []
    if not isinstance(diff, dict) or not diff:
        blockers.append("shadow_diff_missing")
        return blockers
    if diff.get("classification") == "new_less_safe":
        blockers.append("shadow_diff_new_less_safe")
    if diff.get("classification") == "incomparable":
        blockers.append("shadow_diff_incomparable")
    return blockers
