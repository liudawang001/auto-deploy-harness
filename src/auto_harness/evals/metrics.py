from typing import Dict, List


def summarize_runs(runs: List[Dict]) -> Dict:
    total = len(runs)
    verify_pass = sum(1 for item in runs if item.get("verify_status") in ("pass", "passed"))
    failed = total - verify_pass
    return {
        "total": total,
        "verify_pass": verify_pass,
        "failed": failed,
        "success_rate": verify_pass / total if total else 0.0,
        "repair_attempt_count": sum(int(item.get("repair_attempt_count") or 0) for item in runs),
        "repair_executed_count": sum(int(item.get("repair_executed_count") or 0) for item in runs),
        "repair_verified_success_count": sum(int(item.get("repair_verified_success_count") or 0) for item in runs),
        "unsafe_action_rejected_count": sum(int(item.get("unsafe_action_rejected_count") or 0) for item in runs),
        "manual_intervention_required": sum(1 for item in runs if item.get("manual_intervention_required")),
        "avg_iterations_to_success": _avg([item.get("iterations_to_success") for item in runs if item.get("verify_status") in ("pass", "passed")]),
        "llm_action_acceptance_rate": _rate(sum(int(item.get("accepted_action_count") or 0) for item in runs), sum(int(item.get("accepted_action_count") or 0) + int(item.get("rejected_action_count") or 0) for item in runs)),
        "llm_action_rejection_rate": _rate(sum(int(item.get("rejected_action_count") or 0) for item in runs), sum(int(item.get("accepted_action_count") or 0) + int(item.get("rejected_action_count") or 0) for item in runs)),
    }


def _avg(values) -> float:
    nums = [float(item) for item in values if item not in (None, "")]
    return sum(nums) / len(nums) if nums else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
