from typing import Any, Dict

from auto_harness.context.assembler import summarize_stage_results


def build_replan_delta(
    previous_plan: Dict[str, Any],
    stage_results: Dict[str, Any],
    failure_context: Dict[str, Any],
) -> Dict[str, Any]:
    summaries = summarize_stage_results(stage_results)
    completed = [
        stage
        for stage, result in summaries.items()
        if str(result.get("status", "")).lower() in {"passed", "success", "ok"}
    ]
    return {
        "previous_plan_id": str(previous_plan.get("plan_id", "")),
        "previous_plan_summary": str(previous_plan.get("summary", ""))[:1500],
        "completed_stages": completed,
        "failed_stage": str(failure_context.get("stage", "")),
        "failure_signature": str(failure_context.get("failure_signature", "")),
        "failure": {
            key: failure_context.get(key)
            for key in ("error_type", "error", "exit_code", "stderr_tail")
            if key in failure_context
        },
        "previous_stage_summary": summaries,
    }
