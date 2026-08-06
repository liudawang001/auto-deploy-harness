"""Route functions for the LangGraph deployment StateGraph.

Routes are pure functions: they only read state and return a string
identifying the next node. They never call LLM, execute shell,
read/write files, or modify the input state.

Per Phase 8 spec: all routes must be pure, read-only, no side effects,
and default to "stop" for unknown states.
"""


def route_after_parse(state):
    """Route after parse: valid plan goes to policy, invalid to stop."""
    return "valid" if not state.get("stop_reason") else "invalid"


def route_after_policy(state):
    """Route after policy: allowed goes to compile, rejected to stop."""
    return "compile" if not state.get("stop_reason") else "stop"


def route_after_stage(state):
    """Route after a non-verify stage execution.

    Returns:
        "continue" - next stage in sequence
        "observe_failure" - stage failed, observe before diagnose
    """
    current = state.get("current_stage", "")
    result = state.get("stage_results", {}).get(current, {})
    status = result.get("status", "failed")
    if status in ("failed", "uncertain"):
        return "observe_failure"
    return "continue"


def route_after_verify(state):
    """Route after verify without treating a dry-run as verified execution.

    A dry-run is an inspection/planning result.  Its expected ``uncertain``
    verification outcome is terminal and reportable, not a repair signal.
    """
    if state.get("verify_status") in ("pass", "passed"):
        return "report"
    if state.get("dry_run"):
        return "report"
    return "observe_failure"


def route_resume_stage(state):
    """Route from select_resume to the appropriate stage node.

    Only allows whitelisted stages. Unknown values fall back to analyze.
    Side-effect stages route through their recovery gate.
    """
    requested = state.get("resume_from_stage", "analyze")
    allowed = {
        "analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy",
        "model_prepare", "runner", "verify",
    }
    return requested if requested in allowed else "analyze"


def route_after_llm_plan(state):
    """Route after LLM plan/replan: valid plan goes to parse, failure to stop.

    Catches both explicit stop_reason from LLM failure and missing raw_plan_path.
    """
    if state.get("stop_reason") or not state.get("raw_plan_path"):
        return "stop"
    return "parse"


def route_after_replan(state):
    """Route after replan: new plan goes to parse, exhausted or LLM failure to stop."""
    if state.get("stop_reason"):
        return "stop"
    count = int(state.get("replan_count", 0))
    maximum = int(state.get("max_replans", 0))
    if not state.get("raw_plan_path") or count > maximum:
        return "stop"
    return "parse"


def route_after_diagnose(state):
    """Route after diagnose: repair_plan if actions, replan if plan change, stop if exhausted/invalid."""
    if state.get("stop_reason"):
        return "stop"
    diagnosis = state.get("diagnosis", {})
    if diagnosis.get("accepted_actions"):
        return "repair_plan"
    if diagnosis.get("rerun_from") or diagnosis.get("plan_change_required"):
        return "replan"
    if diagnosis.get("status") in {"invalid", "failed", "unavailable"}:
        return "replan"  # Fallback to replan when diagnosis can't help
    return "replan"


def route_after_repair_policy(state):
    """Route after repair_policy: apply if allowed, approval if needed, stop if rejected."""
    if state.get("stop_reason"):
        return "stop"
    if state.get("pending_approval"):
        return "approval"
    policy_result = state.get("repair_policy_result", {})
    if policy_result.get("allowed"):
        return "apply"
    return "stop"


def route_after_approval(state):
    """Route after approval: apply repair, cleanup, retry, or stop on reject.

    Per Phase 8 spec:
    - reject → stop
    - approve for repair_apply → repair_apply
    - approve for cleanup → cleanup
    - approve for retry → select_repair_resume
    """
    approval_history = state.get("approval_history", [])
    if approval_history:
        last = approval_history[-1] if isinstance(approval_history, list) else {}
        decision = last.get("decision", "")
        if decision == "reject":
            return "stop"
        if decision == "approve":
            approved_action = state.get("approved_action", "")
            if approved_action == "cleanup_then_retry":
                return "cleanup"
            resume_target = state.get("approval_resume_target", "repair_apply")
            if resume_target == "repair_apply":
                return "repair_apply"
            return "retry"
    return "stop"


def route_repair_resume_stage(state):
    """Route from select_repair_resume to the appropriate stage node.

    Only allows whitelisted safe rerun stages.
    Side-effect stages route through their recovery gate.
    """
    requested = state.get("resume_from_stage", "verify")
    allowed = {
        "analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy",
        "model_prepare", "runner", "verify",
    }
    return requested if requested in allowed else "analyze"
