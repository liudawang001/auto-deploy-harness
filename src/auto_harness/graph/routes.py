"""Route functions for the LangGraph deployment StateGraph.

Routes are pure functions: they only read state and return a string
identifying the next node. They never call LLM, execute shell,
read/write files, or modify the input state.
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
        "replan" - stage failed and replan count not exhausted
        "stop" - stage failed and replan exhausted
    """
    current = state.get("current_stage", "")
    result = state.get("stage_results", {}).get(current, {})
    status = result.get("status", "failed")
    if status in ("failed", "uncertain"):
        if int(state.get("replan_count", 0)) < int(state.get("max_replans", 0)):
            return "replan"
        return "stop"
    return "continue"


def route_after_verify(state):
    """Route after verify: passed → report, retryable → replan, terminal → stop."""
    if state.get("verify_status") in ("pass", "passed"):
        return "report"
    if int(state.get("replan_count", 0)) < int(state.get("max_replans", 0)):
        return "replan"
    return "stop"


def route_resume_stage(state):
    """Route from select_resume to the appropriate stage node.

    Only allows whitelisted stages. Unknown values fall back to analyze.
    """
    requested = state.get("resume_from_stage", "analyze")
    allowed = {
        "analyze", "resource_plan", "env_solve", "env_deploy",
        "model_prepare", "runner", "verify",
    }
    return requested if requested in allowed else "analyze"


def route_after_replan(state):
    """Route after replan: new plan goes to parse, exhausted to stop."""
    count = int(state.get("replan_count", 0))
    maximum = int(state.get("max_replans", 0))
    if not state.get("raw_plan_path") or count > maximum:
        return "stop"
    return "parse"
