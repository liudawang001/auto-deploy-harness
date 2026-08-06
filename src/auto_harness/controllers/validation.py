"""Controller validation: pre-run checks before any stage side effects.

Ensures LangGraph controller requirements are met (LLM provider available,
mock not used in execute mode, etc.). Fail-fast, never silent fallback.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerValidation:
    """Result of controller pre-run validation."""
    allowed: bool
    reason: str = ""


def resolve_planner_mode(*, requested_mode, provider_name, require_llm=False):
    """Resolve ``auto`` before graph construction or any stage side effect.

    ``auto`` selects the deterministic planner for the built-in mock (or an
    empty provider name), and the LLM planner for an explicitly configured
    real provider.  ``require_llm`` keeps a fail-closed deployment available
    for operators that deliberately require an LLM.
    """
    requested = str(requested_mode or "auto").strip().lower()
    if requested not in {"auto", "llm", "deterministic"}:
        raise ValueError("unsupported langgraph planner mode: %s" % requested)
    if requested != "auto":
        return requested
    provider = str(provider_name or "").strip().lower()
    if require_llm or (provider and provider != "mock"):
        return "llm"
    return "deterministic"


def validate_controller_run(
    *, controller, dry_run, provider_name, config, planner_mode=None
):
    """Validate that the controller can run with the given configuration.

    Args:
        controller: Controller name ("legacy" or "langgraph").
        dry_run: Whether this is a dry-run.
        provider_name: LLM provider name (e.g. "mock", "xunfei").
        config: HarnessConfig instance.

    Returns:
        ControllerValidation with allowed=True if the run may proceed,
        or allowed=False with a reason if it must be rejected.
    """
    if controller != "langgraph":
        return ControllerValidation(True)

    resolved_mode = planner_mode or resolve_planner_mode(
        requested_mode=getattr(config, "langgraph_planner_mode", "auto"),
        provider_name=provider_name,
        require_llm=getattr(config, "langgraph_require_llm", False),
    )
    if resolved_mode == "deterministic":
        return ControllerValidation(True)

    if provider_name == "mock" and not dry_run and not config.langgraph_allow_mock_in_execute:
        return ControllerValidation(False, "mock_provider_not_allowed_for_execute")

    if provider_name == "mock" and dry_run and not config.langgraph_allow_mock_in_dry_run:
        return ControllerValidation(False, "mock_provider_not_allowed")

    return ControllerValidation(True)
