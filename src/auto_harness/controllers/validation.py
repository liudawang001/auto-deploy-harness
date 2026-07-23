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


def validate_controller_run(*, controller, dry_run, provider_name, config):
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

    if not config.langgraph_require_llm:
        # LLM not required for langgraph — unusual but allowed
        return ControllerValidation(True)

    if provider_name == "mock" and not dry_run and not config.langgraph_allow_mock_in_execute:
        return ControllerValidation(False, "mock_provider_not_allowed_for_execute")

    if provider_name == "mock" and dry_run and not config.langgraph_allow_mock_in_dry_run:
        return ControllerValidation(False, "mock_provider_not_allowed")

    return ControllerValidation(True)
