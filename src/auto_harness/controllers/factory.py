"""Controller factory: selects and builds the appropriate controller."""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_harness.controllers.base import DeploymentController


VALID_CONTROLLERS = {"legacy", "langgraph"}


class ControllerUnavailableError(RuntimeError):
    """Raised when a requested controller backend is not installed."""
    pass


def resolve_controller(
    *,
    explicit: Optional[str],
    configured_default: str,
    stored: Optional[str] = None,
    is_resume: bool = False,
) -> str:
    """Resolve the effective controller name.

    Rules:
    - deploy without --controller -> config.default_controller
    - deploy with explicit --controller -> explicit value
    - resume without --controller -> task.controller (stored)
    - resume with explicit != stored -> raise ValueError

    Raises:
        ValueError: if any controller name is unsupported or resume
                    attempts to switch controller.
    """
    if configured_default not in VALID_CONTROLLERS:
        raise ValueError(
            "unsupported configured controller: %s" % configured_default
        )

    if explicit is not None and explicit not in VALID_CONTROLLERS:
        raise ValueError("unsupported controller: %s" % explicit)

    if is_resume:
        if not stored:
            raise ValueError("stored controller is required for resume")
        if stored not in VALID_CONTROLLERS:
            raise ValueError("unsupported stored controller: %s" % stored)
        if explicit is not None and explicit != stored:
            raise ValueError(
                "controller_switch_on_resume_is_not_allowed: "
                "requested=%s stored=%s" % (explicit, stored)
            )
        return stored

    return explicit or configured_default


def create_controller(name: str, dependencies: Any) -> DeploymentController:
    """Create a deployment controller by name.

    Args:
        name: Controller name, either "legacy" or "langgraph".
        dependencies: Object that provides build methods for each controller.

    Returns:
        A DeploymentController instance.

    Raises:
        ControllerUnavailableError: If the langgraph backend is not installed.
        ValueError: If the controller name is not supported.
    """
    if name == "legacy":
        return dependencies.build_legacy_controller()

    if name == "langgraph":
        try:
            from auto_harness.controllers.langgraph import LangGraphController
        except ImportError as exc:
            if "langgraph" in str(exc).lower():
                raise ControllerUnavailableError(
                    "LangGraph backend is not installed; install the langgraph extra"
                ) from exc
            raise
        return LangGraphController(dependencies.graph_dependencies())

    raise ValueError("unsupported controller: %s" % name)
