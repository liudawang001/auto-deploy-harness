"""Controller factory: selects and builds the appropriate controller."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from auto_harness.controllers.base import DeploymentController


class ControllerUnavailableError(RuntimeError):
    """Raised when a requested controller backend is not installed."""
    pass


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
