"""Deployment controllers: pluggable backends for deployment orchestration.

- LegacyController: wraps existing TaskRunner paths (default)
- LangGraphController: LangGraph StateGraph with checkpoint (optional)
"""
from auto_harness.controllers.base import DeploymentContext, DeploymentController, DeploymentResult
from auto_harness.controllers.factory import ControllerUnavailableError, create_controller

__all__ = [
    "DeploymentContext",
    "DeploymentController",
    "DeploymentResult",
    "ControllerUnavailableError",
    "create_controller",
]
