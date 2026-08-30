"""Extensible, side-effect-free deployment adapter foundation."""

from auto_harness.deployment_adapters.composer import CandidateComposer
from auto_harness.deployment_adapters.registry import DeploymentAdapterRegistry
from auto_harness.deployment_adapters.schemas import (
    AdapterDetection,
    DetectionContext,
    EnvironmentProposal,
    RunProposal,
    VerifyProposal,
)

__all__ = [
    "AdapterDetection",
    "CandidateComposer",
    "DeploymentAdapterRegistry",
    "DetectionContext",
    "EnvironmentProposal",
    "RunProposal",
    "VerifyProposal",
]
