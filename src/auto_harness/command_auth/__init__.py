"""Repository-grounded command discovery and authorization."""

from auto_harness.command_auth.discovery import CommandDiscoveryService
from auto_harness.command_auth.policy import CommandAuthorizationEngine
from auto_harness.command_auth.schemas import (
    CommandCandidate,
    CommandDecision,
    CommandEvidence,
    CommandRegistry,
)
from auto_harness.command_auth.selector import CommandCandidateSelector

__all__ = [
    "CommandAuthorizationEngine",
    "CommandCandidate",
    "CommandCandidateSelector",
    "CommandDecision",
    "CommandDiscoveryService",
    "CommandEvidence",
    "CommandRegistry",
]
