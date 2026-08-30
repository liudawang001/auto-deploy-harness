"""Structured, evidence-backed project deployment capabilities."""

from auto_harness.capabilities.assessor import DeployabilityAssessor
from auto_harness.capabilities.dependency_parser import DependencyManifestParser
from auto_harness.capabilities.detector import CapabilityDetector
from auto_harness.capabilities.legacy_compiler import LegacyAnalysisCompiler
from auto_harness.capabilities.schemas import (
    CapabilityEvidence,
    DependencyManifest,
    DeployabilityAssessment,
    DeploymentCandidate,
    ProjectCapabilities,
)

__all__ = [
    "CapabilityDetector",
    "CapabilityEvidence",
    "DependencyManifest",
    "DependencyManifestParser",
    "DeployabilityAssessment",
    "DeployabilityAssessor",
    "DeploymentCandidate",
    "LegacyAnalysisCompiler",
    "ProjectCapabilities",
]
