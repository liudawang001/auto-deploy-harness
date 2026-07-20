"""Tests for DependencyReconciler.

Phase 7 tests: environment installation reconciliation,
Python version checking, package version checking, and decisions.
"""
import pytest
from pathlib import Path

from auto_harness.recovery.dependency import (
    DependencyReconciler,
    check_python_version,
    check_package_versions,
    sha256_text,
)
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_dependency_operation(
    env_path="/path/to/env",
    python_version="3.10",
    package_specs=None,
    backend="venv",
):
    """Build a dependency install operation dict for testing."""
    if package_specs is None:
        package_specs = ["gradio", "torch"]
    identity = {
        "backend": backend,
        "environment_path": env_path,
        "python_version": python_version,
        "command_hash": sha256_text("pip install " + " ".join(package_specs)),
    }
    normalized_input = {"package_specs": package_specs}
    operation_id = compute_operation_id(
        "task1", "env_deploy", "install", normalized_input, identity,
    )
    return {
        "operation_id": operation_id,
        "task_id": "task1",
        "stage": "env_deploy",
        "action": "install",
        "resource_type": "dependency_install",
        "resource_identity": identity,
        "normalized_input": normalized_input,
        "normalized_input_hash": canonical_json(normalized_input),
        "observed_resource": {},
        "status": "planned",
    }


class FakePythonChecker:
    """Configurable Python version checker for testing."""
    def __init__(self, matches=True, version="3.10.12"):
        self.matches = matches
        self.version = version

    def __call__(self, env_path, expected_version):
        return self.matches, self.version


class FakePackageChecker:
    """Configurable package version checker for testing."""
    def __init__(self, all_satisfied=True, installed=None):
        self.all_satisfied = all_satisfied
        self.installed = installed or {"gradio": "4.0.0", "torch": "2.0.0"}

    def __call__(self, env_path, package_specs):
        return self.all_satisfied, self.installed


# -------------------------------------------------------------------
# sha256_text Tests
# -------------------------------------------------------------------

class TestSha256Text:
    def test_deterministic(self):
        assert sha256_text("pip install gradio") == sha256_text("pip install gradio")

    def test_different(self):
        assert sha256_text("pip install gradio") != sha256_text("pip install torch")


# -------------------------------------------------------------------
# DependencyReconciler Tests
# -------------------------------------------------------------------

class TestDependencyReconciler:
    def test_retry_when_env_missing(self):
        """Environment path doesn't exist → retry."""
        op = make_dependency_operation(env_path="/nonexistent/path")
        reconciler = DependencyReconciler(
            python_checker=FakePythonChecker(),
            package_checker=FakePackageChecker(),
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
        assert "does not exist" in result["reason"]

    def test_retry_when_python_mismatch(self, tmp_path):
        """Python version doesn't match → retry."""
        env_path = str(tmp_path / "env")
        Path(env_path).mkdir()
        op = make_dependency_operation(env_path=env_path, python_version="3.11")
        reconciler = DependencyReconciler(
            python_checker=FakePythonChecker(matches=False, version="3.9.0"),
            package_checker=FakePackageChecker(),
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
        assert "Python version mismatch" in result["reason"]

    def test_retry_when_packages_missing(self, tmp_path):
        """Packages not satisfied → retry."""
        env_path = str(tmp_path / "env")
        Path(env_path).mkdir()
        op = make_dependency_operation(
            env_path=env_path,
            package_specs=["gradio", "nonexistent_pkg"],
        )
        reconciler = DependencyReconciler(
            python_checker=FakePythonChecker(),
            package_checker=FakePackageChecker(all_satisfied=False, installed={"gradio": "4.0.0"}),
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
        assert "packages not satisfied" in result["reason"]

    def test_reuse_when_all_match(self, tmp_path):
        """All constraints satisfied → reuse."""
        env_path = str(tmp_path / "env")
        Path(env_path).mkdir()
        op = make_dependency_operation(env_path=env_path)
        reconciler = DependencyReconciler(
            python_checker=FakePythonChecker(),
            package_checker=FakePackageChecker(),
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"
        assert "satisfies" in result["reason"]

    def test_reuse_without_package_specs(self, tmp_path):
        """No package_specs → skip package check, reuse if Python matches."""
        env_path = str(tmp_path / "env")
        Path(env_path).mkdir()
        op = make_dependency_operation(env_path=env_path, package_specs=[])
        reconciler = DependencyReconciler(
            python_checker=FakePythonChecker(),
            package_checker=FakePackageChecker(),
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"

    def test_resource_type(self):
        assert DependencyReconciler.resource_type == "dependency_install"

    def test_empty_env_path(self):
        """Empty env_path → retry."""
        op = make_dependency_operation(env_path="")
        reconciler = DependencyReconciler(
            python_checker=FakePythonChecker(),
            package_checker=FakePackageChecker(),
        )
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
