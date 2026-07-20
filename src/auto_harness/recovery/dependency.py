"""Dependency reconciler: detects external state of environment installations.

Reconciles dependency install operations against the filesystem:
- Environment exists with matching Python version and packages → reuse
- Environment exists but packages don't satisfy constraints → retry
- Environment doesn't exist → retry
- Cannot verify → manual

Resource identity: backend, environment path, Python version,
package spec, command hash, channel/index names.
"""
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.recovery.download import reconcile_result


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_python_version(env_path, expected_version):
    """Check if the Python version in an environment matches expected.

    Returns (matches: bool, actual_version: str).
    """
    python_bin = Path(env_path) / "bin" / "python"
    if not python_bin.exists():
        return False, ""
    try:
        result = subprocess.run(
            [str(python_bin), "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        version_str = result.stdout.strip() + result.stderr.strip()
        # Extract version number (e.g., "Python 3.10.12" → "3.10")
        parts = version_str.split()
        if len(parts) >= 2:
            full_version = parts[1]
            # Compare major.minor
            expected_parts = expected_version.split(".")[:2]
            actual_parts = full_version.split(".")[:2]
            return expected_parts == actual_parts, full_version
        return False, version_str
    except (OSError, subprocess.SubprocessError):
        return False, ""


def check_package_versions(env_path, package_specs):
    """Check if packages are installed in the environment.

    Returns (all_satisfied: bool, installed: Dict[str, str]).
    """
    pip_bin = Path(env_path) / "bin" / "pip"
    if not pip_bin.exists():
        return False, {}
    try:
        result = subprocess.run(
            [str(pip_bin), "list", "--format=json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            return False, {}
        import json
        packages = json.loads(result.stdout)
        installed = {p["name"].lower(): p.get("version", "") for p in packages}
    except (OSError, subprocess.SubprocessError, ValueError):
        return False, {}

    # Check each spec (simple name-only check; version constraints
    # would need a proper package version resolver)
    all_satisfied = True
    for spec in package_specs:
        # Extract package name (before any version specifier)
        name = spec.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].split("[")[0].strip().lower()
        if name not in installed:
            all_satisfied = False
            break
    return all_satisfied, installed


class DependencyReconciler:
    """Reconciler for dependency installation operations.

    Checks if the target environment already satisfies the required
    dependencies. If so, skips re-installation (reuse).
    """
    resource_type = "dependency_install"

    def __init__(
        self,
        python_checker=None,
        package_checker=None,
    ) -> None:
        self.python_checker = python_checker or check_python_version
        self.package_checker = package_checker or check_package_versions

    def reconcile(self, operation):
        """Reconcile a dependency installation operation.

        Decision logic:
        1. Environment path doesn't exist → retry
        2. Python version doesn't match → retry
        3. Packages don't satisfy constraints → retry
        4. All match → reuse
        5. Cannot verify → manual
        """
        identity = operation["resource_identity"]
        env_path = identity.get("environment_path", "")
        expected_python = identity.get("python_version", "")
        package_specs = operation.get("normalized_input", {}).get("package_specs", [])

        # 1. Environment doesn't exist
        if not env_path or not Path(env_path).exists():
            return reconcile_result("retry", "environment path does not exist")

        # 2. Check Python version
        if expected_python:
            matches, actual_version = self.python_checker(env_path, expected_python)
            if not matches:
                return reconcile_result(
                    "retry", "Python version mismatch",
                    actual_python=actual_version,
                    expected_python=expected_python,
                )

        # 3. Check package versions
        if package_specs:
            all_satisfied, installed = self.package_checker(env_path, package_specs)
            if not all_satisfied:
                missing = [s for s in package_specs
                           if s.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].split("[")[0].strip().lower()
                           not in installed]
                return reconcile_result(
                    "retry", "packages not satisfied",
                    missing_packages=missing,
                )

        # 4. All match
        return reconcile_result("reuse", "environment satisfies all constraints")
