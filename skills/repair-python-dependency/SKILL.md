---
name: repair-python-dependency
version: 1.0.0
type: repair_skill
stages:
  - repair
  - replan
  - env_deploy
failure_categories:
  - dependency_missing
  - version_conflict
  - module_not_found
risk_level: medium
side_effects: false
allowed_tools:
  - apply_dependency_constraint
  - propose_repair
success_signals:
  - missing package installed successfully
  - version conflict resolved
  - import error no longer occurs
regression_cases:
  - dependency_missing_install
  - version_conflict_constraint
---

# Purpose

Guide repair of Python dependency failures during deployment. Covers missing packages, version conflicts, and import errors that prevent the target project from running.

# When To Use

Use when:
- A deployment fails with `ModuleNotFoundError` or `ImportError`.
- Log diagnosis classifies the failure as `dependency_missing`, `version_conflict`, or `module_not_found`.
- The deterministic `LogClassifier` identifies a missing or conflicting package.
- The repair loop needs a skill-specific strategy for dependency issues.

# Guidance

- Extract the exact package name from the error message. Do not guess.
- For `ModuleNotFoundError: No module named 'X'`, install `X` via pip.
- For version conflicts (e.g., `numpy.dtype size changed`, `pydantic v1 vs v2`), apply a version constraint that matches the project's requirements.
- Prefer minimal constraints: `numpy<2`, `pydantic>=1.10,<2`, `opencv-python-headless` instead of `opencv-python`.
- Always propose the repair through `apply_dependency_constraint` tool call. Do not execute pip directly.
- The repair action must still pass through `RepairPolicy`: package name must match safe regex, no URL/path/shell metacharacters, no arbitrary index URLs.
- After repair, the resume should restart from `env_deploy` to re-run the installation.

# Allowed Plan Effects

- Add or modify dependency constraints in the install plan.
- Propose `install_package` action for the missing or conflicting package.
- Suggest `rerun_from_stage: env_deploy` to re-install after repair.

# Forbidden

- Do not execute pip or conda commands directly.
- Do not modify source files to remove or change imports.
- Do not propose arbitrary pip index URLs or `--trusted-host` flags.
- Do not install packages from git URLs or local paths.
- Do not bypass `RepairPolicy` validation.
