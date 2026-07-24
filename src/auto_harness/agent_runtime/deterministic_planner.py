"""Deterministic Deployment Planner for baseline evaluation.

Implements the same minimal interface as LLMDeploymentPlanner but
uses only fixed rules from snapshot data — no LLM calls whatsoever.

Used by the LLMNecessityEvaluator for baseline runs where
langgraph_planner_mode = "deterministic".
"""
import json
from typing import Dict, Optional

from auto_harness.providers.base import LLMResult


class DeterministicDeploymentPlanner:
    """Deterministic planner using only snapshot signals and fixed rules.

    Never calls any LLMProvider. Only uses detected_signals from the
    project snapshot to generate a valid DeploymentPlan JSON.

    Prohibited:
    - Calling any LLMProvider
    - Making network requests
    - Reading files beyond snapshot
    """

    def plan(self, snapshot: Dict, mode: str = "planner", skill_context=None) -> LLMResult:
        """Generate a deployment plan from snapshot using fixed rules.

        Fixed rules:
        - requirements.txt -> pip install -r requirements.txt
        - pyproject.toml/setup.py -> pip install -e .
        - No dep file -> pip --version (no-op)
        - Entrypoint: first from detected_signals.entrypoint_candidates
        - Port: first from detected_signals.ports, default 8000
        - Verify: GET /?trace_id={{trace_id}}

        Args:
            snapshot: Project snapshot dict with file_tree and detected_signals.
            mode: Planner mode (ignored, always deterministic).
            skill_context: Skill context (ignored).

        Returns:
            LLMResult with DeploymentPlan JSON, protocol="deterministic".
        """
        signals = snapshot.get("detected_signals", {}) if isinstance(snapshot.get("detected_signals"), dict) else {}
        raw_file_tree = snapshot.get("file_tree", [])
        file_tree = raw_file_tree if isinstance(raw_file_tree, (list, dict)) else []

        # Determine install commands
        install_commands = self._install_commands(file_tree, signals)

        # Determine entrypoint
        entrypoint_candidates = signals.get("entrypoint_candidates", [])
        entrypoint = entrypoint_candidates[0] if entrypoint_candidates else None

        if not entrypoint:
            plan = {
                "status": "no_safe_plan",
                "summary": "deterministic baseline: no entrypoint candidate found",
                "grounding": [],
                "environment": {
                    "backend": "venv",
                    "python": "3.10",
                    "install_commands": install_commands,
                },
            }
            return LLMResult(
                text=json.dumps(plan, ensure_ascii=False),
                raw={"source": "deterministic_rules"},
                protocol="deterministic",
            )

        # Determine port
        ports = signals.get("ports", [])
        port = int(ports[0]) if ports else 8000

        plan = {
            "status": "ok",
            "plan_id": "plan_deterministic_1",
            "summary": "Deterministic plan from detected signals.",
            "grounding": [
                {
                    "claim": "%s is the service entrypoint" % entrypoint,
                    "file": entrypoint,
                    "reason": "first entrypoint candidate from detected_signals",
                }
            ],
            "environment": {
                "backend": "venv",
                "python": "3.10",
                "install_commands": install_commands,
            },
            "model_assets": {
                "required": False,
                "strategy": "none",
                "env_vars": [],
            },
            "run": {
                "candidates": [
                    {
                        "id": "det_%s" % entrypoint.replace(".", "_"),
                        "cmd": [".venv/bin/python", entrypoint],
                        "expected_port": port,
                        "reason": "deterministic: %s on port %d" % (entrypoint, port),
                    }
                ],
                "selected_candidate_id": "det_%s" % entrypoint.replace(".", "_"),
            },
            "verify": {
                "service_type": "http",
                "request": {
                    "method": "GET",
                    "path": "/?_auto_harness_trace={{trace_id}}",
                },
                "success_evidence": "response contains current trace_id",
            },
            "risks": [],
            "fallbacks": [],
        }

        return LLMResult(
            text=json.dumps(plan, ensure_ascii=False),
            raw={"source": "deterministic_rules"},
            protocol="deterministic",
        )

    def replan(
        self,
        snapshot: Dict,
        previous_plan: Dict,
        failure: Dict,
        skill_context=None,
    ) -> LLMResult:
        """Deterministic baseline does not support adaptive replan.

        Always returns status=no_safe_plan. The repair loop is disabled
        in deterministic mode, so this should not normally be called.

        Args:
            snapshot: Project snapshot.
            previous_plan: Previous deployment plan.
            failure: Failure context.
            skill_context: Skill context.

        Returns:
            LLMResult with no_safe_plan status.
        """
        plan = {
            "status": "no_safe_plan",
            "summary": "deterministic baseline does not support adaptive replan",
            "grounding": [],
        }
        return LLMResult(
            text=json.dumps(plan, ensure_ascii=False),
            raw={"source": "deterministic_rules"},
            protocol="deterministic",
        )

    def _install_commands(self, file_tree, signals: Dict):
        """Determine install commands from file tree.

        Fixed rules:
        - requirements.txt -> pip install -r requirements.txt
        - pyproject.toml or setup.py -> pip install -e .
        - Neither -> pip --version (no-op check)
        """
        files = (
            set(file_tree.keys())
            if isinstance(file_tree, dict)
            else set(file_tree)
            if isinstance(file_tree, list)
            else set()
        )
        # Also check signals for dep files
        dep_files = signals.get("dependency_files", [])
        if isinstance(dep_files, list):
            files.update(dep_files)

        venv_cmd = ["python3", "-m", "venv", ".venv"]
        if "requirements.txt" in files:
            return [
                venv_cmd,
                [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
            ]
        if "pyproject.toml" in files or "setup.py" in files:
            return [
                venv_cmd,
                [".venv/bin/python", "-m", "pip", "install", "-e", "."],
            ]
        return [
            venv_cmd,
            [".venv/bin/python", "-m", "pip", "--version"],
        ]
