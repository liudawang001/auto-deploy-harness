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
        source_build_commands = [
            list(item.get("cmd") or [])
            for item in signals.get("source_build_commands", [])
            if isinstance(item, dict) and item.get("cmd")
        ]
        install_commands.extend(source_build_commands)
        setup_commands = [
            list(item.get("cmd") or [])
            for item in signals.get("documented_setup_commands", [])
            if isinstance(item, dict) and item.get("cmd")
        ]
        install_commands.extend(
            [[".venv/bin/%s" % cmd[0]] + cmd[1:] for cmd in setup_commands]
        )

        # Determine entrypoint. Prefer a documented command backed by a
        # declared PEP 621 console script, then fall back to root Python files.
        documented = signals.get("documented_run_commands", [])
        console_scripts = signals.get("console_scripts", [])
        script_names = {
            str(item.get("name")) for item in console_scripts
            if isinstance(item, dict) and item.get("name")
        }
        documented_candidate = next(
            (
                item for item in documented
                if isinstance(item, dict)
                and isinstance(item.get("cmd"), list)
                and item["cmd"]
                and item["cmd"][0] in script_names
            ),
            None,
        )
        entrypoint_candidates = signals.get("entrypoint_candidates", [])
        entrypoint = entrypoint_candidates[0] if entrypoint_candidates else None

        if not entrypoint and not documented_candidate:
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

        ports = signals.get("ports", [])
        if documented_candidate:
            documented_cmd = list(documented_candidate["cmd"])
            command = [".venv/bin/%s" % documented_cmd[0]] + documented_cmd[1:]
            port = (
                self._command_port(command)
                or int(documented_candidate.get("expected_port") or 0)
                or (int(ports[0]) if ports else 8000)
            )
            candidate_id = "det_console_%s" % documented_cmd[0].replace("-", "_")
            grounding = [
                {
                    "claim": "%s is a declared public console script" % documented_cmd[0],
                    "file": "pyproject.toml",
                    "reason": "PEP 621 [project.scripts] declaration",
                },
                {
                    "claim": "%s is a documented service command" % " ".join(documented_cmd),
                    "file": documented_candidate["source"],
                    "reason": "README service launch example",
                },
            ]
            if source_build_commands:
                build_source = next(
                    (
                        str(item.get("source"))
                        for item in signals.get("source_build_commands", [])
                        if isinstance(item, dict) and item.get("source")
                    ),
                    "README.md",
                )
                grounding.append({
                    "claim": "the source checkout requires its documented frontend build",
                    "file": build_source,
                    "reason": "build-frontend creates the missing packaged dashboard artifact",
                })
            if setup_commands:
                setup_source = next(
                    (
                        str(item.get("source"))
                        for item in signals.get("documented_setup_commands", [])
                        if isinstance(item, dict) and item.get("source")
                    ),
                    documented_candidate["source"],
                )
                grounding.append({
                    "claim": "%s is the documented non-interactive initialization command" % " ".join(setup_commands[0]),
                    "file": setup_source,
                    "reason": "public source deployment instructions require initialization before launch",
                })
            reason = "deterministic: declared console script and documented service command on port %d" % port
        else:
            port = int(ports[0]) if ports else 8000
            command = [".venv/bin/python", entrypoint]
            candidate_id = "det_%s" % entrypoint.replace(".", "_")
            grounding = [
                {
                    "claim": "%s is the service entrypoint" % entrypoint,
                    "file": entrypoint,
                    "reason": "first entrypoint candidate from detected_signals",
                }
            ]
            reason = "deterministic: %s on port %d" % (entrypoint, port)

        python_version = self._minimum_python(signals.get("python_requires", "")) or "3.10"

        plan = {
            "status": "ok",
            "plan_id": "plan_deterministic_1",
            "summary": "Deterministic plan from detected signals.",
            "grounding": grounding,
            "environment": {
                "backend": "venv",
                "python": python_version,
                "channels": ["conda-forge"],
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
                        "id": candidate_id,
                        "cmd": command,
                        "expected_port": port,
                        "reason": reason,
                    }
                ],
                "selected_candidate_id": candidate_id,
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

    @staticmethod
    def _command_port(command) -> int:
        for index, arg in enumerate(command[:-1]):
            if arg in ("--port", "-p"):
                try:
                    port = int(command[index + 1])
                except (TypeError, ValueError):
                    return 0
                return port if 0 < port <= 65535 else 0
        return 0

    @staticmethod
    def _minimum_python(constraint: str) -> str:
        import re

        matches = re.findall(r">=?\s*(\d+)\.(\d+)", str(constraint or ""))
        if not matches:
            return ""
        major, minor = max((int(major), int(minor)) for major, minor in matches)
        return "%d.%d" % (major, minor)

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

        if "uv.lock" in files and "pyproject.toml" in files:
            return [
                ["python3", "-m", "venv", ".venv"],
                [".venv/bin/python", "-m", "pip", "install", "uv"],
                [".venv/bin/uv", "sync", "--frozen", "--no-dev"],
            ]

        venv_cmd = ["python3", "-m", "venv", ".venv"]
        if "requirements.txt" in files:
            return [
                venv_cmd,
                [".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"],
                [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
            ]
        if "pyproject.toml" in files or "setup.py" in files:
            return [
                venv_cmd,
                [".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"],
                [".venv/bin/python", "-m", "pip", "install", "-e", "."],
            ]
        return [
            venv_cmd,
            [".venv/bin/python", "-m", "pip", "--version"],
        ]
