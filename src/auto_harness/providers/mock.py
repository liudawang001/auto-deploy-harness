import json
from typing import List

from auto_harness.providers.base import LLMResult, Message


class MockLLMProvider:
    """Mock LLM provider for testing.

    Returns stage-specific decisions that demonstrate LLM helping:
    - runner: selects correct candidate
    - verify: selects probe with trace
    - env: proposes dependency constraint
    - repair: proposes repair action
    - plan_first: returns valid DeploymentPlan JSON
    - replan: returns revised DeploymentPlan JSON
    """

    def __init__(self) -> None:
        self._call_count = 0

    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
        self._call_count += 1
        # Extract stage context from messages to make stage-specific decisions
        stage_hint = self._extract_stage_hint(messages)

        if stage_hint == "plan_first":
            content = self._plan_first_decision(messages)
        elif stage_hint == "replan":
            content = self._replan_decision(messages)
        elif stage_hint == "runner":
            content = self._runner_decision(messages)
        elif stage_hint == "verify":
            content = self._verify_decision(messages)
        elif stage_hint == "env_solve":
            content = self._env_decision(messages)
        elif stage_hint == "repair":
            content = self._repair_decision(messages)
        else:
            content = {
                "status": "ok",
                "summary": "mock provider response",
                "message_count": len(messages),
            }

        return LLMResult(text=json.dumps(content, ensure_ascii=False), raw=content, usage={})

    def _extract_stage_hint(self, messages: List[Message]) -> str:
        """Extract stage hint from message content."""
        for msg in messages:
            if hasattr(msg, 'content'):
                content = str(msg.content).lower()
                # Plan-first detection (highest priority)
                if "deployment plan" in content or "project snapshot" in content:
                    return "plan_first"
                if "previous deployment plan" in content or "failure context" in content:
                    return "replan"
                if "runner" in content or "run_candidates" in content:
                    return "runner"
                elif "verify" in content or "probe" in content:
                    return "verify"
                elif "env" in content or "dependency" in content or "install" in content:
                    return "env_solve"
                elif "repair" in content:
                    return "repair"
        return ""

    def _runner_decision(self, messages: List[Message]) -> dict:
        """Return runner decision - select first candidate."""
        return {
            "status": "ok",
            "tool_call": {
                "name": "select_runner_candidate",
                "input": {
                    "candidate_id": "cand_0",
                    "reason": "Selected first candidate based on entry point analysis",
                },
            },
            "hypothesis": "The first run candidate is likely the correct entry point",
            "confidence": 0.8,
        }

    def _verify_decision(self, messages: List[Message]) -> dict:
        """Return verify decision - use HTTP probe with trace."""
        return {
            "status": "ok",
            "tool_call": {
                "name": "probe_http",
                "input": {
                    "url": "http://127.0.0.1:{{port}}/",
                    "method": "GET",
                    "trace_template": "_auto_harness_trace={{trace_id}}",
                },
            },
            "hypothesis": "HTTP probe with trace will verify the service is responding correctly",
            "confidence": 0.9,
        }

    def _env_decision(self, messages: List[Message]) -> dict:
        """Return env decision - propose dependency constraint."""
        return {
            "status": "ok",
            "tool_call": {
                "name": "propose_dependency_constraint",
                "input": {
                    "package": "requests",
                    "version_spec": "",
                    "reason": "Missing dependency detected in import error",
                },
            },
            "hypothesis": "Adding missing dependency will resolve import error",
            "confidence": 0.95,
        }

    def _repair_decision(self, messages: List[Message]) -> dict:
        """Return repair decision - propose install_package."""
        return {
            "status": "ok",
            "tool_call": {
                "name": "apply_repair",
                "input": {
                    "action_type": "install_package",
                    "package": "requests",
                    "reason": "Install missing dependency to fix import error",
                },
            },
            "hypothesis": "Installing the missing package will fix the import error",
            "confidence": 0.95,
        }

    def _plan_first_decision(self, messages: List[Message]) -> dict:
        """Return a valid DeploymentPlan JSON for plan-first mode."""
        # Try to extract port from snapshot
        port = 8917
        entrypoint = "app.py"
        for msg in messages:
            content = str(getattr(msg, 'content', ''))
            # Look for port signals
            import re
            port_match = re.search(r'"ports"\s*:\s*\[\s*(\d+)', content)
            if port_match:
                port = int(port_match.group(1))
            # Look for entrypoint signals
            if "server.py" in content.lower() and "app.py" not in content.lower():
                entrypoint = "server.py"

        return {
            "status": "ok",
            "plan_id": "plan_mock_%s" % (self._call_count),
            "summary": "Run the HTTP service in a venv (mock plan).",
            "grounding": [
                {
                    "claim": "%s is the service entrypoint" % entrypoint,
                    "file": entrypoint,
                    "reason": "%s contains the HTTP service" % entrypoint,
                }
            ],
            "environment": {
                "backend": "venv",
                "python": "3.10",
                "install_commands": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ],
            },
            "model_assets": {
                "required": False,
                "strategy": "none",
                "env_vars": [],
            },
            "run": {
                "candidates": [
                    {
                        "id": "llm_%s" % entrypoint.replace(".", "_"),
                        "cmd": [".venv/bin/python", entrypoint],
                        "expected_port": port,
                        "reason": "%s starts HTTPServer on %d" % (entrypoint, port),
                    }
                ],
                "selected_candidate_id": "llm_%s" % entrypoint.replace(".", "_"),
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
            "fallbacks": [
                {
                    "trigger": "runner_exited",
                    "next_action": "inspect runner log and replan",
                }
            ],
        }

    def _replan_decision(self, messages: List[Message]) -> dict:
        """Return a revised DeploymentPlan for replan mode."""
        # If previous plan used app.py and it failed, try server.py
        entrypoint = "server.py"
        port = 8917
        for msg in messages:
            content = str(getattr(msg, 'content', ''))
            if "app.py" in content:
                entrypoint = "server.py"
            if "server.py" in content and "failed" in content.lower():
                entrypoint = "app.py"
            import re
            port_match = re.search(r'"expected_port"\s*:\s*(\d+)', content)
            if port_match:
                port = int(port_match.group(1))

        return {
            "status": "ok",
            "plan_id": "plan_replan_%s" % (self._call_count),
            "summary": "Revised plan: try %s instead." % entrypoint,
            "grounding": [
                {
                    "claim": "%s is the service entrypoint" % entrypoint,
                    "file": entrypoint,
                    "reason": "previous entrypoint failed, %s is the correct entrypoint" % entrypoint,
                }
            ],
            "environment": {
                "backend": "venv",
                "python": "3.10",
                "install_commands": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ],
            },
            "model_assets": {
                "required": False,
                "strategy": "none",
                "env_vars": [],
            },
            "run": {
                "candidates": [
                    {
                        "id": "llm_%s" % entrypoint.replace(".", "_"),
                        "cmd": [".venv/bin/python", entrypoint],
                        "expected_port": port,
                        "reason": "revised: %s is the correct entrypoint" % entrypoint,
                    }
                ],
                "selected_candidate_id": "llm_%s" % entrypoint.replace(".", "_"),
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
