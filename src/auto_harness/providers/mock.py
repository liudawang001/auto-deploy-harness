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
    """

    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
        # Extract stage context from messages to make stage-specific decisions
        stage_hint = self._extract_stage_hint(messages)

        if stage_hint == "runner":
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
