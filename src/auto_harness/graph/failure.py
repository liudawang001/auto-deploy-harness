"""Failure observation: deterministic fact extraction from failed stages.

FailureObserver builds a structured failure context from graph state,
without calling LLM. It extracts error details, truncates output,
sanitizes secrets, and computes a stable failure signature.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from auto_harness.agent.safety import AgentInputSanitizer


class FailureObserver:
    """Observe failure facts from graph state — deterministic, no LLM."""

    MAX_FIELD_CHARS = 4000

    def __init__(self, max_input_chars: int = 20000) -> None:
        self.max_input_chars = max_input_chars

    def build(self, state: dict) -> dict:
        """Build failure context from state.

        Returns a dict with:
          - failed_stage, status, summary, error
          - stdout_tail, stderr_tail
          - checks, evidence_paths
          - previous_plan_id, selected_candidate_id
          - runtime_backend, replan_count, repair_count
        """
        failed_stage = state.get("failed_stage", "")
        stage_results = state.get("stage_results", {})
        failed_result = stage_results.get(failed_stage, {})

        # Extract basic fields with truncation
        summary = self._truncate(str(failed_result.get("summary", "")))
        error = self._truncate(str(failed_result.get("error", "")))
        status = failed_result.get("status", "failed")

        # Extract stdout/stderr tails
        data = failed_result.get("data", {}) if isinstance(failed_result, dict) else {}
        if not isinstance(data, dict):
            data = {}
        stdout_tail = self._truncate(str(data.get("stdout", ""))[-2000:])
        stderr_tail = self._truncate(str(data.get("stderr", ""))[-2000:])

        # Extract checks and evidence paths
        checks = data.get("checks", [])
        if not isinstance(checks, list):
            checks = []
        evidence_paths = list(failed_result.get("evidence", []))
        if not isinstance(evidence_paths, list):
            evidence_paths = []

        # Context from plan
        compiled = state.get("compiled_analysis", {})
        selected_candidate_id = ""
        candidates = compiled.get("run_candidates", [])
        if isinstance(candidates, list):
            for c in candidates:
                if isinstance(c, dict) and c.get("selected"):
                    selected_candidate_id = c.get("id", "")
                    break

        # Determine runtime backend
        runtime_policy = state.get("runtime_policy", {})
        runtime_backend = "docker" if runtime_policy.get("execution_backend") == "docker" else "local"

        context = {
            "failed_stage": failed_stage,
            "status": status,
            "summary": summary,
            "error": error,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "checks": checks[:20],  # limit check count
            "evidence_paths": evidence_paths[:20],
            "previous_plan_id": state.get("parsed_plan_path", "").rsplit("/", 1)[-1] if state.get("parsed_plan_path") else "",
            "selected_candidate_id": selected_candidate_id,
            "runtime_backend": runtime_backend,
            "replan_count": int(state.get("replan_count", 0)),
            "repair_count": int(state.get("repair_count", 0)),
        }

        # Sanitize secrets
        try:
            sanitizer = AgentInputSanitizer()
            context = sanitizer.redact_value(context)
        except Exception:
            # If sanitizer fails, do basic sanitization
            context = self._basic_sanitize(context)

        # Enforce total size limit
        total = len(json.dumps(context, default=str))
        if total > self.max_input_chars:
            # Truncate text fields proportionally
            for key in ("summary", "error", "stdout_tail", "stderr_tail"):
                if isinstance(context.get(key), str) and len(context[key]) > 500:
                    context[key] = context[key][:500] + "...[truncated]"

        return context

    def compute_signature(self, context: dict) -> str:
        """Compute stable failure signature from context.

        Uses deterministic fields only — no timestamps, PIDs, or random IDs.
        """
        # Determine error category deterministically
        error = context.get("error", "")
        category = self._categorize_error(error)

        sig_input = {
            "stage": context.get("failed_stage", ""),
            "category": category,
            "error_class": self._normalize_error_class(error),
            "selected_candidate_id": context.get("selected_candidate_id", ""),
        }
        canonical = json.dumps(sig_input, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    @staticmethod
    def _truncate(text: str, limit: int = 4000) -> str:
        """Truncate text to limit chars."""
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    @staticmethod
    def _categorize_error(error: str) -> str:
        """Categorize error into a stable category."""
        error_lower = error.lower()
        if "modulenotfound" in error_lower or "importerror" in error_lower:
            return "dependency_missing"
        if "filenotfound" in error_lower or "no such file" in error_lower:
            return "file_missing"
        if "permission" in error_lower:
            return "permission_denied"
        if "timeout" in error_lower:
            return "timeout"
        if "connection" in error_lower or "refused" in error_lower:
            return "connection_error"
        if "port" in error_lower and ("bind" in error_lower or "in use" in error_lower):
            return "port_conflict"
        if "cuda" in error_lower or "gpu" in error_lower:
            return "gpu_error"
        if "docker" in error_lower:
            return "docker_error"
        if "exit code" in error_lower or "returncode" in error_lower:
            return "process_error"
        return "unknown"

    @staticmethod
    def _normalize_error_class(error: str) -> str:
        """Extract and normalize the error class name."""
        # Try to find Python exception class name
        match = re.search(r'(\w+Error|\w+Exception)', error)
        if match:
            return match.group(1)
        return "UnknownError"

    @staticmethod
    def _basic_sanitize(context: dict) -> dict:
        """Basic sanitization if AgentInputSanitizer is unavailable."""
        sanitized = dict(context)
        for key, value in sanitized.items():
            if isinstance(value, str):
                value = re.sub(r'(api[_-]?key|token|secret|password|credential)["\s:=]+\S+', r'\1=***', value, flags=re.IGNORECASE)
                sanitized[key] = value
        return sanitized
