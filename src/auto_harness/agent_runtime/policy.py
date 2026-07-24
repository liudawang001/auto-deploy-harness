"""Agent-level tool policy for the verify agent.

Validates tool calls before execution. This is a hard gate:
if policy rejects, the tool must NOT execute.

Checks in strict order:
1. tool exists in registry
2. tool is implemented
3. stage is allowed
4. agent_mode is allowed
5. category/risk/runtime permission
6. secret-like key
7. path traversal (using PurePosixPath for robustness)
8. shell metacharacter
9. host allowlist
10. trace requirement
11. normalized input
"""
import re
import urllib.parse
from pathlib import PurePosixPath
from typing import Dict, List, Optional

from auto_harness.agent_runtime.schemas import PolicyDecision, ToolCall, VERIFY_TOOLS
from auto_harness.agent_runtime.stage_schemas import ALL_STAGE_TOOLS
from auto_harness.tools import ToolRegistry


# Shell metacharacters that should never appear in command-like tool inputs
SHELL_META_PATTERN = re.compile(r'[;&|>`\$]')
# Fields where shell metacharacters are dangerous (command/script-like)
COMMAND_LIKE_FIELDS = frozenset({"command", "script", "shell", "cmd", "exec", "run", "bash", "sh"})
# Secret-like field names
SECRET_FIELD_NAMES = frozenset({
    "api_key", "token", "password", "secret", "credential",
    "auth_token", "access_token", "private_key", "bearer",
})
# Tools that require trace_template in their input
TRACE_REQUIRED_TOOLS = frozenset({"probe_http", "discover_gradio_api", "discover_openapi_schema", "probe_browser_dom"})

DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Risk level ordering
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Default runtime policy: all side-effect actions disabled unless explicitly allowed
DEFAULT_RUNTIME_POLICY = {
    "allow_dependency_install": False,
    "allow_service_start": False,
    "allow_source_edit": False,
    "allow_external_network": False,
}

# Mapping from tool names to runtime policy keys
SIDE_EFFECT_RUNTIME_POLICY_MAP = {
    "install_environment": "allow_dependency_install",
    "start_service": "allow_service_start",
    "apply_repair": "allow_source_edit",  # source edit variant
    "resume_from_stage": "allow_service_start",
}


class ToolPolicy:
    """Validates tool calls for the verify agent."""

    def __init__(
        self,
        registry: ToolRegistry = None,
        allowed_hosts: List[str] = None,
        max_risk: str = "medium",
        runtime_policy: Dict = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.allowed_hosts = frozenset(allowed_hosts) if allowed_hosts else DEFAULT_ALLOWED_HOSTS
        self.max_risk = max_risk
        self.runtime_policy = dict(DEFAULT_RUNTIME_POLICY)
        if runtime_policy:
            self.runtime_policy.update(runtime_policy)

    def validate(
        self,
        tool_call: ToolCall,
        stage: str = "verify",
        agent_mode: str = "gated_actor",
        trace_id: str = "",
    ) -> PolicyDecision:
        """Validate a tool call and return a PolicyDecision.

        Checks are applied in strict order so that the most specific
        rejection reason is returned first.

        If allowed, normalized_input contains the sanitized input.
        If rejected, normalized_input is None.
        """
        name = tool_call.name
        tool_input = dict(tool_call.input) if tool_call.input else {}

        # 1. Tool must exist in registry
        tool_schema = self.registry.get(name)
        if not tool_schema:
            return PolicyDecision(allowed=False, reason="tool not found in registry: %s" % name, risk="high")

        # 2. Tool must be implemented
        if not tool_schema.get("implemented", False):
            return PolicyDecision(
                allowed=False,
                reason="tool is not implemented: %s" % name,
                risk="high",
            )

        # 3. Tool must be allowed for the current stage context
        if stage == "verify":
            if name not in VERIFY_TOOLS:
                return PolicyDecision(
                    allowed=False,
                    reason="tool not allowed for verify stage: %s" % name,
                    risk="high",
                )
        else:
            # For non-verify stages, check if the tool declares this stage
            # or is in the ALL_STAGE_TOOLS set
            tool_stages = tool_schema.get("stages", [])
            if tool_stages and stage not in tool_stages:
                return PolicyDecision(
                    allowed=False,
                    reason="tool not allowed for stage %s: %s" % (stage, name),
                    risk="high",
                )
            elif not tool_stages and name not in ALL_STAGE_TOOLS:
                return PolicyDecision(
                    allowed=False,
                    reason="tool not allowed for stage %s: %s" % (stage, name),
                    risk="high",
                )

        # 4. agent_mode check: planner mode may propose but not execute
        if agent_mode == "planner":
            # read_only tools are allowed for observation in planner mode
            tool_category = tool_schema.get("category", "read_only")
            if tool_category != "read_only":
                return PolicyDecision(
                    allowed=False,
                    reason="planner mode does not execute non-read-only tools",
                    risk="low",
                )

        # 4b. Check tool allowed_modes if present
        allowed_modes = tool_schema.get("allowed_modes", [])
        if allowed_modes and agent_mode not in allowed_modes:
            return PolicyDecision(
                allowed=False,
                reason="tool not allowed in agent_mode %s" % agent_mode,
                risk=tool_schema.get("risk_level", "low"),
            )

        # 5. Category/risk/runtime permission check
        tool_risk = str(tool_schema.get("risk_level", "low"))
        tool_category = tool_schema.get("category", "read_only")

        if RISK_ORDER.get(tool_risk, 0) > RISK_ORDER.get(self.max_risk, 1):
            return PolicyDecision(
                allowed=False,
                reason="tool risk level %s exceeds max %s" % (tool_risk, self.max_risk),
                risk=tool_risk,
            )

        # Runtime permission check for side-effect tools
        if tool_category == "side_effect":
            policy_key = SIDE_EFFECT_RUNTIME_POLICY_MAP.get(name)
            if policy_key and not self.runtime_policy.get(policy_key, False):
                return PolicyDecision(
                    allowed=False,
                    reason="side-effect tool requires runtime permission: %s" % policy_key,
                    risk=tool_risk,
                )

        # 6. Check for secret-like fields
        for key in tool_input:
            if key.lower() in SECRET_FIELD_NAMES:
                return PolicyDecision(
                    allowed=False,
                    reason="secret-like field '%s' not allowed in tool input" % key,
                    risk="high",
                )

        # 7. Path traversal check (robust, using PurePosixPath)
        for key, value in tool_input.items():
            if not isinstance(value, str):
                continue
            # Check for NUL byte
            if "\0" in value:
                return PolicyDecision(
                    allowed=False,
                    reason="NUL byte in input field '%s'" % key,
                    risk="high",
                )
            # Check for ~ expansion
            if value.startswith("~") or "/~" in value:
                return PolicyDecision(
                    allowed=False,
                    reason="home directory expansion in input field '%s'" % key,
                    risk="high",
                )
            # Check for absolute path
            if value.startswith("/"):
                return PolicyDecision(
                    allowed=False,
                    reason="absolute path in input field '%s'" % key,
                    risk="high",
                )
            # Robust path traversal using PurePosixPath parts
            parts = PurePosixPath(value.replace("\\", "/")).parts
            if ".." in parts:
                return PolicyDecision(
                    allowed=False,
                    reason="path traversal in input field '%s'" % key,
                    risk="high",
                )

        # 8. Check for shell metacharacters in command-like fields only
        for key, value in tool_input.items():
            if isinstance(value, str) and SHELL_META_PATTERN.search(value):
                if key.lower() in COMMAND_LIKE_FIELDS:
                    return PolicyDecision(
                        allowed=False,
                        reason="shell metacharacters in command-like field '%s'" % key,
                        risk="high",
                    )
                # For non-command fields, only flag pipe/backtick/semicolon as risky
                if re.search(r'[;|`]', value):
                    return PolicyDecision(
                        allowed=False,
                        reason="potentially dangerous metacharacters in input field '%s'" % key,
                        risk="high",
                    )

        # 9. Host allowlist check
        url_fields = self._extract_urls(tool_input)
        for url in url_fields:
            host = self._extract_host(url)
            if host and host not in self.allowed_hosts:
                return PolicyDecision(
                    allowed=False,
                    reason="external host not allowed: %s" % host,
                    risk="high",
                )

        # 10. Trace requirement for verify probes
        if name in TRACE_REQUIRED_TOOLS:
            trace_template = tool_input.get("trace_template", "")
            if not trace_template or "{{trace_id}}" not in str(trace_template):
                return PolicyDecision(
                    allowed=False,
                    reason="verify probe tool must include trace_template with {{trace_id}}",
                    risk="medium",
                )

        # 11. All checks passed - normalize input
        normalized = self._normalize_input(tool_input, trace_id)

        return PolicyDecision(
            allowed=True,
            reason="tool call passes all policy checks",
            risk=tool_risk,
            normalized_input=normalized,
        )

    def _extract_urls(self, tool_input: Dict) -> List[str]:
        """Extract URL-like values from tool input."""
        urls = []
        for key, value in tool_input.items():
            if isinstance(value, str) and (value.startswith("http://") or value.startswith("https://")):
                urls.append(value)
            elif key in ("endpoint", "url", "base_url") and isinstance(value, str):
                if "://" in value:
                    urls.append(value)
        return urls

    def _extract_host(self, url: str) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.hostname or ""
        except Exception:
            return ""

    def _normalize_input(self, tool_input: Dict, trace_id: str = "") -> Dict:
        """Normalize tool input. Strip any potential dangerous values and
        replace {{trace_id}} placeholders with the actual trace_id."""
        normalized = {}
        for key, value in tool_input.items():
            if isinstance(value, str) and trace_id:
                value = value.replace("{{trace_id}}", trace_id)
            normalized[key] = value
        return normalized
