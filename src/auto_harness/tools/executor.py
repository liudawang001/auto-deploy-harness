"""ToolExecutor for the LLM-driven verify agent.

Dispatches approved tool calls to registered handler functions.
Only accepts policy-normalized input, never raw LLM output.

Registry-based dispatch replaces the old if/elif chain, ensuring
that ToolRegistry declarations and executor handlers stay in sync
via validate_contract().
"""
from typing import Callable, Dict

from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.tools import verify_tools
from auto_harness.tools.registry import ToolRegistry


class ToolExecutor:
    """Execute approved tool calls for the verify agent.

    Uses a registered handler pattern instead of if/elif dispatch.
    On construction, validates that all implemented tools in the
    registry have a registered handler and vice versa.
    """

    def __init__(self, registry=None, handlers=None):
        self.registry = registry or ToolRegistry()
        self.handlers: Dict[str, Callable] = {}
        self._register_defaults()
        for name, handler in (handlers or {}).items():
            self.register(name, handler)

    def register(self, name: str, handler: Callable) -> None:
        """Register a handler function for a tool name.

        Args:
            name: Tool name (must match a ToolRegistry entry).
            handler: Callable(tool_input: dict, context: dict) -> ToolResult.

        Raises:
            TypeError: If handler is not callable.
        """
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self.handlers[name] = handler

    def validate_contract(self) -> None:
        """Validate that registry and handlers are consistent.

        Raises:
            RuntimeError: If implemented tools lack handlers or
                handlers exist for unimplemented tools.
        """
        implemented = {
            name
            for name, schema in self.registry.tools.items()
            if schema.implemented and schema.executor == "verify"
        }
        registered = set(self.handlers)

        missing = sorted(implemented - registered)
        unknown = sorted(registered - implemented)

        if missing or unknown:
            raise RuntimeError(
                "tool registry/executor mismatch: "
                "missing=%s unknown=%s" % (missing, unknown)
            )

    def execute(self, tool_call: ToolCall, context: Dict) -> ToolResult:
        """Execute an approved tool call.

        Args:
            tool_call: The approved tool call (with normalized input from policy).
            context: Execution context containing trace_id, evidence_dir, etc.

        Returns:
            ToolResult with status, evidence, and strong_verify_pass flag.
        """
        schema = self.registry.tools.get(tool_call.name)
        if schema is None or not schema.implemented or schema.executor != "verify":
            return ToolResult(
                status="rejected",
                tool_name=tool_call.name,
                category="read_only",
                error="tool is not implemented",
            )

        handler = self.handlers.get(tool_call.name)
        if handler is None:
            return ToolResult(
                status="error",
                tool_name=tool_call.name,
                category=schema.category,
                error="implemented tool has no executor",
            )

        result = handler(tool_call.input or {}, context)
        result.category = schema.category
        return result

    def _register_defaults(self) -> None:
        """Register default handlers for implemented verify tools."""
        self.register("probe_http", verify_tools.probe_http)
        self.register("discover_gradio_api", verify_tools.discover_gradio_api)
        self.register("discover_openapi_schema", verify_tools.discover_openapi_schema)
        self.register("probe_browser_dom", verify_tools.probe_browser_dom)
