"""ToolExecutor for the LLM-driven verify agent.

Dispatches approved tool calls to the appropriate verify tool implementation.
Only accepts policy-normalized input, never raw LLM output.
"""
from typing import Dict

from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.tools import verify_tools


class ToolExecutor:
    """Execute approved tool calls for the verify agent.

    Only dispatches to tools registered in the ToolRegistry and
    allowed for the verify stage. Uses only policy-normalized input.
    """

    def execute(self, tool_call: ToolCall, context: Dict) -> ToolResult:
        """Execute an approved tool call.

        Args:
            tool_call: The approved tool call (with normalized input from policy).
            context: Execution context containing trace_id, evidence_dir, etc.

        Returns:
            ToolResult with status, evidence, and strong_verify_pass flag.
        """
        name = tool_call.name
        tool_input = tool_call.input or {}

        if name == "probe_http":
            return verify_tools.probe_http(tool_input, context)
        if name == "discover_gradio_api":
            return verify_tools.discover_gradio_api(tool_input, context)
        if name == "discover_openapi_schema":
            return verify_tools.discover_openapi_schema(tool_input, context)
        if name == "probe_browser_dom":
            return verify_tools.probe_browser_dom(tool_input, context)

        return ToolResult(
            status="rejected",
            tool_name=name,
            evidence={},
            error="unknown tool: %s" % name,
            started_at="",
            ended_at="",
        )
