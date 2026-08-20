"""Provider protocol adapters for JSON Action and native tool calling."""

from auto_harness.providers.protocols.normalizer import (
    ToolCallConflictError,
    ToolCallProtocolError,
    normalize_json_action_call,
    normalize_provider_tool_call,
)
from auto_harness.providers.protocols.router import (
    ProtocolSelection,
    ProviderProtocolError,
    select_provider_protocol,
)
from auto_harness.providers.protocols.schemas import (
    NormalizedToolCall,
    NormalizedToolResult,
    canonical_json_hash,
    tool_operation_id,
)
from auto_harness.providers.protocols.tool_messages import (
    assistant_tool_call_message,
    redact_tool_payload,
    tool_result_message,
)
from auto_harness.providers.protocols.tool_schema_adapter import (
    ToolSchemaProjection,
    ToolSchemaProjectionError,
    ToolArgumentsValidationError,
    project_provider_tools,
    validate_tool_arguments,
)

__all__ = [
    "NormalizedToolCall",
    "NormalizedToolResult",
    "ProtocolSelection",
    "ProviderProtocolError",
    "ToolCallConflictError",
    "ToolCallProtocolError",
    "ToolSchemaProjection",
    "ToolSchemaProjectionError",
    "ToolArgumentsValidationError",
    "assistant_tool_call_message",
    "redact_tool_payload",
    "canonical_json_hash",
    "normalize_json_action_call",
    "normalize_provider_tool_call",
    "project_provider_tools",
    "validate_tool_arguments",
    "select_provider_protocol",
    "tool_operation_id",
    "tool_result_message",
]
