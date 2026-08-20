from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class Message:
    role: str
    content: str = ""
    reasoning_content: str = field(default="", repr=False, compare=False)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""


@dataclass
class LLMResult:
    text: str
    raw: Dict[str, Any] = None
    usage: Dict[str, Any] = None
    latency_ms: int = 0
    protocol: str = "json_action"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    reasoning_content: str = field(default="", repr=False, compare=False)
    finish_reason: str = ""
    request_id: str = ""
    provider_name: str = ""
    provider_model: str = ""
    retry_count: int = 0


@dataclass
class ProviderRequestContext:
    """Safe context passed from executor to provider.

    Contains operational metadata that the provider needs for request
    construction and telemetry. Must NOT contain:
    - Repository URLs
    - Usernames
    - Tenant privacy data
    - API keys or secrets
    """

    call_id: str = ""
    call_site: str = ""
    stage: str = ""
    purpose: str = ""
    task_scope_hash: str = ""
    requested_output_tokens: int = 4096
    deadline_at: str = ""


class LLMProvider(Protocol):
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_output_tokens: int = None,
        request_context: Optional[ProviderRequestContext] = None,
    ) -> LLMResult:
        ...


class ToolCallingLLMProvider(Protocol):
    """Protocol for providers that support native tool calling.

    Only implement this when the provider API actually supports:
    - Sending tools schema
    - Receiving tool_use/tool_call blocks
    - Preserving tool_call_id
    - Sending tool result messages
    - Clear protocol for parallel or serial tool calls

    Do NOT create a fake adapter that returns fixed values.
    """
    def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict],
        *,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_output_tokens: int = None,
        request_context: Optional[ProviderRequestContext] = None,
    ) -> LLMResult:
        ...
