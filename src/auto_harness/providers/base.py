from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LLMResult:
    text: str
    raw: Dict[str, Any] = None
    usage: Dict[str, Any] = None
    latency_ms: int = 0
    protocol: str = "json_action"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class LLMProvider(Protocol):
    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
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
        tool_choice: str = "auto",
    ) -> LLMResult:
        ...

