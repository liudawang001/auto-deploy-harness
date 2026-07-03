from dataclasses import dataclass
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


class LLMProvider(Protocol):
    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
        ...

