import json
from typing import List

from auto_harness.providers.base import LLMResult, Message


class MockLLMProvider:
    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
        content = {
            "status": "ok",
            "summary": "mock provider response",
            "message_count": len(messages),
        }
        return LLMResult(text=json.dumps(content, ensure_ascii=False), raw=content, usage={})

