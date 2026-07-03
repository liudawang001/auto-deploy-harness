from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol


@dataclass
class AgentRequest:
    stage: str
    prompt: str
    workdir: Path
    timeout_seconds: int = 900
    metadata: Dict[str, str] = None


@dataclass
class AgentResult:
    status: str
    text: str
    raw: Dict = None
    session_id: Optional[str] = None
    error: Optional[str] = None


class AgentExecutor(Protocol):
    def run(self, request: AgentRequest) -> AgentResult:
        ...

    def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        ...

