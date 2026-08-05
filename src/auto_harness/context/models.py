from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

from auto_harness.providers.base import LLMResult, Message


class ContextPriority(IntEnum):
    REQUIRED = 0
    FAILURE_FACT = 1
    RELEVANT_EVIDENCE = 2
    EXPERIENCE = 3
    OPTIONAL_HISTORY = 4


class TrustLevel(str, Enum):
    TRUSTED_INSTRUCTION = "trusted_instruction"
    RUNTIME_FACT = "runtime_fact"
    UNTRUSTED_REPOSITORY = "untrusted_repository"
    UNTRUSTED_LOG = "untrusted_log"
    UNTRUSTED_MEMORY = "untrusted_memory"


@dataclass
class ContextSection:
    name: str
    content: Any
    priority: ContextPriority
    trust_level: TrustLevel
    content_type: str
    required: bool = False
    max_tokens: Optional[int] = None
    min_tokens: int = 0
    source: str = ""
    source_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_name: str
    model: str
    context_window_tokens: int
    max_output_tokens: int
    tokenizer_id: Optional[str]
    supports_tool_calling: bool
    usage_format: str
    message_overhead_tokens: int = 4
    request_overhead_tokens: int = 8
    source: str = "fallback"


@dataclass(frozen=True)
class ContextProfile:
    name: str
    version: str
    total_input_cap_tokens: int
    reserved_output_tokens: int
    required_sections: tuple = ()
    fallback_behavior: str = "fail"


@dataclass
class PromptEnvelope:
    messages: List[Message]
    candidate_messages: Optional[List[Message]] = None
    retry_messages: Optional[List[Message]] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    sections: List[ContextSection] = field(default_factory=list)
    candidate_sections: Optional[List[ContextSection]] = None
    retry_sections: Optional[List[ContextSection]] = None
    required_fragments: List[str] = field(default_factory=list)
    requested_output_tokens: Optional[int] = None


@dataclass(frozen=True)
class ContextBudget:
    call_site: str
    provider_name: str
    model: str
    context_window_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    max_input_tokens: int


@dataclass
class ContextBuildResult:
    envelope: PromptEnvelope
    estimated_input_tokens: int
    original_estimated_input_tokens: int
    candidate_estimated_input_tokens: int
    section_metrics: List[Dict[str, Any]]
    truncated: bool
    dropped_sections: List[str]
    shrink_events: List[Dict[str, Any]]
    included_files: List[str]
    skill_count: int
    memory_count: int
    mode: str
    selected_variant: str = "original"
    warnings: List[str] = field(default_factory=list)
    stop_reason: str = ""


@dataclass
class NormalizedUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    source: str = "estimated"
    cache_hit_tokens: Optional[int] = None
    cache_miss_tokens: Optional[int] = None


@dataclass
class LLMCallResult:
    call_id: str
    provider_result: Optional[LLMResult]
    context_result: ContextBuildResult
    usage: NormalizedUsage
    attempts: int
    stop_reason: str = ""
