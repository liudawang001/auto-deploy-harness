"""Protocol-independent tool call and result schemas.

Provider response objects are untrusted transport data.  They are normalized
into these small immutable contracts before reaching policy or executors.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation for identity hashes."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NormalizedToolCall:
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    arguments_hash: str
    provider_protocol: str
    provider_name: str = ""
    provider_model: str = ""
    turn_index: int = 0
    call_index: int = 0
    raw_call_hash: str = ""

    @property
    def fingerprint(self) -> str:
        return canonical_json_hash({
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        })

    def to_dict(self) -> Dict[str, Any]:
        # Deliberately contains no raw provider response.
        return asdict(self)


@dataclass(frozen=True)
class NormalizedToolResult:
    call_id: str
    operation_id: str
    tool_name: str
    status: str
    category: str
    policy_allowed: bool
    executed: bool
    applied: bool
    result: Dict[str, Any] = field(default_factory=dict)
    result_hash: str = ""
    evidence_paths: List[str] = field(default_factory=list)
    error: str = ""
    reused: bool = False
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tool_operation_id(
    *,
    task_id: str,
    stage: str,
    tool_name: str,
    arguments: Dict[str, Any],
    repository_fingerprint: str = "",
    runtime_policy_fingerprint: str = "",
) -> str:
    """Build semantic identity independent of a provider-generated call id."""
    digest = canonical_json_hash({
        "task_id": str(task_id),
        "stage": str(stage),
        "tool_name": str(tool_name),
        "arguments": dict(arguments or {}),
        "repository_fingerprint": str(repository_fingerprint),
        "runtime_policy_fingerprint": str(runtime_policy_fingerprint),
    })
    return "tool-" + digest.split(":", 1)[1][:24]
