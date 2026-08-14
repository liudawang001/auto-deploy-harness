"""Model preparation Artifact Writer.

Writes the four stable Document A artifacts:
    runs/<task-id>/reports/model/resolved_model.json
    runs/<task-id>/reports/model/model_file_plan.json
    runs/<task-id>/reports/model/resource_decision.json

The complete marker lives in the model cache directory and is written by the
download/cache flow, not here.

Before writing, payloads are scanned for secret-like fields and redacted
tokens. Non-finite floats, Path, Exception, and HTTP response objects are
rejected rather than serialized.
"""
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from auto_harness.models.base import write_json
from auto_harness.model_runtime.schemas import (
    InferenceResourceDecision,
    ModelFilePlan,
    ResolvedModelSpec,
)
from auto_harness.utils.redaction import check_redaction

# Field names that must never appear in a persisted Artifact.
_FORBIDDEN_FIELD_NAMES = {
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "authorization",
    "credential",
    "hf_token",
    "modelscope_token",
    "bearer",
}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scan_forbidden_fields(value: Any, path: str = "") -> List[str]:
    """Recursively collect paths to secret-like fields.

    Any key whose normalized name contains a forbidden token-like word is
    flagged. Values themselves are not inspected for content here — that is
    the job of :func:`check_redaction` on the serialized text.
    """
    problems: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = "%s.%s" % (path, key) if path else str(key)
            normalized = str(key).lower().replace("-", "_").replace(".", "_")
            if any(word in normalized for word in _FORBIDDEN_FIELD_NAMES):
                problems.append(child_path)
            problems.extend(scan_forbidden_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(scan_forbidden_fields(child, "%s[%d]" % (path, index)))
    return problems


def validate_serializable(value: Any, path: str = "root") -> List[str]:
    """Reject non-finite floats, Path, Exception, and HTTP response objects."""
    problems: List[str] = []
    if isinstance(value, float) and value != value:
        problems.append("%s: NaN float is not serializable" % path)
    elif isinstance(value, float) and value in (float("inf"), float("-inf")):
        problems.append("%s: infinite float is not serializable" % path)
    elif isinstance(value, Path):
        problems.append("%s: Path object is not serializable" % path)
    elif isinstance(value, BaseException):
        problems.append("%s: Exception object is not serializable" % path)
    elif isinstance(value, dict):
        for key, child in value.items():
            problems.extend(validate_serializable(child, "%s.%s" % (path, key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            problems.extend(validate_serializable(child, "%s[%d]" % (path, index)))
    return problems


def _to_payload(value: Any) -> Any:
    """Convert a schema dataclass to a plain dict (already plain, defensive)."""
    if is_dataclass(value):
        return value.to_dict()
    return value


class ModelArtifactWriter:
    """Write the three run-dir model Artifacts atomically with secret scans."""

    def __init__(self, run_dir: Path) -> None:
        self.root = Path(run_dir) / "reports" / "model"

    def _write(self, name: str, payload: Dict[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        serializable = validate_serializable(payload)
        if serializable:
            raise ValueError(
                "artifact %s contains non-serializable values: %s"
                % (name, ", ".join(serializable))
            )
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        forbidden = scan_forbidden_fields(payload)
        if forbidden:
            raise ValueError(
                "artifact %s contains forbidden secret-like fields: %s"
                % (name, ", ".join(forbidden))
            )
        redacted = check_redaction(serialized)
        if redacted:
            raise ValueError(
                "artifact %s contains unredacted sensitive content: %s"
                % (name, ", ".join(item["pattern_name"] for item in redacted))
            )
        path = self.root / ("%s.json" % name)
        write_json(path, payload)
        return str(path)

    def write_resolved_model(self, spec: ResolvedModelSpec) -> str:
        return self._write("resolved_model", _to_payload(spec))

    def write_file_plan(self, plan: ModelFilePlan) -> str:
        return self._write("model_file_plan", _to_payload(plan))

    def write_resource_decision(self, decision: InferenceResourceDecision) -> str:
        return self._write("resource_decision", _to_payload(decision))
