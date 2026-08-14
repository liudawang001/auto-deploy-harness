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
import hashlib
import json
import secrets
import time
import urllib.request
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.model_runtime.schemas import (
    InferenceResourceDecision,
    ModelFilePlan,
    ModelInferenceEvidence,
    ResolvedModelSpec,
)
from auto_harness.utils.redaction import check_redaction
from auto_harness.utils.time import utc_now_iso

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

# Legitimate token-count / latency fields that must NOT be treated as secrets
# even though they contain the substring "token".
_LEGIT_TOKEN_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "first_token_at",
    "max_tokens",
    "token_count",
    "tokens",
    "tokens_per_second",
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
            if normalized not in _LEGIT_TOKEN_FIELDS and any(word in normalized for word in _FORBIDDEN_FIELD_NAMES):
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


class ModelRuntimeEvidenceWriter:
    """Write Document B runtime plans and evidence atomically.

    Produces:
        runs/<task-id>/reports/model/runtime_plan.json
        runs/<task-id>/reports/model/startup_evidence.json
        runs/<task-id>/reports/model/inference_non_stream_evidence.json
        runs/<task-id>/reports/model/inference_stream_evidence.json
        runs/<task-id>/reports/model/performance_summary.json

    Every payload passes the same secret / non-serializable scans as the
    Document A writer before an atomic replace.
    """

    def __init__(self, run_dir) -> None:
        self.root = Path(run_dir) / "reports" / "model"

    def _write(self, name: str, payload: Any) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        serializable = validate_serializable(payload)
        if serializable:
            raise ValueError("evidence %s contains non-serializable values: %s" % (name, ", ".join(serializable)))
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        forbidden = scan_forbidden_fields(payload)
        if forbidden:
            raise ValueError("evidence %s contains forbidden secret-like fields: %s" % (name, ", ".join(forbidden)))
        redacted = check_redaction(serialized)
        if redacted:
            raise ValueError("evidence %s contains unredacted content: %s" % (name, ", ".join(i["pattern_name"] for i in redacted)))
        path = self.root / ("%s.json" % name)
        write_json(path, payload)
        return str(path)

    def write_runtime_plan(self, plan) -> str:
        return self._write("runtime_plan", _to_payload(plan))

    def write_startup_evidence(self, evidence) -> str:
        return self._write("startup_evidence", _to_payload(evidence))

    def write_inference_evidence(self, evidence, kind: str) -> str:
        return self._write("inference_%s_evidence" % kind, _to_payload(evidence))

    def write_performance_summary(self, summary) -> str:
        return self._write("performance_summary", _to_payload(summary))


# ----------------------------------------------------------------------
# Inference evidence gate (Document B Phase B6/B7)
# ----------------------------------------------------------------------

MAX_SSE_BYTES = 8 * 1024 * 1024
MAX_SSE_FRAMES = 10_000
MAX_SSE_FRAME_BYTES = 1 * 1024 * 1024


def _sha256_hex(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_content(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


class ModelInferenceGate:
    """Execute and evidence a current-trace inference request against a
    served vLLM endpoint (non-stream and SSE).

    The trace id is unpredictable and regenerated on every call. Success is
    decided by the current trace being found in the reconstructed response
    plus non-empty usage — never by HTTP status alone.
    """

    def __init__(
        self,
        urlopen: Optional[Any] = None,
        now=None,
        clock=None,
        rng=None,
    ) -> None:
        self.urlopen = urlopen or urllib.request.urlopen
        self._now = now or utc_now_iso
        self._clock = clock or time.monotonic
        self._rng = rng or secrets.token_hex

    def generate_trace_id(self) -> str:
        suffix = self._rng(8)
        stamp = self._now().replace(":", "").replace("-", "").replace(".", "").replace("+", "")
        return "infer_%s_%s" % (stamp, suffix)

    # -- request helpers -------------------------------------------------

    @staticmethod
    def _chat_url(runtime_plan) -> str:
        return "http://%s:%s/v1/chat/completions" % (
            runtime_plan.expected_host,
            runtime_plan.expected_port,
        )

    @staticmethod
    def _request_payload(served_model: str, trace_id: str, stream: bool) -> dict:
        payload = {
            "model": served_model,
            "messages": [
                {"role": "system", "content": "Follow the requested output format exactly."},
                {"role": "user", "content": "Reply with exactly this token and nothing else: %s" % trace_id},
            ],
            "temperature": 0,
            "max_tokens": 32,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    # -- non-stream ------------------------------------------------------

    def verify_non_stream(
        self,
        *,
        runtime_plan,
        startup_evidence=None,
        task_id: str = "",
        operation_id: str = "",
        trace_id: str = "",
        gpu_before: Optional[Dict] = None,
        gpu_after: Optional[Dict] = None,
    ) -> ModelInferenceEvidence:
        trace_id = trace_id or self.generate_trace_id()
        request_started_at = self._now()
        start_mono = self._clock()

        base = {
            "schema_version": 1,
            "status": "failed",
            "task_id": task_id,
            "operation_id": operation_id,
            "container_id": startup_evidence.container_id if startup_evidence else "",
            "container_created_at": startup_evidence.container_created_at if startup_evidence else "",
            "runtime_plan_hash": runtime_plan.plan_hash,
            "model_identity": runtime_plan.model_identity,
            "served_model_name": runtime_plan.served_model_name,
            "trace_id": trace_id,
            "request_started_at": request_started_at,
            "stream": False,
            "gpu_before": dict(gpu_before or {}),
            "gpu_after": dict(gpu_after or {}),
            "evidence_paths": [],
        }

        ready_at = startup_evidence.ready_at if startup_evidence else ""
        if ready_at and request_started_at <= ready_at:
            base["failure_reason"] = "request started before startup ready time"
            return ModelInferenceEvidence(**base)

        payload = self._request_payload(runtime_plan.served_model_name, trace_id, stream=False)
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self._chat_url(runtime_plan), data=body_bytes, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with self.urlopen(req, timeout=runtime_plan.request_timeout_seconds) as resp:
                status = getattr(resp, "status", None) or getattr(resp, "code", None)
                raw_body = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - stored as evidence
            base["failure_reason"] = "request failed: %s" % str(exc)[:300]
            return ModelInferenceEvidence(**base)

        base["http_status"] = int(status or 0)
        base["response_body_sha256"] = _sha256_hex(raw_body)
        base["request_sha256"] = _sha256_hex(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        base["response_completed_at"] = self._now()
        base["total_latency_ms"] = int((self._clock() - start_mono) * 1000)

        if not (isinstance(status, int) and 200 <= status < 300):
            base["failure_reason"] = "non-2xx HTTP status %s" % status
            return ModelInferenceEvidence(**base)

        parsed, parse_error = _parse_chat_json(raw_body)
        if parse_error:
            base["failure_reason"] = "response is not valid OpenAI chat JSON: %s" % parse_error
            return ModelInferenceEvidence(**base)

        base["response_model"] = parsed.get("model", "")
        content = _normalize_content(((parsed.get("choices") or [{}])[0].get("message") or {}).get("content", ""))
        usage = parsed.get("usage") or {}
        prompt_tokens = _int_or_zero(usage.get("prompt_tokens"))
        completion_tokens = _int_or_zero(usage.get("completion_tokens"))

        base["prompt_tokens"] = prompt_tokens
        base["completion_tokens"] = completion_tokens
        base["trace_found"] = trace_id in content

        if base["response_model"] != runtime_plan.served_model_name:
            base["status"] = "failed"
            base["failure_reason"] = "response model %r != served model %r" % (
                base["response_model"], runtime_plan.served_model_name,
            )
        elif not base["trace_found"]:
            base["status"] = "failed"
            base["failure_reason"] = "current trace id not found in response content"
        elif prompt_tokens <= 0 or completion_tokens <= 0:
            base["status"] = "uncertain"
            base["failure_reason"] = "usage tokens missing or zero"
        else:
            base["status"] = "passed"
            base["failure_reason"] = ""

        return ModelInferenceEvidence(**base)

    # -- SSE stream ------------------------------------------------------

    def verify_stream(
        self,
        *,
        runtime_plan,
        startup_evidence=None,
        task_id: str = "",
        operation_id: str = "",
        trace_id: str = "",
        gpu_before: Optional[Dict] = None,
        gpu_after: Optional[Dict] = None,
    ) -> ModelInferenceEvidence:
        trace_id = trace_id or self.generate_trace_id()
        request_started_at = self._now()
        start_mono = self._clock()

        base = {
            "schema_version": 1,
            "status": "failed",
            "task_id": task_id,
            "operation_id": operation_id,
            "container_id": startup_evidence.container_id if startup_evidence else "",
            "container_created_at": startup_evidence.container_created_at if startup_evidence else "",
            "runtime_plan_hash": runtime_plan.plan_hash,
            "model_identity": runtime_plan.model_identity,
            "served_model_name": runtime_plan.served_model_name,
            "trace_id": trace_id,
            "request_started_at": request_started_at,
            "stream": True,
            "gpu_before": dict(gpu_before or {}),
            "gpu_after": dict(gpu_after or {}),
            "evidence_paths": [],
        }

        ready_at = startup_evidence.ready_at if startup_evidence else ""
        if ready_at and request_started_at <= ready_at:
            base["failure_reason"] = "request started before startup ready time"
            return ModelInferenceEvidence(**base)

        payload = self._request_payload(runtime_plan.served_model_name, trace_id, stream=True)
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self._chat_url(runtime_plan), data=body_bytes, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with self.urlopen(req, timeout=runtime_plan.request_timeout_seconds) as resp:
                status = getattr(resp, "status", None) or getattr(resp, "code", None)
                sse = parse_sse_stream(
                    resp,
                    max_bytes=MAX_SSE_BYTES,
                    max_frames=MAX_SSE_FRAMES,
                    max_frame_bytes=MAX_SSE_FRAME_BYTES,
                    clock=self._clock,
                    started_mono=start_mono,
                )
        except Exception as exc:  # noqa: BLE001 - stored as evidence
            base["failure_reason"] = "stream request failed: %s" % str(exc)[:300]
            return ModelInferenceEvidence(**base)

        base["http_status"] = int(status or 0)
        base["request_sha256"] = _sha256_hex(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        base["response_completed_at"] = self._now()
        base["total_latency_ms"] = int((self._clock() - start_mono) * 1000)

        if not (isinstance(status, int) and 200 <= status < 300):
            base["failure_reason"] = "non-2xx HTTP status %s" % status
            return ModelInferenceEvidence(**base)

        if sse.error:
            base["failure_reason"] = sse.error
            return ModelInferenceEvidence(**base)
        if not sse.terminal:
            base["failure_reason"] = "SSE stream did not terminate with [DONE]"
            return ModelInferenceEvidence(**base)

        base["response_model"] = sse.model
        base["prompt_tokens"] = sse.prompt_tokens
        base["completion_tokens"] = sse.completion_tokens
        base["trace_found"] = trace_id in _normalize_content(sse.content)
        base["ttft_ms"] = sse.ttft_ms
        base["response_body_sha256"] = _sha256_hex(sse.content)

        if not sse.content:
            base["failure_reason"] = "no content delta received"
        elif base["response_model"] and base["response_model"] != runtime_plan.served_model_name:
            base["failure_reason"] = "response model %r != served model %r" % (
                base["response_model"], runtime_plan.served_model_name,
            )
        elif not base["trace_found"]:
            base["failure_reason"] = "current trace id not found in reconstructed content"
        elif sse.prompt_tokens <= 0 or sse.completion_tokens <= 0:
            base["status"] = "uncertain"
            base["failure_reason"] = "usage tokens missing or zero"
        else:
            base["status"] = "passed"
            base["failure_reason"] = ""

        return ModelInferenceEvidence(**base)


def _parse_chat_json(body: str):
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("choices"), list) or not parsed["choices"]:
        return None, "choices missing"
    return parsed, ""


def _int_or_zero(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _SSEResult:
    def __init__(self):
        self.content = ""
        self.model = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.ttft_ms = 0
        self.terminal = False
        self.error = ""


def parse_sse_stream(
    resp,
    *,
    max_bytes: int = MAX_SSE_BYTES,
    max_frames: int = MAX_SSE_FRAMES,
    max_frame_bytes: int = MAX_SSE_FRAME_BYTES,
    clock=None,
    started_mono: float = 0.0,
) -> _SSEResult:
    """Parse an OpenAI-compatible SSE stream with hard byte/frame/length bounds.

    Reads incrementally from a file-like ``resp``. Frames are delimited by
    blank lines; only ``data:`` fields are consumed (comments and other field
    lines are ignored). Reconstructs ``choices[].delta.content`` and reads
    usage from the final frame. ``[DONE]`` marks the terminal event.
    """
    result = _SSEResult()
    clock = clock or time.monotonic
    total_bytes = 0
    frames = 0
    frame_bytes = 0
    frame_lines: List[str] = []
    first_content = False

    def flush_frame():
        nonlocal frames, frame_bytes, frame_lines, first_content
        data_text = "\n".join(frame_lines)
        frame_lines = []
        frame_bytes = 0
        if data_text.strip() == "[DONE]":
            result.terminal = True
            return
        try:
            event = json.loads(data_text)
        except ValueError as exc:
            result.error = "invalid SSE JSON frame: %s" % str(exc)[:200]
            raise _SSEAbort()
        if not isinstance(event, dict):
            result.error = "invalid SSE frame (not an object)"
            raise _SSEAbort()
        frames += 1
        if isinstance(event.get("model"), str) and event["model"]:
            result.model = event["model"]
        choices = event.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                result.content += piece
                if not first_content:
                    first_content = True
                    result.ttft_ms = int((clock() - started_mono) * 1000)
        usage = event.get("usage")
        if isinstance(usage, dict):
            result.prompt_tokens = _int_or_zero(usage.get("prompt_tokens"))
            result.completion_tokens = _int_or_zero(usage.get("completion_tokens"))

    try:
        for raw_line in resp:
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = raw_line
            total_bytes += len(line.encode("utf-8"))
            if total_bytes > max_bytes:
                result.error = "SSE stream exceeded byte limit"
                break
            line = line.rstrip("\r\n")
            if not line:
                # Blank line terminates the current event frame.
                if frame_lines:
                    flush_frame()
                    if result.error or result.terminal:
                        break
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                frame_lines.append(line[5:].lstrip())
                frame_bytes += len(line)
                if frame_bytes > max_frame_bytes:
                    result.error = "SSE frame exceeded length limit"
                    break
                continue
            # Other field lines (event:, id:, retry:) are ignored.
            continue
        if not result.error and frame_lines:
            flush_frame()
    except _SSEAbort:
        pass

    if frames > max_frames:
        result.error = "SSE stream exceeded frame limit"
    return result


class _SSEAbort(Exception):
    pass
