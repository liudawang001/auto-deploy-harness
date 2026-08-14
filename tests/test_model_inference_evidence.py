"""Phase B6/B7 tests: non-stream and SSE inference evidence gates.

Covers correct/old/missing trace, wrong model, missing usage, request-before-
ready, token/header redaction, SSE trace split across frames, [DONE] terminal,
missing terminal, invalid JSON frame, byte limit, unicode, and TTFT monotonicity.
"""
import json

import pytest

from auto_harness.model_runtime.evidence import ModelInferenceGate, parse_sse_stream
from auto_harness.model_runtime.schemas import (
    InferenceRuntimePlan,
    ModelRuntimeStartupEvidence,
)

DIGEST = "sha256:" + "d" * 64
MODEL_ID = "huggingface:org/model@" + "c" * 40
NOW_AFTER = "2000-01-01T01:00:00Z"
NOW_BEFORE = "2000-01-01T00:00:00Z"
READY_AT = "2000-01-01T00:30:00Z"


def _plan(**overrides):
    data = dict(
        runtime="vllm",
        deployment_mode="managed_vllm",
        image="vllm/vllm-openai:v0.6.1@" + DIGEST,
        image_digest=DIGEST,
        model_identity=MODEL_ID,
        resolved_model_hash="sha256:model",
        model_host_path="/tmp/model_cache/key",
        model_container_path="/models/current",
        served_model_name="org/model",
        command=["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/current"],
        expected_host="127.0.0.1",
        expected_port=8000,
        startup_timeout_seconds=900,
        request_timeout_seconds=120,
        health_path="/v1/models",
        container_name="auto-harness-abc-vllm",
        gpu_indexes=[0],
        security_profile="model_runtime_v1",
    )
    data.update(overrides)
    plan = InferenceRuntimePlan(**data)
    plan.plan_hash = plan.compute_plan_hash()
    return plan


def _startup(ready_at=READY_AT, container_id="cid", served="org/model"):
    return ModelRuntimeStartupEvidence(
        status="ready",
        ready_at=ready_at,
        container_id=container_id,
        container_created_at="2000-01-01T00:00:00Z",
        served_model_name=served,
        runtime_plan_hash="sha256:plan",
    )


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status = status
        self.code = status
        self.headers = dict(headers or {})
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class FakeStreamResponse:
    def __init__(self, status, lines):
        self.status = status
        self.code = status
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


def _gate(urlopen, now=None):
    return ModelInferenceGate(urlopen=urlopen, now=now or (lambda: NOW_AFTER))


def _chat(trace_id, model="org/model", pt=18, ct=6, content=None):
    content = content if content is not None else "the token is %s" % trace_id
    return json.dumps({
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    })


def _sse(trace_id, model="org/model", pt=18, ct=6, done=True, usage=True, split=True):
    lines = []

    def frame(payload):
        lines.append("data: " + json.dumps(payload))
        lines.append("")

    frame({"choices": [{"delta": {"role": "assistant"}}], "model": model})
    if split:
        half = len(trace_id) // 2
        frame({"choices": [{"delta": {"content": trace_id[:half]}}]})
        frame({"choices": [{"delta": {"content": trace_id[half:]}}]})
    else:
        frame({"choices": [{"delta": {"content": trace_id}}]})
    if usage:
        frame({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": ct}})
    if done:
        lines.append("data: [DONE]")
        lines.append("")
    return lines


# -------------------------------------------------------------------
# Non-stream
# -------------------------------------------------------------------

class TestNonStreamEvidence:
    def test_correct_trace_passed(self):
        trace = "infer_abc_12345678"
        captured = {}

        def urlopen(req, timeout=5):
            captured["headers"] = dict(req.headers or {})
            captured["data"] = req.data
            return FakeResponse(200, _chat(trace))

        ev = _gate(urlopen).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(),
            task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "passed"
        assert ev.trace_found is True
        assert ev.prompt_tokens == 18
        assert ev.completion_tokens == 6
        assert ev.response_model == "org/model"

    def test_http_200_no_trace_failed(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeResponse(200, _chat(trace, content="hello world"))).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"
        assert ev.trace_found is False

    def test_old_trace_failed(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeResponse(200, _chat("infer_OLD_99999999"))).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"

    def test_wrong_model_failed(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeResponse(200, _chat(trace, model="other/model"))).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"
        assert "model" in ev.failure_reason

    def test_usage_missing_uncertain(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeResponse(200, _chat(trace, pt=0, ct=0))).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "uncertain"

    def test_request_before_ready_failed(self):
        trace = "infer_abc_12345678"
        ev = _gate(
            lambda req, timeout=5: FakeResponse(200, _chat(trace)),
            now=lambda: NOW_BEFORE,
        ).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"
        assert "before" in ev.failure_reason

    def test_no_auth_header_or_token(self):
        trace = "infer_abc_12345678"
        captured = {}

        def urlopen(req, timeout=5):
            captured["headers"] = dict(req.headers or {})
            captured["data"] = req.data.decode("utf-8") if isinstance(req.data, bytes) else ""
            return FakeResponse(200, _chat(trace))

        ev = _gate(urlopen).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert "Authorization" not in captured["headers"]
        assert "Bearer" not in captured["data"]
        assert "hf_" not in captured["data"]
        assert ev.request_sha256.startswith("sha256:")

    def test_container_binding(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeResponse(200, _chat(trace))).verify_non_stream(
            runtime_plan=_plan(), startup_evidence=_startup(container_id="cid-123"),
            task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.container_id == "cid-123"


# -------------------------------------------------------------------
# SSE stream
# -------------------------------------------------------------------

class TestStreamEvidence:
    def test_trace_split_across_frames(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, _sse(trace))).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "passed"
        assert ev.stream is True
        assert ev.trace_found is True
        assert ev.prompt_tokens == 18
        assert ev.completion_tokens == 6
        assert 0 <= ev.ttft_ms <= ev.total_latency_ms

    def test_done_terminal_received(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, _sse(trace))).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "passed"

    def test_no_terminal_failed(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, _sse(trace, done=False))).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"
        assert "terminate" in ev.failure_reason

    def test_invalid_json_frame_failed(self):
        trace = "infer_abc_12345678"
        lines = _sse(trace)
        lines.insert(2, "data: {not valid json")
        lines.insert(3, "")
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, lines)).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"

    def test_old_trace_stream_failed(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, _sse("infer_OLD_99999999"))).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"

    def test_usage_final_frame(self):
        trace = "infer_abc_12345678"
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, _sse(trace, pt=7, ct=3))).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.prompt_tokens == 7
        assert ev.completion_tokens == 3

    def test_byte_limit(self):
        trace = "infer_abc_12345678"
        # A frame larger than the single-frame byte limit.
        huge = "data: " + json.dumps({"choices": [{"delta": {"content": "x" * 2 * 1024 * 1024}}]}) + "\n"
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, [huge, ""])).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.status == "failed"

    def test_unicode_content_reconstructed(self):
        trace = "infer_abc_12345678"
        # Unicode prefix and the current trace split across frames.
        unicode_prefix = "你好，世界🎉"
        lines = []
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": unicode_prefix}}]}))
        lines.append("")
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": trace}}]}))
        lines.append("")
        lines.append("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
        lines.append("")
        lines.append("data: [DONE]")
        lines.append("")
        ev = _gate(lambda req, timeout=5: FakeStreamResponse(200, lines)).verify_stream(
            runtime_plan=_plan(), startup_evidence=_startup(), task_id="t", operation_id="o", trace_id=trace,
        )
        assert ev.trace_found is True
        assert ev.status == "passed"


class TestSSEParser:
    def test_bytes_lines_decoded(self):
        trace = "infer_abc_12345678"
        lines = [
            b"data: " + json.dumps({"choices": [{"delta": {"content": trace}}]}).encode(),
            b"",
            b"data: [DONE]",
            b"",
        ]
        result = parse_sse_stream(iter(lines))
        assert result.terminal is True
        assert trace in result.content

    def test_frame_limit(self):
        # many frames exceeding the limit -> error
        result = parse_sse_stream(iter([]), max_frames=0)
        assert result.content == ""
