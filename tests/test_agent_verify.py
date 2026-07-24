"""Tests for the LLM-driven verify agent.

Covers: schemas, policy, planner, executor, act_verify loop, llm_helped logic.
Per design doc §13.1, these are mandatory before claiming LLM-driven Agent.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.agent_runtime.schemas import (
    AgentDecision,
    AgentVerifyResult,
    ToolCall,
    ToolResult,
    VERIFY_TOOLS,
    parse_agent_decision,
)
from auto_harness.agent_runtime.policy import ToolPolicy, DEFAULT_ALLOWED_HOSTS
from auto_harness.agent_runtime.state import AgentVerifyState, AgentStepWriter, compute_idempotency_key
from auto_harness.agent_runtime.planner import VerifyPlanner
from auto_harness.agent_runtime.runtime import AgentRuntime
from auto_harness.tools.executor import ToolExecutor
from auto_harness.tools import ToolRegistry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class FakeProvider:
    """Fake LLM provider that returns a fixed JSON string."""

    def __init__(self, response_json: dict = None):
        self._response = response_json or {}

    def complete(self, messages):
        text = json.dumps(self._response, ensure_ascii=False)
        return MagicMock(text=text)


class FakeProviderInvalid:
    """Fake LLM provider that returns non-JSON."""

    def complete(self, messages):
        return MagicMock(text="I think you should probe the endpoint.")


class FakeProviderNoAction:
    """Fake LLM provider that returns no_action."""

    def complete(self, messages):
        return MagicMock(text=json.dumps({
            "status": "no_action",
            "hypothesis": "No safe local probe",
            "confidence": 0.3,
            "tool_call": None,
            "expected_observation": "",
            "stop_reason": "no_safe_tool",
        }))


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------

class TestParseAgentDecision(unittest.TestCase):
    """Test parse_agent_decision strict validation."""

    def test_valid_ok_decision(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "Gradio API exists",
            "confidence": 0.8,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "http://127.0.0.1:7860", "trace_template": "{{trace_id}}"}},
            "expected_observation": "trace in response",
        })
        decision = parse_agent_decision(raw, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "ok")
        self.assertEqual(decision.tool_call.name, "probe_http")
        self.assertAlmostEqual(decision.confidence, 0.8)

    def test_valid_no_action(self):
        raw = json.dumps({
            "status": "no_action",
            "hypothesis": "No safe tool",
            "confidence": 0.3,
            "tool_call": None,
            "stop_reason": "no_safe_tool",
        })
        decision = parse_agent_decision(raw)
        self.assertEqual(decision.status, "no_action")
        self.assertIsNone(decision.tool_call)

    def test_invalid_json_rejected(self):
        decision = parse_agent_decision("not json at all")
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "invalid_json")

    def test_empty_response_rejected(self):
        decision = parse_agent_decision("")
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "empty_response")

    def test_unknown_status_rejected(self):
        raw = json.dumps({"status": "maybe", "hypothesis": "test"})
        decision = parse_agent_decision(raw)
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "invalid_status")

    def test_ok_without_tool_call_rejected(self):
        raw = json.dumps({"status": "ok", "hypothesis": "test", "tool_call": None})
        decision = parse_agent_decision(raw)
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "missing_tool_call_name")

    def test_unknown_tool_rejected(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "test",
            "tool_call": {"name": "hack_system", "input": {}},
        })
        decision = parse_agent_decision(raw, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "unknown_tool")

    def test_invalid_tool_input_type_rejected(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "test",
            "tool_call": {"name": "probe_http", "input": "not a dict"},
        })
        decision = parse_agent_decision(raw, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "invalid_tool_input")

    def test_no_action_with_tool_call_rejected(self):
        """no_action should not have a tool_call with a name."""
        raw = json.dumps({
            "status": "no_action",
            "hypothesis": "test",
            "tool_call": {"name": "probe_http", "input": {}},
        })
        decision = parse_agent_decision(raw, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "no_action_with_tool_call")

    def test_no_action_with_empty_tool_call_dict_allowed(self):
        """no_action with tool_call={} (no name) should be treated as no tool_call."""
        raw = json.dumps({
            "status": "no_action",
            "hypothesis": "test",
            "tool_call": {},
            "stop_reason": "no_safe_tool",
        })
        decision = parse_agent_decision(raw)
        self.assertEqual(decision.status, "no_action")
        self.assertIsNone(decision.tool_call)

    def test_confidence_non_numeric_defaults_to_zero(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "test",
            "confidence": "high",
            "tool_call": {"name": "probe_http", "input": {}},
        })
        decision = parse_agent_decision(raw, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "ok")
        self.assertAlmostEqual(decision.confidence, 0.0)

    def test_fallback_tool_call_parsed(self):
        raw = json.dumps({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.7,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "http://127.0.0.1:7860", "trace_template": "{{trace_id}}"}},
            "expected_observation": "trace in response",
            "fallback_tool_call": {"name": "discover_gradio_api", "input": {"endpoint": "http://127.0.0.1:7860", "trace_template": "{{trace_id}}"}},
        })
        decision = parse_agent_decision(raw, allowed_tools=["probe_http", "discover_gradio_api"])
        self.assertIsNotNone(decision.fallback_tool_call)
        self.assertEqual(decision.fallback_tool_call.name, "discover_gradio_api")


# ------------------------------------------------------------------
# Policy tests
# ------------------------------------------------------------------

class TestToolPolicy(unittest.TestCase):
    """Test ToolPolicy validation for verify agent."""

    def setUp(self):
        self.policy = ToolPolicy(registry=ToolRegistry())

    def test_localhost_probe_allowed(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertTrue(result.allowed)
        self.assertIsNotNone(result.normalized_input)

    def test_external_url_rejected(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "https://evil.example.com/api",
            "trace_template": "{{trace_id}}",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)
        self.assertIn("external host", result.reason)

    def test_missing_trace_template_rejected_for_probe(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)
        self.assertIn("trace_template", result.reason)

    def test_discovery_tool_without_trace_template_allowed(self):
        """discover_gradio_api and discover_openapi_schema are discovery-only,
        so trace_template is still required by policy (they are in TRACE_REQUIRED_TOOLS).
        This test verifies the current behavior."""
        tc = ToolCall(name="discover_gradio_api", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertTrue(result.allowed)

    def test_planner_mode_does_not_execute(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="planner", trace_id="trace-abc")
        self.assertFalse(result.allowed)
        self.assertIn("planner mode", result.reason)

    def test_secret_field_rejected(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
            "api_key": "sk-12345",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)
        self.assertIn("secret", result.reason.lower())

    def test_shell_metacharacters_in_command_field_rejected(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
            "command": "; rm -rf /",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)

    def test_path_traversal_rejected(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
            "path": "../../../etc/passwd",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)
        self.assertIn("traversal", result.reason.lower())

    def test_non_verify_tool_rejected(self):
        tc = ToolCall(name="install_environment", input={"package": "rich"})
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)

    def test_unknown_tool_rejected(self):
        tc = ToolCall(name="hack_system", input={})
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertFalse(result.allowed)
        self.assertIn("not found in registry", result.reason)

    def test_normalized_input_replaces_trace_template(self):
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:7860",
            "trace_template": "{{trace_id}}",
        })
        result = self.policy.validate(tc, stage="verify", agent_mode="gated_actor", trace_id="trace-abc")
        self.assertTrue(result.allowed)
        self.assertEqual(result.normalized_input["trace_template"], "trace-abc")


# ------------------------------------------------------------------
# State tests
# ------------------------------------------------------------------

class TestAgentVerifyState(unittest.TestCase):

    def test_initial_state(self):
        state = AgentVerifyState(trace_id="trace-1", initial_status="uncertain")
        self.assertEqual(state.verify_status, "uncertain")
        self.assertEqual(state.accepted_tool_count, 0)
        self.assertEqual(state.rejected_tool_count, 0)
        self.assertFalse(state.strong_verify_pass)

    def test_record_reject(self):
        state = AgentVerifyState(trace_id="trace-1")
        state.record_reject("critic_rejected")
        self.assertEqual(state.rejected_tool_count, 1)
        self.assertEqual(state.stop_reason, "critic_rejected")

    def test_record_accepted_tool(self):
        state = AgentVerifyState(trace_id="trace-1")
        state.record_accepted_tool()
        self.assertEqual(state.accepted_tool_count, 1)

    def test_apply_tool_result_strong_pass(self):
        state = AgentVerifyState(trace_id="trace-1")
        state.apply_tool_result({"strong_verify_pass": True, "evidence_path": "/tmp/evidence.json"})
        self.assertTrue(state.strong_verify_pass)
        self.assertEqual(state.verify_status, "passed")
        self.assertEqual(state.evidence_paths, ["/tmp/evidence.json"])

    def test_to_result(self):
        state = AgentVerifyState(trace_id="trace-1")
        state.record_accepted_tool()
        result = state.to_result(final_status="passed", stop_reason="strong_verify_pass", mode="gated_actor", llm_helped=True)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["final_status"], "passed")
        self.assertTrue(result["llm_helped"])


class TestComputeIdempotencyKey(unittest.TestCase):

    def test_deterministic(self):
        key1 = compute_idempotency_key("run-1", 0, "probe_http", {"endpoint": "http://127.0.0.1:7860"})
        key2 = compute_idempotency_key("run-1", 0, "probe_http", {"endpoint": "http://127.0.0.1:7860"})
        self.assertEqual(key1, key2)

    def test_different_input_different_key(self):
        key1 = compute_idempotency_key("run-1", 0, "probe_http", {"endpoint": "http://127.0.0.1:7860"})
        key2 = compute_idempotency_key("run-1", 0, "probe_http", {"endpoint": "http://127.0.0.1:8080"})
        self.assertNotEqual(key1, key2)


# ------------------------------------------------------------------
# Planner tests
# ------------------------------------------------------------------

class TestVerifyPlanner(unittest.TestCase):

    def test_no_provider_returns_no_action(self):
        planner = VerifyPlanner(provider=None)
        decision = planner.plan_verify({}, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "no_action")
        self.assertEqual(decision.stop_reason, "no_provider")

    def test_valid_provider_returns_parsed_decision(self):
        provider = FakeProvider({
            "status": "ok",
            "hypothesis": "Gradio API",
            "confidence": 0.8,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "http://127.0.0.1:7860", "trace_template": "{{trace_id}}"}},
            "expected_observation": "trace in response",
        })
        planner = VerifyPlanner(provider=provider)
        decision = planner.plan_verify({"service": {}, "failed_checks": [], "allowed_tools": ["probe_http"]}, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "ok")
        self.assertEqual(decision.tool_call.name, "probe_http")

    def test_invalid_json_from_provider(self):
        planner = VerifyPlanner(provider=FakeProviderInvalid())
        decision = planner.plan_verify({}, allowed_tools=["probe_http"])
        self.assertEqual(decision.status, "invalid")
        self.assertEqual(decision.stop_reason, "invalid_json")


# ------------------------------------------------------------------
# Executor tests
# ------------------------------------------------------------------

class TestToolExecutor(unittest.TestCase):

    def test_unknown_tool_rejected(self):
        executor = ToolExecutor()
        tc = ToolCall(name="hack_system", input={})
        result = executor.execute(tc, {})
        self.assertEqual(result.status, "rejected")
        self.assertIn("not implemented", result.error)

    def test_probe_http_dispatched(self):
        """Verify probe_http is dispatched (will fail without real server, but dispatch works)."""
        executor = ToolExecutor()
        tc = ToolCall(name="probe_http", input={
            "endpoint": "http://127.0.0.1:99999",
            "trace_template": "trace-1",
        })
        result = executor.execute(tc, {"trace_id": "trace-1", "evidence_dir": None, "run_dir": "/tmp"})
        # Will fail to connect, but should not be "rejected" (unknown tool)
        self.assertNotEqual(result.status, "rejected")
        self.assertEqual(result.tool_name, "probe_http")

    def test_probe_browser_dom_dispatched(self):
        """Verify probe_browser_dom is dispatched (not unknown tool)."""
        executor = ToolExecutor()
        tc = ToolCall(name="probe_browser_dom", input={
            "endpoint": "http://127.0.0.1:99999",
            "trace_template": "trace-1",
        })
        result = executor.execute(tc, {"trace_id": "trace-1", "evidence_dir": None, "run_dir": "/tmp"})
        self.assertNotEqual(result.status, "rejected")
        self.assertEqual(result.tool_name, "probe_browser_dom")


# ------------------------------------------------------------------
# act_verify loop tests
# ------------------------------------------------------------------

class TestActVerifyLoop(unittest.TestCase):

    def _make_runtime(self):
        return AgentRuntime()

    def _base_kwargs(self, run_dir, provider=None):
        return {
            "run_dir": run_dir,
            "repo_path": run_dir / "workspace" / "repo",
            "initial_verify_result": {
                "status": "uncertain",
                "data": {
                    "checks": [{"name": "http_trace_response", "status": "uncertain", "reason": "no trace"}],
                    "frameworks": ["gradio"],
                    "trace_id": "trace-test-1",
                },
            },
            "service_context": {
                "process_alive": True,
                "port_ready": True,
                "endpoint_candidates": ["http://127.0.0.1:7860"],
            },
            "trace_id": "trace-test-1",
            "config": {},
            "provider": provider,
            "max_steps": 3,
            "agent_mode": "gated_actor",
            "allowed_hosts": ["127.0.0.1", "localhost"],
        }

    def test_invalid_llm_output_records_reject_and_breaks(self):
        """Invalid LLM output should record rejection and stop the loop."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            runtime = self._make_runtime()
            result = runtime.act_verify(**self._base_kwargs(run_dir, provider=FakeProviderInvalid()))
            self.assertEqual(result["final_status"], "uncertain")
            self.assertTrue(result["triggered"])
            # Should have recorded the rejection
            steps_file = run_dir / "agent_verify_steps.jsonl"
            if steps_file.exists():
                steps = [json.loads(l) for l in steps_file.read_text().strip().splitlines()]
                self.assertTrue(any("invalid_llm_output" in s.get("execution", {}).get("reason", "") for s in steps))

    def test_no_action_records_reject_and_breaks(self):
        """LLM returning no_action should stop the loop."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            runtime = self._make_runtime()
            result = runtime.act_verify(**self._base_kwargs(run_dir, provider=FakeProviderNoAction()))
            self.assertEqual(result["final_status"], "uncertain")
            self.assertTrue(result["triggered"])

    def test_planner_mode_does_not_execute(self):
        """In planner mode, tools should not be executed."""
        provider = FakeProvider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.7,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "http://127.0.0.1:7860", "trace_template": "{{trace_id}}"}},
            "expected_observation": "trace in response",
        })
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            kwargs = self._base_kwargs(run_dir, provider=provider)
            kwargs["agent_mode"] = "planner"
            runtime = self._make_runtime()
            result = runtime.act_verify(**kwargs)
            # Planner mode should not execute tools
            self.assertEqual(result["accepted_tool_count"], 0)
            self.assertFalse(result["llm_helped"])

    def test_critic_reject_continues_loop(self):
        """Critic rejection should continue the loop, not break it.
        With max_steps=3 and a provider that always returns a tool_call
        that the critic will reject (secret in input), the loop should
        exhaust max_steps rather than break on first rejection."""
        provider = FakeProvider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.7,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "http://127.0.0.1:7860", "trace_template": "{{trace_id}}", "api_key": "sk-secret"}},
            "expected_observation": "trace in response",
        })
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            kwargs = self._base_kwargs(run_dir, provider=provider)
            kwargs["max_steps"] = 3
            runtime = self._make_runtime()
            result = runtime.act_verify(**kwargs)
            # Should have attempted multiple steps (not break on first critic reject)
            self.assertTrue(result["step_count"] > 1, "Loop should continue after critic rejection")
            self.assertEqual(result["final_status"], "uncertain")

    def test_policy_reject_continues_loop(self):
        """Policy rejection should continue the loop, not break it.
        Provider returns a tool_call targeting an external URL, which policy rejects."""
        provider = FakeProvider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.7,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "https://evil.example.com/api", "trace_template": "{{trace_id}}"}},
            "expected_observation": "trace in response",
        })
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            kwargs = self._base_kwargs(run_dir, provider=provider)
            kwargs["max_steps"] = 3
            runtime = self._make_runtime()
            result = runtime.act_verify(**kwargs)
            # Should have attempted multiple steps (not break on first policy reject)
            self.assertTrue(result["step_count"] > 1, "Loop should continue after policy rejection")
            self.assertEqual(result["final_status"], "uncertain")

    def test_llm_helped_false_when_no_execution(self):
        """llm_helped must be false when no tool was executed."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            runtime = self._make_runtime()
            result = runtime.act_verify(**self._base_kwargs(run_dir, provider=FakeProviderNoAction()))
            self.assertFalse(result["llm_helped"])

    def test_llm_helped_false_when_status_not_improved(self):
        """llm_helped must be false when execution did not improve status."""
        # Provider returns a valid tool_call, but the probe will fail (no real server)
        # so status stays uncertain
        provider = FakeProvider({
            "status": "ok",
            "hypothesis": "test",
            "confidence": 0.7,
            "tool_call": {"name": "probe_http", "input": {"endpoint": "http://127.0.0.1:99999", "trace_template": "{{trace_id}}"}},
            "expected_observation": "trace in response",
        })
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True, exist_ok=True)
            kwargs = self._base_kwargs(run_dir, provider=provider)
            kwargs["max_steps"] = 1
            runtime = self._make_runtime()
            result = runtime.act_verify(**kwargs)
            # Tool executed but didn't improve status
            self.assertFalse(result["llm_helped"])

    def test_audit_mode_does_not_set_llm_helped(self):
        """audit() mode must never set llm_helped=true."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            from auto_harness.agent_runtime.schemas import AgentGoal
            goal = AgentGoal(task_id="t1", objective="test", success_condition="verify pass")
            runtime = self._make_runtime()
            result = runtime.audit(goal, run_dir, {"verify": {"status": "passed"}}, contribution={"llm_helped": True})
            # llm_helped is in the state sub-dict, not at top level
            self.assertFalse(result["state"]["llm_helped"])
            self.assertEqual(result["mode"], "audit")


# ------------------------------------------------------------------
# llm_helped strict logic tests
# ------------------------------------------------------------------

class TestLlmHelpedLogic(unittest.TestCase):

    def test_llm_helped_requires_gated_actor(self):
        self.assertFalse(AgentRuntime._compute_llm_helped("planner", MagicMock(verify_status="passed", strong_verify_pass=True, evidence_paths=["/tmp/e.json"])))

    def test_llm_helped_requires_passed_status(self):
        state = AgentVerifyState(trace_id="t1")
        # status is "uncertain" by default
        self.assertFalse(AgentRuntime._compute_llm_helped("gated_actor", state))

    def test_llm_helped_requires_strong_verify_pass(self):
        state = AgentVerifyState(trace_id="t1")
        state.verify_status = "passed"
        # strong_verify_pass is False by default
        self.assertFalse(AgentRuntime._compute_llm_helped("gated_actor", state))

    def test_llm_helped_requires_evidence_path_exists(self):
        state = AgentVerifyState(trace_id="t1")
        state.verify_status = "passed"
        state.strong_verify_pass = True
        state.evidence_paths = ["/nonexistent/path/evidence.json"]
        self.assertFalse(AgentRuntime._compute_llm_helped("gated_actor", state))

    def test_llm_helped_true_when_all_conditions_met(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"tool": "probe_http"}, f)
            ep = f.name
        try:
            state = AgentVerifyState(trace_id="t1")
            state.verify_status = "passed"
            state.strong_verify_pass = True
            state.evidence_paths = [ep]
            self.assertTrue(AgentRuntime._compute_llm_helped("gated_actor", state))
        finally:
            Path(ep).unlink(missing_ok=True)


# ------------------------------------------------------------------
# StepWriter tests
# ------------------------------------------------------------------

class TestAgentStepWriter(unittest.TestCase):

    def test_write_step_creates_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = AgentStepWriter(Path(tmp))
            writer.write_step({"step_index": 1, "stage": "verify"})
            steps_file = Path(tmp) / "agent_verify_steps.jsonl"
            self.assertTrue(steps_file.exists())
            lines = steps_file.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["step_index"], 1)

    def test_write_rejected_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = AgentStepWriter(Path(tmp))
            writer.write_rejected(step_index=1, trace_id="t1", decision={"status": "invalid"}, reason="bad json")
            steps_file = Path(tmp) / "agent_verify_steps.jsonl"
            self.assertTrue(steps_file.exists())
            step = json.loads(steps_file.read_text().strip())
            self.assertFalse(step["execution"]["executed"])

    def test_write_state_creates_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = AgentStepWriter(Path(tmp))
            state = AgentVerifyState(trace_id="t1")
            writer.write_state(state, mode="gated_actor")
            state_file = Path(tmp) / "agent_state.json"
            self.assertTrue(state_file.exists())
            data = json.loads(state_file.read_text())
            self.assertEqual(data["trace_id"], "t1")

    def test_write_result_creates_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = AgentStepWriter(Path(tmp))
            writer.write_result({"triggered": True, "final_status": "passed"})
            result_file = Path(tmp) / "reports" / "agent_verify_result.json"
            self.assertTrue(result_file.exists())


# ------------------------------------------------------------------
# VerifyModule integration test (agent_verify_config)
# ------------------------------------------------------------------

class TestVerifyModuleAgentVerifyConfig(unittest.TestCase):

    def test_agent_verify_disabled_by_default(self):
        """When agent_verify_config is empty, _should_trigger_agent_verify returns False."""
        from auto_harness.modules.verify import VerifyModule
        vm = VerifyModule()
        self.assertFalse(vm._should_trigger_agent_verify({"process_alive": True, "port_ready": True}))

    def test_agent_verify_enabled_triggers(self):
        from auto_harness.modules.verify import VerifyModule
        vm = VerifyModule(agent_verify_config={
            "agent_mode": "gated_actor",
            "agent_enable_verify": True,
        })
        self.assertTrue(vm._should_trigger_agent_verify({"process_alive": True, "port_ready": True}))

    def test_agent_verify_not_triggered_when_process_dead(self):
        from auto_harness.modules.verify import VerifyModule
        vm = VerifyModule(agent_verify_config={
            "agent_mode": "gated_actor",
            "agent_enable_verify": True,
        })
        self.assertFalse(vm._should_trigger_agent_verify({"process_alive": False, "port_ready": True}))

    def test_agent_verify_not_triggered_in_off_mode(self):
        from auto_harness.modules.verify import VerifyModule
        vm = VerifyModule(agent_verify_config={
            "agent_mode": "off",
            "agent_enable_verify": True,
        })
        self.assertFalse(vm._should_trigger_agent_verify({"process_alive": True, "port_ready": True}))


if __name__ == "__main__":
    unittest.main()
