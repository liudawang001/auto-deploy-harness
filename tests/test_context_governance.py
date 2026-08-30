import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auto_harness.agent.engine import AgentDecisionEngine
from auto_harness.agent.schemas import AgentObservation
from auto_harness.agent.traces import AgentTraceWriter
from auto_harness.context import (
    ContextGovernanceError,
    LLMCallExecutor,
    PromptEnvelope,
    compact_agent_observation,
    get_context_profile,
)
from auto_harness.context.assembler import fit_items_to_budget
from auto_harness.context.logs import LogCompactor
from auto_harness.context.repository import RepoEvidenceSelector
from auto_harness.context.tokens import ConservativeTokenEstimator
from auto_harness.providers.base import LLMResult, Message


class _Config:
    agent_context_mode = "enforce"
    agent_context_window_tokens = None
    agent_context_reserved_output_tokens = 512
    agent_context_safety_margin_tokens = 256
    agent_context_unknown_model_fallback_tokens = 8192
    agent_context_max_overflow_retries = 1
    agent_context_skill_budget_tokens = 2000
    agent_context_memory_budget_tokens = 2000


class _RecordingProvider:
    context_window_tokens = 16384
    max_tokens = 1024
    model = "test-model"

    def __init__(self, overflow_count=0):
        self.calls = []
        self.overflow_count = overflow_count

    def complete(self, messages, temperature=0.2, max_output_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        if len(self.calls) <= self.overflow_count:
            raise RuntimeError("context length exceeded")
        return LLMResult(
            text='{"status":"ok"}',
            usage={"input_tokens": 12, "output_tokens": 3},
        )


class _UnknownProvider:
    model = "unregistered-model"

    def __init__(self):
        self.calls = []

    def complete(self, messages, temperature=0.2, max_output_tokens=None):
        self.calls.append(messages)
        return LLMResult(text='{"status":"ok"}')


class _AuditedProvider:
    provider_name = "deepseek"
    purpose = "plan_first"
    context_window_tokens = 16384
    max_tokens = 1024
    model = "test-model"

    def __init__(self):
        self.request_context = None

    def complete(
        self,
        messages,
        temperature=0.2,
        max_output_tokens=None,
        request_context=None,
    ):
        self.request_context = request_context
        return LLMResult(
            text='{"status":"ok"}',
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "prompt_cache_hit_tokens": 5,
            },
            protocol="json_action",
            context={"reasoning_present": True, "reasoning_sha256": "abc"},
            finish_reason="stop",
            request_id="req-1",
            provider_name="deepseek",
            provider_model="test-model",
            retry_count=1,
        )


class ContextGovernanceTest(unittest.TestCase):
    def test_planning_profile_caps(self):
        self.assertEqual(get_context_profile("plan").total_input_cap_tokens, 50000)
        self.assertEqual(get_context_profile("replan").total_input_cap_tokens, 30000)
        self.assertEqual(get_context_profile("diagnose").total_input_cap_tokens, 30000)

    def test_provider_request_context_and_audit_metadata_are_preserved(self):
        provider = _AuditedProvider()
        call = LLMCallExecutor(_Config()).execute(
            call_site="plan_first.plan",
            stage="plan",
            provider=provider,
            envelope=PromptEnvelope(
                messages=[Message(role="user", content="Return JSON")],
                requested_output_tokens=512,
            ),
            profile=get_context_profile("plan", 512),
        )
        self.assertEqual(provider.request_context.call_site, "plan_first.plan")
        self.assertTrue(provider.request_context.deadline_at)
        audit = call.provider_result.context["provider_response"]
        self.assertEqual(audit["request_id"], "req-1")
        self.assertEqual(audit["reasoning_sha256"], "abc")
        self.assertEqual(audit["retry_count"], 1)
        self.assertEqual(
            call.provider_result.context["usage"]["cache_hit_tokens"], 5
        )

    def test_estimator_counts_messages_tools_and_schema(self):
        provider = _RecordingProvider()
        from auto_harness.context.capabilities import resolve_provider_capabilities

        capabilities = resolve_provider_capabilities(provider, _Config())
        estimator = ConservativeTokenEstimator()
        base = estimator.estimate_request(
            [Message(role="user", content="hello")], capabilities
        )
        full = estimator.estimate_request(
            [Message(role="system", content="guard"), Message(role="user", content="hello")],
            capabilities,
            tools=[{"name": "probe"}],
            output_schema={"type": "object"},
        )
        self.assertGreater(full, base)

    def test_enforce_sends_compacted_candidate(self):
        provider = _RecordingProvider()
        call = LLMCallExecutor(_Config()).execute(
            call_site="gate.runner",
            stage="runner",
            provider=provider,
            envelope=PromptEnvelope(
                messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="x" * 12000),
                ],
                candidate_messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="small"),
                ],
            ),
            profile=get_context_profile("runner", 512),
        )
        self.assertEqual(provider.calls[0]["messages"][1].content, "small")
        self.assertTrue(call.context_result.truncated)
        self.assertLessEqual(
            call.context_result.estimated_input_tokens,
            6000,
        )

    def test_enforce_preserves_original_when_it_is_below_compaction_threshold(self):
        provider = _RecordingProvider()
        LLMCallExecutor(_Config()).execute(
            call_site="gate.runner",
            stage="runner",
            provider=provider,
            envelope=PromptEnvelope(
                messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="complete original"),
                ],
                candidate_messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="smaller"),
                ],
            ),
            profile=get_context_profile("runner", 512),
        )
        self.assertEqual(
            provider.calls[0]["messages"][1].content,
            "complete original",
        )

    def test_observe_and_shadow_send_original_request(self):
        for mode in ("observe", "shadow"):
            config = _Config()
            config.agent_context_mode = mode
            provider = _RecordingProvider()
            call = LLMCallExecutor(config).execute(
                call_site="gate.runner",
                stage="runner",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[Message(role="user", content="original")],
                    candidate_messages=[Message(role="user", content="candidate")],
                ),
                profile=get_context_profile("runner", 512),
            )
            self.assertEqual(provider.calls[0]["messages"][0].content, "original")
            self.assertEqual(call.context_result.mode, mode)

    def test_unknown_provider_fails_closed_in_enforce_mode(self):
        provider = _UnknownProvider()
        with self.assertRaises(ContextGovernanceError) as caught:
            LLMCallExecutor(_Config()).execute(
                call_site="agent.plan",
                stage="plan",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[Message(role="user", content="plan")]
                ),
                profile=get_context_profile("plan", 512),
            )
        self.assertEqual(
            caught.exception.stop_reason, "context_capability_unknown"
        )
        self.assertEqual(provider.calls, [])

    def test_enforce_rejects_candidate_without_system_guardrail(self):
        provider = _RecordingProvider()
        with self.assertRaises(ContextGovernanceError) as caught:
            LLMCallExecutor(_Config()).execute(
                call_site="agent.plan",
                stage="plan",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[
                        Message(role="system", content="trusted guardrail"),
                        Message(role="user", content="x" * 20000),
                    ],
                    candidate_messages=[
                        Message(role="user", content="task")
                    ],
                ),
                profile=get_context_profile("plan", 512),
            )
        self.assertEqual(
            caught.exception.stop_reason, "context_required_section_missing"
        )
        self.assertEqual(provider.calls, [])

    def test_enforce_rejects_candidate_without_required_task(self):
        provider = _RecordingProvider()
        with self.assertRaises(ContextGovernanceError) as caught:
            LLMCallExecutor(_Config()).execute(
                call_site="gate.runner",
                stage="runner",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[
                        Message(role="system", content="trusted guardrail"),
                        Message(role="user", content="x" * 12000),
                    ],
                    candidate_messages=[
                        Message(role="system", content="trusted guardrail"),
                    ],
                ),
                profile=get_context_profile("runner", 512),
            )
        self.assertEqual(
            caught.exception.stop_reason,
            "context_required_section_missing",
        )
        self.assertEqual(provider.calls, [])

    def test_enforce_rejects_candidate_without_required_user_contract(self):
        from auto_harness.context.models import (
            ContextPriority,
            ContextSection,
            TrustLevel,
        )

        provider = _RecordingProvider()
        with self.assertRaises(ContextGovernanceError) as caught:
            LLMCallExecutor(_Config()).execute(
                call_site="agent.plan",
                stage="plan",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[
                        Message(role="system", content="trusted guardrail"),
                        Message(role="user", content="required output contract"),
                        Message(role="user", content="x" * 20000),
                    ],
                    candidate_messages=[
                        Message(role="system", content="trusted guardrail"),
                        Message(role="user", content="small candidate"),
                    ],
                    sections=[
                        ContextSection(
                            name="output_contract",
                            content="required output contract",
                            priority=ContextPriority.REQUIRED,
                            trust_level=TrustLevel.TRUSTED_INSTRUCTION,
                            content_type="schema",
                            required=True,
                        )
                    ],
                ),
                profile=get_context_profile("plan", 512),
            )
        self.assertEqual(
            caught.exception.stop_reason,
            "context_required_section_missing",
        )
        self.assertEqual(provider.calls, [])

    def test_budget_exceeded_fails_before_provider_call(self):
        provider = _RecordingProvider()
        with self.assertRaises(ContextGovernanceError) as caught:
            LLMCallExecutor(_Config()).execute(
                call_site="gate.runner",
                stage="runner",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[Message(role="user", content="x" * 10000)],
                    candidate_messages=[Message(role="user", content="x" * 10000)],
                ),
                profile=get_context_profile("runner", 512),
            )
        self.assertEqual(caught.exception.stop_reason, "context_budget_exceeded")
        self.assertEqual(provider.calls, [])

    def test_provider_overflow_retries_once_with_smaller_request(self):
        provider = _RecordingProvider(overflow_count=1)
        call = LLMCallExecutor(_Config()).execute(
            call_site="agent.diagnose",
            stage="diagnose",
            provider=provider,
            envelope=PromptEnvelope(
                messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="normal"),
                ],
                candidate_messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="compact"),
                ],
                retry_messages=[
                    Message(role="system", content="guard"),
                    Message(role="user", content="retry"),
                ],
            ),
            profile=get_context_profile("diagnose", 512),
        )
        self.assertEqual(call.attempts, 2)
        self.assertEqual(provider.calls[1]["messages"][1].content, "retry")

    def test_provider_overflow_never_retries_third_time(self):
        provider = _RecordingProvider(overflow_count=3)
        with self.assertRaises(ContextGovernanceError) as caught:
            LLMCallExecutor(_Config()).execute(
                call_site="agent.diagnose",
                stage="diagnose",
                provider=provider,
                envelope=PromptEnvelope(
                    messages=[
                        Message(role="system", content="guard"),
                        Message(role="user", content="normal"),
                    ],
                    candidate_messages=[
                        Message(role="system", content="guard"),
                        Message(role="user", content="compact"),
                    ],
                    retry_messages=[
                        Message(role="system", content="guard"),
                        Message(role="user", content="retry"),
                    ],
                ),
                profile=get_context_profile("diagnose", 512),
            )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            caught.exception.stop_reason,
            "provider_context_limit_exceeded",
        )
        self.assertEqual(caught.exception.context["attempts"], 2)
        self.assertEqual(
            caught.exception.context["selected_variant"],
            "retry",
        )
        self.assertIn(
            "provider_overflow_retry",
            {
                event["strategy"]
                for event in caught.exception.context["shrink_events"]
            },
        )

    def test_observation_compaction_removes_unbounded_history(self):
        observation = AgentObservation(
            task_id="task",
            stage="runner",
            file_tree=["f%s.py" % index for index in range(10000)],
            selected_files={
                "app.py": {"content": "a" * 10000},
                "requirements.txt": {"content": "requests\n"},
                "other.py": {"content": "b" * 10000},
            },
            deterministic_result={"stderr": "noise\n" * 10000 + "RuntimeError: boom"},
            previous_results={
                "runner": {
                    "status": "failed",
                    "summary": "boom",
                    "stdout": "x" * 100000,
                }
            },
            memory_hits=[{"id": str(index), "symptom": "x" * 5000} for index in range(10)],
        )
        compacted = compact_agent_observation(
            observation, profile="diagnose"
        )
        self.assertLessEqual(len(compacted.file_tree), 201)
        self.assertLessEqual(len(compacted.selected_files), 4)
        self.assertNotIn("stdout", compacted.previous_results["runner"])
        self.assertLessEqual(len(compacted.memory_hits), 3)

    def test_skill_and_memory_lists_have_independent_hard_budgets(self):
        skills = [
            {"name": "skill-%s" % index, "content": "技" * 5000}
            for index in range(10)
        ]
        memories = [
            {"id": "memory-%s" % index, "symptom": "忆" * 5000}
            for index in range(10)
        ]
        for items, limit in ((skills, 700), (memories, 900)):
            fitted = fit_items_to_budget(items, limit)
            serialized = json.dumps(
                fitted,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            self.assertLessEqual(len(serialized), limit)

    def test_agent_engine_compacts_large_observation_before_provider_call(self):
        provider = _RecordingProvider()
        observation = AgentObservation(
            task_id="large-task",
            stage="runner",
            file_tree=["src/f%s.py" % index for index in range(10000)],
            selected_files={
                "src/app.py": {"content": "a" * 100000},
                "requirements.txt": {"content": "requests\n" * 10000},
                "src/other.py": {"content": "b" * 100000},
            },
            deterministic_result={
                "stderr": "noise\n" * 50000 + "RuntimeError: boom"
            },
            previous_results={
                "runner": {
                    "status": "failed",
                    "summary": "boom",
                    "stdout": "x" * 200000,
                }
            },
            memory_hits=[
                {"id": str(index), "symptom": "m" * 10000}
                for index in range(20)
            ],
            extra={"raw_history": "h" * 200000},
        )
        with tempfile.TemporaryDirectory() as tmp:
            decision = AgentDecisionEngine(
                provider,
                config=_Config(),
                trace_writer=AgentTraceWriter(Path(tmp)),
            ).decide(observation)
            trace = json.loads(Path(decision.trace_path).read_text(encoding="utf-8"))

        self.assertEqual(decision.status, "ok", decision.rationale)
        self.assertEqual(len(provider.calls), 1)
        self.assertLess(
            sum(
                len(message.content)
                for message in provider.calls[0]["messages"]
            ),
            20000,
        )
        self.assertEqual(trace["context"]["mode"], "enforce")
        self.assertEqual(trace["context"]["selected_variant"], "retry")
        self.assertIn(
            "preflight_budget_retry",
            {
                event["strategy"]
                for event in trace["context"]["shrink_events"]
            },
        )
        self.assertLessEqual(
            trace["context"]["estimated_input_tokens"],
            trace["context"]["max_input_tokens"],
        )
        self.assertEqual(trace["context"]["memory_count"], 1)

    def test_log_compactor_keeps_first_error_tail_and_redacts_secret(self):
        text = (
            "start\napi_key=topsecret\nValueError: first\n"
            + "noise\n" * 100
            + "RuntimeError: final\n"
        )
        result = LogCompactor().compact(text, max_chars=2000, exit_code=1)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("ValueError", result["first_error_line"])
        self.assertIn("RuntimeError: final", result["stack_tail"])
        self.assertNotIn("topsecret", str(result))

    def test_log_compactor_enforces_serialized_hard_limit(self):
        text = "\n".join(
            "RuntimeError: line-%03d %s" % (index, "x" * 3000)
            for index in range(100)
        )
        for limit in (1000, 2000, 4000):
            result = LogCompactor().compact(text, max_chars=limit)
            self.assertLessEqual(
                len(json.dumps(result, ensure_ascii=False, sort_keys=True)),
                limit,
            )
            self.assertTrue(result.get("truncated"))

    def test_xunfei_context_environment_does_not_register_other_provider(self):
        from auto_harness.context.capabilities import resolve_provider_capabilities

        with patch.dict(
            os.environ,
            {"XUNFEI_CONTEXT_WINDOW_TOKENS": "9999"},
        ):
            capabilities = resolve_provider_capabilities(
                _UnknownProvider(),
                _Config(),
            )
        self.assertEqual(capabilities.source, "fallback")
        self.assertNotEqual(capabilities.context_window_tokens, 9999)

    def test_stage_planner_uses_injected_enforce_config(self):
        from auto_harness.agent_runtime.stage_planners import PlanPlanner

        provider = _RecordingProvider()
        decision = PlanPlanner(config=_Config()).plan(
            {
                "analysis_summary": {},
                "frameworks": [],
                "previous_results": {},
                "uncertainties": [],
                "constraints": [],
            },
            provider=provider,
        )
        self.assertEqual(decision.context.get("mode"), "enforce")

    def test_memory_evolution_passes_config_to_curator(self):
        from auto_harness.memory.evolution import MemoryEvolutionManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MemoryEvolutionManager(
                root / "memory",
                root / "skills",
                provider=_RecordingProvider(),
                config=_Config(),
            )
        self.assertIs(manager.curator.config.__class__, _Config)

    def test_repo_selector_prioritizes_stack_file(self):
        selected = {
            "misc.py": {"content": "nothing"},
            "pkg/service.py": {"content": "raise RuntimeError()"},
            "requirements.txt": {"content": "requests"},
        }
        result = RepoEvidenceSelector().select(
            selected,
            'File "pkg/service.py", line 3, in run',
            max_files=1,
        )
        self.assertEqual(list(result), ["pkg/service.py"])

    def test_business_code_has_no_direct_provider_calls(self):
        root = Path(__file__).resolve().parents[1] / "src" / "auto_harness"
        targets = [
            root / "agent",
            root / "agent_runtime",
            root / "graph",
            root / "modules",
            root / "memory",
            root / "orchestrator.py",
        ]
        findings = []
        for target in targets:
            paths = [target] if target.is_file() else target.rglob("*.py")
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"complete", "complete_with_tools"}
                    ):
                        findings.append(
                            "%s:%s"
                            % (path.relative_to(root), node.lineno)
                        )
        self.assertEqual(findings, [])

    def test_project_snapshot_tree_is_bounded(self):
        from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(12):
                (root / ("f%02d.py" % index)).write_text("pass\n", encoding="utf-8")
            snapshot = ProjectSnapshotBuilder(
                max_files=2, max_tree_entries=5
            ).build(root)
        self.assertEqual(len(snapshot["file_tree"]), 5)
        self.assertEqual(snapshot["file_tree_summary"]["total_file_count"], 12)
        self.assertEqual(snapshot["file_tree_summary"]["omitted_file_count"], 7)

    def test_aggressive_snapshot_compaction_bounds_command_registry(self):
        from auto_harness.context.assembler import compact_project_snapshot

        snapshot = {
            "command_registry": {
                "schema_version": 1,
                "repository_fingerprint": "abc",
                "evidence": [
                    {"path": "README.md", "content": "x" * 2000}
                    for _ in range(40)
                ],
                "candidates": [
                    {"argv": ["python", "main.py"], "reason": "y" * 1000}
                    for _ in range(20)
                ],
            },
            "file_tree": ["main.py"],
            "selected_files": {"main.py": {"content": "print('ok')"}},
            "detected_signals": {"entrypoint_candidates": ["main.py"]},
        }

        compacted = compact_project_snapshot(snapshot, aggressive=True)

        registry = compacted["command_registry"]
        self.assertNotIn("evidence", registry)
        self.assertEqual(registry["evidence_count"], 40)
        self.assertEqual(len(registry["candidates"]), 8)
        self.assertLess(len(json.dumps(compacted)), 12000)

    def test_aggressive_snapshot_prioritizes_serve_only_command(self):
        from auto_harness.context.assembler import compact_project_snapshot

        snapshot = {
            "command_registry": {
                "candidates": [
                    {"phase": "install", "argv": ["npm", "ci"]},
                    {"phase": "run", "argv": ["python", "main.py"]},
                    {
                        "phase": "run",
                        "candidate_id": "serve",
                        "argv": ["python", "main.py", "--serve-only"],
                    },
                ]
            },
            "file_tree": ["main.py"],
            "selected_files": {"main.py": {"content": "serve"}},
        }

        compacted = compact_project_snapshot(snapshot, aggressive=True)

        first = compacted["command_registry"]["candidates"][0]
        self.assertEqual(first["candidate_id"], "serve")

    def test_aggressive_snapshot_prioritizes_application_run_target_over_docs(self):
        from auto_harness.context.assembler import compact_project_snapshot

        snapshot = {
            "command_registry": {
                "candidates": [
                    {
                        "phase": "run",
                        "candidate_id": "docs",
                        "argv": ["npm", "run", "serve"],
                        "cwd": "docs",
                    },
                    {
                        "phase": "run",
                        "candidate_id": "app",
                        "argv": ["make", "-f", "Makefile", "run_cli"],
                        "cwd": ".",
                    },
                ]
            },
            "file_tree": ["README.md", "pyproject.toml"],
            "selected_files": {
                "README.md": {"content": "make run_cli"},
                "pyproject.toml": {"content": "[project]"},
            },
        }

        compacted = compact_project_snapshot(snapshot, aggressive=True)

        assert compacted["command_registry"]["candidates"][0]["candidate_id"] == "app"
        assert "README.md" in compacted["selected_files"]


if __name__ == "__main__":
    unittest.main()
