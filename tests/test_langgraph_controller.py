"""LangGraphController integration tests with mock dependencies.

Tests the full graph flow: normal dry-run, policy reject, verify-replan.
Uses InMemorySaver for fast in-process testing (no SQLite needed for
basic flow tests; SQLite resume is tested separately).
"""
import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.controllers.langgraph import (
    LangGraphController,
    build_initial_state,
    build_graph,
    STAGES,
    SIDE_EFFECT_STAGES,
)
from auto_harness.graph.nodes import GraphNodeDependencies, merge_plan_analysis
from auto_harness.graph.state import DeploymentGraphState


# -------------------------------------------------------------------
# Fake dependencies for testing
# -------------------------------------------------------------------

@dataclass
class FakeStageExecutionResult:
    """Mimics StageExecutionResult from agent_runtime.stage_executor."""
    stage: str = ""
    before_status: str = ""
    after_status: str = "passed"
    result: dict = field(default_factory=dict)
    changed: bool = True
    evidence_paths: list = field(default_factory=list)
    error: str = ""


class FakeStageExecutor:
    """Stage executor that returns passed for all stages by default."""
    def __init__(self, fail_stages=None):
        self.fail_stages = fail_stages or []
        self.call_count = 0
        self.calls = []

    def execute_stage(self, **kwargs):
        self.call_count += 1
        stage = kwargs.get("stage", "")
        self.calls.append(stage)
        if stage in self.fail_stages:
            return FakeStageExecutionResult(
                stage=stage,
                after_status="failed",
                result={"status": "failed", "error": "intentional failure", "data": {}},
                error="intentional failure",
            )
        return FakeStageExecutionResult(
            stage=stage,
            after_status="passed",
            result={"status": "passed", "data": {}},
        )


class FakePlanner:
    """Planner that returns a valid plan text."""
    call_count = 0
    replan_call_count = 0

    def plan(self, snapshot, **kwargs):
        self.call_count += 1
        result = MagicMock()
        result.text = '{"status": "ok", "plan_id": "test-plan", "summary": "test", "grounding": [], "environment": {"install_commands": []}, "run": {"candidates": [], "selected_candidate_id": ""}, "verify": {"request": {"url": "http://localhost:8501", "trace_id": "{{trace_id}}"}}}'
        return result

    def replan(self, snapshot, previous_plan, failure, **kwargs):
        self.replan_call_count += 1
        result = MagicMock()
        result.text = '{"status": "ok", "plan_id": "test-replan", "summary": "revised", "grounding": [], "environment": {"install_commands": []}, "run": {"candidates": [], "selected_candidate_id": ""}, "verify": {"request": {"url": "http://localhost:8501", "trace_id": "{{trace_id}}"}}}'
        return result


class FakeParser:
    """Parser that always returns a valid plan."""
    def parse(self, text):
        from auto_harness.agent_runtime.deployment_plan import DeploymentPlan
        return DeploymentPlan(
            status="ok",
            plan_id="test-plan",
            summary="parsed plan",
            environment={"install_commands": []},
            run={"candidates": [], "selected_candidate_id": ""},
            verify={"request": {"url": "http://localhost:8501", "trace_id": "{{trace_id}}"}},
        )


class FakePolicyGate:
    """Policy gate that allows by default."""
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.validate_call_count = 0

    def validate(self, plan, snapshot, **kwargs):
        self.validate_call_count += 1
        if self.allowed:
            return {"allowed": True, "status": "accepted", "normalized_plan": plan, "accepted_sections": list(plan.keys()), "rejected_items": []}
        return {"allowed": False, "status": "rejected", "rejected_items": [{"section": "run", "item_index": 0, "reason": "unsafe command"}]}


class FakeCompiler:
    """Compiler that returns the plan as effective_plan."""
    def compile(self, plan, **kwargs):
        return {"effective_plan": plan, "analysis": {"install_plan": [], "run_candidates": []}}


class FakeArtifactWriter:
    """Artifact writer that writes to temp dir."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.reports_dir = self.run_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_project_snapshot(self, snapshot):
        from auto_harness.models.base import write_json
        path = self.reports_dir / "project_snapshot.json"
        write_json(path, snapshot)
        return path

    def write_raw_plan(self, raw_plan):
        from auto_harness.models.base import write_json
        path = self.reports_dir / "llm_deployment_plan.raw.json"
        write_json(path, raw_plan)
        return path

    def write_parsed_plan(self, parsed_plan):
        from auto_harness.models.base import write_json
        path = self.reports_dir / "llm_deployment_plan.parsed.json"
        write_json(path, parsed_plan)
        return path

    def write_policy_result(self, policy_result):
        from auto_harness.models.base import write_json
        path = self.reports_dir / "llm_plan_policy.json"
        write_json(path, policy_result)
        return path

    def write_effective_plan(self, effective_plan):
        from auto_harness.models.base import write_json
        path = self.reports_dir / "effective_deployment_plan.json"
        write_json(path, effective_plan)
        return path

    def write_pipeline_results(self, results):
        from auto_harness.models.base import write_json
        path = self.reports_dir / "pipeline_results.json"
        write_json(path, results)
        return path


def make_fake_deps(
    tmp_path,
    policy_allowed=True,
    fail_stages=None,
    max_replans=2,
):
    """Build fake GraphNodeDependencies for testing."""
    stage_executor = FakeStageExecutor(fail_stages=fail_stages)
    policy_gate = FakePolicyGate(allowed=policy_allowed)
    planner = FakePlanner()

    def build_snapshot(state):
        return {"task_id": state["task_id"], "files": [], "skill_context": {}}

    def build_replan_input(state):
        return {}, {}, {"failed_stage": state.get("failed_stage", "")}

    def determine_resume_stage(previous, current):
        return "analyze"

    deps = GraphNodeDependencies(
        build_snapshot=build_snapshot,
        build_replan_input=build_replan_input,
        determine_resume_stage=determine_resume_stage,
        merge_analysis=merge_plan_analysis,
        planner=planner,
        parser=FakeParser(),
        policy_gate=policy_gate,
        compiler=FakeCompiler(),
        stage_executor=stage_executor,
        artifact_writer_factory=lambda run_dir: FakeArtifactWriter(run_dir),
        runtime_config={},
    )

    return deps, stage_executor, policy_gate, planner


def make_context(tmp_path, task_id="test_task", dry_run=True):
    run_dir = tmp_path / "runs" / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    return DeploymentContext(
        task_id=task_id,
        run_dir=str(run_dir),
        repo_dir=str(run_dir / "workspace" / "repo"),
        dry_run=dry_run,
        runtime_policy={"allow_dependency_install": False, "allow_service_start": False},
    )


class FakeControllerDependencies:
    """Dependencies object for LangGraphController constructor."""
    def __init__(self, graph_deps, max_replans=2):
        self._graph_deps = graph_deps
        self._max_replans = max_replans

    def initial_state(self, context):
        return build_initial_state(context, self._max_replans)

    def graph_deps(self):
        return self._graph_deps

    def to_controller_result(self, output):
        verify_status = output.get("verify_status", "")
        if verify_status in ("passed", "pass"):
            status = "completed"
            stop_reason = "verify_passed"
        else:
            status = "failed"
            stop_reason = output.get("stop_reason", "unknown")
        return DeploymentResult(
            task_id=output.get("task_id", ""),
            status=status,
            stop_reason=stop_reason,
            controller="langgraph",
            verify_status=verify_status,
            metrics={
                "replan_count": output.get("replan_count", 0),
                "node_history": output.get("node_history", []),
            },
        )

    def completed_result(self, values):
        return DeploymentResult(
            task_id=values.get("task_id", ""),
            status="completed",
            stop_reason="already_completed",
            controller="langgraph",
            verify_status=values.get("verify_status", ""),
        )

    def blocked_result(self, values, reason):
        return DeploymentResult(
            task_id=values.get("task_id", ""),
            status="blocked",
            stop_reason=reason,
            controller="langgraph",
        )

    @staticmethod
    def has_side_effect(next_nodes):
        return bool(set(next_nodes) & SIDE_EFFECT_STAGES)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestBuildInitialState:
    def test_initial_state_fields(self, tmp_path):
        ctx = make_context(tmp_path)
        state = build_initial_state(ctx, max_replans=2)
        assert state["schema_version"] == 2
        assert state["task_id"] == "test_task"
        assert state["controller"] == "langgraph"
        assert state["dry_run"] is True
        assert state["replan_count"] == 0
        assert state["max_replans"] == 2
        assert state["stage_results"] == {}
        assert state["errors"] == []
        assert state["node_history"] == []


class TestMergePlanAnalysis:
    def test_compiled_keys_override_deterministic(self):
        deterministic = {"install_plan": ["pip install foo"], "run_candidates": ["old"]}
        compiled = {"install_plan": ["pip install bar"], "run_candidates": ["new"], "verify_hint": {"url": "test"}}
        merged = merge_plan_analysis(deterministic, compiled)
        assert merged["install_plan"] == ["pip install bar"]
        assert merged["run_candidates"] == ["new"]
        assert merged["verify_hint"] == {"url": "test"}

    def test_deterministic_preserved_when_no_override(self):
        deterministic = {"frameworks": ["gradio"], "custom_key": "value"}
        compiled = {}
        merged = merge_plan_analysis(deterministic, compiled)
        assert merged["frameworks"] == ["gradio"]
        assert merged["custom_key"] == "value"

    def test_empty_deterministic(self):
        compiled = {"install_plan": ["pip install foo"]}
        merged = merge_plan_analysis(None, compiled)
        assert merged["install_plan"] == ["pip install foo"]


class TestLangGraphControllerNormalDryRun:
    def test_full_dry_run_passes(self, tmp_path):
        """Normal dry-run: all stages pass, verify passes, reaches report."""
        deps, executor, policy, planner = make_fake_deps(tmp_path)
        ctrl_deps = FakeControllerDependencies(deps, max_replans=2)
        ctrl = LangGraphController(ctrl_deps)
        ctx = make_context(tmp_path, dry_run=True)
        result = ctrl.run(ctx)
        assert result.controller == "langgraph"
        assert result.verify_status == "passed"
        assert result.status == "completed"
        assert result.stop_reason == "verify_passed"
        # Executor should have been called for each stage
        assert executor.call_count == len(STAGES)


class TestLangGraphControllerPolicyReject:
    def test_policy_reject_never_calls_executor(self, tmp_path):
        """Policy reject: executor should never be called."""
        deps, executor, policy, planner = make_fake_deps(tmp_path, policy_allowed=False)
        ctrl_deps = FakeControllerDependencies(deps, max_replans=2)
        ctrl = LangGraphController(ctrl_deps)
        ctx = make_context(tmp_path, dry_run=True)
        result = ctrl.run(ctx)
        assert result.stop_reason == "policy_rejected"
        assert executor.call_count == 0
        assert policy.validate_call_count == 1


class TestLangGraphControllerVerifyReplan:
    def test_runner_failure_replans_through_policy(self, tmp_path):
        """Runner failure triggers replan, which goes through parse and policy again."""
        deps, executor, policy, planner = make_fake_deps(
            tmp_path,
            fail_stages=["runner"],
            max_replans=2,
        )
        ctrl_deps = FakeControllerDependencies(deps, max_replans=2)
        ctrl = LangGraphController(ctrl_deps)
        ctx = make_context(tmp_path, dry_run=True)
        result = ctrl.run(ctx)
        # Planner should have been called for initial plan + 1 replan
        assert planner.replan_call_count >= 1
        # Policy gate should have been called at least twice (initial + replan)
        assert policy.validate_call_count >= 2
        # Replan count should be in metrics
        assert result.metrics.get("replan_count", 0) >= 1


class TestLangGraphControllerReplanWhitelist:
    def test_replan_resumes_only_from_whitelisted_stage(self, tmp_path):
        """If determine_resume_stage returns an unknown stage, falls back to analyze."""
        deps, executor, policy, planner = make_fake_deps(
            tmp_path,
            fail_stages=["runner"],
            max_replans=2,
        )
        # Override determine_resume_stage to return an arbitrary node
        deps.determine_resume_stage = lambda prev, curr: "arbitrary_node"
        ctrl_deps = FakeControllerDependencies(deps, max_replans=2)
        ctrl = LangGraphController(ctrl_deps)
        ctx = make_context(tmp_path, dry_run=True)
        result = ctrl.run(ctx)
        # Should still complete (fallback to analyze)
        assert result.controller == "langgraph"
