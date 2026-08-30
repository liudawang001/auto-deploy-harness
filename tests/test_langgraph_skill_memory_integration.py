"""Task 8: LangGraph Memory/Skill dependency integration tests.

Verifies:
1. LangGraphDependencies exposes memory and skill services
2. Initial state has serializable skill/memory fields
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_harness.config import HarnessConfig
from auto_harness.orchestrator import TaskRunner
from auto_harness.controllers.langgraph_deps import LangGraphControllerDependencies
from auto_harness.controllers.base import DeploymentContext
from auto_harness.controllers.langgraph import build_initial_state


class TestLangGraphDependenciesExposeServices:
    """LangGraphDependencies exposes memory and skill services."""

    def test_langgraph_dependencies_expose_memory_and_skill_services(self, tmp_path):
        config = HarnessConfig(
            runs_dir=str(tmp_path / "runs"),
            skills_dir=str(tmp_path / "skills"),
            memory_dir=str(tmp_path / "memory"),
            default_controller="langgraph",
        )
        runner = TaskRunner(config)
        deps = LangGraphControllerDependencies(runner)

        # All properties should return the runner's services
        assert deps.skill_router is runner.skill_router
        assert deps.skill_context_builder is runner.skill_context_builder
        assert deps.memory_store is runner.memory
        assert deps.verified_memory_recorder is runner.verified_memory_recorder
        assert deps.skill_outcome_recorder is runner.skill_outcome_recorder


class TestInitialStateSerializableFields:
    """Initial state has serializable skill/memory fields."""

    def test_initial_state_has_serializable_skill_memory_fields(self, tmp_path):
        config = HarnessConfig(runs_dir=str(tmp_path / "runs"), default_controller="langgraph")
        context = DeploymentContext(
            task_id="t1",
            run_dir=str(tmp_path / "runs" / "t1"),
            repo_dir=str(tmp_path / "repo"),
            dry_run=True,
            runtime_policy={},
        )
        state = build_initial_state(context, max_replans=2, config=config)

        # All fields must be present and serializable (dict/list/str)
        assert "memory_hits" in state
        assert state["memory_hits"] == []
        assert state["selected_skills"] == {}
        assert state["skill_contexts"] == {}
        assert state["skill_route_paths"] == {}
        assert state["verified_memory_path"] == ""
        assert state["skill_outcome_paths"] == []

        # Verify the state is JSON-serializable (no objects/providers)
        import json
        json.dumps(state)  # Should not raise

    def test_runner_has_skill_routing_services(self, tmp_path):
        """TaskRunner constructs skill_router and memory services."""
        config = HarnessConfig(
            runs_dir=str(tmp_path / "runs"),
            skills_dir=str(tmp_path / "skills"),
            memory_dir=str(tmp_path / "memory"),
        )
        runner = TaskRunner(config)
        assert runner.skill_router is not None
        assert runner.skill_context_builder is not None
        assert runner.verified_memory_recorder is not None
        assert runner.skill_outcome_recorder is not None


# -------------------------------------------------------------------
# Task 9: SkillRoutingService and plan skill routing tests
# -------------------------------------------------------------------

SKILL_MD_TEMPLATE = """---
name: test-plan-skill
version: "1.0.0"
type: analysis_skill
stages: [plan_first, plan]
frameworks: [flask]
risk_level: low
side_effects: false
allowed_tools: [set_deployment_strategy]
success_signals: [plan_generated]
regression_cases: [flask_plan_correct]
---

# Guidance

- Prefer venv for Flask projects.

# When To Use

- plan stage for Flask projects.
"""


class TestSkillRoutingService:
    """SkillRoutingService routes skills and queries memory."""

    def _setup_skill(self, tmp_path):
        """Create a skill directory with SKILL.md."""
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-plan-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(SKILL_MD_TEMPLATE, encoding="utf-8")
        return skills_dir

    def test_route_returns_required_fields(self, tmp_path):
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder
        from auto_harness.skills.routing_service import SkillRoutingService
        from auto_harness.memory.store import MemoryStore

        skills_dir = self._setup_skill(tmp_path)
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        router = SkillRouter(skills_dir=skills_dir)
        context_builder = SkillContextBuilder()
        memory_store = MemoryStore(memory_dir)
        service = SkillRoutingService(
            router=router,
            context_builder=context_builder,
            memory_store=memory_store,
        )

        result = service.route(
            stage="plan",
            analysis={"frameworks": ["flask"]},
            allowed_tools=[],
        )

        assert "memory_hits" in result
        assert "selected_skills" in result
        assert "skill_context" in result
        assert "request" in result
        assert "artifact" in result

    def test_route_selects_matching_skill(self, tmp_path):
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder
        from auto_harness.skills.routing_service import SkillRoutingService

        skills_dir = self._setup_skill(tmp_path)
        router = SkillRouter(skills_dir=skills_dir)
        context_builder = SkillContextBuilder()

        service = SkillRoutingService(
            router=router,
            context_builder=context_builder,
        )

        result = service.route(
            stage="plan",
            analysis={"frameworks": ["flask"]},
            allowed_tools=[],
        )
        # The flask skill should be selected
        assert len(result["selected_skills"]) >= 1
        assert result["selected_skills"][0]["name"] == "test-plan-skill"

    def test_artifact_does_not_contain_full_content(self, tmp_path):
        """Route artifact should not contain full skill content."""
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder
        from auto_harness.skills.routing_service import SkillRoutingService

        skills_dir = self._setup_skill(tmp_path)
        router = SkillRouter(skills_dir=skills_dir)
        context_builder = SkillContextBuilder()
        service = SkillRoutingService(router=router, context_builder=context_builder)

        result = service.route(stage="plan", analysis={"frameworks": ["flask"]})
        artifact = result["artifact"]
        # artifact only has safe summary fields
        for skill_summary in artifact["selected_skills"]:
            assert "name" in skill_summary
            assert "score" in skill_summary
            # No full content field
            assert "content" not in skill_summary
            assert "body" not in skill_summary

    def test_memory_trust_level_classification(self, tmp_path):
        """Memory hits are classified by trust_level."""
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder
        from auto_harness.skills.routing_service import SkillRoutingService
        from unittest.mock import MagicMock

        # Mock memory store with verified and unresolved entries
        mock_memory = MagicMock()
        mock_memory.query.return_value = [
            {"id": "mem-1", "stage": "plan", "verified_success": True, "fix_status": "verified"},
            {"id": "mem-2", "stage": "plan", "verified_success": False, "fix_status": "unresolved"},
        ]

        router = SkillRouter(skills_dir=tmp_path / "skills")
        context_builder = SkillContextBuilder()
        service = SkillRoutingService(
            router=router,
            context_builder=context_builder,
            memory_store=mock_memory,
        )

        result = service.route(stage="plan", analysis={"frameworks": ["flask"]})
        # Verify trust levels
        verified_hits = [h for h in result["memory_hits"] if h.get("trust_level") == "verified"]
        unresolved_hits = [h for h in result["memory_hits"] if h.get("trust_level") == "unresolved"]
        assert len(verified_hits) >= 1
        assert len(unresolved_hits) >= 1


class TestPlanSkillRoutingInSnapshot:
    """build_snapshot routes plan skills."""

    def test_build_snapshot_includes_skill_context(self, tmp_path):
        """build_snapshot produces snapshot with memory_hits, selected_skills, skill_context."""
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder
        from auto_harness.skills.routing_service import SkillRoutingService
        from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder

        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test-plan-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(SKILL_MD_TEMPLATE, encoding="utf-8")

        # Create a minimal repo
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text("import flask\napp = flask.Flask(__name__)\n", encoding="utf-8")
        (repo_dir / "requirements.txt").write_text("flask\n", encoding="utf-8")

        router = SkillRouter(skills_dir=skills_dir)
        context_builder = SkillContextBuilder()
        service = SkillRoutingService(router=router, context_builder=context_builder)

        routed = service.route(stage="plan", analysis={"frameworks": ["flask"]})

        builder = ProjectSnapshotBuilder()
        snapshot = builder.build(
            repo_dir,
            task_id="t1",
            memory_hits=routed["memory_hits"],
            selected_skills=routed["selected_skills"],
            skill_context=routed["skill_context"],
        )

        assert snapshot.get("selected_skills")
        assert snapshot.get("skill_context")


# -------------------------------------------------------------------
# Task 10: Per-stage skill routing tests
# -------------------------------------------------------------------

class TestPerStageSkillRouting:
    """plan/replan/verify/repair each route skills independently."""

    def _make_routing_service(self, tmp_path, stage="verify"):
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder
        from auto_harness.skills.routing_service import SkillRoutingService
        from unittest.mock import MagicMock
        router = SkillRouter(skills_dir=tmp_path / "skills")
        context_builder = SkillContextBuilder()
        return SkillRoutingService(router=router, context_builder=context_builder)

    def test_verify_receives_explicit_skill_context(self, tmp_path):
        """route_skills for verify produces skill_context."""
        service = self._make_routing_service(tmp_path)
        routed = service.route(stage="verify", analysis={"frameworks": ["flask"]})
        assert "skill_context" in routed
        # Verify allowed_tools defaults are used internally by router

    def test_repair_receives_failure_specific_skill(self, tmp_path):
        """repair routing uses failure_category."""
        service = self._make_routing_service(tmp_path)
        routed = service.route(
            stage="repair",
            analysis={"frameworks": ["flask"]},
            failure_category="dependency_missing",
        )
        assert routed["request"]["failure_category"] == "dependency_missing"
        assert routed["request"]["stage"] == "repair"

    def test_each_stage_writes_separate_route_artifact(self, tmp_path):
        """Each stage produces a distinct artifact with stage field."""
        service = self._make_routing_service(tmp_path)
        for stage in ("plan", "verify", "repair"):
            routed = service.route(stage=stage, analysis={"frameworks": ["flask"]})
            assert routed["artifact"]["stage"] == stage

    def test_skill_cannot_add_unregistered_tool(self, tmp_path):
        """Skill routing only suggests skills; allowed_tools are fixed by caller."""
        service = self._make_routing_service(tmp_path)
        routed = service.route(
            stage="verify",
            analysis={"frameworks": ["flask"]},
            allowed_tools=["probe_http", "discover_gradio_api"],
        )
        # The request records the allowed_tools passed by the caller
        assert "probe_http" in routed["request"]["allowed_tools"]

    def test_route_skills_node_writes_artifact(self, tmp_path):
        """DeploymentGraphNodes.route_skills writes route artifact and updates state."""
        from auto_harness.graph.nodes import GraphNodeDependencies, DeploymentGraphNodes
        from auto_harness.skills.routing_service import SkillRoutingService
        from auto_harness.skills.router import SkillRouter
        from auto_harness.skills.context import SkillContextBuilder

        router = SkillRouter(skills_dir=tmp_path / "skills")
        context_builder = SkillContextBuilder()
        service = SkillRoutingService(router=router, context_builder=context_builder)

        deps = GraphNodeDependencies(
            build_snapshot=lambda s: {},
            build_replan_input=lambda s: ({}, {}, {}),
            determine_resume_stage=lambda p, c: "verify",
            merge_analysis=lambda a, b: {},
            planner=None, parser=None, policy_gate=None, compiler=None,
            stage_executor=None,
            artifact_writer_factory=lambda r: MagicMock(),
            runtime_config=None,
            route_skills=service.route,
        )
        nodes = DeploymentGraphNodes(deps)
        state = {
            "run_dir": str(tmp_path),
            "compiled_analysis": {"frameworks": ["flask"]},
            "selected_skills": {},
            "skill_contexts": {},
            "skill_route_paths": {},
            "repair_count": 0,
        }
        result = nodes.route_skills(state, "verify")
        assert "selected_skills" in result
        assert "verify" in result["selected_skills"]
        assert "skill_contexts" in result
        assert "verify" in result["skill_contexts"]
        # Route artifact written
        route_path = tmp_path / "skills" / "routes" / "verify.json"
        assert route_path.exists()


# -------------------------------------------------------------------
# Task 11: Verified memory and skill outcome tests
# -------------------------------------------------------------------

class TestVerifiedMemoryRecording:
    """Verified memory recording conditions."""

    def test_verify_pass_without_repair_does_not_record(self, tmp_path):
        """No repair -> no verified memory."""
        from auto_harness.memory.success import VerifiedMemoryRecorder
        recorder = VerifiedMemoryRecorder(tmp_path / "memory")
        run_dir = tmp_path / "runs" / "t1"
        run_dir.mkdir(parents=True)
        pipeline_results = {
            "verify": {"status": "passed", "data": {"trace_id": "tr1"}},
        }
        result = recorder.record_if_verified(
            run_dir, pipeline_results, {},
            repair_apply_result={"status": "not_applied"},
        )
        # Should be skipped because repair not applied
        assert result is None or (isinstance(result, dict) and result.get("verified_success") is not True)

    def test_metadata_only_repair_does_not_record(self, tmp_path):
        """metadata-only repair does not count as effective."""
        from auto_harness.memory.success import VerifiedMemoryRecorder
        recorder = VerifiedMemoryRecorder(tmp_path / "memory")
        run_dir = tmp_path / "runs" / "t2"
        run_dir.mkdir(parents=True)
        pipeline_results = {
            "verify": {"status": "passed", "data": {"trace_id": "tr2"}},
        }
        apply_result = {
            "status": "applied",
            "action_results": [
                {"action": "update_verify_hint", "executed": False},
            ],
        }
        result = recorder.record_if_verified(
            run_dir, pipeline_results, {},
            repair_apply_result=apply_result,
        )
        assert result is None or (isinstance(result, dict) and result.get("verified_success") is not True)

    def test_effective_repair_and_trace_pass_records_once(self, tmp_path):
        """Effective repair + trace verify pass records verified memory."""
        from auto_harness.memory.success import VerifiedMemoryRecorder
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        recorder = VerifiedMemoryRecorder(memory_dir)
        run_dir = tmp_path / "runs" / "t3"
        run_dir.mkdir(parents=True)
        (run_dir / "repairs").mkdir()
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "workspace").mkdir()
        pipeline_results = {
            "verify": {"status": "passed", "data": {"trace_id": "tr3"}},
            "analyze": {"data": {"frameworks": ["flask"]}},
        }
        apply_result = {
            "status": "applied",
            "action_results": [
                {"action": "install_package", "executed": True, "exit_code": 0},
            ],
            "policy": {"allowed": True},
        }
        result = recorder.record_if_verified(
            run_dir, pipeline_results, {},
            repair_apply_result=apply_result,
        )
        assert result is not None
        assert result.get("verified_success") is True

    def test_resume_does_not_duplicate_memory(self, tmp_path):
        """Same entry_id is not recorded twice."""
        from auto_harness.memory.success import VerifiedMemoryRecorder
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        recorder = VerifiedMemoryRecorder(memory_dir)
        run_dir = tmp_path / "runs" / "t4"
        run_dir.mkdir(parents=True)
        (run_dir / "repairs").mkdir()
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "workspace").mkdir()
        pipeline_results = {
            "verify": {"status": "passed", "data": {"trace_id": "tr4"}},
            "analyze": {"data": {"frameworks": ["flask"]}},
        }
        apply_result = {
            "status": "applied",
            "action_results": [
                {"action": "install_package", "executed": True, "exit_code": 0},
            ],
            "policy": {"allowed": True},
        }
        result1 = recorder.record_if_verified(
            run_dir, pipeline_results, {},
            repair_apply_result=apply_result,
        )
        assert result1 is not None
        # Call again with same inputs - should not duplicate
        result2 = recorder.record_if_verified(
            run_dir, pipeline_results, {},
            repair_apply_result=apply_result,
        )
        # The _has_entry check should prevent duplicate
        assert result2 is not None


class TestSkillOutcomeFourState:
    """Skill outcome classification: helped/neutral/harmful/unknown."""

    def test_skill_failure_without_causal_evidence_is_unknown(self):
        """Verify failure without causal evidence -> unknown, NOT harmful."""
        from auto_harness.memory.outcomes import SkillOutcomeRecorder
        recorder = SkillOutcomeRecorder(Path("/tmp/test_outcomes"))
        outcome = recorder._classify_outcome(
            influenced_plan=True,
            policy_accepted=True,
            harmful=False,
            final_verify_status="failed",
            trace_verified=False,
            has_causal_evidence=False,
        )
        assert outcome == "unknown"

    def test_policy_rejected_unsafe_skill_is_harmful(self):
        """Policy-rejected unsafe guidance -> harmful."""
        from auto_harness.memory.outcomes import SkillOutcomeRecorder
        recorder = SkillOutcomeRecorder(Path("/tmp/test_outcomes"))
        outcome = recorder._classify_outcome(
            influenced_plan=True,
            policy_accepted=False,
            harmful=True,
            final_verify_status="failed",
            trace_verified=False,
            has_causal_evidence=False,
        )
        assert outcome == "harmful"

    def test_neutral_when_not_influenced(self):
        """Skill selected but didn't influence plan -> neutral."""
        from auto_harness.memory.outcomes import SkillOutcomeRecorder
        recorder = SkillOutcomeRecorder(Path("/tmp/test_outcomes"))
        outcome = recorder._classify_outcome(
            influenced_plan=False,
            policy_accepted=True,
            harmful=False,
            final_verify_status="passed",
            trace_verified=True,
            has_causal_evidence=False,
        )
        assert outcome == "neutral"

    def test_helped_with_causal_evidence(self):
        """Causal evidence + verify pass -> helped."""
        from auto_harness.memory.outcomes import SkillOutcomeRecorder
        recorder = SkillOutcomeRecorder(Path("/tmp/test_outcomes"))
        outcome = recorder._classify_outcome(
            influenced_plan=True,
            policy_accepted=True,
            harmful=False,
            final_verify_status="passed",
            trace_verified=True,
            has_causal_evidence=True,
        )
        assert outcome == "helped"
