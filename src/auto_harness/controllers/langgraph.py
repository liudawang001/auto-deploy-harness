"""LangGraphController: deployment controller using LangGraph StateGraph.

Implements the plan-first deployment flow as an explicit state graph
with nodes, conditional edges, and SQLite checkpointing.

The graph flow (final topology per Phase 8):
  snapshot → plan → parse → policy → compile → select_resume → [stages] → verify → report/stop
  Side-effect stages: env_solve → recover_env_deploy → env_deploy → recover_model_prepare → ...
  Failure: stage/verify failure → observe_failure → diagnose → repair_plan/replan
  Repair: repair_plan → repair_policy → approval(optional) → repair_apply → select_repair_resume
  Recovery: recover_* gates enforce journal lifecycle before side-effect execution
  Cleanup: approval → cleanup → select_repair_resume (only for approved cleanup)
"""
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.graph.checkpoint import SqliteCheckpointManager
from auto_harness.graph.nodes import DeploymentGraphNodes, make_stage_node, make_recovery_gate_node
from auto_harness.graph.repair_nodes import (
    repair_plan_node,
    repair_policy_node,
    repair_apply_node,
    select_repair_resume_node,
)
from auto_harness.graph.routes import (
    route_after_parse,
    route_after_policy,
    route_after_stage,
    route_after_verify,
    route_after_replan,
    route_after_llm_plan,
    route_after_diagnose,
    route_after_repair_policy,
    route_after_approval,
    route_repair_resume_stage,
    route_resume_stage,
)
from auto_harness.graph.state import DeploymentGraphState
from auto_harness.models.base import read_json, write_json


# Pipeline stages in execution order
STAGES = (
    "analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy",
    "model_prepare", "runner", "verify",
)


def build_initial_state(context: DeploymentContext, max_replans: int, config=None) -> Dict:
    """Build the initial graph state for a new deployment run.

    Only called by run(), never by resume().
    """
    return {
        "schema_version": 2,
        "task_id": context.task_id,
        "controller": "langgraph",
        "run_dir": str(context.run_dir),
        "repo_dir": str(context.repo_dir),
        "dry_run": bool(context.dry_run),
        "runtime_policy": dict(context.runtime_policy),
        "current_stage": "created",
        "stage_results": {},
        "verify_status": "",
        "verify_evidence_paths": [],
        "failed_stage": "",
        "resume_from_stage": "analyze",
        "plan_revision_paths": {},
        "replan_count": 0,
        "max_replans": int(max_replans),
        "stop_reason": "",
        "errors": [],
        "node_history": [],
        # Recovery fields (Phase 2)
        "operation_ids": {},
        "recovery_capabilities": {
            "download": False,
            "local_process": False,
            "docker_service": False,
            "dependency_install": False,
        },
        "recovery_events": [],
        "pending_approval": None,
        "approval_history": [],
        "approved_operation_id": "",
        "approved_action": "",
        # Agent failure reasoning
        "failure_context": {},
        "failure_signature": "",
        "same_failure_count": 0,
        "diagnosis": {},
        "diagnosis_path": "",
        "diagnose_count": 0,
        "max_diagnoses": getattr(config, "langgraph_max_diagnoses", 2) if config else 2,
        # Controlled repair
        "repair_plan": {},
        "repair_plan_path": "",
        "repair_policy_result": {},
        "repair_policy_path": "",
        "repair_apply_result": {},
        "repair_apply_path": "",
        "repair_overlay": {},
        "repair_count": 0,
        "max_repairs": getattr(config, "langgraph_max_repairs", 2) if config else 2,
        "repair_resume_stage": "",
        "repair_resume_executed": False,
        # Recovery
        "pending_operation": None,
        "pending_operation_id": "",
        "recovery_stage": "",
        "recovery_decision": "",
        "recovery_result": {},
        "recovery_skip_stage": False,
        # Approval routing
        "approval_kind": "",
        "approval_resume_target": "",
        # Agent audit
        "agent_call_count": 0,
        "agent_trace_paths": [],
        "llm_required": True,
        "llm_provider": "",
        "llm_error": "",
        # Memory/Skill fields (Task 8)
        "memory_hits": [],
        "selected_skills": {},
        "skill_contexts": {},
        "skill_route_paths": {},
        "verified_memory_path": "",
        "skill_outcome_paths": [],
    }


def build_graph(deps, checkpointer):
    """Build and compile the LangGraph StateGraph.

    Args:
        deps: GraphNodeDependencies instance with injected callables.
        checkpointer: LangGraph checkpointer (SqliteSaver or InMemorySaver for tests).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    from langgraph.graph import END, START, StateGraph

    nodes = DeploymentGraphNodes(deps)
    builder = StateGraph(DeploymentGraphState)

    # Add all nodes
    builder.add_node("snapshot", nodes.build_snapshot)
    builder.add_node("plan", nodes.plan)
    builder.add_node("parse", nodes.parse)
    builder.add_node("policy", nodes.policy)
    builder.add_node("compile", nodes.compile)
    builder.add_node("select_resume", nodes.select_resume)
    for stage in STAGES:
        builder.add_node(stage, make_stage_node(stage, deps))
    builder.add_node("replan", nodes.replan)
    builder.add_node("observe_failure", nodes.observe_failure)
    builder.add_node("diagnose", nodes.diagnose)

    # Repair nodes
    def _repair_plan(state):
        return repair_plan_node(state, deps)
    def _repair_policy(state):
        return repair_policy_node(state, deps)
    def _repair_apply(state):
        return repair_apply_node(state, deps, config=deps.runtime_config)
    def _select_repair_resume(state):
        return select_repair_resume_node(state, deps)

    builder.add_node("repair_plan", _repair_plan)
    builder.add_node("repair_policy", _repair_policy)
    builder.add_node("repair_apply", _repair_apply)
    builder.add_node("select_repair_resume", _select_repair_resume)

    # Recovery gate nodes for side-effect stages
    for se_stage in ("env_deploy", "model_prepare", "runner"):
        builder.add_node("recover_%s" % se_stage, make_recovery_gate_node(se_stage, deps))

    # Approval node (reuse existing approval_node)
    from auto_harness.graph.approval import approval_node
    builder.add_node("approval", approval_node)

    # Cleanup node (only reachable from approval approve for cleanup_then_retry)
    def _cleanup(state):
        """Cleanup node: execute approved cleanup of external resources.

        Only runs when approved_action is "cleanup_then_retry".
        Re-verifies ownership before cleanup via OwnedResourceCleanupExecutor.
        """
        from auto_harness.graph.approval import cleanup_node
        recovery_adapter = deps.recovery_adapter
        if not recovery_adapter:
            return {"stop_reason": "cleanup_no_recovery_adapter"}
        # Build a minimal recovery object for cleanup_node
        run_dir = Path(state["run_dir"])
        from auto_harness.recovery.journal import OperationJournal
        journal = OperationJournal(run_dir)
        recovery = type("RecoveryHandle", (), {
            "journal": journal,
            "reconcilers": getattr(recovery_adapter, "reconcilers", {}),
        })()
        # Use the real OwnedResourceCleanupExecutor from dependencies,
        # or construct one if dependencies don't provide it.
        cleanup_executor = getattr(deps, "cleanup_executor", None)
        if cleanup_executor is None:
            from auto_harness.recovery.cleanup import OwnedResourceCleanupExecutor
            cleanup_executor = OwnedResourceCleanupExecutor()
        return cleanup_node(state, recovery, cleanup_executor)

    builder.add_node("cleanup", _cleanup)

    # Skill routing nodes (Task 10)
    builder.add_node(
        "route_verify_skills",
        lambda state: nodes.route_skills(state, "verify"),
    )
    builder.add_node(
        "route_repair_skills",
        lambda state: nodes.route_skills(state, "repair"),
    )

    builder.add_node("report", nodes.report)
    builder.add_node("stop", nodes.stop)

    # Route function for recovery gate decisions
    def _route_after_recovery(state):
        """Route after recovery gate: execute stage, reuse result, approval, or stop."""
        if state.get("stop_reason"):
            return "stop"
        if state.get("pending_approval"):
            return "approval"
        decision = state.get("recovery_decision", "execute")
        if decision == "reuse":
            return "reuse"
        if decision in ("execute", "continue", "retry"):
            return "execute"
        return "stop"

    # ---- Add edges (final topology per Phase 8) ----

    builder.add_edge(START, "snapshot")
    builder.add_edge("snapshot", "plan")
    builder.add_conditional_edges("plan", route_after_llm_plan, {
        "parse": "parse",
        "stop": "stop",
    })
    builder.add_conditional_edges(
        "parse", route_after_parse,
        {"valid": "policy", "invalid": "stop"},
    )
    builder.add_conditional_edges(
        "policy", route_after_policy,
        {"compile": "compile", "stop": "stop"},
    )
    builder.add_edge("compile", "select_resume")
    builder.add_conditional_edges(
        "select_resume",
        route_resume_stage,
        {
            "analyze": "analyze",
            "resource_plan": "resource_plan",
            "host_preflight": "host_preflight",
            "env_solve": "env_solve",
            "env_deploy": "recover_env_deploy",
            "model_prepare": "recover_model_prepare",
            "runner": "recover_runner",
            "verify": "route_verify_skills",
        },
    )

    # Non-side-effect stages: analyze -> resource_plan -> host_preflight -> env_solve
    builder.add_conditional_edges("analyze", route_after_stage,
        {"continue": "resource_plan", "observe_failure": "observe_failure"})
    builder.add_conditional_edges("resource_plan", route_after_stage,
        {"continue": "host_preflight", "observe_failure": "observe_failure"})
    builder.add_conditional_edges("host_preflight", route_after_stage,
        {"continue": "env_solve", "observe_failure": "observe_failure"})

    # env_solve -> recover_env_deploy -> env_deploy
    builder.add_conditional_edges(
        "env_solve", route_after_stage,
        {"continue": "recover_env_deploy", "observe_failure": "observe_failure"},
    )
    builder.add_conditional_edges(
        "recover_env_deploy", _route_after_recovery,
        {"execute": "env_deploy", "reuse": "recover_model_prepare", "approval": "approval", "stop": "stop"},
    )
    builder.add_conditional_edges(
        "env_deploy", route_after_stage,
        {"continue": "recover_model_prepare", "observe_failure": "observe_failure"},
    )

    # env_deploy -> recover_model_prepare -> model_prepare
    builder.add_conditional_edges(
        "recover_model_prepare", _route_after_recovery,
        {"execute": "model_prepare", "reuse": "recover_runner", "approval": "approval", "stop": "stop"},
    )
    builder.add_conditional_edges(
        "model_prepare", route_after_stage,
        {"continue": "recover_runner", "observe_failure": "observe_failure"},
    )

    # model_prepare -> recover_runner -> runner
    builder.add_conditional_edges(
        "recover_runner", _route_after_recovery,
        {"execute": "runner", "reuse": "route_verify_skills", "approval": "approval", "stop": "stop"},
    )
    builder.add_conditional_edges(
        "runner", route_after_stage,
        {"continue": "route_verify_skills", "observe_failure": "observe_failure"},
    )

    # Skill routing before verify (Task 10)
    builder.add_edge("route_verify_skills", "verify")

    # Verify has special routing
    builder.add_conditional_edges(
        "verify", route_after_verify,
        {"report": "finalize_learning", "observe_failure": "observe_failure"},
    )
    # finalize_learning -> report (Task 11)
    builder.add_node("finalize_learning", nodes.finalize_learning)
    builder.add_edge("finalize_learning", "report")
    # Replan routes back to parse
    builder.add_conditional_edges(
        "replan", route_after_replan,
        {"parse": "parse", "stop": "stop"},
    )
    # Failure observation → diagnose
    builder.add_edge("observe_failure", "diagnose")
    builder.add_conditional_edges(
        "diagnose", route_after_diagnose,
        {"repair_plan": "route_repair_skills", "replan": "replan", "stop": "stop"},
    )
    # Skill routing before repair (Task 10)
    builder.add_edge("route_repair_skills", "repair_plan")
    # Repair pipeline
    builder.add_edge("repair_plan", "repair_policy")
    builder.add_conditional_edges(
        "repair_policy", route_after_repair_policy,
        {"apply": "repair_apply", "approval": "approval", "stop": "stop"},
    )
    builder.add_edge("repair_apply", "select_repair_resume")
    builder.add_conditional_edges(
        "select_repair_resume",
        route_repair_resume_stage,
        {
            "analyze": "analyze",
            "resource_plan": "resource_plan",
            "host_preflight": "host_preflight",
            "env_solve": "env_solve",
            "env_deploy": "recover_env_deploy",
            "model_prepare": "recover_model_prepare",
            "runner": "recover_runner",
            "verify": "route_verify_skills",
        },
    )
    # Approval routing (includes cleanup path)
    builder.add_conditional_edges(
        "approval", route_after_approval,
        {"repair_apply": "repair_apply", "cleanup": "cleanup", "retry": "select_repair_resume", "stop": "stop"},
    )
    # Cleanup routes back to select_repair_resume for retry
    builder.add_edge("cleanup", "select_repair_resume")
    # Terminal edges
    builder.add_edge("report", END)
    builder.add_edge("stop", END)

    return builder.compile(checkpointer=checkpointer)


# Stages that have external side effects (Docker, processes, file downloads)
SIDE_EFFECT_STAGES = frozenset({"env_deploy", "model_prepare", "runner"})

# Map from side-effect stage to recovery capability key
STAGE_TO_CAPABILITY = {
    "model_prepare": "download",
    "runner": "local_process",  # or "docker_service" — checked at runtime
    "env_deploy": "dependency_install",
}


def can_resume_stage(stage, capabilities, dry_run):
    """Check if a stage can be safely resumed.

    Non-side-effect stages can always resume.
    Side-effect stages can resume if:
    - dry_run is True, OR
    - The corresponding capability is enabled in the recovery map

    Per Phase 7 spec:
    - Recovery gate nodes (recover_*) are always safe to resume
    - Raw side-effect stages without recovery gate are treated as old checkpoints
    """
    if stage not in SIDE_EFFECT_STAGES:
        return True
    if dry_run:
        return True
    cap_key = STAGE_TO_CAPABILITY.get(stage)
    if cap_key and capabilities.get(cap_key):
        return True
    # For runner, also check docker_service capability
    if stage == "runner" and capabilities.get("docker_service"):
        return True
    return False


class LangGraphController:
    """Deployment controller backed by LangGraph StateGraph with checkpoint.

    Implements the plan-first deployment flow as an explicit state graph.
    Each node does one responsibility and returns a state delta.
    Routes are pure functions that determine the next node based on state.
    """

    name = "langgraph"

    def __init__(self, dependencies) -> None:
        """Initialize with graph dependencies.

        Args:
            dependencies: Object with initial_state(), to_controller_result(),
                completed_result(), blocked_result(), has_side_effect(),
                and graph_dependencies() methods.
        """
        self.dependencies = dependencies

    def _prepare_planner_selection(self, context, *, reuse_selection=False):
        """Resolve and freeze planner selection before provider construction."""
        from auto_harness.config import HarnessConfig
        from auto_harness.controllers.validation import resolve_planner_mode

        prebuilt = None
        configured = getattr(self.dependencies, "runtime_config", None)
        if configured is None:
            # Compatibility for the small dependency fakes used by the graph
            # contract tests. Production dependencies expose runtime_config.
            prebuilt = self.dependencies.graph_deps()
            configured = getattr(prebuilt, "runtime_config", None)

        requested = getattr(configured, "langgraph_planner_mode", "auto")
        if requested not in {"auto", "llm", "deterministic"}:
            requested = "auto"
        provider_name = getattr(configured, "agent_plan_first_provider", "mock")
        if not isinstance(provider_name, str) or not provider_name.strip():
            provider_name = "mock"
        require_llm = getattr(configured, "langgraph_require_llm", False)
        if not isinstance(require_llm, bool):
            require_llm = False

        reports_dir = Path(context.run_dir) / "reports"
        selection_path = reports_dir / "controller_selection.json"
        selection = None
        if reuse_selection and selection_path.exists():
            try:
                candidate = read_json(selection_path)
                if candidate.get("resolved_planner_mode") in {
                    "llm", "deterministic"
                }:
                    selection = candidate
            except (OSError, ValueError):
                selection = None

        if selection is None:
            resolved = resolve_planner_mode(
                requested_mode=requested,
                provider_name=provider_name,
                require_llm=require_llm,
            )
            selection = {
                "controller": "langgraph",
                "requested_planner_mode": requested,
                "resolved_planner_mode": resolved,
                "provider": provider_name,
                "dry_run": bool(context.dry_run),
                "reason": (
                    "explicit_mode"
                    if requested != "auto"
                    else "real_provider_configured"
                    if resolved == "llm" and not require_llm
                    else "llm_required"
                    if resolved == "llm"
                    else "mock_or_missing_provider"
                ),
            }
            write_json(selection_path, selection)
            runner = getattr(self.dependencies, "runner", None)
            store = getattr(runner, "store", None)
            if store is not None:
                store.events(context.task_id).append(
                    "controller", "controller_selected", selection
                )

        effective_config = configured
        if not hasattr(effective_config, "langgraph_allow_mock_in_execute"):
            effective_config = HarnessConfig(
                langgraph_planner_mode=selection["resolved_planner_mode"]
            )
        return effective_config, selection, prebuilt

    def _build_graph_dependencies(self, selection, prebuilt=None):
        """Construct provider-bearing graph dependencies after validation."""
        import inspect

        if prebuilt is not None:
            return prebuilt
        graph_deps_factory = self.dependencies.graph_deps
        parameters = inspect.signature(graph_deps_factory).parameters
        if "planner_mode" in parameters:
            return graph_deps_factory(
                planner_mode=selection["resolved_planner_mode"]
            )
        return graph_deps_factory()

    def run(self, context: DeploymentContext) -> DeploymentResult:
        """Run a new deployment from scratch using the StateGraph."""
        # Pre-run validation
        from auto_harness.controllers.validation import validate_controller_run

        config, selection, prebuilt = self._prepare_planner_selection(
            context
        )
        provider_name = selection["provider"]
        validation = validate_controller_run(
            controller="langgraph",
            dry_run=context.dry_run,
            provider_name=provider_name,
            config=config,
            planner_mode=selection["resolved_planner_mode"],
        )
        if not validation.allowed:
            reports_dir = Path(context.run_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            write_json(reports_dir / "controller_validation.json", {
                "allowed": False,
                "reason": validation.reason,
            })
            return DeploymentResult(
                task_id=context.task_id,
                status="stopped",
                stop_reason=validation.reason,
                controller="langgraph",
            )

        graph_dependencies = self._build_graph_dependencies(selection, prebuilt)
        initial_state = self.dependencies.initial_state(context)
        initial_state["llm_required"] = (
            selection["resolved_planner_mode"] == "llm"
        )
        initial_state["llm_provider"] = provider_name
        with SqliteCheckpointManager(Path(context.run_dir)) as checkpoint:
            graph = build_graph(graph_dependencies, checkpoint.saver)
            output = graph.invoke(initial_state, config=checkpoint.config(context.task_id))
        return self.dependencies.to_controller_result(output)

    def resume(self, context: DeploymentContext, resume_input: Optional[Dict[str, Any]] = None) -> DeploymentResult:
        """Resume a deployment from the latest checkpoint.

        Two resume kinds:
        1. Ordinary checkpoint resume: graph.invoke(None, config=config)
        2. Approval interrupt resume: graph.invoke(Command(resume=resume_input), config=config)

        Recovery gate nodes (recover_*) are always safe to resume since
        they only inspect/reconcile, never execute side effects.

        Per Phase 7 spec:
        - checkpoint next = recover_<stage> -> allow entering recovery node
        - checkpoint next = raw side-effect stage -> treat as old checkpoint,
          route to recover_<stage> or safe stop
        - recovery node capability unavailable -> manual/stop
        - resume must not switch controller
        - approval resolve must truly resume graph, not just update JSON

        Writes reports/resume_audit.json on every resume attempt.
        """
        from auto_harness.utils.time import utc_now_iso

        run_dir = Path(context.run_dir)
        audit = {
            "task_id": context.task_id,
            "controller": "langgraph",
            "checkpoint_next": [],
            "resume_kind": "",
            "operation_id": "",
            "reconcile_decision": "",
            "duplicate_execution_prevented": False,
            "resumed_at": utc_now_iso(),
            "final_stop_reason": "",
        }

        config, selection, prebuilt = self._prepare_planner_selection(
            context, reuse_selection=True
        )
        from auto_harness.controllers.validation import validate_controller_run
        validation = validate_controller_run(
            controller="langgraph",
            dry_run=context.dry_run,
            provider_name=selection["provider"],
            config=config,
            planner_mode=selection["resolved_planner_mode"],
        )
        if not validation.allowed:
            audit["final_stop_reason"] = validation.reason
            self._write_resume_audit(run_dir, audit)
            return DeploymentResult(
                task_id=context.task_id,
                status="stopped",
                stop_reason=validation.reason,
                controller="langgraph",
            )

        graph_dependencies = self._build_graph_dependencies(selection, prebuilt)
        with SqliteCheckpointManager(run_dir) as checkpoint:
            graph = build_graph(graph_dependencies, checkpoint.saver)
            config = checkpoint.config(context.task_id)
            snapshot = graph.get_state(config)

            if not snapshot.next:
                audit["final_stop_reason"] = "already_completed"
                self._write_resume_audit(run_dir, audit)
                return self.dependencies.completed_result(snapshot.values)

            audit["checkpoint_next"] = list(snapshot.next)

            # Check each next node against capability map
            # Recovery gate nodes (recover_*) are always resumable
            capabilities = snapshot.values.get("recovery_capabilities", {})
            blocked_stages = []
            for node in snapshot.next:
                # Recovery gate nodes are always safe to resume
                if node.startswith("recover_"):
                    continue
                # Raw side-effect stage without recovery gate = old checkpoint
                # Must route through recovery gate or safe stop
                if node in SIDE_EFFECT_STAGES:
                    blocked_stages.append(node)
                    continue
                if not can_resume_stage(node, capabilities, context.dry_run):
                    blocked_stages.append(node)

            if blocked_stages:
                audit["final_stop_reason"] = "external_recovery_not_ready"
                audit["blocked_stages"] = blocked_stages
                audit["recovery_capabilities"] = capabilities
                self._write_resume_audit(run_dir, audit)
                return self.dependencies.blocked_result(
                    snapshot.values,
                    "external_recovery_not_ready",
                )

            # Track recovery state for audit
            recovery_stage = snapshot.values.get("recovery_stage", "")
            recovery_decision = snapshot.values.get("recovery_decision", "")
            pending_op_id = snapshot.values.get("pending_operation_id", "")
            if recovery_stage:
                audit["operation_id"] = pending_op_id
                audit["reconcile_decision"] = recovery_decision
                if recovery_decision == "reuse":
                    audit["duplicate_execution_prevented"] = True

            # Determine resume kind
            if resume_input:
                # Approval interrupt resume — validate before resuming
                if not isinstance(resume_input, dict):
                    audit["final_stop_reason"] = "invalid_resume_input"
                    self._write_resume_audit(run_dir, audit)
                    return DeploymentResult(
                        task_id=context.task_id,
                        status="stopped",
                        stop_reason="invalid_resume_input",
                        controller="langgraph",
                    )
                from langgraph.types import Command
                output = graph.invoke(Command(resume=resume_input), config=config)
                audit["resume_kind"] = "approval"
            else:
                # Ordinary checkpoint resume
                output = graph.invoke(None, config=config)
                audit["resume_kind"] = "checkpoint"

        # Determine final stop reason from output
        result = self.dependencies.to_controller_result(output)
        audit["final_stop_reason"] = result.stop_reason or "resumed"
        self._write_resume_audit(run_dir, audit)
        return result

    @staticmethod
    def _write_resume_audit(run_dir: Path, audit: Dict) -> None:
        """Write resume audit JSON to reports/resume_audit.json."""
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        from auto_harness.models.base import write_json
        write_json(reports_dir / "resume_audit.json", audit)
