"""LangGraphController: deployment controller using LangGraph StateGraph.

Implements the plan-first deployment flow as an explicit state graph
with nodes, conditional edges, and SQLite checkpointing.

The graph flow:
  snapshot → plan → parse → policy → compile → select_resume → [stages] → verify → report/stop
  On failure: replan → parse → policy → compile → select_resume → ...
"""
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.graph.checkpoint import SqliteCheckpointManager
from auto_harness.graph.nodes import DeploymentGraphNodes, make_stage_node
from auto_harness.graph.routes import (
    route_after_parse,
    route_after_policy,
    route_after_stage,
    route_after_verify,
    route_after_replan,
    route_resume_stage,
)
from auto_harness.graph.state import DeploymentGraphState
from auto_harness.models.base import write_json


# Pipeline stages in execution order
STAGES = (
    "analyze", "resource_plan", "env_solve", "env_deploy",
    "model_prepare", "runner", "verify",
)


def build_initial_state(context: DeploymentContext, max_replans: int) -> Dict:
    """Build the initial graph state for a new deployment run.

    Only called by run(), never by resume().
    """
    return {
        "schema_version": 1,
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
    builder.add_node("report", nodes.report)
    builder.add_node("stop", nodes.stop)

    # Add edges
    builder.add_edge(START, "snapshot")
    builder.add_edge("snapshot", "plan")
    builder.add_edge("plan", "parse")
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
        {stage: stage for stage in STAGES},
    )
    # Stage-to-stage edges (except verify)
    for current, following in zip(STAGES[:-1], STAGES[1:]):
        if current != "verify":
            builder.add_conditional_edges(
                current,
                route_after_stage,
                {"continue": following, "replan": "replan", "stop": "stop"},
            )
    # Verify has special routing
    builder.add_conditional_edges(
        "verify", route_after_verify,
        {"report": "report", "replan": "replan", "stop": "stop"},
    )
    # Replan routes back to parse
    builder.add_conditional_edges(
        "replan", route_after_replan,
        {"parse": "parse", "stop": "stop"},
    )
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

    def run(self, context: DeploymentContext) -> DeploymentResult:
        """Run a new deployment from scratch using the StateGraph."""
        initial_state = self.dependencies.initial_state(context)
        with SqliteCheckpointManager(Path(context.run_dir)) as checkpoint:
            graph = build_graph(self.dependencies.graph_deps(), checkpoint.saver)
            output = graph.invoke(initial_state, config=checkpoint.config(context.task_id))
        return self.dependencies.to_controller_result(output)

    def resume(self, context: DeploymentContext, resume_input: Optional[Dict[str, Any]] = None) -> DeploymentResult:
        """Resume a deployment from the latest checkpoint.

        Uses the capability map to determine if a side-effect stage
        can be safely resumed. Supported resource types allow resume;
        unsupported types remain blocked (safe stop).

        Flow: load checkpoint → inspect next node → check capability
        → resume if allowed → VerifyModule is the only success judge

        Writes reports/resume_audit.json on every resume attempt.
        """
        from auto_harness.utils.time import utc_now_iso

        run_dir = Path(context.run_dir)
        audit = {
            "task_id": context.task_id,
            "dry_run": context.dry_run,
            "resumed_at": utc_now_iso(),
            "controller": "langgraph",
            "blocked": False,
            "stop_reason": "",
        }

        with SqliteCheckpointManager(run_dir) as checkpoint:
            graph = build_graph(self.dependencies.graph_deps(), checkpoint.saver)
            config = checkpoint.config(context.task_id)
            snapshot = graph.get_state(config)

            if not snapshot.next:
                audit["stop_reason"] = "already_completed"
                self._write_resume_audit(run_dir, audit)
                return self.dependencies.completed_result(snapshot.values)

            # Check each next node against capability map
            capabilities = snapshot.values.get("recovery_capabilities", {})
            blocked_stages = []
            for node in snapshot.next:
                if not can_resume_stage(node, capabilities, context.dry_run):
                    blocked_stages.append(node)

            if blocked_stages:
                audit["blocked"] = True
                audit["stop_reason"] = "external_recovery_not_ready"
                audit["next_nodes"] = list(snapshot.next)
                audit["blocked_stages"] = blocked_stages
                audit["recovery_capabilities"] = capabilities
                self._write_resume_audit(run_dir, audit)
                return self.dependencies.blocked_result(
                    snapshot.values,
                    "external_recovery_not_ready",
                )

            output = graph.invoke(None, config=config)

        audit["stop_reason"] = "resumed"
        self._write_resume_audit(run_dir, audit)
        return self.dependencies.to_controller_result(output)

    @staticmethod
    def _write_resume_audit(run_dir: Path, audit: Dict) -> None:
        """Write resume audit JSON to reports/resume_audit.json."""
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        from auto_harness.models.base import write_json
        write_json(reports_dir / "resume_audit.json", audit)
