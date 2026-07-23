"""LangGraphController dependencies: wires real modules to graph nodes.

This module provides the LangGraphControllerDependencies class that
TaskRunner.graph_dependencies() returns. It builds GraphNodeDependencies
from real modules and provides the adapter methods that
LangGraphController needs (initial_state, to_controller_result, etc).
"""
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.controllers.base import DeploymentContext, DeploymentResult
from auto_harness.controllers.langgraph import (
    SIDE_EFFECT_STAGES,
    build_initial_state,
)
from auto_harness.graph.nodes import GraphNodeDependencies, merge_plan_analysis


class LangGraphControllerDependencies:
    """Dependencies object for LangGraphController.

    Wires real TaskRunner modules to the graph node interface.
    """

    def __init__(self, runner) -> None:
        self.runner = runner

    def initial_state(self, context: DeploymentContext) -> Dict:
        """Build initial graph state for a new deployment."""
        max_replans = getattr(self.runner.config, "agent_plan_first_max_replans", 2)
        return build_initial_state(context, max_replans, config=self.runner.config)

    def graph_deps(self) -> GraphNodeDependencies:
        """Build GraphNodeDependencies with real modules."""
        from auto_harness.agent_runtime.deployment_plan import DeploymentPlanParser
        from auto_harness.agent_runtime.plan_compiler import PlanCompiler
        from auto_harness.agent_runtime.plan_policy import PlanPolicyGate
        from auto_harness.agent_runtime.plan_artifacts import PlanArtifactWriter
        from auto_harness.agent_runtime.plan_first_loop import LLMDeploymentPlanner
        from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
        from auto_harness.agent_runtime.stage_executor import AgentStageExecutor

        provider = self.runner._create_plan_first_provider()
        planner = LLMDeploymentPlanner(provider)
        parser = DeploymentPlanParser()
        policy_gate = PlanPolicyGate()
        compiler = PlanCompiler()

        stage_executor = AgentStageExecutor(
            config=self.runner.config,
            store=self.runner.store,
            model_prepare=self.runner.model_prepare,
            repair_components={
                "planner": self.runner.repair_planner,
                "policy": self.runner.repair_policy,
                "applier": self.runner.repair_applier,
                "loop": self.runner.repair_loop,
                "overlay": self.runner.repair_overlay,
            },
            provider_factory=lambda: self.runner._create_plan_first_provider(),
            runtime_policy={},
            # Phase 5: Agent Verify integration factories
            verify_planner_factory=self._verify_planner_factory(),
            agent_verify_config_factory=self._agent_verify_config_factory(),
        )

        def build_snapshot(state):
            snapshot_builder = ProjectSnapshotBuilder(
                max_files=getattr(self.runner.config, "agent_plan_first_max_files", 80),
                max_file_chars=getattr(self.runner.config, "agent_plan_first_max_file_chars", 6000),
            )
            return snapshot_builder.build(
                Path(state["repo_dir"]),
                task_id=state["task_id"],
            )

        def build_replan_input(state):
            failed_stage = state.get("failed_stage", "")
            stage_results = state.get("stage_results", {})
            failed_result = stage_results.get(failed_stage, {})
            snapshot_path = state.get("snapshot_path", "")
            snapshot = {}
            if snapshot_path:
                from auto_harness.models.base import read_json
                try:
                    snapshot = read_json(Path(snapshot_path))
                except (OSError, ValueError):
                    pass
            previous_plan_path = state.get("parsed_plan_path", "")
            previous_plan = {}
            if previous_plan_path:
                from auto_harness.models.base import read_json
                try:
                    previous_plan = read_json(Path(previous_plan_path))
                except (OSError, ValueError):
                    pass
            failure = {
                "failed_stage": failed_stage,
                "stage_status": failed_result.get("status", ""),
                "summary": failed_result.get("summary", ""),
                "error": str(failed_result.get("error", ""))[:2000],
            }
            return snapshot, previous_plan, failure

        def determine_resume_stage(previous, current):
            # Reuse PlanFirstDeploymentLoop's logic
            old_env = previous.get("environment", {})
            new_env = current.get("environment", {})
            if old_env.get("install_commands") != new_env.get("install_commands"):
                return "env_deploy"
            old_assets = previous.get("model_assets", {})
            new_assets = current.get("model_assets", {})
            if old_assets != new_assets:
                return "model_prepare"
            old_run = previous.get("run", {})
            new_run = current.get("run", {})
            if old_run.get("candidates") != new_run.get("candidates"):
                return "runner"
            old_verify = previous.get("verify", {})
            new_verify = current.get("verify", {})
            if old_verify != new_verify:
                return "verify"
            return "runner"

        # Phase 3: diagnoser_factory creates a per-state AgentDiagnoser
        def diagnoser_factory(state):
            from auto_harness.agent import AgentDiagnoser, AgentTraceWriter
            trace_dir = Path(state["run_dir"]) / "logs" / "agent_calls"
            return AgentDiagnoser(
                provider=self.runner._create_plan_first_provider(),
                config=self.runner.config,
                trace_writer=AgentTraceWriter(trace_dir),
            )

        # Phase 6: recovery adapter with available reconcilers
        from auto_harness.recovery.graph_adapter import GraphRecoveryAdapter
        reconcilers = self._build_reconcilers()
        recovery_adapter = GraphRecoveryAdapter(reconcilers=reconcilers)

        return GraphNodeDependencies(
            build_snapshot=build_snapshot,
            build_replan_input=build_replan_input,
            determine_resume_stage=determine_resume_stage,
            merge_analysis=merge_plan_analysis,
            planner=planner,
            parser=parser,
            policy_gate=policy_gate,
            compiler=compiler,
            stage_executor=stage_executor,
            artifact_writer_factory=lambda run_dir: PlanArtifactWriter(run_dir),
            runtime_config=self.runner.config,
            # Phase 3 additions
            diagnoser_factory=diagnoser_factory,
            failure_observer=self._failure_observer(),
            # Phase 4 additions
            repair_planner=self.runner.repair_planner,
            repair_policy=self.runner.repair_policy,
            repair_applier=self.runner.repair_applier,
            repair_loop=self.runner.repair_loop,
            repair_overlay=self.runner.repair_overlay,
            # Phase 6 additions
            recovery_adapter=recovery_adapter,
            runtime_policy_factory=lambda state: state.get("runtime_policy", {}),
        )

    def _failure_observer(self):
        """Phase 3: FailureObserver for deterministic failure extraction."""
        from auto_harness.graph.failure import FailureObserver
        return FailureObserver()

    def _verify_planner_factory(self):
        """Phase 5: factory creating AgentVerifyPlanner per call."""
        def factory():
            from auto_harness.agent import AgentVerifyPlanner, AgentTraceWriter
            return AgentVerifyPlanner(
                self.runner._create_plan_first_provider(),
                config=self.runner.config,
                trace_writer=AgentTraceWriter(Path("/tmp/_verify_traces")),
            )
        return factory

    def _agent_verify_config_factory(self):
        """Phase 5: factory creating agent verify config dict."""
        def factory():
            config = self.runner.config
            return {
                "agent_mode": "gated_actor",
                "agent_enable_verify": getattr(config, "langgraph_enable_agent_verify", True),
                "agent_verify_max_steps": getattr(config, "agent_verify_max_steps", 5),
                "agent_allowed_hosts": getattr(config, "agent_allowed_hosts", ["localhost", "127.0.0.1"]),
                "provider": self.runner._create_plan_first_provider(),
            }
        return factory

    def _build_reconcilers(self):
        """Phase 6: build available reconcilers based on config."""
        reconcilers = {}
        config = self.runner.config
        try:
            if getattr(config, "langgraph_enable_recovery", True):
                from auto_harness.recovery.download import DownloadReconciler
                from auto_harness.recovery.process import ProcessReconciler
                from auto_harness.recovery.docker import DockerReconciler
                from auto_harness.recovery.dependency import DependencyReconciler
                reconcilers["model_download"] = DownloadReconciler()
                reconcilers["local_process"] = ProcessReconciler()
                reconcilers["docker_service"] = DockerReconciler()
                reconcilers["dependency_install"] = DependencyReconciler()
        except Exception:
            # Reconcilers may not be available in all environments
            pass
        return reconcilers

    def to_controller_result(self, output: Dict) -> DeploymentResult:
        """Convert graph output to DeploymentResult.

        Per Phase 10 spec:
        - verify pass/passed -> completed
        - pending approval -> interrupted
        - explicit stop_reason -> stopped
        - graph ended without verify pass -> stopped/failed
        """
        verify_status = output.get("verify_status", "")
        if verify_status in ("passed", "pass"):
            status = "completed"
            stop_reason = "verify_passed"
        elif output.get("pending_approval"):
            status = "interrupted"
            stop_reason = "approval_pending"
        elif output.get("stop_reason"):
            status = "stopped"
            stop_reason = output["stop_reason"]
        else:
            status = "stopped"
            stop_reason = "graph_ended_without_verify_pass"
        return DeploymentResult(
            task_id=output.get("task_id", ""),
            status=status,
            stop_reason=stop_reason,
            controller="langgraph",
            verify_status=verify_status,
            metrics={
                "replan_count": output.get("replan_count", 0),
                "node_history_count": len(output.get("node_history", [])),
            },
        )

    def completed_result(self, values: Dict) -> DeploymentResult:
        """Result when graph is already completed (no next nodes)."""
        return DeploymentResult(
            task_id=values.get("task_id", ""),
            status="completed",
            stop_reason="already_completed",
            controller="langgraph",
            verify_status=values.get("verify_status", ""),
        )

    def blocked_result(self, values: Dict, reason: str) -> DeploymentResult:
        """Result when resume is blocked due to side effects."""
        return DeploymentResult(
            task_id=values.get("task_id", ""),
            status="blocked",
            stop_reason=reason,
            controller="langgraph",
        )

    @staticmethod
    def has_side_effect(next_nodes) -> bool:
        """Check if any next node has external side effects."""
        return bool(set(next_nodes) & SIDE_EFFECT_STAGES)
