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


def _build_replan_failure(state: Dict, failed_stage: str, failed_result: Dict) -> Dict:
    """Preserve the sanitized failure evidence collected before diagnosis."""
    observed = state.get("failure_context")
    failure = dict(observed) if isinstance(observed, dict) else {}
    failure.setdefault("failed_stage", failed_stage)
    failure.setdefault("stage_status", failed_result.get("status", ""))
    failure.setdefault("summary", failed_result.get("summary", ""))
    failure.setdefault("error", str(failed_result.get("error") or "")[:2000])
    diagnosis = state.get("diagnosis")
    if isinstance(diagnosis, dict):
        failure["diagnosis"] = {
            "status": diagnosis.get("status", ""),
            "summary": diagnosis.get("summary", ""),
            "diagnosis": diagnosis.get("diagnosis", {}),
            "rerun_from": diagnosis.get("rerun_from", ""),
            "rerun_reason": diagnosis.get("rerun_reason", ""),
        }
    return failure


class LangGraphControllerDependencies:
    """Dependencies object for LangGraphController.

    Wires real TaskRunner modules to the graph node interface.
    """

    def __init__(self, runner) -> None:
        self.runner = runner

    @property
    def runtime_config(self):
        """Expose configuration without constructing provider dependencies."""
        return self.runner.config

    @property
    def skill_router(self):
        return self.runner.skill_router

    @property
    def skill_context_builder(self):
        return self.runner.skill_context_builder

    @property
    def memory_store(self):
        return self.runner.memory

    @property
    def verified_memory_recorder(self):
        return self.runner.verified_memory_recorder

    @property
    def skill_outcome_recorder(self):
        return self.runner.skill_outcome_recorder

    @property
    def skill_routing_service(self):
        return self.runner.skill_routing_service

    def initial_state(self, context: DeploymentContext) -> Dict:
        """Build initial graph state for a new deployment."""
        max_replans = getattr(self.runner.config, "agent_plan_first_max_replans", 2)
        return build_initial_state(context, max_replans, config=self.runner.config)

    def graph_deps(self, planner_mode=None) -> GraphNodeDependencies:
        """Build GraphNodeDependencies with real modules."""
        from auto_harness.agent_runtime.deployment_plan import DeploymentPlanParser
        from auto_harness.agent_runtime.plan_compiler import PlanCompiler
        from auto_harness.agent_runtime.plan_policy import PlanPolicyGate
        from auto_harness.agent_runtime.plan_artifacts import PlanArtifactWriter
        from auto_harness.agent_runtime.plan_first_loop import LLMDeploymentPlanner
        from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
        from auto_harness.agent_runtime.stage_executor import AgentStageExecutor

        # Support deterministic planner mode for baseline evaluation
        selected_planner = planner_mode or getattr(
            self.runner.config, "langgraph_planner_mode", "auto"
        )
        if selected_planner == "deterministic":
            from auto_harness.agent_runtime.deterministic_planner import DeterministicDeploymentPlanner
            planner = DeterministicDeploymentPlanner()
        else:
            provider = self.runner._create_plan_first_provider()
            planner = LLMDeploymentPlanner(provider, config=self.runner.config)
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
            # Document B: managed vLLM runtime controller (only when enabled).
            model_runtime_controller=self._model_runtime_controller(),
        )

        def build_snapshot(state):
            snapshot_builder = ProjectSnapshotBuilder(
                max_files=getattr(self.runner.config, "agent_plan_first_max_files", 80),
                max_file_chars=getattr(self.runner.config, "agent_plan_first_max_file_chars", 6000),
                max_tree_entries=getattr(self.runner.config, "agent_repo_tree_max_entries", 5000),
                context_mode=getattr(self.runner.config, "agent_repo_context_mode", "layered"),
                core_budget_tokens=getattr(self.runner.config, "agent_repo_core_budget_tokens", 12000),
            )
            # Build first so routing uses signals from the actual repository.
            snapshot = snapshot_builder.build(
                Path(state["repo_dir"]),
                task_id=state["task_id"],
            )
            analysis = snapshot.get("detected_signals", {})

            routed = self.runner.skill_routing_service.route(
                stage="plan",
                analysis=analysis,
                allowed_tools=[],
            )

            snapshot["memory_hits"] = routed["memory_hits"]
            snapshot["selected_skills"] = routed["selected_skills"]
            snapshot["skill_context"] = routed["skill_context"]
            snapshot["_skill_route"] = routed["request"]

            # Write route artifact
            route_dir = Path(state["run_dir"]) / "skills" / "routes"
            route_dir.mkdir(parents=True, exist_ok=True)
            from auto_harness.models.base import write_json
            write_json(route_dir / "plan.json", routed["artifact"])

            return snapshot

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
            failure = _build_replan_failure(state, failed_stage, failed_result)
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
        from auto_harness.recovery.faults import FaultInjector
        reconcilers = self._build_reconcilers()
        recovery_adapter = GraphRecoveryAdapter(reconcilers=reconcilers)
        fault_injector = FaultInjector(
            getattr(self.runner.config, "langgraph_fault_injection_points", [])
        )

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
            # Task 10: skill routing
            route_skills=self.runner.skill_routing_service.route,
            fault_injector=fault_injector,
        )

    def _failure_observer(self):
        """Phase 3: FailureObserver for deterministic failure extraction."""
        from auto_harness.graph.failure import FailureObserver
        return FailureObserver()

    def _verify_planner_factory(self):
        """Phase 5: factory creating AgentVerifyPlanner per call."""
        def factory():
            from auto_harness.agent import AgentVerifyPlanner, AgentTraceWriter
            try:
                provider = self.runner._create_plan_first_provider()
            except Exception:
                # Deterministic verification must remain usable when a task is
                # resumed after its intentionally ephemeral provider secret is
                # gone.  The planner is only consulted if deterministic trace
                # probes are uncertain.
                return None
            return AgentVerifyPlanner(
                provider,
                config=self.runner.config,
                trace_writer=AgentTraceWriter(Path("/tmp/_verify_traces")),
            )
        return factory

    def _agent_verify_config_factory(self):
        """Phase 5: factory creating agent verify config dict."""
        def factory():
            config = self.runner.config
            try:
                provider = self.runner._create_plan_first_provider()
            except Exception:
                provider = None
            return {
                "agent_mode": "gated_actor",
                "agent_enable_verify": getattr(config, "langgraph_enable_agent_verify", True),
                "agent_verify_max_steps": getattr(config, "agent_verify_max_steps", 5),
                "agent_allowed_hosts": getattr(config, "agent_allowed_hosts", ["localhost", "127.0.0.1"]),
                "provider": provider,
                "agent_context_mode": getattr(config, "agent_context_mode", "enforce"),
                "agent_context_window_tokens": getattr(config, "agent_context_window_tokens", None),
                "agent_context_reserved_output_tokens": getattr(config, "agent_context_reserved_output_tokens", 4096),
                "agent_context_safety_margin_tokens": getattr(config, "agent_context_safety_margin_tokens", 2048),
                "agent_context_unknown_model_fallback_tokens": getattr(config, "agent_context_unknown_model_fallback_tokens", 8192),
                "agent_context_max_overflow_retries": getattr(config, "agent_context_max_overflow_retries", 1),
                "agent_context_skill_budget_tokens": getattr(config, "agent_context_skill_budget_tokens", 2000),
                "agent_context_memory_budget_tokens": getattr(config, "agent_context_memory_budget_tokens", 2000),
            }
        return factory

    def _model_runtime_controller(self):
        """Document B: build a managed vLLM controller only when enabled."""
        if not getattr(self.runner.config, "model_inference_enabled", False):
            return None
        from auto_harness.model_runtime.controller import ModelRuntimeController

        return ModelRuntimeController()

    def _build_reconcilers(self):
        """Phase 6: build available reconcilers based on config."""
        reconcilers = {}
        config = self.runner.config
        try:
            if getattr(config, "langgraph_enable_recovery", True):
                import socket
                import subprocess
                from auto_harness.recovery.download import DownloadReconciler
                from auto_harness.recovery.process import ProcessProbe, ProcessReconciler
                from auto_harness.recovery.docker import DockerReconciler
                from auto_harness.recovery.dependency import DependencyReconciler

                def port_probe(host, port):
                    try:
                        with socket.create_connection((host, int(port)), timeout=1):
                            return True
                    except OSError:
                        return False

                def docker_command_runner(cmd):
                    try:
                        completed = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False,
                        )
                        return {
                            "exit_code": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                        }
                    except (OSError, subprocess.SubprocessError) as exc:
                        return {
                            "exit_code": 1,
                            "stdout": "",
                            "stderr": str(exc)[:500],
                        }

                reconcilers["model_download"] = DownloadReconciler()
                reconcilers["local_process"] = ProcessReconciler(
                    ProcessProbe(),
                    port_probe,
                )
                reconcilers["docker_service"] = DockerReconciler(
                    docker_command_runner,
                )
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
        elif output.get("dry_run") and output.get("current_stage") == "report":
            status = "completed_dry_run"
            stop_reason = "dry_run_completed"
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
        if not values or not values.get("task_id"):
            return DeploymentResult(
                task_id="",
                status="stopped",
                stop_reason="checkpoint_not_found",
                controller="langgraph",
            )
        verify_status = values.get("verify_status", "")
        is_dry_run = bool(values.get("dry_run"))
        verified = verify_status in ("passed", "pass")
        return DeploymentResult(
            task_id=values.get("task_id", ""),
            status=(
                "completed_dry_run"
                if is_dry_run and not verified
                else "completed" if verified else "stopped"
            ),
            stop_reason=(
                "dry_run_completed"
                if is_dry_run and not verified
                else "already_completed" if verified else "graph_ended_without_verify_pass"
            ),
            controller="langgraph",
            verify_status=verify_status,
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
