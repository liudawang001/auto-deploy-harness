"""LegacyController: wraps existing TaskRunner paths.

Only migrates the routing decision and result adaptation.
The actual execution logic stays in the original methods
(PlanFirstDeploymentLoop.run, DeploymentAgentLoop.run,
TaskRunner._run_existing_once) — they are injected as callables.
"""
from typing import Any, Callable, Dict, Optional

from auto_harness.controllers.base import DeploymentContext, DeploymentResult


class LegacyController:
    """Controller that preserves the existing TaskRunner routing logic.

    Does NOT copy the implementation of plan_first, agent_loop, or
    pipeline. Instead, it delegates to injected callables that point
    to the existing methods on TaskRunner.
    """

    name = "legacy"

    def __init__(
        self,
        config: Any,
        run_plan_first: Callable,
        run_agent_loop: Callable,
        run_pipeline: Callable,
        resume_existing: Callable,
        result_adapter: Callable,
    ) -> None:
        self.config = config
        self.run_plan_first = run_plan_first
        self.run_agent_loop = run_agent_loop
        self.run_pipeline = run_pipeline
        self.resume_existing = resume_existing
        self.result_adapter = result_adapter

    def run(self, context: DeploymentContext) -> DeploymentResult:
        if self.config.agent_plan_first:
            self.run_plan_first(context.task_id, dry_run=context.dry_run)
            strategy = "plan_first"
        elif (
            self.config.agent_enable_runtime_loop
            and self.config.agent_mode == "gated_actor"
            and self.config.agent_runtime_loop_position == "primary"
        ):
            self.run_agent_loop(context.task_id, dry_run=context.dry_run)
            strategy = "agent_loop"
        else:
            self.run_pipeline(context.task_id, dry_run=context.dry_run)
            strategy = "pipeline"
        return self.result_adapter(context, controller=self.name, strategy=strategy)

    def resume(self, context: DeploymentContext, resume_input: Optional[Dict[str, Any]] = None) -> DeploymentResult:
        self.resume_existing(
            context.task_id,
            dry_run=context.dry_run,
            resume_input=resume_input,
        )
        return self.result_adapter(context, controller=self.name, strategy="resume")
