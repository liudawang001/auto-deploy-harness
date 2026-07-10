"""DeploymentAgentLoop: the unified observe-plan-act-verify loop.

This is the core agent loop that orchestrates the full deployment process:
1. Load current pipeline state
2. Observe current stage
3. Ask LLM for next decision
4. Validate schema
5. Critic check
6. Policy check
7. Execute tool or apply state delta
8. Write artifacts
9. Decide next stage or stop

The deterministic pipeline is not deleted but serves as the controlled
execution layer that the agent can invoke.
"""
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent_runtime.artifacts import AgentArtifactWriter
from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
from auto_harness.agent_runtime.stage_schemas import (
    RUNNER_TOOLS, ENV_TOOLS, MODEL_TOOLS, REPAIR_TOOLS, PLAN_TOOLS,
    PIPELINE_STAGES, SAFE_RERUN_STAGES,
)
from auto_harness.agent_runtime.state import AgentState
from auto_harness.agent_runtime.stop import StopCondition
from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


# Stage -> allowed tools mapping
STAGE_TOOLS = {
    "analyze": PLAN_TOOLS,
    "env_solve": ENV_TOOLS,
    "model_prepare": MODEL_TOOLS,
    "runner": RUNNER_TOOLS,
    "verify": RUNNER_TOOLS + ENV_TOOLS + MODEL_TOOLS,
    "repair": REPAIR_TOOLS,
}


class DeploymentAgentLoop:
    """Unified agent loop for LLM-driven deployment.

    Runs the observe-plan-act-verify cycle across pipeline stages.
    Each iteration observes the current state, asks LLM for a decision,
    validates through schema/critic/policy, executes if allowed, and
    decides the next action.
    """

    def __init__(
        self,
        *,
        provider=None,
        config=None,
        max_iterations: int = 5,
        stop_on_verify_pass: bool = True,
    ) -> None:
        self.provider = provider
        self.config = config
        self.max_iterations = max_iterations
        self.stop_on_verify_pass = stop_on_verify_pass

    def run(
        self,
        *,
        task_id: str,
        run_dir: Path,
        repo_dir: Path,
        initial_results: Dict,
        dry_run: bool = True,
    ) -> Dict:
        """Run the agent loop.

        Args:
            task_id: Task identifier
            run_dir: Run directory for artifacts
            repo_dir: Repository directory
            initial_results: Results from deterministic pipeline
            dry_run: If True, don't execute real commands

        Returns:
            Dict with final status, agent contributions, and evidence
        """
        run_dir = Path(run_dir)
        repo_dir = Path(repo_dir)

        # Initialize state
        state = AgentState(
            task_id=task_id,
            run_dir=str(run_dir),
            mode="gated_actor" if self.provider else "planner",
            objective="Deploy AI demo project successfully",
            current_stage=self._determine_start_stage(initial_results),
            max_iterations=self.max_iterations,
        )

        # Initialize artifacts writer
        artifacts = AgentArtifactWriter(run_dir)

        # Initialize stop condition
        stop = StopCondition(
            max_iterations=self.max_iterations,
            stop_on_verify_pass=self.stop_on_verify_pass,
        )

        # Initialize decision gate
        gate = AgentDecisionGate(provider=self.provider)

        # Load existing plan if any
        state.plan = self._load_existing_plan(run_dir)

        # Main loop
        for iteration in range(self.max_iterations):
            state.iteration = iteration

            # 1. Observe current stage
            observation = self._observe_stage(state, initial_results, run_dir, repo_dir)
            state.record_observation(state.current_stage, observation)

            # 2. Determine allowed tools for current stage
            allowed_tools = list(STAGE_TOOLS.get(state.current_stage, ()))

            # 3. Run decision gate (LLM -> schema -> critic -> policy -> execute)
            gate_result = gate.decide(
                stage=state.current_stage,
                observation=observation,
                allowed_tools=allowed_tools,
                mode=state.mode,
                run_dir=run_dir,
            )

            # 4. Record decision
            decision_record = {
                "stage": state.current_stage,
                "decision_status": gate_result.decision_status,
                "hypothesis": gate_result.hypothesis,
                "tool_call": gate_result.tool_call,
                "policy_allowed": gate_result.policy.get("allowed", False),
                "executed": gate_result.execution.get("executed", False),
                "applied": gate_result.execution.get("applied", False),
                "llm_helped": gate_result.llm_helped,
            }
            state.record_decision(state.current_stage, decision_record)

            # 5. Record tool result if executed
            if gate_result.execution.get("executed") or gate_result.execution.get("applied"):
                state.record_tool_result(state.current_stage, gate_result.execution)

            # 6. Update stage status
            if gate_result.llm_helped:
                state.update_stage_status(state.current_stage, "improved")
            elif gate_result.decision_status == "no_action":
                state.update_stage_status(state.current_stage, "no_change")

            # 7. Write artifacts
            step_record = {
                "step_id": iteration + 1,
                "phase": "decide",
                "stage": state.current_stage,
                "decision_status": gate_result.decision_status,
                "policy_allowed": gate_result.policy.get("allowed", False),
                "executed": gate_result.execution.get("executed", False),
                "applied": gate_result.execution.get("applied", False),
                "llm_helped": gate_result.llm_helped,
                "recorded_at": utc_now_iso(),
            }
            artifacts.write_step(step_record)

            # 8. Check stop conditions
            policy_results = [gate_result.policy]
            should_stop, reason = stop.check(
                iteration=iteration + 1,
                verify_status=state.verify.get("status", ""),
                policy_results=policy_results,
                stage_status=state.stage_status,
                last_error=gate_result.error or "",
            )

            if should_stop:
                state.stop_reason = reason
                break

            # 9. Determine next stage
            state.current_stage = self._next_stage(state, gate_result)

        # Save final state
        artifacts.write_state(state.to_dict())

        # Build result
        return self._build_result(state, run_dir)

    def _determine_start_stage(self, initial_results: Dict) -> str:
        """Determine which stage to start the agent loop at."""
        # Find the first failed or uncertain stage
        for stage in PIPELINE_STAGES:
            result = initial_results.get(stage, {})
            status = result.get("status", "") if isinstance(result, dict) else ""
            if status in ("failed", "uncertain"):
                return stage
        # If all passed, start at verify for final confirmation
        return "verify"

    def _observe_stage(self, state: AgentState, initial_results: Dict,
                       run_dir: Path, repo_dir: Path) -> Dict:
        """Observe the current stage state."""
        stage = state.current_stage
        result = initial_results.get(stage, {})

        observation = {
            "stage": stage,
            "iteration": state.iteration,
            "previous_results": result,
            "repo_dir": str(repo_dir),
        }

        # Add stage-specific observation data
        if stage == "runner":
            observation["run_candidates"] = result.get("data", {}).get("run_candidates", [])
        elif stage == "env_solve":
            observation["env_solution"] = result.get("data", {}).get("env_solution", {})
            observation["constraints"] = self._load_constraints(run_dir)
        elif stage == "model_prepare":
            observation["model_assets"] = result.get("data", {}).get("model_assets", {})
        elif stage == "verify":
            observation["verify_status"] = result.get("status", "")
            observation["verify_evidence"] = result.get("data", {}).get("evidence", {})

        return observation

    def _load_constraints(self, run_dir: Path) -> List[str]:
        """Load constraints from repair overlay."""
        constraints_path = run_dir / "repair_overlay" / "constraints.txt"
        if constraints_path.exists():
            return [line.strip() for line in constraints_path.read_text().splitlines() if line.strip()]
        return []

    def _load_existing_plan(self, run_dir: Path) -> Dict:
        """Load existing deployment plan if available."""
        plan_path = run_dir / "agent_plan.json"
        if plan_path.exists():
            try:
                return read_json(plan_path)
            except (OSError, ValueError):
                return {}
        return {}

    def _next_stage(self, state: AgentState, gate_result) -> str:
        """Determine the next stage based on current state and gate result."""
        current = state.current_stage

        # If repair was applied, resume from the specified stage
        if gate_result.execution.get("resume_from_stage"):
            return gate_result.execution["resume_from_stage"]

        # Move to next stage in pipeline
        if current in PIPELINE_STAGES:
            idx = PIPELINE_STAGES.index(current)
            if idx + 1 < len(PIPELINE_STAGES):
                return PIPELINE_STAGES[idx + 1]

        return "verify"

    def _build_result(self, state: AgentState, run_dir: Path) -> Dict:
        """Build the final result dict."""
        return {
            "status": "completed",
            "final_stage": state.current_stage,
            "iteration_count": state.iteration + 1,
            "stop_reason": state.stop_reason,
            "mode": state.mode,
            "stage_status": state.stage_status,
            "decision_count": len(state.decisions),
            "tool_result_count": len(state.tool_results),
            "repair_count": len(state.repairs),
            "verify": state.verify,
            "artifacts": {
                "agent_steps": "agent_steps.jsonl",
                "agent_state": "agent_state.json",
                "agent_plan": "agent_plan.json",
            },
        }
