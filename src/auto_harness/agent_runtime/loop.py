"""DeploymentAgentLoop: the unified observe-plan-act-verify loop.

This is the core agent loop that orchestrates the full deployment process:
1. Load current pipeline state
2. Observe current stage
3. Ask LLM for next decision (if needed)
4. Execute stage through AgentStageExecutor
5. Record before/after status
6. Stop on verify passed or enter repair loop
7. Resume from safe stage after repair

The deterministic pipeline is not deleted but serves as the controlled
execution layer that the agent can invoke.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent_runtime.artifacts import AgentArtifactWriter
from auto_harness.agent_runtime.contribution import compute_llm_helped
from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
from auto_harness.agent_runtime.evidence import LLMContributionEvidenceWriter
from auto_harness.agent_runtime.stage_executor import AgentStageExecutor, StageExecutionResult
from auto_harness.agent_runtime.stage_schemas import (
    RUNNER_TOOLS, ENV_TOOLS, MODEL_TOOLS, REPAIR_TOOLS, PLAN_TOOLS,
    PIPELINE_STAGES, SAFE_RERUN_STAGES, VERIFY_TOOLS,
)
from auto_harness.agent_runtime.state import AgentState
from auto_harness.agent_runtime.stop import StopCondition
from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


# Stage -> allowed tools mapping
STAGE_TOOLS = {
    "analyze": PLAN_TOOLS,
    "resource_plan": PLAN_TOOLS,
    "env_solve": ENV_TOOLS,
    "env_deploy": ENV_TOOLS,
    "model_prepare": MODEL_TOOLS,
    "runner": RUNNER_TOOLS,
    "verify": VERIFY_TOOLS,
    "repair": REPAIR_TOOLS,
}


class DeploymentAgentLoop:
    """Unified agent loop for LLM-driven deployment.

    Runs the observe-plan-act-verify cycle across pipeline stages.
    Each iteration observes the current state, optionally asks LLM for a
    decision, executes the stage through AgentStageExecutor, and decides
    whether to continue, repair, or stop.
    """

    def __init__(
        self,
        *,
        provider=None,
        config=None,
        stage_executor: AgentStageExecutor = None,
        max_iterations: int = 5,
        stop_on_verify_pass: bool = True,
        runtime_policy: Dict = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.stage_executor = stage_executor
        self.max_iterations = max_iterations
        self.stop_on_verify_pass = stop_on_verify_pass
        self.runtime_policy = runtime_policy or {}

    def run(
        self,
        *,
        task_id: str,
        run_dir: Path,
        repo_dir: Path,
        initial_results: Dict,
        dry_run: bool = True,
    ) -> Dict:
        """Run the agent loop as the primary deployment controller.

        Iterates through pipeline stages, executing each through
        AgentStageExecutor. On failure/uncertain, enters repair loop.
        Stops when verify passes or max iterations reached.

        Args:
            task_id: Task identifier
            run_dir: Run directory for artifacts
            repo_dir: Repository directory
            initial_results: Results from any prior pipeline run
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

        # Merge initial results into stage_status
        for stage_name, result in initial_results.items():
            if isinstance(result, dict) and "status" in result:
                state.stage_status[stage_name] = {
                    "status": result["status"],
                    "iteration": -1,
                    "updated_at": utc_now_iso(),
                }

        # Initialize artifacts writer
        artifacts = AgentArtifactWriter(run_dir)

        # Initialize decision gate (for uncertain stages)
        gate = AgentDecisionGate(provider=self.provider)

        # Load existing plan if any
        state.plan = self._load_existing_plan(run_dir)

        # Load analysis and resource data for stage executor
        analysis = initial_results.get("analyze", {}).get("data", {})
        resource_data = initial_results.get("resource_plan", {}).get("data", {})

        # Main loop: iterate through pipeline stages
        iteration = 0
        stage_index = PIPELINE_STAGES.index(state.current_stage) if state.current_stage in PIPELINE_STAGES else 0

        while iteration < self.max_iterations and stage_index < len(PIPELINE_STAGES):
            state.iteration = iteration
            stage = PIPELINE_STAGES[stage_index]
            state.current_stage = stage

            # 1. Observe current stage
            observation = self._observe_stage(state, initial_results, run_dir, repo_dir)
            state.record_observation(stage, observation)

            # 2. For uncertain stages, optionally call LLM decision gate
            current_status = self._get_stage_status(stage, state)
            gate_decision_applied = False

            if current_status in ("uncertain", "failed", "") and self.provider:
                allowed_tools = list(STAGE_TOOLS.get(stage, ()))
                gate_result = gate.decide(
                    stage=stage,
                    observation=observation,
                    allowed_tools=allowed_tools,
                    mode=state.mode,
                    run_dir=run_dir,
                )

                # Record decision
                decision_record = {
                    "stage": stage,
                    "decision_status": gate_result.decision_status,
                    "hypothesis": gate_result.hypothesis,
                    "tool_call": gate_result.tool_call,
                    "policy_allowed": gate_result.policy.get("allowed", False),
                    "executed": gate_result.execution.get("executed", False),
                    "applied": gate_result.execution.get("applied", False),
                    "llm_helped": False,  # Gate never self-declares
                }
                state.record_decision(stage, decision_record)

                # Apply gate hints to stage_hints if provided
                if gate_result.execution.get("applied"):
                    gate_decision_applied = True
                    state.record_tool_result(stage, gate_result.execution)

            # 3. Execute stage through AgentStageExecutor
            stage_hints = state.plan.get("stage_hints", {}).get(stage, {})
            repair_overlay = self._load_repair_overlay(run_dir)

            # Determine deploy_analysis and runner_data from results so far
            # deploy_analysis should come from env_solve's data (which contains the merged analysis)
            env_solve_result = state.stage_status.get("env_solve", {}).get("result", {})
            deploy_analysis = {}
            if isinstance(env_solve_result, dict):
                # env_solve result structure: {stage, status, summary, data: {..., analysis: {...}}}
                env_solve_data = env_solve_result.get("data", {})
                if isinstance(env_solve_data, dict):
                    # Check for nested analysis field
                    analysis_nested = env_solve_data.get("analysis", {})
                    if isinstance(analysis_nested, dict):
                        # analysis has structure: {stage, status, summary, data: {run_candidates, ...}}
                        analysis_data = analysis_nested.get("data", {})
                        if isinstance(analysis_data, dict):
                            deploy_analysis = {
                                "run_candidates": analysis_data.get("run_candidates", []),
                                "verify_hint": analysis_data.get("verify_hint", {}),
                                "install_plan": analysis_data.get("install_plan", []),
                            }
                            # Also include env_solve specific data
                            deploy_analysis["environment_strategy"] = env_solve_data.get("environment_strategy", {})
                            deploy_analysis["env_solution"] = env_solve_data
                    # If no nested analysis, use env_solve_data itself
                    if not deploy_analysis:
                        deploy_analysis = env_solve_data

            # Fallback to initial_results if no deploy_analysis from state
            if not deploy_analysis:
                env_solve_initial = initial_results.get("env_solve", {})
                if isinstance(env_solve_initial, dict):
                    env_solve_data = env_solve_initial.get("data", {})
                    if isinstance(env_solve_data, dict):
                        analysis_nested = env_solve_data.get("analysis", {})
                        if isinstance(analysis_nested, dict):
                            analysis_data = analysis_nested.get("data", {})
                            if isinstance(analysis_data, dict):
                                deploy_analysis = {
                                    "run_candidates": analysis_data.get("run_candidates", []),
                                    "verify_hint": analysis_data.get("verify_hint", {}),
                                    "install_plan": analysis_data.get("install_plan", []),
                                }
                        if not deploy_analysis:
                            deploy_analysis = env_solve_data

            # Also merge from analyze stage results
            analyze_data = analysis if isinstance(analysis, dict) else {}
            # If analyze_data has data field (from StageResult), extract it
            if "data" in analyze_data and isinstance(analyze_data["data"], dict):
                analyze_data = analyze_data["data"]
            if not deploy_analysis.get("run_candidates") and analyze_data.get("run_candidates"):
                deploy_analysis = dict(deploy_analysis)
                deploy_analysis["run_candidates"] = analyze_data["run_candidates"]
                deploy_analysis["verify_hint"] = analyze_data.get("verify_hint", {})
                deploy_analysis["install_plan"] = analyze_data.get("install_plan", [])

            runner_data = state.stage_status.get("runner", {}).get("result", {})
            if not runner_data:
                runner_data = initial_results.get("runner", {}).get("data", {})

            if self.stage_executor:
                stage_result = self.stage_executor.execute_stage(
                    task_id=task_id,
                    run_dir=run_dir,
                    repo_dir=repo_dir,
                    stage=stage,
                    state=state.to_dict(),
                    analysis=analysis,
                    resource_data=resource_data,
                    deploy_analysis=deploy_analysis,
                    runner_data=runner_data,
                    dry_run=dry_run,
                    stage_hints=stage_hints,
                    repair_overlay=repair_overlay,
                )

                # 4. Record stage result
                self._record_stage_result(state, stage, stage_result, artifacts, iteration)

                # 5. Check if verify passed -> stop
                if stage == "verify" and stage_result.after_status in ("passed", "pass"):
                    state.stop_reason = "verify_passed"
                    state.verify = {
                        "status": stage_result.after_status,
                        "evidence_paths": stage_result.evidence_paths,
                    }
                    # Record verify step before breaking
                    stage_status_info = state.stage_status.get(stage, {})
                    step_record = {
                        "step_id": iteration + 1,
                        "phase": "execute",
                        "stage": stage,
                        "before_status": current_status,
                        "after_status": stage_status_info.get("status", ""),
                        "observation": observation,
                        "decision": state.decisions[-1] if state.decisions else None,
                        "evidence_paths": stage_result.evidence_paths,
                        "next_stage": None,
                        "stop_reason": state.stop_reason,
                        "recorded_at": utc_now_iso(),
                    }
                    artifacts.write_step(step_record)
                    break

                # 6. If stage failed/uncertain, enter repair loop
                if stage_result.after_status in ("failed", "uncertain"):
                    repair_result = self._run_repair_loop(
                        state=state,
                        stage=stage,
                        stage_result=stage_result,
                        gate=gate,
                        artifacts=artifacts,
                        task_id=task_id,
                        run_dir=run_dir,
                        repo_dir=repo_dir,
                        dry_run=dry_run,
                        iteration=iteration,
                    )

                    if repair_result.get("should_resume"):
                        # Resume from safe stage
                        resume_stage = repair_result.get("resume_from", "env_deploy")
                        if resume_stage in PIPELINE_STAGES:
                            stage_index = PIPELINE_STAGES.index(resume_stage)
                            iteration += 1
                            continue
                    else:
                        # Repair failed - for non-critical stages, continue to next stage
                        # Critical stages that must pass: verify
                        # Non-critical stages that can be uncertain: env_solve, model_prepare
                        critical_stages = ("verify",)
                        if stage in critical_stages:
                            state.stop_reason = repair_result.get("stop_reason", "repair_failed")
                            # Record step before breaking
                            stage_status_info = state.stage_status.get(stage, {})
                            step_record = {
                                "step_id": iteration + 1,
                                "phase": "execute",
                                "stage": stage,
                                "before_status": current_status,
                                "after_status": stage_status_info.get("status", ""),
                                "observation": observation,
                                "decision": state.decisions[-1] if state.decisions else None,
                                "evidence_paths": stage_status_info.get("evidence_paths", []),
                                "next_stage": None,
                                "stop_reason": state.stop_reason,
                                "recorded_at": utc_now_iso(),
                            }
                            artifacts.write_step(step_record)
                            break
                        else:
                            # Log repair failure but continue to next stage
                            state.record_observation(stage, {
                                "repair_failed": True,
                                "stop_reason": repair_result.get("stop_reason", "repair_failed"),
                                "continuing_to_next_stage": True,
                            })
                elif stage_result.after_status in ("passed", "pass"):
                    # Stage passed, update analysis/resource_data for next stages
                    if stage == "analyze":
                        analysis = stage_result.result or analysis
                    elif stage == "resource_plan":
                        resource_data = stage_result.result or resource_data
            else:
                # No stage executor, just record observation and move on
                state.update_stage_status(stage, current_status or "no_executor")

            # 7. Write step artifact with full context
            stage_status_info = state.stage_status.get(stage, {})
            step_record = {
                "step_id": iteration + 1,
                "phase": "execute",
                "stage": stage,
                "before_status": current_status,
                "after_status": stage_status_info.get("status", ""),
                "observation": observation,
                "decision": state.decisions[-1] if state.decisions else None,
                "evidence_paths": stage_status_info.get("evidence_paths", []),
                "next_stage": PIPELINE_STAGES[stage_index + 1] if stage_index + 1 < len(PIPELINE_STAGES) else None,
                "stop_reason": state.stop_reason,
                "recorded_at": utc_now_iso(),
            }
            artifacts.write_step(step_record)

            # 8. Move to next stage
            stage_index += 1
            iteration += 1

        # Save final state
        artifacts.write_state(state.to_dict())

        # Write agent plan
        artifacts.write_plan(state.plan)

        # Write agent_plan_revisions.jsonl (empty if no revisions)
        revisions_path = run_dir / "agent_plan_revisions.jsonl"
        if not revisions_path.exists():
            revisions_path.touch()

        # Build result
        result = self._build_result(state, run_dir)

        # Write pipeline_results.json for compatibility with existing tools
        self._write_pipeline_results(run_dir, state)

        # Write LLM contribution evidence
        self._write_llm_contribution_evidence(run_dir, task_id, state)

        # Write report.md
        self._write_report(run_dir, task_id, state)

        return result

    def _run_repair_loop(
        self,
        *,
        state: AgentState,
        stage: str,
        stage_result: StageExecutionResult,
        gate: AgentDecisionGate,
        artifacts: AgentArtifactWriter,
        task_id: str,
        run_dir: Path,
        repo_dir: Path,
        dry_run: bool,
        iteration: int,
    ) -> Dict:
        """Run repair loop when a stage fails or is uncertain.

        Returns:
            Dict with should_resume, resume_from, stop_reason
        """
        repair_record = {
            "trigger_stage": stage,
            "before_status": stage_result.after_status,
            "iteration": iteration,
            "started_at": utc_now_iso(),
        }

        # Ask LLM for repair decision
        if not self.provider:
            repair_record["stop_reason"] = "no_provider_for_repair"
            state.record_repair(repair_record)
            return {"should_resume": False, "stop_reason": "no_provider_for_repair"}

        observation = {
            "stage": "repair",
            "trigger_stage": stage,
            "failure_status": stage_result.after_status,
            "error": stage_result.error or "",
            "evidence_paths": stage_result.evidence_paths,
        }

        allowed_tools = list(STAGE_TOOLS.get("repair", ()))
        gate_result = gate.decide(
            stage="repair",
            observation=observation,
            allowed_tools=allowed_tools,
            mode=state.mode,
            run_dir=run_dir,
        )

        # Record repair decision with full state machine
        repair_record["planned"] = gate_result.decision_status == "ok"
        repair_record["decision_status"] = gate_result.decision_status
        repair_record["policy_allowed"] = gate_result.policy.get("allowed", False)
        repair_record["applied"] = gate_result.execution.get("applied", False)
        repair_record["executed"] = gate_result.execution.get("executed", False)

        # Check if repair action was effective (not metadata_only)
        action_type = gate_result.execution.get("action_type", "")
        metadata_only = action_type in ("update_verify_hint", "rerun_from_stage")
        repair_record["metadata_only"] = metadata_only
        effective = gate_result.execution.get("executed") and not metadata_only

        repair_record["effective"] = effective

        # Check if repair includes resume
        resume_from = gate_result.execution.get("resume_from_stage", "")
        repair_record["resumed"] = bool(resume_from)

        if effective and resume_from:
            repair_record["should_resume"] = True
            repair_record["resume_from"] = resume_from
            repair_record["completed_at"] = utc_now_iso()
            state.record_repair(repair_record)
            # Write repair artifacts
            self._write_repair_artifacts(run_dir, repair_record, iteration)
            return {
                "should_resume": True,
                "resume_from": resume_from,
            }

        repair_record["stop_reason"] = "repair_not_effective"
        repair_record["completed_at"] = utc_now_iso()
        state.record_repair(repair_record)
        # Write repair artifacts
        self._write_repair_artifacts(run_dir, repair_record, iteration)
        return {"should_resume": False, "stop_reason": "repair_not_effective"}

    def _record_stage_result(
        self,
        state: AgentState,
        stage: str,
        stage_result: StageExecutionResult,
        artifacts: AgentArtifactWriter,
        iteration: int,
    ) -> None:
        """Record stage execution result in state and artifacts."""
        # Update stage status
        if stage_result.error:
            state.update_stage_status(stage, "failed")
        elif stage_result.after_status:
            state.update_stage_status(stage, stage_result.after_status)
        else:
            state.update_stage_status(stage, "no_status")

        # Store result data
        if stage_result.result:
            state.stage_status[stage]["result"] = stage_result.result

        # Record evidence paths
        if stage_result.evidence_paths:
            state.stage_status[stage]["evidence_paths"] = stage_result.evidence_paths

    def _get_stage_status(self, stage: str, state: AgentState) -> str:
        """Get current status of a stage from state."""
        stage_info = state.stage_status.get(stage, {})
        if isinstance(stage_info, dict):
            return stage_info.get("status", "")
        return ""

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
            # run_candidates are in analyze stage, not runner stage
            analyze_result = initial_results.get("analyze", {})
            analyze_data = analyze_result.get("data", analyze_result) if isinstance(analyze_result, dict) else {}
            candidates = analyze_data.get("run_candidates", [])
            # Also check state.stage_status for analyze result
            if not candidates:
                analyze_status = state.stage_status.get("analyze", {})
                if isinstance(analyze_status, dict):
                    analyze_result_data = analyze_status.get("result", {})
                    if isinstance(analyze_result_data, dict):
                        analyze_inner = analyze_result_data.get("data", analyze_result_data)
                        if isinstance(analyze_inner, dict):
                            candidates = analyze_inner.get("run_candidates", [])
            # Normalize candidates with IDs (same as orchestrator._build_runner_observation)
            normalized = []
            for i, c in enumerate(candidates):
                cand = dict(c)
                if "id" not in cand:
                    cand["id"] = "cand_%d" % i
                normalized.append(cand)
            observation["run_candidates"] = normalized
        elif stage == "env_solve":
            observation["env_solution"] = result.get("data", {}).get("env_solution", {})
            observation["constraints"] = self._load_constraints(run_dir)
        elif stage == "model_prepare":
            observation["model_assets"] = result.get("data", {}).get("model_assets", {})
        elif stage == "verify":
            observation["verify_status"] = result.get("status", "")
            observation["verify_evidence"] = result.get("data", {}).get("evidence", {})

        return observation

    def _write_repair_artifacts(self, run_dir: Path, repair_record: Dict, iteration: int) -> None:
        """Write repair artifacts to disk.

        Writes:
        - repair_hypothesis.json
        - repair_tool_call.json
        - repair_policy.json
        - repair_execution.json
        - resume_result.json (if resumed)
        - repair_effectiveness.json
        """
        repairs_dir = run_dir / "repairs"
        repairs_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"repair_{iteration}"

        # Write hypothesis
        write_json(repairs_dir / f"{prefix}_hypothesis.json", {
            "iteration": iteration,
            "trigger_stage": repair_record.get("trigger_stage", ""),
            "hypothesis": repair_record.get("hypothesis", ""),
            "recorded_at": utc_now_iso(),
        })

        # Write policy
        write_json(repairs_dir / f"{prefix}_policy.json", {
            "iteration": iteration,
            "policy_allowed": repair_record.get("policy_allowed", False),
            "decision_status": repair_record.get("decision_status", ""),
            "recorded_at": utc_now_iso(),
        })

        # Write execution
        write_json(repairs_dir / f"{prefix}_execution.json", {
            "iteration": iteration,
            "applied": repair_record.get("applied", False),
            "executed": repair_record.get("executed", False),
            "metadata_only": repair_record.get("metadata_only", False),
            "recorded_at": utc_now_iso(),
        })

        # Write resume result if resumed
        if repair_record.get("resumed"):
            write_json(repairs_dir / f"{prefix}_resume_result.json", {
                "iteration": iteration,
                "resumed": True,
                "resume_from": repair_record.get("resume_from", ""),
                "recorded_at": utc_now_iso(),
            })

        # Write effectiveness
        write_json(repairs_dir / f"{prefix}_effectiveness.json", {
            "iteration": iteration,
            "effective": repair_record.get("effective", False),
            "metadata_only": repair_record.get("metadata_only", False),
            "resumed": repair_record.get("resumed", False),
            "repair_verified": repair_record.get("repair_verified", False),
            "recorded_at": utc_now_iso(),
        })

    def _load_constraints(self, run_dir: Path) -> List[str]:
        """Load constraints from repair overlay."""
        constraints_path = run_dir / "repair_overlay" / "constraints.txt"
        if constraints_path.exists():
            return [line.strip() for line in constraints_path.read_text().splitlines() if line.strip()]
        return []

    def _load_repair_overlay(self, run_dir: Path) -> Dict:
        """Load repair overlay if exists."""
        overlay_path = run_dir / "repair_overlay" / "overlay.json"
        if overlay_path.exists():
            try:
                return read_json(overlay_path)
            except (OSError, ValueError):
                return {}
        return {}

    def _load_existing_plan(self, run_dir: Path) -> Dict:
        """Load existing deployment plan if available."""
        plan_path = run_dir / "agent_plan.json"
        if plan_path.exists():
            try:
                return read_json(plan_path)
            except (OSError, ValueError):
                return {}
        return {}

    def _determine_start_stage(self, initial_results: Dict) -> str:
        """Determine which stage to start the agent loop at."""
        # Find the first failed or uncertain stage
        for stage in PIPELINE_STAGES:
            result = initial_results.get(stage, {})
            status = result.get("status", "") if isinstance(result, dict) else ""
            if status in ("failed", "uncertain"):
                return stage
        # If all passed, start at the beginning for full execution
        return PIPELINE_STAGES[0] if PIPELINE_STAGES else "analyze"

    def _write_pipeline_results(self, run_dir: Path, state: AgentState) -> None:
        """Write pipeline_results.json for compatibility with existing tools.

        Converts agent state to pipeline format so CLI status/report works.
        The result stored in stage_status is a StageResult wrapper with
        {stage, status, summary, data, evidence, error}. For compatibility
        with the original pipeline format, we store it as-is (matching to_plain output).
        """
        results = {}
        for stage, status_info in state.stage_status.items():
            if isinstance(status_info, dict):
                result_data = status_info.get("result", {})
                if isinstance(result_data, dict) and "status" in result_data:
                    # result_data is already a StageResult wrapper from to_plain()
                    # Store it directly to match original pipeline format
                    results[stage] = result_data
                else:
                    stage_status = status_info.get("status", "")
                    results[stage] = {
                        "status": stage_status,
                        "data": result_data if isinstance(result_data, dict) else {},
                    }
        # Add report stage if verify passed
        verify_status = state.verify.get("status", "")
        if verify_status in ("passed", "pass"):
            results["report"] = {"status": "passed", "data": {"report_path": str(run_dir / "reports" / "report.md")}}
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / "pipeline_results.json", results)

    def _write_llm_contribution_evidence(self, run_dir: Path, task_id: str, state: AgentState) -> None:
        """Write LLM contribution evidence JSON."""
        # Collect agent steps from JSONL
        agent_steps = []
        steps_path = run_dir / "agent_steps.jsonl"
        if steps_path.exists():
            try:
                for line in steps_path.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        agent_steps.append(json.loads(line))
            except (OSError, ValueError):
                pass

        # Build agent result from state
        agent_result = {
            "mode": state.mode,
            "changed_stage": state.current_stage,
            "decision_type": "",
            "accepted_tool_count": len(state.tool_results),
            "rejected_tool_count": sum(1 for d in state.decisions if not d.get("decision", {}).get("policy_allowed")),
        }

        # Extract decision type from decisions
        for decision in state.decisions:
            d = decision.get("decision", {})
            if d.get("tool_call"):
                tool_name = d["tool_call"].get("name", "")
                if tool_name:
                    agent_result["decision_type"] = tool_name
                    break

        # Load pipeline results
        pipeline_results = {}
        pipeline_path = run_dir / "reports" / "pipeline_results.json"
        if pipeline_path.exists():
            try:
                pipeline_results = read_json(pipeline_path)
            except (OSError, ValueError):
                pass

        # Write evidence
        writer = LLMContributionEvidenceWriter()
        writer.write(
            run_dir=run_dir,
            task_id=task_id,
            agent_result=agent_result,
            agent_steps=agent_steps,
            pipeline_results=pipeline_results,
        )

    def _write_report(self, run_dir: Path, task_id: str, state: AgentState) -> None:
        """Write report.md with deployment summary and LLM contribution evidence."""
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Extract key information
        verify_status = state.verify.get("status", "unknown")
        trace_id = ""
        evidence_paths = state.verify.get("evidence_paths", [])

        # Try to get trace_id from verify data in stage_status
        verify_stage = state.stage_status.get("verify", {})
        if isinstance(verify_stage, dict):
            verify_result = verify_stage.get("result", {})
            if isinstance(verify_result, dict):
                verify_data = verify_result.get("data", verify_result)
                if isinstance(verify_data, dict):
                    trace_id = verify_data.get("trace_id", "")
                    if not evidence_paths:
                        evidence_paths = verify_data.get("evidence_paths", verify_data.get("evidence", []))

        # Build report content
        lines = [
            "# Deployment Report",
            "",
            "## Summary",
            "",
            f"- Task ID: `{task_id}`",
            f"- Agent Mode: `{state.mode}`",
            f"- Stop Reason: `{state.stop_reason}`",
            f"- Iteration Count: {state.iteration + 1}",
            f"- Final Status: `{verify_status}`",
            "",
        ]

        # Stage status table
        lines.extend([
            "## Stage Status",
            "",
            "| Stage | Status |",
            "|-------|--------|",
        ])
        for stage_name in PIPELINE_STAGES:
            stage_info = state.stage_status.get(stage_name, {})
            if isinstance(stage_info, dict):
                stage_status = stage_info.get("status", "not_run")
                lines.append(f"| {stage_name} | `{stage_status}` |")

        lines.extend(["", ""])

        # Verify details
        if verify_status in ("passed", "pass"):
            lines.extend([
                "## Verify Details",
                "",
                f"Final status: `{verify_status}`",
                "",
            ])
            if trace_id:
                lines.append(f"Trace ID: `{trace_id}`")
                lines.append("")
            if evidence_paths:
                lines.append("Evidence:")
                for path in evidence_paths:
                    lines.append(f"- `{path}`")
                lines.append("")

        # Agent decisions summary
        if state.decisions:
            lines.extend([
                "## Agent Decisions",
                "",
                f"Total decisions: {len(state.decisions)}",
                "",
            ])

        # Repairs summary
        if state.repairs:
            lines.extend([
                "## Repairs",
                "",
                f"Total repairs attempted: {len(state.repairs)}",
                "",
            ])

        # LLM Contribution Evidence
        evidence_path = reports_dir / "llm_contribution_evidence.json"
        if evidence_path.exists():
            try:
                evidence = read_json(evidence_path)
                lines.extend([
                    "## LLM Contribution Evidence",
                    "",
                    f"- Agent mode: `{evidence.get('agent', {}).get('mode', state.mode)}`",
                    f"- Baseline status: `{evidence.get('baseline', {}).get('final_status', 'unknown')}`",
                    f"- Agent status: `{evidence.get('agent', {}).get('final_status', verify_status)}`",
                    f"- LLM helped: `{evidence.get('llm_helped', False)}`",
                    f"- LLM required: `{evidence.get('llm_required', False)}`",
                    f"- Changed stage: `{evidence.get('agent', {}).get('changed_stage', '')}`",
                    f"- Decision type: `{evidence.get('agent', {}).get('decision_type', '')}`",
                    f"- Trace ID: `{evidence.get('trace_id', '')}`",
                    "",
                ])
                if evidence.get("evidence_paths"):
                    lines.append("Evidence:")
                    for path in evidence["evidence_paths"]:
                        lines.append(f"- `{path}`")
                    lines.append("")
            except (OSError, ValueError):
                pass
        else:
            lines.extend([
                "## LLM Contribution Evidence",
                "",
                f"- Agent mode: `{state.mode}`",
                f"- Decision count: {len(state.decisions)}",
                f"- Tool result count: {len(state.tool_results)}",
                f"- Repair count: {len(state.repairs)}",
                "",
            ])

        report_content = "\n".join(lines)
        report_path = reports_dir / "report.md"
        report_path.write_text(report_content, encoding="utf-8")

    def _build_result(self, state: AgentState, run_dir: Path) -> Dict:
        """Build the final result dict."""
        # Compute llm_helped across all decisions
        total_llm_helped = False
        for decision in state.decisions:
            d = decision.get("decision", {})
            if d.get("llm_helped"):
                total_llm_helped = True
                break

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
            "llm_helped": total_llm_helped,
            "verify": state.verify,
            "artifacts": {
                "agent_steps": "agent_steps.jsonl",
                "agent_state": "agent_state.json",
                "agent_plan": "agent_plan.json",
            },
        }
