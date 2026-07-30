import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent_runtime.critic import AgentCritic
from auto_harness.agent_runtime.planner import VerifyPlanner
from auto_harness.agent_runtime.policy import ToolPolicy
from auto_harness.agent_runtime.schemas import (
    AgentDecision,
    AgentGoal,
    AgentRuntimeStep,
    AgentVerifyResult,
    ToolCall,
    ToolResult,
    VERIFY_TOOLS,
    parse_agent_decision,
)
from auto_harness.agent_runtime.state import AgentStepWriter, AgentVerifyState, compute_idempotency_key
from auto_harness.models.base import to_plain, write_json
from auto_harness.tools import ToolRegistry
from auto_harness.tools.executor import ToolExecutor
from auto_harness.utils.time import utc_now_iso


class AgentRuntime:
    """Records an explicit observe-plan-tool-observe loop around the existing runtime.

    Provides two modes:
    - audit(): post-hoc artifact generation (backward compatible with old run())
    - act_verify(): runtime LLM-driven verify agent loop (true Agent mode)
    """

    STAGE_TO_TOOL = {
        "analyze": "inspect_repo_tree",
        "resource_plan": "parse_dependency_files",
        "env_solve": "solve_environment",
        "env_deploy": "install_environment",
        "model_prepare": "prepare_model_assets",
        "runner": "start_service",
        "verify": "verify_evidence",
        "report": "verify_evidence",
    }

    def __init__(self, registry: ToolRegistry = None, critic: AgentCritic = None) -> None:
        self.registry = registry or ToolRegistry()
        self.critic = critic or AgentCritic()
        self._sanitizer = None  # Lazy-init in _build_observation

    # ------------------------------------------------------------------
    # Backward-compatible entry point
    # ------------------------------------------------------------------

    def run(self, goal: AgentGoal, run_dir: Path, results: Dict, contribution: Dict = None) -> Dict:
        """Backward-compatible wrapper. Delegates to audit()."""
        return self.audit(goal, run_dir, results, contribution=contribution)

    # ------------------------------------------------------------------
    # Phase 1: audit mode (post-hoc artifact generation)
    # ------------------------------------------------------------------

    def audit(self, goal: AgentGoal, run_dir: Path, results: Dict, contribution: Dict = None) -> Dict:
        """Post-hoc artifact generation. Does NOT represent true LLM-driven execution.

        All artifacts are tagged mode='audit'. llm_helped is never set to true
        from audit output alone.
        """
        run_dir = Path(run_dir)
        steps = self._steps(goal, results)
        # Tag all steps with mode=audit
        for step in steps:
            step["mode"] = "audit"
        self._write_jsonl(run_dir / "agent_steps.jsonl", steps)
        state = self._state(goal, steps, results, contribution or {})
        state["mode"] = "audit"
        # audit mode never claims llm_helped
        state["llm_helped"] = False
        plan = self._plan(goal, results, contribution or {})
        plan["mode"] = "audit"
        write_json(run_dir / "agent_state.json", state)
        write_json(run_dir / "agent_plan.json", plan)
        self._write_plan_revisions(run_dir / "agent_plan_revisions.jsonl", steps)
        return {"state": state, "plan": plan, "step_count": len(steps), "mode": "audit"}

    # ------------------------------------------------------------------
    # Phase 5: act_verify (runtime LLM-driven verify agent loop)
    # ------------------------------------------------------------------

    def act_verify(
        self,
        *,
        run_dir: Path,
        repo_path: Path,
        initial_verify_result: Dict,
        service_context: Dict,
        trace_id: str,
        config: Dict,
        provider=None,
        max_steps: int = 3,
        agent_mode: str = "gated_actor",
        allowed_hosts: List[str] = None,
        skill_context: Optional[Dict] = None,
    ) -> Dict:
        """Runtime LLM-driven verify agent loop.

        Called when deterministic verify returns uncertain and agent verify is enabled.
        This is the true Agent mode: LLM selects tool_call at runtime,
        passes through schema/critic/policy gates, and tool results can change
        the final verify status.

        Flow: observe -> planner -> schema validation -> critic -> policy -> executor
              -> observe result -> update state -> repeat until pass/reject/max_steps

        Returns AgentVerifyResult dict.
        """
        run_dir = Path(run_dir)
        state = AgentVerifyState(trace_id=trace_id, initial_status="uncertain")
        writer = AgentStepWriter(run_dir)
        planner = VerifyPlanner(provider=provider, config=config)
        policy = ToolPolicy(
            registry=self.registry,
            allowed_hosts=allowed_hosts or ["127.0.0.1", "localhost"],
        )
        executor = ToolExecutor(registry=self.registry)
        executor.validate_contract()

        # Store skill_context for use in observations
        self._skill_context = skill_context or {}

        # Determine allowed tools based on service context
        allowed_tools = self._allowed_tools_for_verify(service_context, initial_verify_result)

        for step_index in range(max_steps):
            # Build observation
            observation = self._build_observation(
                run_dir=run_dir,
                repo_path=repo_path,
                trace_id=trace_id,
                service_context=service_context,
                initial_verify_result=initial_verify_result,
                allowed_tools=allowed_tools,
                previous_steps=state.steps,
            )

            # Call LLM planner
            decision = planner.plan_verify(observation, allowed_tools=allowed_tools)

            # Schema validation already done in parse_agent_decision.
            # If decision is invalid, record and break.
            if decision.status == "invalid":
                decision_dict = {
                    "status": decision.status,
                    "hypothesis": decision.hypothesis,
                    "confidence": decision.confidence,
                    "stop_reason": decision.stop_reason,
                    "raw_response": decision.raw_response[:500] if decision.raw_response else "",
                }
                writer.write_rejected(
                    step_index=step_index + 1,
                    trace_id=trace_id,
                    decision=decision_dict,
                    reason="invalid_llm_output: %s" % decision.stop_reason,
                )
                state.record_reject("invalid_llm_output")
                state.steps.append(decision_dict)
                break

            # no_action from LLM
            if decision.status == "no_action":
                decision_dict = {
                    "status": decision.status,
                    "hypothesis": decision.hypothesis,
                    "confidence": decision.confidence,
                    "tool_call": None,
                    "stop_reason": decision.stop_reason,
                }
                writer.write_rejected(
                    step_index=step_index + 1,
                    trace_id=trace_id,
                    decision=decision_dict,
                    reason="no_action: %s" % (decision.stop_reason or "LLM determined no safe action"),
                )
                state.record_reject("no_action")
                state.steps.append(decision_dict)
                break

            # Decision is ok with a tool_call
            tool_call = decision.tool_call
            decision_dict = {
                "status": decision.status,
                "hypothesis": decision.hypothesis,
                "confidence": decision.confidence,
                "tool_call": {"name": tool_call.name, "input": tool_call.input},
                "expected_observation": decision.expected_observation,
            }

            # Critic gate
            critic_result = self._critic_evaluate(tool_call, observation)
            if not critic_result.get("allowed", False):
                writer.write_rejected(
                    step_index=step_index + 1,
                    trace_id=trace_id,
                    decision=decision_dict,
                    critic_result=critic_result,
                    reason="critic_rejected: %s" % critic_result.get("reason", ""),
                )
                state.record_reject("critic_rejected")
                state.steps.append(decision_dict)
                # Continue loop so LLM can try a different tool
                continue

            # Policy gate
            policy_result = policy.validate(
                tool_call=tool_call,
                stage="verify",
                agent_mode=agent_mode,
                trace_id=trace_id,
            )
            if not policy_result.allowed:
                writer.write_rejected(
                    step_index=step_index + 1,
                    trace_id=trace_id,
                    decision=decision_dict,
                    critic_result=critic_result,
                    policy_result={"allowed": False, "reason": policy_result.reason, "risk": policy_result.risk},
                    reason="policy_rejected: %s" % policy_result.reason,
                )
                state.record_reject("policy_rejected")
                state.steps.append(decision_dict)
                # Continue loop so LLM can try a different tool
                continue

            # In planner mode, record would_execute but don't actually execute
            if agent_mode == "planner":
                writer.write_rejected(
                    step_index=step_index + 1,
                    trace_id=trace_id,
                    decision=decision_dict,
                    critic_result=critic_result,
                    policy_result={"allowed": True, "reason": policy_result.reason, "risk": policy_result.risk},
                    reason="planner_mode_would_execute",
                )
                state.steps.append(decision_dict)
                # Continue loop in planner mode to see what else LLM would do
                continue

            # gated_actor mode: execute the approved tool
            normalized_tool_call = ToolCall(
                name=tool_call.name,
                input=policy_result.normalized_input or tool_call.input,
                idempotency_key=compute_idempotency_key(
                    run_id=str(run_dir.name),
                    step_index=step_index,
                    tool_name=tool_call.name,
                    tool_input=tool_call.input,
                ),
            )

            # Build execution context
            exec_context = {
                "trace_id": trace_id,
                "evidence_dir": str(run_dir / "evidence"),
                "run_dir": str(run_dir),
                "step_index": step_index + 1,
                "urlopen": config.get("urlopen"),  # No-proxy urlopen from VerifyModule
            }

            # Execute
            tool_result = executor.execute(normalized_tool_call, exec_context)
            tool_result_dict = {
                "status": tool_result.status,
                "tool_name": tool_result.tool_name,
                "evidence": tool_result.evidence,
                "evidence_path": tool_result.evidence_path,
                "strong_verify_pass": tool_result.strong_verify_pass,
                "error": tool_result.error,
                "started_at": tool_result.started_at,
                "ended_at": tool_result.ended_at,
            }

            state.record_accepted_tool()

            # Write executed step
            state_delta = {
                "verify_status_before": state.verify_status,
            }
            state.apply_tool_result(tool_result_dict)
            state_delta["verify_status_after"] = state.verify_status

            writer.write_executed(
                step_index=step_index + 1,
                trace_id=trace_id,
                decision=decision_dict,
                critic_result=critic_result,
                policy_result={"allowed": True, "reason": policy_result.reason, "risk": policy_result.risk},
                tool_result=tool_result_dict,
                state_delta=state_delta,
            )
            state.steps.append(decision_dict)

            # If strong_verify_pass, we're done
            if tool_result.strong_verify_pass:
                llm_helped = self._compute_llm_helped(agent_mode, state)
                result = state.to_result(
                    final_status="passed",
                    stop_reason="strong_verify_pass",
                    mode=agent_mode,
                    llm_helped=llm_helped,
                )
                writer.write_state(state, mode=agent_mode)
                writer.write_result(result)
                return result

            # Check if service is still alive
            if not service_context.get("process_alive") or not service_context.get("port_ready"):
                result = state.to_result(
                    final_status="uncertain",
                    stop_reason="service_dead",
                    mode=agent_mode,
                    llm_helped=False,
                )
                writer.write_state(state, mode=agent_mode)
                writer.write_result(result)
                return result

        # Exited loop without strong_verify_pass
        llm_helped = self._compute_llm_helped(agent_mode, state)
        result = state.to_result(
            final_status=state.verify_status,
            stop_reason=state.stop_reason or "max_steps",
            mode=agent_mode,
            llm_helped=llm_helped,
        )
        writer.write_state(state, mode=agent_mode)
        writer.write_result(result)
        return result

    # ------------------------------------------------------------------
    # act_verify helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_llm_helped(agent_mode: str, state: "AgentVerifyState") -> bool:
        """Determine if LLM genuinely helped improve the verify outcome.

        Per design doc §15.1, llm_helped requires ALL of:
        - agent_mode == gated_actor
        - LLM produced valid tool_call
        - critic allowed
        - policy allowed
        - executor executed tool
        - tool_result improved final status
        - evidence path exists

        Simplified: gated_actor mode + status improved + strong evidence exists.
        """
        if agent_mode != "gated_actor":
            return False
        if state.verify_status != "passed":
            return False
        if not state.strong_verify_pass:
            return False
        # At least one evidence path must exist on disk
        if not state.evidence_paths:
            return False
        if not any(Path(ep).exists() for ep in state.evidence_paths):
            return False
        return True

    def _allowed_tools_for_verify(self, service_context: Dict, verify_result: Dict) -> List[str]:
        """Determine which tools the LLM can choose from based on context."""
        tools = ["probe_http"]
        frameworks = set()
        analysis = verify_result.get("data", {}) if isinstance(verify_result.get("data"), dict) else {}
        if isinstance(analysis, dict):
            frameworks = set(analysis.get("frameworks") or [])

        # If Gradio framework detected, add Gradio discovery
        if "gradio" in frameworks or verify_result.get("data", {}).get("service", {}).get("type") == "webui":
            tools.append("discover_gradio_api")

        # If FastAPI/Flask detected, add OpenAPI discovery
        if frameworks.intersection({"fastapi", "flask"}) or verify_result.get("data", {}).get("service", {}).get("type") == "api":
            tools.append("discover_openapi_schema")

        # Browser probe is always an option but medium risk
        tools.append("probe_browser_dom")

        return tools

    def _build_observation(
        self,
        run_dir: Path,
        repo_path: Path,
        trace_id: str,
        service_context: Dict,
        initial_verify_result: Dict,
        allowed_tools: List[str],
        previous_steps: List[Dict],
    ) -> Dict:
        """Build observation dict for the LLM planner."""
        # Extract failed checks from verify result
        failed_checks = []
        data = initial_verify_result.get("data", {}) if isinstance(initial_verify_result.get("data"), dict) else {}
        checks = data.get("checks", [])
        for check in checks:
            if isinstance(check, dict) and check.get("status") != "pass":
                failed_checks.append({
                    "name": check.get("name", ""),
                    "status": check.get("status", ""),
                    "reason": check.get("reason", ""),
                })

        # Build evidence summary
        evidence_summary = {"trace_id": trace_id}
        if data.get("trace_id"):
            evidence_summary["trace_id"] = data["trace_id"]

        # Selected files (truncated)
        if self._sanitizer is None:
            from auto_harness.agent.safety import AgentInputSanitizer
            self._sanitizer = AgentInputSanitizer()
        sanitizer = self._sanitizer
        selected_files = {}
        for name in ("README.md", "app.py", "main.py"):
            path = repo_path / name
            if path.is_file():
                content = path.read_text(encoding="utf-8", errors="ignore")[:3000]
                selected_files[name] = "UNTRUSTED REPO CONTENT:\n" + content
        selected_files = sanitizer.sanitize_selected_files(selected_files)

        # Constraints
        constraints = [
            "Only call localhost or 127.0.0.1 endpoints.",
            "Do not call external URLs.",
            "Do not include secret values.",
            "Any verification request must include {{trace_id}} or explain why discovery-only tool is used.",
            "Success requires current trace evidence, not HTTP 200 alone.",
        ]

        # Previous steps (max 3)
        prev = previous_steps[-3:] if len(previous_steps) > 3 else previous_steps

        return {
            "service": {
                "endpoint_candidates": service_context.get("endpoint_candidates", []),
                "process_alive": service_context.get("process_alive", False),
                "port_ready": service_context.get("port_ready", False),
                "framework_hint": ", ".join(
                    (initial_verify_result.get("data", {}).get("frameworks") or [])
                    if isinstance(initial_verify_result.get("data", {}).get("frameworks"), list) else []
                ),
            },
            "failed_checks": failed_checks,
            "evidence_summary": evidence_summary,
            "selected_files": selected_files,
            "allowed_tools": allowed_tools,
            "constraints": constraints,
            "previous_steps": prev,
            "skill_context": getattr(self, '_skill_context', {}),
        }

    def _critic_evaluate(self, tool_call: ToolCall, observation: Dict) -> Dict:
        """Evaluate a tool call through the critic gate.

        The critic checks relevance, hallucination, and authorization.
        """
        name = tool_call.name
        tool_input = tool_call.input or {}
        input_text = json.dumps(tool_input, ensure_ascii=False).lower()

        # Check for secret values
        for token in ("api_key=", "token=", "password", "bearer "):
            if token in input_text:
                return {"allowed": False, "reason": "tool input appears to contain secret value", "issues": ["secret_in_input"]}

        # Check for verify-relevance: verify stage tools should not install/repair
        if name in ("install_environment", "start_service", "apply_repair", "resume_from_stage"):
            return {"allowed": False, "reason": "tool is not relevant for verify stage", "issues": ["stage_mismatch"]}

        # Check that verify tools reference trace
        if name in VERIFY_TOOLS and "trace" not in input_text and "{{trace_id}}" not in input_text:
            # Discovery-only tools are ok without trace in input
            if name not in ("discover_gradio_api", "discover_openapi_schema"):
                return {"allowed": False, "reason": "verify tool should reference current trace", "issues": ["no_trace_reference"]}

        return {"allowed": True, "reason": "tool call is consistent with policy and evidence requirements", "issues": []}

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _steps(self, goal: AgentGoal, results: Dict) -> List[Dict]:
        steps = []
        belief = {"known_stage_status": {}, "open_hypotheses": []}
        for index, (stage, result) in enumerate(results.items(), start=1):
            if not isinstance(result, dict):
                continue
            tool_name = self.STAGE_TO_TOOL.get(stage, "inspect_log")
            tool = self.registry.get(tool_name)
            before = dict(belief)
            status = result.get("status", "")
            belief = {
                "known_stage_status": {**belief.get("known_stage_status", {}), stage: status},
                "open_hypotheses": self._hypotheses(result),
            }
            step = to_plain(AgentRuntimeStep(
                step_id=index,
                goal=to_plain(goal),
                observation={"stage": stage, "status": status, "summary": result.get("summary", "")},
                belief_state_before=before,
                llm_decision=self._llm_decision(stage, result),
                policy_result=self._policy_result(result),
                tool_call={"name": tool_name, "input": {"stage": stage}, "schema": tool},
                tool_result={"status": status, "summary": result.get("summary", ""), "evidence": result.get("evidence", [])},
                belief_state_after=belief,
                next_step=self._next_step(stage, status),
                termination_reason="final_report_generated" if stage == "report" else "",
            ))
            step["critique"] = self.critic.critique(step)
            steps.append(step)
        return steps

    def _state(self, goal: AgentGoal, steps: List[Dict], results: Dict, contribution: Dict) -> Dict:
        verify = results.get("verify", {}) if isinstance(results.get("verify"), dict) else {}
        return {
            "task_id": goal.task_id,
            "objective": goal.objective,
            "updated_at": utc_now_iso(),
            "step_count": len(steps),
            "final_verify_status": verify.get("status", ""),
            "success_condition": goal.success_condition,
            "llm_helped": bool(contribution.get("llm_helped")),
            "llm_required": bool(contribution.get("llm_required")),
            "termination_reason": "verify_passed" if verify.get("status") in ("pass", "passed") else "pipeline_finished",
        }

    def _plan(self, goal: AgentGoal, results: Dict, contribution: Dict) -> Dict:
        return {
            "task_id": goal.task_id,
            "goal": to_plain(goal),
            "strategy": "observe -> plan -> policy gate -> tool call -> observe -> verify",
            "tool_registry": self.registry.list(),
            "llm_contribution": contribution,
            "stages": [{"stage": stage, "tool": self.STAGE_TO_TOOL.get(stage, "inspect_log"), "status": result.get("status", "")} for stage, result in results.items() if isinstance(result, dict)],
        }

    def _write_plan_revisions(self, path: Path, steps: List[Dict]) -> None:
        revisions = []
        for step in steps:
            if step.get("critique", {}).get("decision") in ("reject", "revise") or step.get("observation", {}).get("status") in ("failed", "uncertain"):
                revisions.append({
                    "step_id": step["step_id"],
                    "reason": step.get("critique", {}).get("critique") or "stage did not pass",
                    "next_step": step.get("next_step"),
                    "created_at": utc_now_iso(),
                })
        self._write_jsonl(path, revisions)

    def _write_jsonl(self, path: Path, rows: List[Dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(to_plain(row), ensure_ascii=False) + "\n")

    def _llm_decision(self, stage: str, result: Dict) -> Dict:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        if stage == "analyze":
            return data.get("agent_decision") or {}
        return data.get("agent_diagnosis") or data.get("llm_verify_planner") or {}

    def _policy_result(self, result: Dict) -> Dict:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        loop = data.get("agent_loop") if isinstance(data.get("agent_loop"), dict) else {}
        if loop:
            return {"allowed": bool(loop.get("policy_allowed")), "executed_action_count": loop.get("executed_action_count", 0)}
        return {}

    def _hypotheses(self, result: Dict) -> List[Dict]:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        diagnosis = data.get("diagnosis") if isinstance(data.get("diagnosis"), dict) else {}
        agent_diagnosis = data.get("agent_diagnosis") if isinstance(data.get("agent_diagnosis"), dict) else {}
        hypotheses = []
        if diagnosis:
            hypotheses.append({"source": "deterministic", "hypothesis": diagnosis.get("root_cause") or diagnosis.get("category", ""), "confidence": diagnosis.get("confidence", 0)})
        if agent_diagnosis:
            parsed = agent_diagnosis.get("diagnosis") if isinstance(agent_diagnosis.get("diagnosis"), dict) else {}
            hypotheses.append({"source": "llm", "hypothesis": parsed.get("root_cause") or agent_diagnosis.get("summary", ""), "confidence": agent_diagnosis.get("confidence", 0)})
        return hypotheses

    def _next_step(self, stage: str, status: str) -> str:
        if stage == "verify" and status in ("pass", "passed"):
            return "stop"
        if status in ("failed", "uncertain"):
            return "repair"
        return "continue"
