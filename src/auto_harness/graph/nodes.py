"""Graph nodes for the LangGraph deployment StateGraph.

Each node does ONE responsibility and returns a state delta.
Nodes never modify the input state directly — they return a dict
that LangGraph merges according to the state's reducer annotations.

GraphNodeDependencies holds all injectable callables so that nodes
can be tested with mocks.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_harness.agent_runtime.deployment_plan import DeploymentPlan
from auto_harness.graph.failure import FailureObserver
from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


@dataclass
class GraphNodeDependencies:
    """Dependencies injected into graph nodes."""
    build_snapshot: Any
    build_replan_input: Any
    determine_resume_stage: Any
    merge_analysis: Any
    planner: Any
    parser: Any
    policy_gate: Any
    compiler: Any
    stage_executor: Any
    artifact_writer_factory: Any
    runtime_config: Any
    diagnoser_factory: Any = None
    failure_observer: Any = None
    repair_planner: Any = None
    repair_policy: Any = None
    repair_applier: Any = None
    repair_loop: Any = None
    repair_overlay: Any = None
    recovery_adapter: Any = None
    runtime_policy_factory: Any = None
    route_skills: Any = None
    fault_injector: Any = None


def stage_data(state, stage):
    """Extract stage result data from the state."""
    result = state.get("stage_results", {}).get(stage, {})
    data = result.get("data", {}) if isinstance(result, dict) else {}
    return data if isinstance(data, dict) else {}


def _sanitize_error(exc):
    """Sanitize exception for state storage — no secrets or full prompts."""
    msg = str(exc)[:500]
    import re
    msg = re.sub(r'(api[_-]?key|token|secret|password|credential)["\s:=]+\S+', r'\1=***', msg, flags=re.IGNORECASE)
    return msg


def effective_stage_hints(state):
    """Compute effective stage hints from compiled analysis and repair overlay."""
    analysis = state.get("compiled_analysis", {})
    hints = dict(analysis.get("verify_hint") or {})
    overlay = state.get("repair_overlay", {})
    for item in overlay.get("verify_hints") or []:
        if isinstance(item, dict):
            hints.update(item.get("verify_hint", item))
    return hints


def make_recovery_gate_node(stage, deps):
    """Create a recovery gate node for a side-effect stage."""
    def recovery_gate(state):
        adapter = deps.recovery_adapter
        if not adapter:
            return {
                "recovery_stage": stage,
                "recovery_decision": "execute",
                "pending_operation_id": "",
                "recovery_result": {},
                "recovery_skip_stage": False,
                "current_stage": "recover_%s" % stage,
            }
        decision = adapter.prepare_or_reconcile(state, stage)
        update = {
            "recovery_stage": stage,
            "recovery_decision": decision.decision,
            "pending_operation_id": decision.operation.get("operation_id", ""),
            "recovery_result": decision.reconcile_result,
            "pending_operation": decision.operation,
            "current_stage": "recover_%s" % stage,
        }
        if decision.decision == "reuse":
            hydrated = decision.hydrated_stage_result
            if hydrated:
                results = dict(state.get("stage_results", {}))
                results[stage] = hydrated
                update["stage_results"] = results
                update["recovery_skip_stage"] = True
                if stage == "verify" and hydrated.get("status") in ("pass", "passed"):
                    update["verify_status"] = hydrated["status"]
        elif decision.decision == "stop":
            update["stop_reason"] = decision.stop_reason or "recovery_gate_blocked"
        elif decision.decision == "approval":
            update["approval_kind"] = "recovery"
            update["approval_resume_target"] = stage
            operation_id = decision.operation.get("operation_id", "")
            if operation_id:
                existing = state.get("pending_approval")
                if existing and existing.get("operation_id") == operation_id:
                    # Reuse existing request on checkpoint replay
                    update["pending_approval"] = existing
                else:
                    from auto_harness.graph.approval import build_approval_request
                    update["pending_approval"] = build_approval_request(
                        approval_id="recovery-%s-%s" % (stage, operation_id[:8]),
                        operation_id=operation_id,
                        approval_kind="recovery",
                        requested_action="cleanup_then_retry",
                        risk="high",
                        reason=decision.stop_reason or "manual_recovery_required",
                    )
            else:
                update["stop_reason"] = "approval_operation_id_missing"
        return update
    return recovery_gate


def make_stage_node(stage, deps):
    """Create a node function for a specific pipeline stage."""
    SIDE_EFFECT_STAGES = {"env_deploy", "model_prepare", "runner"}

    def execute_stage(state):
        if stage in SIDE_EFFECT_STAGES:
            recovery_stage = state.get("recovery_stage", "")
            recovery_decision = state.get("recovery_decision", "")
            if recovery_stage != stage:
                return {"stop_reason": "recovery_gate_missing", "current_stage": stage}
            if state.get("recovery_skip_stage"):
                return {"current_stage": stage, "node_history": [{"node": stage, "status": "skipped_recovery_reuse", "at": utc_now_iso()}]}
            if recovery_decision not in {"execute", "continue", "retry"}:
                return {"stop_reason": "recovery_execution_not_allowed", "current_stage": stage}
            if deps.fault_injector:
                deps.fault_injector.raise_if_configured(
                    run_dir=Path(state["run_dir"]),
                    task_id=state["task_id"],
                    stage=stage,
                    window="before_side_effect",
                    operation_id=state.get("pending_operation_id", ""),
                )

        deterministic_analysis = stage_data(state, "analyze")
        analysis = deps.merge_analysis(deterministic_analysis, state.get("compiled_analysis", {}))
        resource_data = stage_data(state, "resource_plan")
        deploy_analysis = stage_data(state, "env_solve").get("analysis", analysis)
        runner_data = stage_data(state, "runner")
        current_overlay = state.get("repair_overlay", {})
        hints = effective_stage_hints(state)

        executed = deps.stage_executor.execute_stage(
            task_id=state["task_id"], run_dir=Path(state["run_dir"]),
            repo_dir=Path(state["repo_dir"]), stage=stage, state=state,
            analysis=analysis, resource_data=resource_data,
            deploy_analysis=deploy_analysis, runner_data=runner_data,
            dry_run=state["dry_run"], stage_hints=hints,
            repair_overlay=current_overlay,
            runtime_policy=state.get("runtime_policy", {}),
            skill_context=state.get("skill_contexts", {}).get(stage, {}),
        )
        results = dict(state.get("stage_results", {}))
        results[stage] = executed.result or {"status": executed.after_status, "error": executed.error, "data": {}}
        update = {
            "current_stage": stage,
            "stage_results": results,
            "node_history": [{"node": stage, "status": executed.after_status, "at": utc_now_iso()}],
        }
        if stage == "verify":
            update["verify_status"] = executed.after_status
            update["verify_evidence_paths"] = list(executed.evidence_paths)
        if executed.after_status in ("failed", "uncertain"):
            update["failed_stage"] = stage

        if stage in SIDE_EFFECT_STAGES and deps.recovery_adapter:
            if executed.after_status in ("failed", "uncertain"):
                if deps.fault_injector:
                    deps.fault_injector.raise_if_configured(
                        run_dir=Path(state["run_dir"]),
                        task_id=state["task_id"],
                        stage=stage,
                        window="after_side_effect_before_commit",
                        operation_id=state.get("pending_operation_id", ""),
                    )
                deps.recovery_adapter.fail(state, stage, executed.error or "stage_failed")
            else:
                result_for_journal = executed.result or {"status": executed.after_status, "data": {}}
                result_artifact = deps.recovery_adapter.persist_result(
                    state, stage, result_for_journal
                )
                if deps.fault_injector:
                    deps.fault_injector.raise_if_configured(
                        run_dir=Path(state["run_dir"]),
                        task_id=state["task_id"],
                        stage=stage,
                        window="after_side_effect_before_commit",
                        operation_id=state.get("pending_operation_id", ""),
                    )
                deps.recovery_adapter.commit(
                    state,
                    stage,
                    result_for_journal,
                    artifact_path=result_artifact,
                )
                if deps.fault_injector:
                    deps.fault_injector.raise_if_configured(
                        run_dir=Path(state["run_dir"]),
                        task_id=state["task_id"],
                        stage=stage,
                        window="after_commit_before_checkpoint",
                        operation_id=state.get("pending_operation_id", ""),
                    )
        return update
    return execute_stage


def merge_plan_analysis(deterministic, compiled):
    """Merge deterministic analysis with compiled plan analysis."""
    merged = dict(deterministic or {})
    plan_owned_keys = (
        "install_plan", "run_candidates", "verify_hint", "environment_strategy",
        "selected_candidate", "selection_source", "llm_plan", "llm_candidates",
        "merged_candidates", "llm_required_reason",
    )
    for key in plan_owned_keys:
        if key in compiled:
            merged[key] = compiled[key]
    return merged


class DeploymentGraphNodes:
    """Collection of node functions for the deployment StateGraph."""

    def __init__(self, deps):
        self.deps = deps

    def _artifacts(self, state):
        return self.deps.artifact_writer_factory(Path(state["run_dir"]))

    def build_snapshot(self, state):
        snapshot = self.deps.build_snapshot(state)
        path = self._artifacts(state).write_project_snapshot(snapshot)
        # Update state with skill/memory fields from snapshot (Task 9)
        route_path = Path(state["run_dir"]) / "skills" / "routes" / "plan.json"
        update = {
            "snapshot_path": str(path),
            "current_stage": "snapshot",
            "memory_hits": snapshot.get("memory_hits", []),
            "selected_skills": {
                **state.get("selected_skills", {}),
                "plan": snapshot.get("selected_skills", []),
            },
            "skill_contexts": {
                **state.get("skill_contexts", {}),
                "plan": snapshot.get("skill_context", {}),
            },
            "skill_route_paths": {
                **state.get("skill_route_paths", {}),
                "plan": str(route_path) if route_path.exists() else "",
            },
        }
        return update

    def route_skills(self, state, stage):
        """Route skills for a pipeline stage (Task 10).

        Routes skills for verify or repair stages, writes a route artifact,
        and updates state with selected skills, skill contexts, and route paths.
        Returns a stop_reason dict if route_skills dependency is unavailable.
        """
        if not self.deps.route_skills:
            return {"stop_reason": "skill_router_unavailable"}

        analysis = state.get("compiled_analysis", {})
        failure_category = (
            state.get("diagnosis", {}).get("category", "")
            if isinstance(state.get("diagnosis"), dict)
            else ""
        )
        allowed_tools = {
            "verify": [
                "probe_http",
                "discover_gradio_api",
                "discover_openapi_schema",
                "probe_browser_dom",
            ],
            "repair": [
                "install_package",
                "install_conda_package",
                "update_verify_hint",
                "rerun_from_stage",
                "select_run_candidate",
            ],
        }.get(stage, [])

        routed = self.deps.route_skills(
            stage=stage,
            analysis=analysis,
            failure_category=failure_category,
            allowed_tools=allowed_tools,
        )

        route_dir = Path(state["run_dir"]) / "skills" / "routes"
        route_dir.mkdir(parents=True, exist_ok=True)
        suffix = ""
        if stage == "repair":
            suffix = "_%s" % (int(state.get("repair_count", 0)) + 1)
        path = route_dir / ("%s%s.json" % (stage, suffix))
        write_json(path, routed["artifact"])

        selected = dict(state.get("selected_skills", {}))
        selected[stage] = routed["selected_skills"]
        contexts = dict(state.get("skill_contexts", {}))
        contexts[stage] = routed["skill_context"]
        paths = dict(state.get("skill_route_paths", {}))
        paths[stage] = str(path)

        return {
            "selected_skills": selected,
            "skill_contexts": contexts,
            "skill_route_paths": paths,
            "current_stage": "route_%s_skills" % stage,
        }

    def plan(self, state):
        snapshot = read_json(Path(state["snapshot_path"]))
        try:
            raw = self.deps.planner.plan(snapshot, skill_context=snapshot.get("skill_context", {}))
        except Exception as exc:
            return {"llm_error": _sanitize_error(exc), "stop_reason": "llm_plan_failed", "raw_plan_path": "", "current_stage": "plan"}
        path = self._artifacts(state).write_raw_plan({"raw_text": raw.text[:10000]})
        return {"raw_plan_path": str(path), "current_stage": "plan"}

    def parse(self, state):
        raw = read_json(Path(state["raw_plan_path"]))
        try:
            parsed = self.deps.parser.parse(raw.get("raw_text", ""))
            stop_reason = "" if parsed.status == "ok" else "plan_not_ok"
        except ValueError as exc:
            parsed = DeploymentPlan(status="invalid", summary=str(exc))
            stop_reason = "plan_parse_failed"
        path = self._artifacts(state).write_parsed_plan(parsed.to_dict())
        return {"parsed_plan_path": str(path), "stop_reason": stop_reason}

    def policy(self, state):
        parsed = read_json(Path(state["parsed_plan_path"]))
        snapshot = read_json(Path(state["snapshot_path"]))
        result = self.deps.policy_gate.validate(parsed, snapshot, runtime_policy=state.get("runtime_policy", {}), config=self.deps.runtime_config)
        path = self._artifacts(state).write_policy_result(result)
        return {"policy_result_path": str(path), "stop_reason": "" if result.get("allowed") else "policy_rejected"}

    def compile(self, state):
        parsed = read_json(Path(state["parsed_plan_path"]))
        policy = read_json(Path(state["policy_result_path"]))
        compiled = self.deps.compiler.compile(policy.get("normalized_plan", parsed))
        path = self._artifacts(state).write_effective_plan(compiled.get("effective_plan", {}))
        return {"effective_plan_path": str(path), "compiled_analysis": compiled.get("analysis", {})}

    def select_resume(self, state):
        if int(state.get("replan_count", 0)) == 0:
            return {"resume_from_stage": "analyze"}
        previous_path = state.get("previous_plan_path", "")
        if not previous_path:
            return {"resume_from_stage": "analyze", "errors": [{"node": "select_resume", "error": "previous_plan_missing"}]}
        previous = read_json(Path(previous_path))
        current = read_json(Path(state["parsed_plan_path"]))
        requested = self.deps.determine_resume_stage(previous, current)
        allowed = {"analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy", "model_prepare", "runner", "verify"}
        selected = requested if requested in allowed else "analyze"
        revision = int(state.get("replan_count", 0))
        revision_path = Path(state["run_dir"]) / "reports" / "replans" / ("replan_%s.revision.json" % revision)
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(revision_path, {"revision": revision, "trigger_stage": state.get("failed_stage", ""), "previous_plan_id": previous.get("plan_id", ""), "new_plan_id": current.get("plan_id", ""), "resume_from": selected, "policy_allowed": True})
        revision_paths = dict(state.get("plan_revision_paths", {}))
        revision_paths[str(revision)] = str(revision_path)
        return {"resume_from_stage": selected, "plan_revision_paths": revision_paths}

    def replan(self, state):
        snapshot, previous_plan, failure = self.deps.build_replan_input(state)
        # Route replan skills (Task 10)
        skill_context = {}
        if self.deps.route_skills:
            try:
                routed = self.deps.route_skills(
                    stage="replan",
                    analysis=snapshot.get("detected_signals", {})
                    if isinstance(snapshot, dict) else {},
                    failure_category=(
                        state.get("diagnosis", {}).get("category", "")
                        if isinstance(state.get("diagnosis"), dict) else ""
                    ),
                    allowed_tools=[],
                )
                skill_context = routed.get("skill_context", {})
                # Write replan route artifact
                route_dir = Path(state["run_dir"]) / "skills" / "routes"
                route_dir.mkdir(parents=True, exist_ok=True)
                write_json(route_dir / "replan.json", routed.get("artifact", {}))
                selected = dict(state.get("selected_skills", {}))
                selected["replan"] = routed.get("selected_skills", [])
                contexts = dict(state.get("skill_contexts", {}))
                contexts["replan"] = skill_context
            except Exception:
                pass
            failure["skill_context"] = skill_context
        try:
            raw = self.deps.planner.replan(
                snapshot, previous_plan, failure,
                skill_context=skill_context,
            )
        except Exception as exc:
            return {"llm_error": _sanitize_error(exc), "stop_reason": "llm_replan_failed", "raw_plan_path": "", "previous_plan_path": "", "replan_count": int(state.get("replan_count", 0)) + 1}
        revision = int(state.get("replan_count", 0)) + 1
        replans_dir = Path(state["run_dir"]) / "reports" / "replans"
        replans_dir.mkdir(parents=True, exist_ok=True)
        path = replans_dir / ("replan_%s.raw.json" % revision)
        previous_path = replans_dir / ("replan_%s.previous.json" % revision)
        write_json(previous_path, previous_plan)
        write_json(path, {"raw_text": raw.text[:10000], "revision": revision})
        update = {
            "raw_plan_path": str(path),
            "previous_plan_path": str(previous_path),
            "replan_count": revision,
            "stop_reason": "",
        }
        if skill_context:
            update["skill_contexts"] = {
                **state.get("skill_contexts", {}),
                "replan": skill_context,
            }
            update["selected_skills"] = {
                **state.get("selected_skills", {}),
                "replan": selected.get("replan", []) if 'selected' in dir() else [],
            }
        return update

    def observe_failure(self, state):
        observer = self.deps.failure_observer or FailureObserver()
        context = observer.build(state)
        signature = observer.compute_signature(context)
        same_count = int(state.get("same_failure_count", 0))
        prev_signature = state.get("failure_signature", "")
        if signature == prev_signature:
            same_count += 1
        else:
            same_count = 1
        run_dir = Path(state["run_dir"])
        failures_dir = run_dir / "reports" / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        write_json(failures_dir / ("%s.json" % same_count), {"failure_context": context, "failure_signature": signature, "same_failure_count": same_count})
        update = {"failure_context": context, "failure_signature": signature, "same_failure_count": same_count, "current_stage": "observe_failure"}
        max_same = int(self.deps.runtime_config.langgraph_max_same_failure) if self.deps.runtime_config and hasattr(self.deps.runtime_config, "langgraph_max_same_failure") else 2
        if same_count >= max_same:
            update["stop_reason"] = "same_failure_limit_reached"
        return update

    def diagnose(self, state):
        if not self.deps.diagnoser_factory:
            return {"diagnosis": {"status": "unavailable", "accepted_actions": []}, "diagnose_count": int(state.get("diagnose_count", 0)) + 1, "current_stage": "diagnose"}
        diagnose_count = int(state.get("diagnose_count", 0))
        max_diagnoses = int(state.get("max_diagnoses", 2))
        if diagnose_count >= max_diagnoses:
            return {"stop_reason": "diagnose_limit_reached", "current_stage": "diagnose"}
        from auto_harness.agent.schemas import AgentObservation
        snapshot = {}
        snapshot_path = state.get("snapshot_path", "")
        if snapshot_path:
            try:
                snapshot = read_json(Path(snapshot_path))
            except (OSError, ValueError):
                pass
        observation = AgentObservation(
            task_id=state["task_id"], stage=state.get("failed_stage", ""),
            repo_dir=state.get("repo_dir", ""), file_tree=snapshot.get("file_tree", []),
            selected_files=snapshot.get("selected_files", {}),
            deterministic_result=state.get("failure_context", {}),
            previous_results=state.get("stage_results", {}),
            memory_hits=snapshot.get("memory_hits", []),
            selected_skills=snapshot.get("selected_skills", []),
            runtime_policy=state.get("runtime_policy", {}),
            allowed_action_types=[
                "install_package",
                "install_pip_package",
                "pin_dependency",
                "install_conda_package",
                "update_verify_hint",
                "rerun_from_stage",
                "set_env_var_name_only",
            ],
            extra={"compiled_analysis": state.get("compiled_analysis", {}), "replan_count": state.get("replan_count", 0), "repair_count": state.get("repair_count", 0)},
        )
        try:
            diagnoser = self.deps.diagnoser_factory(state)
            diagnosis = diagnoser.diagnose(observation)
        except Exception as exc:
            return {"llm_error": _sanitize_error(exc), "stop_reason": "llm_diagnosis_failed", "diagnose_count": diagnose_count + 1, "agent_call_count": int(state.get("agent_call_count", 0)) + 1, "current_stage": "diagnose"}
        run_dir = Path(state["run_dir"])
        diag_dir = run_dir / "reports" / "diagnoses"
        diag_dir.mkdir(parents=True, exist_ok=True)
        new_count = diagnose_count + 1
        write_json(diag_dir / ("diagnosis_%s.json" % new_count), diagnosis)
        trace_paths = []
        if hasattr(diagnoser, 'trace_writer') and diagnoser.trace_writer:
            trace_dir = Path(state["run_dir"]) / "logs" / "agent_calls"
            if trace_dir.exists():
                trace_paths = [str(f) for f in sorted(trace_dir.glob("*.json"))[-1:]]
        return {"diagnosis": diagnosis, "diagnosis_path": str(diag_dir / ("diagnosis_%s.json" % new_count)), "diagnose_count": new_count, "agent_call_count": int(state.get("agent_call_count", 0)) + 1, "agent_trace_paths": trace_paths, "current_stage": "diagnose"}

    def generate_agent_contribution(self, state):
        plan_calls = 1 if state.get("raw_plan_path") else 0
        replan_count = int(state.get("replan_count", 0))
        diagnose_calls = int(state.get("diagnose_count", 0))
        repair_attempts = int(state.get("repair_count", 0))
        agent_call_count = int(state.get("agent_call_count", 0))
        policy_result = state.get("repair_policy_result", {})
        decisions = policy_result.get("decisions", [])
        rejected_actions = sum(1 for d in decisions if isinstance(d, dict) and not d.get("allowed"))
        accepted_actions = sum(1 for d in decisions if isinstance(d, dict) and d.get("allowed"))
        recovery_events = state.get("recovery_events", [])
        recovery_decisions = {}
        for event in recovery_events:
            if isinstance(event, dict):
                decision = event.get("decision", "unknown")
                recovery_decisions[decision] = recovery_decisions.get(decision, 0) + 1
        contribution = {
            "controller": "langgraph", "llm_required": state.get("llm_required", True),
            "provider": state.get("llm_provider", ""), "plan_calls": plan_calls,
            "diagnose_calls": diagnose_calls, "replan_calls": replan_count,
            "agent_verify_calls": 0, "accepted_actions": accepted_actions,
            "rejected_actions": rejected_actions, "repair_attempts": repair_attempts,
            "recovery_decisions": recovery_decisions,
            "final_verify_status": state.get("verify_status", ""),
            "llm_claimed_success": False, "success_decided_by": "evidence_gate",
            "agent_call_count": agent_call_count, "generated_at": utc_now_iso(),
        }
        run_dir = Path(state["run_dir"])
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / "agent_contribution.json", contribution)
        return {"current_stage": "generate_contribution"}

    def _write_recovery_summary(self, state):
        """Phase 10: write recovery_summary.json from actual recovery events."""
        recovery_events = state.get("recovery_events", [])
        recovery_capabilities = state.get("recovery_capabilities", {})
        summary = {
            "total_recovery_events": len(recovery_events),
            "capabilities": recovery_capabilities,
            "decisions": {},
            "operations": [],
        }
        for event in recovery_events:
            if isinstance(event, dict):
                decision = event.get("decision", "unknown")
                summary["decisions"][decision] = summary["decisions"].get(decision, 0) + 1
                op_id = event.get("operation_id", "")
                if op_id:
                    summary["operations"].append(op_id)
        run_dir = Path(state["run_dir"])
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / "recovery_summary.json", summary)

    def finalize_learning(self, state):
        """finalize_learning node: record verified memory and skill outcomes.

        Only runs when verify passes (or after verify failed for neutral/harmful
        outcome recording). Does NOT write verified success memory for
        verify failures.

        Writes:
        - reports/verified_memory.json
        - reports/skill_outcomes.json
        """
        from auto_harness.memory.success import VerifiedMemoryRecorder
        from auto_harness.memory.outcomes import SkillOutcomeRecorder

        run_dir = Path(state["run_dir"])
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Assemble pipeline_results from state
        pipeline_results = dict(state.get("stage_results", {}))
        repair_apply_result = dict(state.get("repair_apply_result", {}) or {})
        repair_plan = dict(state.get("repair_plan", {}) or {})

        # Finalize the latest repair attempt only after the post-repair verify.
        repair_count = int(state.get("repair_count", 0))
        if repair_count > 0 and repair_apply_result.get("status") == "applied":
            from auto_harness.repair.evidence import (
                build_repair_attempt,
                compute_fresh_trace,
                compute_repair_verified,
                is_effective_repair_action,
            )

            verify = pipeline_results.get("verify", {})
            verify_data = verify.get("data", {}) if isinstance(verify, dict) else {}
            after_trace = str(verify_data.get("trace_id") or "")
            before_trace = str(
                repair_apply_result.get("verification_trace_before") or ""
            )
            checks = verify_data.get("checks", [])
            evidence_contains_after_trace = any(
                isinstance(check, dict)
                and check.get("status") in ("pass", "passed")
                and after_trace
                and after_trace in json.dumps(check, ensure_ascii=False)
                for check in checks
            )
            action_results = list(
                repair_apply_result.get("action_results") or []
            )
            effective_action_count = sum(
                1
                for item in action_results
                if is_effective_repair_action(item)
            )
            metadata_only_count = sum(
                1
                for item in action_results
                if item.get("metadata_only")
                or item.get("status") == "metadata_only"
            )
            fresh_trace = compute_fresh_trace(before_trace, after_trace)
            resume_executed = bool(state.get("repair_resume_executed"))
            repair_verified = compute_repair_verified(
                effective_action_count=effective_action_count,
                resume_executed=resume_executed,
                verify_status_after=verify.get("status", ""),
                evidence_contains_after_trace=evidence_contains_after_trace,
                fresh_trace=fresh_trace,
            )
            repair_apply_result.update({
                "effective_action_count": effective_action_count,
                "metadata_only_count": metadata_only_count,
                "repair_effective": effective_action_count > 0,
                "repair_verified": repair_verified,
                "resume_executed": resume_executed,
                "verification_trace_id": after_trace,
                "fresh_trace": fresh_trace,
                "evidence_contains_after_trace": evidence_contains_after_trace,
            })
            repairs_dir = run_dir / "repairs"
            apply_path = state.get("repair_apply_path", "")
            if apply_path:
                write_json(Path(apply_path), repair_apply_result)
            write_json(
                repairs_dir / "repair_apply_result.json",
                repair_apply_result,
            )
            attempt = build_repair_attempt(
                attempt=repair_count,
                failure_signature_before=state.get(
                    "failure_signature", ""
                ),
                diagnosis_path=state.get("diagnosis_path", ""),
                plan_path=state.get("repair_plan_path", ""),
                policy_path=state.get("repair_policy_path", ""),
                apply_path=apply_path,
                resume_from_stage=state.get("repair_resume_stage", ""),
                effective_action_count=effective_action_count,
                metadata_only_count=metadata_only_count,
                verify_status_after=verify.get("status", ""),
                verification_trace_id=after_trace,
                fresh_trace=fresh_trace,
                repair_verified=repair_verified,
            )
            write_json(
                repairs_dir / ("attempt_%s.json" % repair_count),
                attempt,
            )

        # Record verified memory if conditions met
        verified_memory_path = ""
        try:
            config = self.deps.runtime_config
            if config:
                recorder = VerifiedMemoryRecorder(Path(config.memory_dir))
                result = recorder.record_if_verified(
                    run_dir,
                    pipeline_results,
                    {},  # agent_metrics from state if available
                    repair_apply_result=repair_apply_result,
                    repair_plan=repair_plan,
                )
                if result:
                    verified_memory_path = str(reports_dir / "verified_memory.json")
        except Exception:
            pass  # Learning recording must not block the pipeline

        # Record skill outcomes
        skill_outcome_paths = []
        try:
            config = self.deps.runtime_config
            if config:
                outcome_recorder = SkillOutcomeRecorder(Path(config.memory_dir))
                outcome_records = []
                for stage, skills in state.get("selected_skills", {}).items():
                    if not skills:
                        continue
                    outcome = outcome_recorder.record_run(
                        run_id=state.get("task_id", ""),
                        stage=stage,
                        selected_skills=skills,
                        result=pipeline_results.get(stage, {}),
                        agent_metadata={
                            "final_verify_status": state.get("verify_status", ""),
                            "trace_verified": bool(
                                repair_apply_result.get("repair_verified")
                            ),
                        },
                    )
                    outcome_records.extend(outcome.get("records", []))
                outcome_path = str(reports_dir / "skill_outcomes.json")
                write_json(Path(outcome_path), {
                    "task_id": state.get("task_id", ""),
                    "recorded_count": len(outcome_records),
                    "records": outcome_records,
                })
                skill_outcome_paths.append(outcome_path)
        except Exception:
            pass  # Outcome recording must not block the pipeline

        return {
            "current_stage": "finalize_learning",
            "verified_memory_path": verified_memory_path,
            "skill_outcome_paths": skill_outcome_paths,
        }

    def report(self, state):
        self._artifacts(state).write_pipeline_results(state.get("stage_results", {}))
        self.generate_agent_contribution(state)
        self._write_recovery_summary(state)
        return {"current_stage": "report"}

    def stop(self, state):
        return {"current_stage": "stop"}
