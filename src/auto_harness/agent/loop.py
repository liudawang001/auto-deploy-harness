import json
from pathlib import Path
from typing import Callable, Dict

from auto_harness.agent.diagnoser import AgentDiagnoser
from auto_harness.agent.schemas import AgentObservation
from auto_harness.agent.traces import AgentTraceWriter
from auto_harness.models.base import read_json, to_plain, write_json
from auto_harness.utils.time import compact_timestamp, utc_now_iso


class AgentLoopController:
    """Coordinates failure observation, LLM diagnosis, repair policy, action apply, and resume decision."""

    def __init__(
        self,
        config,
        store,
        memory,
        repair_planner,
        repair_policy,
        repair_applier,
        repair_loop,
        provider_factory: Callable,
    ) -> None:
        self.config = config
        self.store = store
        self.memory = memory
        self.repair_planner = repair_planner
        self.repair_policy = repair_policy
        self.repair_applier = repair_applier
        self.repair_loop = repair_loop
        self.provider_factory = provider_factory

    def handle_stage_result(
        self,
        task_id: str,
        stage: str,
        result,
        analysis: Dict,
        runtime_policy,
        last_safe_stage: str = None,
        memory_entry: Dict = None,
        command_runner=None,
    ) -> Dict:
        run_dir = self.store.run_dir(task_id)
        entry = memory_entry or self.memory.remember_issue(task_id, stage, result, analysis)
        if result.status not in ("failed", "uncertain") or not entry:
            return {"handled": False, "stop_reason": "success" if result.status in ("passed", "pass") else "not_recordable"}

        # Route repair skills based on failure category
        repair_skill_context = self._route_repair_skills(stage, result, analysis)

        agent_diagnosis = self._maybe_diagnose(task_id, stage, result, analysis, runtime_policy, run_dir)
        repair_plan = self.repair_planner.propose(stage, result, analysis)

        # Inject skill_context into repair plan if available
        if repair_skill_context and repair_skill_context.get("selected_skills"):
            repair_plan["skill_context"] = repair_skill_context
        # Repair Actuator Gate: LLM proposes repair tool_call through decision gate
        if self._repair_gate_enabled():
            repair_gate_result = self._apply_repair_gate(task_id, stage, result, analysis, run_dir)
            if repair_gate_result and repair_gate_result.get("gate_decision_status") == "ok":
                # Merge gate repair actions into repair plan
                gate_actions = repair_gate_result.get("actions", [])
                if gate_actions:
                    existing = list(repair_plan.get("actions", []))
                    existing.extend(gate_actions)
                    repair_plan["actions"] = existing
                    repair_plan["repair_source"] = "llm_repair_gate"
        approval = self.repair_loop.load_approval(run_dir)
        policy_result = self.repair_policy.check(repair_plan, runtime_policy, operator_approval=approval)
        effective_policy = self.repair_loop.gate(run_dir, stage, entry, repair_plan, policy_result, last_safe_stage)
        apply_result = self.repair_applier.apply(
            run_dir,
            repair_plan,
            effective_policy,
            execute=self._repair_execute_enabled(runtime_policy),
            command_runner=command_runner,
            timeout_seconds=self.config.default_timeout_seconds,
            allowed_commands=self.config.allowed_commands,
            env_context=self._env_context(analysis),
        )
        next_rerun_from = (
            (effective_policy.get("loop") or {}).get("rerun_from_effective")
            or repair_plan.get("rerun_from_effective")
            or repair_plan.get("rerun_from")
            or stage
        )
        stop_reason = self._stop_reason(result, repair_plan, effective_policy, apply_result)
        should_auto_resume = self._should_auto_resume(runtime_policy, repair_plan, effective_policy, apply_result, stop_reason)
        summary = {
            "handled": True,
            "agent_diagnosis": agent_diagnosis or result.data.get("agent_diagnosis", {}),
            "repair_plan": repair_plan,
            "policy": effective_policy,
            "apply_result": apply_result,
            "next_rerun_from": next_rerun_from,
            "should_auto_resume": should_auto_resume,
            "stop_reason": stop_reason,
            "created_at": utc_now_iso(),
        }
        result.data["agent_loop"] = self._compact_summary(summary)
        self._write_loop_trace(run_dir, stage, summary)
        return summary

    def _maybe_diagnose(self, task_id: str, stage: str, result, analysis: Dict, runtime_policy, run_dir: Path) -> Dict:
        if not (self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_log_diagnosis):
            return {}
        data = result.data if isinstance(result.data, dict) else {}
        diagnosis = data.get("diagnosis") if isinstance(data.get("diagnosis"), dict) else {}
        if diagnosis.get("category") not in (None, "", "unknown") and float(diagnosis.get("confidence") or 0) >= 0.75:
            return {}
        observation = AgentObservation(
            task_id=task_id,
            stage=stage,
            repo_dir=str(run_dir / "workspace" / "repo"),
            deterministic_result=to_plain(result),
            previous_results=self._load_previous_results(run_dir),
            memory_hits=self.memory.query(stage, analysis, limit=self.config.max_memory_items),
            runtime_policy=runtime_policy.__dict__,
            allowed_action_types=["install_package", "update_verify_hint", "request_env_var_name_only", "rerun_from_stage"],
            extra={"analysis": analysis},
        )
        diagnoser = AgentDiagnoser(self.provider_factory(), config=self.config, trace_writer=AgentTraceWriter(run_dir / "logs" / "agent_calls"))
        agent_diagnosis = diagnoser.diagnose(observation)
        result.data["agent_diagnosis"] = agent_diagnosis
        return agent_diagnosis

    def _repair_execute_enabled(self, runtime_policy) -> bool:
        return (
            self.config.agent_mode == "gated_actor"
            and self.config.agent_enable_repair_actions
            and bool(getattr(runtime_policy, "allow_dependency_install", False))
        )

    def _repair_gate_enabled(self) -> bool:
        return (
            self.config.agent_mode in ("planner", "gated_actor")
            and getattr(self.config, "agent_enable_repair_gate", False)
        )

    def _route_repair_skills(self, stage: str, result, analysis: Dict) -> Dict:
        """Route repair skills based on failure category and stage.

        Returns skill_context dict or empty dict if routing fails.
        """
        try:
            from auto_harness.skills.router import SkillRouter, SkillRouteRequest
            from auto_harness.skills.context import SkillContextBuilder

            skills_dir = getattr(self.config, 'skills_path', None)
            if not skills_dir:
                return {}
            skills_dir = Path(skills_dir)
            if not skills_dir.exists():
                return {}

            # Determine failure category from diagnosis
            failure_category = ""
            data = result.data if isinstance(result.data, dict) else {}
            diagnosis = data.get("diagnosis", {})
            if isinstance(diagnosis, dict):
                failure_category = diagnosis.get("category", "")

            router = SkillRouter(skills_dir=skills_dir)
            request = SkillRouteRequest(
                stage=stage,
                analysis=analysis,
                frameworks=analysis.get("frameworks", []),
                failure_category=failure_category,
                allowed_tools=["apply_dependency_constraint", "propose_repair"],
                mode=self.config.agent_mode,
            )
            routed = router.route(request, limit=3)
            if not routed:
                return {}

            builder = SkillContextBuilder()
            return builder.build(routed, stage=stage)
        except Exception:
            # Skill routing must not crash the repair loop
            return {}

    def _apply_repair_gate(self, task_id: str, stage: str, result, analysis: Dict, run_dir: Path) -> Dict:
        """Apply repair decision gate: LLM proposes repair action through decision gate."""
        from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
        from auto_harness.agent_runtime.stage_schemas import REPAIR_TOOLS

        data = result.data if isinstance(result.data, dict) else {}
        diagnosis = data.get("diagnosis", {}) if isinstance(data.get("diagnosis"), dict) else {}
        agent_diagnosis = data.get("agent_diagnosis", {}) if isinstance(data.get("agent_diagnosis"), dict) else {}

        observation = {
            "stage": "repair",
            "failure": {
                "stage": stage,
                "status": result.status,
                "error": str(result.error or "")[:2000],
                "summary": str(result.summary or "")[:1000],
            },
            "diagnosis": agent_diagnosis or diagnosis,
            "previous_repairs": [],
            "constraints": [
                "Only propose repair actions allowed by policy.",
                "Do not edit source files directly.",
                "Do not include secret values.",
                "Resume from a safe stage after applying repair.",
            ],
        }

        provider = self.provider_factory() if callable(self.provider_factory) else None
        if not provider:
            return {}

        gate = AgentDecisionGate(provider=provider)
        gate_result = gate.decide(
            stage="repair",
            observation=observation,
            allowed_tools=list(REPAIR_TOOLS),
            mode=self.config.agent_mode,
            run_dir=run_dir,
            max_steps=getattr(self.config, "agent_decision_gate_max_steps", 2),
        )

        # Build repair actions from gate result
        actions = []
        if gate_result.tool_call and gate_result.execution.get("applied"):
            tc = gate_result.tool_call
            if tc.get("name") == "apply_repair":
                actions.append({
                    "type": tc.get("input", {}).get("action_type", "install_package"),
                    "reason": gate_result.hypothesis,
                    "payload": tc.get("input", {}),
                    "source": "llm_repair_gate",
                })
            elif tc.get("name") == "apply_dependency_constraint":
                actions.append({
                    "type": "install_package",
                    "reason": gate_result.hypothesis,
                    "payload": tc.get("input", {}),
                    "source": "llm_repair_gate",
                })

        return {
            "gate_decision_status": gate_result.decision_status,
            "gate_policy_allowed": gate_result.policy.get("allowed", False),
            "gate_state_changed": gate_result.state_delta.get("changed", False),
            "actions": actions,
        }

    def _env_context(self, analysis: Dict) -> Dict:
        env_solution = analysis.get("env_solution") if isinstance(analysis.get("env_solution"), dict) else {}
        backend = env_solution.get("backend") or "venv"
        return {
            "backend": backend,
            "python_executable": env_solution.get("environment_python") or ".venv/bin/python",
            "environment_backend": backend,
            "environment_prefix": env_solution.get("environment_prefix") or "",
            "conda_prefix": env_solution.get("environment_prefix") or "",
        }

    def _stop_reason(self, result, repair_plan: Dict, effective_policy: Dict, apply_result: Dict) -> str:
        if result.status in ("passed", "pass"):
            return "success"
        if self._has_unsafe_request(repair_plan, effective_policy):
            return "unsafe_request"
        loop_reasons = (effective_policy.get("loop") or {}).get("reasons") or []
        if "repair attempt limit reached" in loop_reasons:
            return "max_iterations"
        if not effective_policy.get("allowed"):
            return "policy_rejected"
        if not repair_plan.get("actions"):
            return "no_progress"
        if self._action_failed(apply_result):
            return "action_failed"
        return ""

    def _should_auto_resume(self, runtime_policy, repair_plan: Dict, effective_policy: Dict, apply_result: Dict, stop_reason: str) -> bool:
        if stop_reason:
            return False
        if not (
            self.config.agent_mode == "gated_actor"
            and self.config.agent_enable_log_diagnosis
            and self.config.agent_enable_repair_actions
            and self.config.agent_auto_resume_after_repair
        ):
            return False
        if not effective_policy.get("allowed"):
            return False
        if any((action.get("requires") or {}).get("source_edit") or (action.get("requires") or {}).get("operator_secret") for action in repair_plan.get("actions", [])):
            return False
        if any(item.get("executed") and int(item.get("exit_code") or 0) == 0 for item in apply_result.get("action_results", [])):
            return True
        metadata_actions = {"update_verify_hint", "rerun_from_stage"}
        return any(action.get("type") in metadata_actions for action in repair_plan.get("actions", []))

    def _has_unsafe_request(self, repair_plan: Dict, effective_policy: Dict) -> bool:
        unsafe_terms = ("source edit", "operator secret", "unsafe", "shell", "external URL")
        text = json.dumps({"plan": repair_plan, "policy": effective_policy}, ensure_ascii=False).lower()
        return any(term.lower() in text for term in unsafe_terms)

    def _action_failed(self, apply_result: Dict) -> bool:
        for item in apply_result.get("action_results", []):
            if item.get("executed") and int(item.get("exit_code") or 0) != 0:
                return True
            if item.get("status") == "rejected":
                return True
        return False

    def _compact_summary(self, summary: Dict) -> Dict:
        policy = summary.get("policy") or {}
        apply_result = summary.get("apply_result") or {}
        return {
            "handled": summary.get("handled"),
            "next_rerun_from": summary.get("next_rerun_from"),
            "should_auto_resume": summary.get("should_auto_resume"),
            "stop_reason": summary.get("stop_reason"),
            "agent_diagnosis_status": (summary.get("agent_diagnosis") or {}).get("status"),
            "repair_action_count": len((summary.get("repair_plan") or {}).get("actions") or []),
            "policy_allowed": bool(policy.get("allowed")),
            "executed_action_count": int(apply_result.get("executed_action_count") or 0),
            "created_at": summary.get("created_at"),
        }

    def _write_loop_trace(self, run_dir: Path, stage: str, summary: Dict) -> None:
        path = run_dir / "logs" / "agent_loop" / ("%s_%s.json" % (stage, compact_timestamp()))
        write_json(path, summary)

    def _load_previous_results(self, run_dir: Path) -> Dict:
        pipeline = self._read_optional(run_dir / "reports" / "pipeline_results.json")
        if isinstance(pipeline, dict):
            return pipeline
        return {}

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
