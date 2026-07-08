from pathlib import Path
from typing import Dict, List

from auto_harness.agent_runtime.critic import AgentCritic
from auto_harness.agent_runtime.schemas import AgentGoal, AgentRuntimeStep
from auto_harness.models.base import to_plain, write_json
from auto_harness.tools import ToolRegistry
from auto_harness.utils.time import utc_now_iso


class AgentRuntime:
    """Records an explicit observe-plan-tool-observe loop around the existing runtime."""

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

    def run(self, goal: AgentGoal, run_dir: Path, results: Dict, contribution: Dict = None) -> Dict:
        run_dir = Path(run_dir)
        steps = self._steps(goal, results)
        self._write_jsonl(run_dir / "agent_steps.jsonl", steps)
        state = self._state(goal, steps, results, contribution or {})
        plan = self._plan(goal, results, contribution or {})
        write_json(run_dir / "agent_state.json", state)
        write_json(run_dir / "agent_plan.json", plan)
        self._write_plan_revisions(run_dir / "agent_plan_revisions.jsonl", steps)
        return {"state": state, "plan": plan, "step_count": len(steps)}

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
                import json
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
