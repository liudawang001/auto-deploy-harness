from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json
from auto_harness.models.result import StageResult


class ReportGenerator:
    def generate(self, run_dir: Path, task: Dict, results: Dict[str, Dict], execution_audit: Dict = None) -> StageResult:
        report_path = run_dir / "reports" / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# AI-Auto-Harness Deployment Report",
            "",
            "## Project",
            "",
            "- Name: `%s`" % task.get("project", {}).get("name", ""),
            "- Repo: `%s`" % task.get("project", {}).get("repo_url", ""),
            "",
            "## Stage Results",
            "",
        ]
        for stage, result in results.items():
            context = result.get("data", {}).get("control_context", {}) if isinstance(result.get("data"), dict) else {}
            skill_names = [item.get("name") for item in context.get("selected_skills", [])]
            memory_ids = [item.get("id") for item in context.get("memory_hits", [])]
            lines.extend([
                "### %s" % stage,
                "",
                "- Status: `%s`" % result.get("status", ""),
                "- Summary: %s" % result.get("summary", ""),
                "- Skills: %s" % (", ".join("`%s`" % name for name in skill_names if name) or "`none`"),
                "- Memory hits: %s" % (", ".join("`%s`" % item for item in memory_ids if item) or "`none`"),
                "",
            ])
        verify = results.get("verify", {}).get("data", {})
        run_selection = self._run_candidate_selection(results)
        if run_selection:
            lines.extend([
                "## Run Candidate Selection",
                "",
                "- Command: `%s`" % " ".join(str(part) for part in run_selection.get("cmd", [])),
                "- Score: `%.2f`" % float(run_selection.get("score") or 0),
                "- Selected by: `%s`" % run_selection.get("selected_by", ""),
                "- Reasons:",
            ])
            for reason in run_selection.get("score_reasons") or []:
                lines.append("  - %s" % reason)
            lines.append("")
        if verify:
            lines.extend([
                "## Verify",
                "",
                "- Final status: `%s`" % verify.get("status", ""),
                "- Trace ID: `%s`" % verify.get("trace_id", ""),
                "- Next action: `%s`" % verify.get("next_action", ""),
                "",
            ])
        repair_rerun = self._repair_rerun_summary(run_dir, results)
        if repair_rerun:
            lines.extend([
                "## Repair Rerun Decision",
                "",
                "- Proposed rerun_from: `%s`" % repair_rerun.get("rerun_from_proposed", ""),
                "- Required safe rerun_from: `%s`" % repair_rerun.get("rerun_from_required", ""),
                "- Effective rerun_from: `%s`" % repair_rerun.get("rerun_from_effective", ""),
                "- Source: `%s`" % repair_rerun.get("rerun_from_source", ""),
            ])
            if repair_rerun.get("rerun_reason"):
                lines.append("- Reason: %s" % repair_rerun["rerun_reason"])
            if repair_rerun.get("rerun_from_adjustment_reason"):
                lines.append("- Adjustment: %s" % repair_rerun["rerun_from_adjustment_reason"])
            effectiveness = self._repair_effectiveness(run_dir)
            if effectiveness:
                lines.extend([
                    "- Repair applied: `%s`" % str(bool(effectiveness.get("repair_applied"))).lower(),
                    "- Repair executed: `%s`" % str(bool(effectiveness.get("repair_executed"))).lower(),
                    "- Repair verified: `%s`" % str(bool(effectiveness.get("repair_verified"))).lower(),
                ])
            lines.append("")
        agent_metrics = self._read_optional(run_dir / "reports" / "agent_metrics.json")
        metrics = agent_metrics.get("agent_metrics") if isinstance(agent_metrics, dict) else {}
        if metrics:
            lines.extend([
                "## Agent Metrics",
                "",
                "- LLM calls: `%s`" % metrics.get("llm_call_count", 0),
                "- Accepted actions: `%s`" % metrics.get("accepted_action_count", 0),
                "- Rejected actions: `%s`" % metrics.get("rejected_action_count", 0),
                "- Executed actions: `%s`" % metrics.get("executed_action_count", 0),
                "- Repair attempts: `%s`" % metrics.get("repair_attempt_count", 0),
                "- Auto resume count: `%s`" % metrics.get("auto_resume_count", 0),
                "- Verify candidate count: `%s`" % metrics.get("verify_candidate_count", 0),
                "- Agent helped: `%s`" % str(bool(metrics.get("agent_helped"))).lower(),
                "- Help type: %s" % (", ".join("`%s`" % item for item in metrics.get("help_type") or []) or "`none`"),
                "",
            ])
        contribution = self._read_optional(run_dir / "reports" / "agent_contribution.json")
        if isinstance(contribution, dict) and contribution:
            lines.extend([
                "## Agent Contribution",
                "",
                "- LLM required: `%s`" % str(bool(contribution.get("llm_required"))).lower(),
                "- LLM helped: `%s`" % str(bool(contribution.get("llm_helped"))).lower(),
                "- Selection source: `%s`" % contribution.get("selection_source", ""),
                "- Reason: %s" % contribution.get("llm_required_reason", ""),
                "- Help type: %s" % (", ".join("`%s`" % item for item in contribution.get("help_type") or []) or "`none`"),
                "- Agent artifacts: `agent_steps.jsonl`, `agent_state.json`, `agent_plan.json`, `agent_plan_revisions.jsonl`",
                "",
            ])
        # Agent Summary section
        agent_summary = self._build_agent_summary(run_dir, results, contribution)
        if agent_summary:
            lines.extend([
                "## Agent Summary",
                "",
                "- Mode: `%s`" % agent_summary.get("mode", "off"),
                "- Runtime loop position: `%s`" % agent_summary.get("runtime_loop_position", "off"),
                "- LLM helped: `%s`" % str(bool(agent_summary.get("llm_helped"))).lower(),
                "- LLM required: `%s`" % str(bool(agent_summary.get("llm_required"))).lower(),
                "- LLM required status: `%s`" % agent_summary.get("llm_required_status", "unknown_without_baseline"),
                "- Helped stages: %s" % (", ".join("`%s`" % s for s in agent_summary.get("helped_stages") or []) or "`none`"),
                "- Accepted tool calls: `%s`" % agent_summary.get("accepted_tool_calls", 0),
                "- Rejected tool calls: `%s`" % agent_summary.get("rejected_tool_calls", 0),
                "- Final verify status: `%s`" % agent_summary.get("final_verify_status", ""),
                "",
            ])
        llm_required_evidence = self._read_optional(run_dir / "reports" / "llm_required_evidence.json")
        if isinstance(llm_required_evidence, dict) and llm_required_evidence:
            lines.extend([
                "## LLM Required Evidence",
                "",
                "- Baseline status: `%s`" % llm_required_evidence.get("baseline_status", ""),
                "- Agent status: `%s`" % llm_required_evidence.get("agent_status", ""),
                "- LLM required: `%s`" % str(bool(llm_required_evidence.get("llm_required"))).lower(),
                "",
            ])
        verified_memory = self._read_optional(run_dir / "reports" / "verified_memory.json")
        if isinstance(verified_memory, dict):
            lines.extend([
                "## Verified Memory",
                "",
                "- Recorded: `%s`" % str(bool(verified_memory.get("recorded"))).lower(),
                "- Memory id: `%s`" % verified_memory.get("memory_id", ""),
                "- Repair action hash: `%s`" % verified_memory.get("repair_action_hash", ""),
                "- Verification trace id: `%s`" % verified_memory.get("verification_trace_id", ""),
                "- Reason: %s" % verified_memory.get("reason", ""),
                "",
            ])
        execution_audit = execution_audit or self._read_optional(run_dir / "reports" / "execution_audit.json")
        if isinstance(execution_audit, dict) and execution_audit:
            lines.extend([
                "## Execution Audit",
                "",
                "- Requested start stage: `%s`" % execution_audit.get("requested_start_stage", ""),
                "- Effective start stage: `%s`" % execution_audit.get("effective_start_stage", ""),
                "- Fallback applied: `%s`" % str(bool(execution_audit.get("fallback_applied"))).lower(),
                "- Reused stages: %s" % self._format_stage_list(execution_audit.get("reused_stages") or []),
                "- Rerun stages: %s" % self._format_stage_list(execution_audit.get("rerun_stages") or []),
                "",
            ])
        required_env = self._required_env_vars(run_dir, results)
        if required_env:
            lines.extend([
                "## Required Environment Variables",
                "",
                "The following variable names may be required by the deployment. Values are not recorded in reports.",
                "",
            ])
            for name in required_env:
                lines.append("- `%s`: value required from operator or secret manager" % name)
            lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return StageResult("report", "passed", "report generated", {"report_path": str(report_path)}, evidence=[str(report_path)])

    def _repair_rerun_summary(self, run_dir: Path, results: Dict[str, Dict]) -> Dict:
        plan = self._read_optional(run_dir / "repairs" / "repair_plan.json")
        if isinstance(plan, dict) and (plan.get("rerun_from_proposed") or plan.get("rerun_from_effective") or plan.get("rerun_from")):
            return {
                "rerun_from_proposed": plan.get("rerun_from_proposed", ""),
                "rerun_from_required": plan.get("rerun_from_required", ""),
                "rerun_from_effective": plan.get("rerun_from_effective") or plan.get("rerun_from", ""),
                "rerun_from_source": plan.get("rerun_from_source", ""),
                "rerun_reason": plan.get("rerun_reason", ""),
                "rerun_from_adjustment_reason": plan.get("rerun_from_adjustment_reason", ""),
            }
        for result in results.values():
            data = result.get("data") if isinstance(result, dict) else {}
            loop = data.get("agent_loop") if isinstance(data, dict) else {}
            if isinstance(loop, dict) and loop.get("next_rerun_from"):
                return {
                    "rerun_from_proposed": "",
                    "rerun_from_required": "",
                    "rerun_from_effective": loop.get("next_rerun_from", ""),
                    "rerun_from_source": "agent_loop",
                    "rerun_reason": "",
                    "rerun_from_adjustment_reason": "",
                }
        return {}

    def _build_agent_summary(self, run_dir: Path, results: Dict, contribution: Dict = None) -> Dict:
        """Build agent summary for the report.

        Returns dict with mode, runtime_loop_position, llm_helped, llm_required,
        llm_required_status, helped_stages, accepted_tool_calls, rejected_tool_calls,
        final_verify_status.
        """
        # Get mode from agent_state.json
        agent_state = self._read_optional(run_dir / "agent_state.json")
        mode = "off"
        if isinstance(agent_state, dict):
            mode = agent_state.get("mode", "off")

        # Get runtime_loop_position from config or agent_loop_result
        loop_result = self._read_optional(run_dir / "reports" / "agent_loop_result.json")
        runtime_loop_position = "off"
        if isinstance(loop_result, dict) and loop_result.get("status"):
            runtime_loop_position = "primary" if loop_result.get("mode") == "gated_actor" else "post_pipeline"

        # Get contribution data
        if contribution is None:
            contribution = self._read_optional(run_dir / "reports" / "agent_contribution.json") or {}

        llm_helped = bool(contribution.get("llm_helped"))
        llm_required = bool(contribution.get("llm_required"))
        llm_required_status = contribution.get("llm_required_status", "unknown_without_baseline")

        # Get helped stages from gate results
        gate_results = contribution.get("gate_summary") or []
        helped_stages = [
            g.get("stage") for g in gate_results
            if g.get("policy_allowed") and (g.get("executed") or g.get("applied"))
        ]

        # Get accepted/rejected counts
        accepted_tool_calls = int(contribution.get("accepted_action_count") or 0)
        rejected_tool_calls = int(contribution.get("rejected_action_count") or 0)

        # Get final verify status
        final_verify_status = contribution.get("final_verify_status", "")
        if not final_verify_status:
            verify_result = results.get("verify", {})
            if isinstance(verify_result, dict):
                final_verify_status = verify_result.get("status", "")

        return {
            "mode": mode,
            "runtime_loop_position": runtime_loop_position,
            "llm_helped": llm_helped,
            "llm_required": llm_required,
            "llm_required_status": llm_required_status,
            "helped_stages": helped_stages,
            "accepted_tool_calls": accepted_tool_calls,
            "rejected_tool_calls": rejected_tool_calls,
            "final_verify_status": final_verify_status,
        }

    def _repair_effectiveness(self, run_dir: Path) -> Dict:
        apply_result = self._read_optional(run_dir / "repairs" / "repair_apply_result.json")
        return apply_result if isinstance(apply_result, dict) else {}

    def _run_candidate_selection(self, results: Dict[str, Dict]) -> Dict:
        runner = results.get("runner", {}).get("data", {})
        if isinstance(runner, dict) and isinstance(runner.get("candidate_selection"), dict):
            return runner["candidate_selection"]
        analyze = results.get("analyze", {}).get("data", {})
        candidates = analyze.get("run_candidates") if isinstance(analyze, dict) else []
        if candidates:
            candidate = candidates[0]
            return {
                "cmd": candidate.get("cmd", []),
                "score": float(candidate.get("score") or candidate.get("confidence") or 0),
                "score_reasons": list(candidate.get("score_reasons") or []),
                "selected_by": candidate.get("selected_by") or candidate.get("preferred_by") or "deterministic",
            }
        return {}

    def _format_stage_list(self, stages) -> str:
        names = [stage for stage in stages if isinstance(stage, str) and stage]
        return ", ".join("`%s`" % stage for stage in names) or "`none`"

    def _required_env_vars(self, run_dir: Path, results: Dict[str, Dict]) -> List[str]:
        names = set()
        repair_required = self._read_optional(run_dir / "repairs" / "required_env_vars.json")
        if isinstance(repair_required, dict):
            names.update(self._safe_names(repair_required.get("env_vars") or []))
        repair_plan = self._read_optional(run_dir / "repairs" / "repair_plan.json")
        if isinstance(repair_plan, dict):
            for action in repair_plan.get("actions") or []:
                payload = action.get("payload") if isinstance(action, dict) else {}
                if isinstance(payload, dict):
                    names.update(self._safe_names(payload.get("env_vars") or []))
        for result in results.values():
            data = result.get("data") if isinstance(result, dict) else {}
            if not isinstance(data, dict):
                continue
            diagnosis = data.get("diagnosis")
            if isinstance(diagnosis, dict) and diagnosis.get("category") == "auth_required":
                names.update(self._safe_names(diagnosis.get("required_env_vars") or []))
            names.update(self._safe_names(data.get("external_tokens") or []))
        return sorted(names)

    def _safe_names(self, values) -> List[str]:
        safe = []
        for value in values:
            if not isinstance(value, str):
                continue
            if value and value.upper() == value and all(ch.isalnum() or ch == "_" for ch in value):
                safe.append(value)
        return safe

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
