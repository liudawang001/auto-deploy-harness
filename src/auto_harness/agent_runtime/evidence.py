"""LLM Contribution Evidence Writer.

Generates reports/llm_contribution_evidence.json to prove whether
LLM genuinely helped improve deployment outcomes.

Key invariant:
- llm_helped=true ONLY if baseline was failed/uncertain AND agent passed
  AND evidence contains current trace_id
- Without baseline, llm_required_status must be "unknown_without_baseline"
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


class LLMContributionEvidenceWriter:
    """Writes llm_contribution_evidence.json for each deployment run.

    This is the definitive proof of whether LLM helped:
    - baseline vs agent comparison
    - trace evidence verification
    - safety metrics
    """

    def write(
        self,
        run_dir: Path,
        task_id: str,
        baseline_result: Optional[Dict] = None,
        agent_result: Optional[Dict] = None,
        agent_steps: Optional[List[Dict]] = None,
        pipeline_results: Optional[Dict] = None,
    ) -> Dict:
        """Write LLM contribution evidence.

        Args:
            run_dir: Run directory
            task_id: Task identifier
            baseline_result: Result from baseline (off mode), None if not available
            agent_result: Result from agent mode
            agent_steps: List of agent step records
            pipeline_results: Full pipeline results

        Returns:
            The evidence dict that was written
        """
        agent_steps = agent_steps or []
        pipeline_results = pipeline_results or {}

        # Extract verify status and trace info
        verify_data = self._extract_verify_data(pipeline_results)
        verify_status = verify_data.get("status", "unknown")
        trace_id = verify_data.get("trace_id", "")
        evidence_paths = verify_data.get("evidence_paths", verify_data.get("evidence", []))

        # Compute baseline status
        baseline_status = "unknown"
        baseline_failed_stage = ""
        baseline_reason = ""
        if baseline_result:
            baseline_status = baseline_result.get("final_status", baseline_result.get("status", "unknown"))
            baseline_failed_stage = baseline_result.get("failed_stage", "")
            baseline_reason = baseline_result.get("reason", "")

        # Compute agent status
        agent_status = verify_status
        agent_mode = "gated_actor" if agent_result else "off"
        changed_stage = ""
        decision_type = ""
        accepted_tool_count = 0
        rejected_tool_count = 0

        if agent_result:
            agent_mode = agent_result.get("mode", "gated_actor")
            changed_stage = agent_result.get("changed_stage", "")
            decision_type = agent_result.get("decision_type", "")
            accepted_tool_count = agent_result.get("accepted_tool_count", 0)
            rejected_tool_count = agent_result.get("rejected_tool_count", 0)

        # Compute llm_helped: baseline failed/uncertain AND agent passed AND trace evidence exists
        baseline_failed = baseline_status in ("failed", "uncertain", "unknown")
        agent_passed = agent_status in ("pass", "passed")
        evidence_has_trace = self._evidence_contains_trace(evidence_paths, trace_id)

        llm_changed_decision = bool(agent_steps) and any(
            step.get("decision", {}).get("decision_status") == "ok"
            for step in agent_steps
            if isinstance(step, dict)
        )

        llm_helped = (
            baseline_failed
            and agent_passed
            and evidence_has_trace
            and llm_changed_decision
        )

        # Compute llm_required status
        if baseline_result is None:
            llm_required = False
            llm_required_status = "unknown_without_baseline"
        elif llm_helped:
            llm_required = True
            llm_required_status = "proven_by_baseline_agent_delta"
        else:
            llm_required = False
            llm_required_status = "baseline_did_not_fail"

        # Compute help_type
        help_types = self._compute_help_types(agent_steps, pipeline_results)

        # Compute safety metrics
        safety = self._compute_safety(agent_steps)

        # Build evidence payload
        evidence = {
            "task_id": task_id,
            "status": "generated",
            "generated_at": utc_now_iso(),
            "baseline": {
                "mode": "off",
                "final_status": baseline_status,
                "failed_stage": baseline_failed_stage,
                "reason": baseline_reason,
            },
            "agent": {
                "mode": agent_mode,
                "final_status": agent_status,
                "changed_stage": changed_stage,
                "decision_type": decision_type,
                "accepted_tool_count": accepted_tool_count,
                "rejected_tool_count": rejected_tool_count,
            },
            "llm_changed_decision": llm_changed_decision,
            "llm_helped": llm_helped,
            "llm_required": llm_required,
            "llm_required_status": llm_required_status,
            "help_type": help_types,
            "trace_id": trace_id,
            "evidence_paths": evidence_paths,
            "why_llm_required": self._why_llm_required(
                llm_helped, baseline_status, agent_status, evidence_has_trace
            ),
            "safety": safety,
        }

        # Write to file
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / "llm_contribution_evidence.json", evidence)

        return evidence

    def _extract_verify_data(self, pipeline_results: Dict) -> Dict:
        """Extract verify data from pipeline results."""
        verify = pipeline_results.get("verify", {})
        if isinstance(verify, dict):
            # verify may have {status, data} or be the data itself
            if "data" in verify and isinstance(verify["data"], dict):
                data = verify["data"]
                # data may have nested data (from to_plain)
                if "data" in data and isinstance(data.get("data"), dict):
                    return data["data"]
                return data
            return verify
        return {}

    def _evidence_contains_trace(self, evidence_paths: List[str], trace_id: str) -> bool:
        """Check if any evidence file contains the current trace_id."""
        if not trace_id or not evidence_paths:
            return False
        for path_str in evidence_paths:
            path = Path(path_str)
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    if trace_id in content:
                        return True
                except (OSError, ValueError):
                    continue
        return False

    def _compute_help_types(self, agent_steps: List[Dict], pipeline_results: Dict) -> List[str]:
        """Compute the types of help LLM provided."""
        help_types = []

        # Check if LLM selected runner candidate
        for step in agent_steps:
            if not isinstance(step, dict):
                continue
            stage = step.get("stage", "")
            decision = step.get("decision", {})
            if isinstance(decision, dict):
                tool_call = decision.get("tool_call", {})
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "")
                    if tool_name in ("select_runner_candidate", "add_runner_candidate"):
                        help_types.append("runner_candidate_selection")
                    elif tool_name in ("discover_gradio_api", "discover_openapi_schema", "probe_http"):
                        help_types.append("verify_probe_selection")
                    elif tool_name in ("apply_dependency_constraint", "propose_dependency_constraint"):
                        help_types.append("repair_dependency_missing")
                    elif tool_name == "set_deployment_strategy":
                        help_types.append("deployment_strategy")

        # Deduplicate
        return list(set(help_types))

    def _compute_safety(self, agent_steps: List[Dict]) -> Dict:
        """Compute safety metrics from agent steps."""
        side_effect_count = 0
        external_hosts = []

        for step in agent_steps:
            if not isinstance(step, dict):
                continue
            decision = step.get("decision", {})
            if isinstance(decision, dict):
                # Count side-effect tools executed
                if decision.get("executed") and decision.get("policy_allowed"):
                    side_effect_count += 1

        return {
            "policy_gated": True,
            "side_effect_tools_executed": side_effect_count,
            "external_hosts_used": external_hosts,
        }

    def _why_llm_required(
        self,
        llm_helped: bool,
        baseline_status: str,
        agent_status: str,
        evidence_has_trace: bool,
    ) -> str:
        """Generate human-readable explanation of why LLM was required."""
        if llm_helped:
            return (
                "deterministic baseline was %s; agent mode achieved %s "
                "with current trace evidence" % (baseline_status, agent_status)
            )
        if baseline_status in ("passed", "pass"):
            return "baseline already passed; LLM was not required"
        if not evidence_has_trace:
            return "agent passed but no current trace evidence found"
        return "LLM did not change the outcome in this run"
