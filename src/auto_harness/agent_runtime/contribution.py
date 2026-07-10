from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json, write_json


class AgentContributionAnalyzer:
    """Summarizes where LLM decisions changed or attempted to change the path.

    Computes llm_helped and llm_required based on real state improvement:
    - llm_helped=true only if: accepted LLM decision exists, policy allowed,
      tool/state_delta applied or executed, stage status improved or final verify passed
    - llm_required=true only if: baseline status is failed/uncertain, agent status is passed,
      accepted LLM decision affected execution path, final verify has evidence
    """

    def analyze(self, run_dir: Path, results: Dict, output_path: Path = None) -> Dict:
        analyze = results.get("analyze", {}).get("data", {}) if isinstance(results.get("analyze"), dict) else {}
        decision = analyze.get("agent_decision") if isinstance(analyze.get("agent_decision"), dict) else {}
        merged = decision.get("merged") if isinstance(decision.get("merged"), dict) else {}
        metrics = self._read_optional(Path(run_dir) / "reports" / "agent_metrics.json") or {}
        agent_metrics = metrics.get("agent_metrics") if isinstance(metrics.get("agent_metrics"), dict) else {}
        helped_types: List[str] = []
        if merged.get("preferred_candidate_selected") or merged.get("run_candidates_added"):
            helped_types.append("selected_or_added_run_candidate")
        if merged.get("verify_hint_updated"):
            helped_types.append("updated_verify_hint")
        if merged.get("environment_strategy_updated") or merged.get("torch_variant_updated"):
            helped_types.append("selected_environment_strategy")
        if int(agent_metrics.get("executed_action_count") or 0) > 0:
            helped_types.append("executed_policy_approved_repair")
        if int(agent_metrics.get("rejected_action_count") or 0) > 0:
            helped_types.append("rejected_unsafe_action")
        final_verify = results.get("verify", {}).get("status", "")

        # Collect gate results to check real state improvement
        gate_results = self._collect_gate_results(run_dir)
        any_gate_applied = any(
            g.get("policy_allowed") and (g.get("executed") or g.get("applied"))
            for g in gate_results
        )

        # llm_helped: LLM decision was accepted, policy allowed, state actually changed, and verify passed
        llm_helped = bool(helped_types and any_gate_applied and final_verify in ("pass", "passed"))

        # llm_required: needs baseline comparison - only true when baseline would have failed
        # This is set by evaluate_llm_required() when baseline data is available
        llm_required = llm_helped  # default: if helped, then required

        payload = {
            "status": "generated",
            "llm_required": llm_required,
            "llm_helped": llm_helped,
            "help_type": helped_types,
            "selection_source": self._selection_source(analyze),
            "accepted_action_count": len(decision.get("accepted_actions") or []),
            "rejected_action_count": len(decision.get("rejected_actions") or []),
            "final_verify_status": final_verify,
            "llm_required_reason": self._reason(helped_types),
            "gate_summary": gate_results,
            "evidence": {
                "agent_steps": "agent_steps.jsonl",
                "agent_state": "agent_state.json",
                "agent_plan": "agent_plan.json",
            },
        }
        if output_path:
            write_json(Path(output_path), payload)
        return payload

    def evaluate_llm_required(
        self,
        run_dir: Path,
        baseline_status: str,
        agent_status: str,
        results: Dict,
        output_path: Path = None,
    ) -> Dict:
        """Compute llm_required based on baseline-vs-agent comparison.

        llm_required=true only if:
          1. baseline status is failed/uncertain
          2. agent status is passed
          3. accepted LLM decision exists
          4. policy_allowed=true
          5. tool_result.applied=true or executed=true
          6. final verify passed with evidence
        """
        analyze = results.get("analyze", {}).get("data", {}) if isinstance(results.get("analyze"), dict) else {}
        decision = analyze.get("agent_decision") if isinstance(analyze.get("agent_decision"), dict) else {}
        final_verify = results.get("verify", {}).get("status", "")
        gate_results = self._collect_gate_results(run_dir)

        baseline_failed = baseline_status in ("failed", "uncertain")
        agent_passed = agent_status in ("pass", "passed")
        has_accepted_decision = bool(decision.get("accepted_actions"))
        any_gate_applied = any(
            g.get("policy_allowed") and (g.get("executed") or g.get("applied"))
            for g in gate_results
        )
        verify_passed = final_verify in ("pass", "passed")

        llm_required = (
            baseline_failed
            and agent_passed
            and has_accepted_decision
            and any_gate_applied
            and verify_passed
        )

        evidence = {
            "baseline_status": baseline_status,
            "agent_status": agent_status,
            "has_accepted_decision": has_accepted_decision,
            "any_gate_applied": any_gate_applied,
            "verify_passed": verify_passed,
            "gate_results": gate_results,
        }

        payload = {
            "llm_required": llm_required,
            "baseline_status": baseline_status,
            "agent_status": agent_status,
            "evidence": evidence,
        }

        if output_path:
            write_json(Path(output_path), payload)
        return payload

    def _collect_gate_results(self, run_dir: Path) -> List[Dict]:
        """Collect gate results from agent_decision_gates directory."""
        gates_dir = Path(run_dir) / "agent_decision_gates"
        if not gates_dir.exists():
            return []
        results = []
        for gate_file in sorted(gates_dir.glob("*_gate.json")):
            try:
                gate = read_json(gate_file)
                results.append({
                    "stage": gate.get("stage", ""),
                    "decision_status": gate.get("decision_status", ""),
                    "policy_allowed": gate.get("policy", {}).get("allowed", False),
                    "executed": gate.get("execution", {}).get("executed", False),
                    "applied": gate.get("execution", {}).get("applied", False),
                    "llm_helped": gate.get("llm_helped", False),
                })
            except (OSError, ValueError):
                continue
        return results

    def _selection_source(self, analyze: Dict) -> str:
        candidates = analyze.get("run_candidates") or []
        if candidates:
            selected_by = candidates[0].get("selected_by") or candidates[0].get("preferred_by") or "deterministic"
            return "hybrid" if selected_by == "combined" else selected_by
        return "none"

    def _reason(self, helped_types: List[str]) -> str:
        if not helped_types:
            return "LLM did not materially change the deterministic path in this run."
        return "LLM contributed through: %s" % ", ".join(helped_types)

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
