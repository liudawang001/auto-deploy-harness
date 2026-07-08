from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json, write_json


class AgentContributionAnalyzer:
    """Summarizes where LLM decisions changed or attempted to change the path."""

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
        payload = {
            "status": "generated",
            "llm_required": bool(helped_types),
            "llm_helped": bool(helped_types and final_verify in ("pass", "passed")),
            "help_type": helped_types,
            "selection_source": self._selection_source(analyze),
            "accepted_action_count": len(decision.get("accepted_actions") or []),
            "rejected_action_count": len(decision.get("rejected_actions") or []),
            "final_verify_status": final_verify,
            "llm_required_reason": self._reason(helped_types),
            "evidence": {
                "agent_steps": "agent_steps.jsonl",
                "agent_state": "agent_state.json",
                "agent_plan": "agent_plan.json",
            },
        }
        if output_path:
            write_json(Path(output_path), payload)
        return payload

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
