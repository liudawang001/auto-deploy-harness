from pathlib import Path
from typing import Dict

from auto_harness.models.base import read_json, write_json


class AgentMetricsCollector:
    def collect(self, run_dir: Path, results: Dict = None, output_path: Path = None) -> Dict:
        run_dir = Path(run_dir)
        results = results or self._read_optional(run_dir / "reports" / "pipeline_results.json") or {}
        policy_counts = self._policy_counts(run_dir)
        repair_apply = self._read_optional(run_dir / "repairs" / "repair_apply_result.json") or {}
        loop_state = self._read_optional(run_dir / "repairs" / "repair_loop_state.json") or {}
        metrics = {
            "llm_call_count": len(list((run_dir / "logs" / "agent_calls").glob("*.json"))),
            "accepted_action_count": policy_counts["accepted"],
            "rejected_action_count": policy_counts["rejected"],
            "executed_action_count": int(repair_apply.get("executed_action_count") or 0),
            "repair_attempt_count": len(loop_state.get("history") or []),
            "auto_resume_count": self._auto_resume_count(results),
            "verify_candidate_count": self._verify_candidate_count(results),
            "final_status": results.get("verify", {}).get("status", ""),
            "agent_helped": False,
            "help_type": [],
        }
        help_types = self._help_types(results, metrics)
        metrics["help_type"] = help_types
        metrics["agent_helped"] = bool(help_types)
        payload = {"agent_metrics": metrics}
        if output_path:
            write_json(Path(output_path), payload)
        return payload

    def collect_many(self, runs_dir: Path, output_path: Path = None) -> Dict:
        runs_dir = Path(runs_dir)
        items = []
        if runs_dir.exists():
            for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
                items.append({"task_id": run_dir.name, **self.collect(run_dir)})
        summary = {
            "status": "generated",
            "run_count": len(items),
            "totals": self._totals(items),
            "runs": items,
        }
        if output_path:
            write_json(Path(output_path), summary)
        return summary

    def _policy_counts(self, run_dir: Path) -> Dict:
        accepted = 0
        rejected = 0
        for path in (run_dir / "logs" / "agent_calls").glob("*.json"):
            data = self._read_optional(path) or {}
            policy = data.get("policy_result") if isinstance(data.get("policy_result"), dict) else {}
            accepted += len(policy.get("accepted_actions") or [])
            rejected += len(policy.get("rejected_actions") or [])
        return {"accepted": accepted, "rejected": rejected}

    def _auto_resume_count(self, results: Dict) -> int:
        count = 0
        for result in results.values():
            data = result.get("data") if isinstance(result, dict) else {}
            loop = data.get("agent_loop") if isinstance(data, dict) else {}
            if isinstance(loop, dict) and loop.get("should_auto_resume"):
                count += 1
        return count

    def _verify_candidate_count(self, results: Dict) -> int:
        planner = results.get("verify", {}).get("data", {}).get("llm_verify_planner", {})
        return len(planner.get("accepted_candidates") or planner.get("verify_candidates") or [])

    def _help_types(self, results: Dict, metrics: Dict) -> list:
        help_types = []
        analyze = results.get("analyze", {}).get("data", {})
        candidates = analyze.get("run_candidates") if isinstance(analyze, dict) else []
        if any(candidate.get("selected_by") in ("llm_planner", "combined") for candidate in candidates or []):
            help_types.append("selected_run_candidate")
        if metrics.get("executed_action_count", 0) > 0:
            help_types.append("repaired_dependency")
        if metrics.get("verify_candidate_count", 0) > 0:
            help_types.append("generated_verify_hint")
        if metrics.get("rejected_action_count", 0) > 0:
            help_types.append("rejected_unsafe_action")
        return help_types

    def _totals(self, items) -> Dict:
        totals = {
            "llm_call_count": 0,
            "accepted_action_count": 0,
            "rejected_action_count": 0,
            "executed_action_count": 0,
            "repair_attempt_count": 0,
            "auto_resume_count": 0,
            "verify_candidate_count": 0,
            "agent_helped_count": 0,
        }
        for item in items:
            metrics = item.get("agent_metrics") or {}
            for key in totals:
                if key == "agent_helped_count":
                    totals[key] += 1 if metrics.get("agent_helped") else 0
                else:
                    totals[key] += int(metrics.get(key) or 0)
        return totals

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
