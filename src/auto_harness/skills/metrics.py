"""Skill Metrics Reporter: aggregate skill selection/influence/pass/harm metrics.

Reads skill effects and outcome records to produce per-skill metrics
including selection_count, influence_count, policy_accept_rate,
verify_pass_rate, harm_rate, and llm_helped_count.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


class SkillMetricsReporter:
    """Aggregates and reports skill metrics across deployment runs.

    Can report per-run metrics or aggregate across multiple runs.
    """

    def compute_run_metrics(
        self,
        skill_effects: Dict,
        pipeline_results: Dict,
        agent_metrics: Dict = None,
    ) -> Dict:
        """Compute skill metrics for a single deployment run.

        Args:
            skill_effects: The skill_effects.json data.
            pipeline_results: The pipeline_results.json data.
            agent_metrics: Optional agent_metrics.json data.

        Returns:
            Dict with per-skill metrics.
        """
        effects = skill_effects.get("effects", [])
        verify_result = pipeline_results.get("verify", {})
        verify_status = verify_result.get("status", "")
        agent_metrics = agent_metrics or {}

        # Group effects by skill
        skill_data: Dict[str, Dict] = {}
        for effect in effects:
            name = effect.get("skill_name", "")
            sha = effect.get("skill_sha256", "")
            key = "%s@%s" % (name, sha[:8]) if sha else name

            if key not in skill_data:
                skill_data[key] = {
                    "name": name,
                    "sha256": sha,
                    "selection_count": 1,
                    "influence_count": 0,
                    "policy_accept_count": 0,
                    "policy_reject_count": 0,
                    "harm_count": 0,
                }

            entry = skill_data[key]
            if effect.get("accepted_by_policy"):
                entry["policy_accept_count"] += 1
            else:
                entry["policy_reject_count"] += 1

            # Influence: skill had an actual effect on a plan field
            if effect.get("field_changed"):
                entry["influence_count"] += 1

        # Compute derived metrics and verify/harm indicators
        for key, entry in skill_data.items():
            total = entry["policy_accept_count"] + entry["policy_reject_count"]
            entry["policy_accept_rate"] = (
                entry["policy_accept_count"] / total if total > 0 else 1.0
            )

            # verify_pass: if this skill influenced a field and final verify passed
            if entry["influence_count"] > 0 and verify_status in ("passed", "pass"):
                entry["verify_pass_count"] = entry["influence_count"]
            else:
                entry["verify_pass_count"] = 0

            total_influenced = entry["influence_count"]
            entry["verify_pass_rate"] = (
                entry["verify_pass_count"] / total_influenced
                if total_influenced > 0 else 0.0
            )

            # Harm: skill influenced plan, policy accepted, but verify failed/uncertain
            if (
                entry["influence_count"] > 0
                and entry["policy_accept_count"] > 0
                and verify_status not in ("passed", "pass")
            ):
                entry["harm_count"] = entry["influence_count"]
            else:
                entry["harm_count"] = 0

            entry["harm_rate"] = (
                entry["harm_count"] / entry["influence_count"]
                if entry["influence_count"] > 0 else 0.0
            )

            # llm_helped from agent_metrics
            entry["llm_helped_count"] = 1 if agent_metrics.get("agent_metrics", {}).get("llm_helped") else 0

        return {
            "skills": skill_data,
            "computed_at": utc_now_iso(),
        }

    def aggregate_metrics(
        self,
        runs_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Dict:
        """Aggregate skill metrics across multiple runs.

        Reads skill_effects.json and pipeline_results.json from each run
        directory and computes aggregate metrics.

        Args:
            runs_dir: Path to the runs directory.
            output_path: Optional path to write aggregate metrics.

        Returns:
            Dict with aggregated per-skill metrics.
        """
        runs_dir = Path(runs_dir)
        if not runs_dir.exists():
            return {"skills": {}, "computed_at": utc_now_iso()}

        aggregate: Dict[str, Dict] = {}

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            effects_path = run_dir / "reports" / "skill_effects.json"
            pipeline_path = run_dir / "reports" / "pipeline_results.json"
            metrics_path = run_dir / "reports" / "agent_metrics.json"

            if not effects_path.exists():
                continue

            try:
                effects = read_json(effects_path)
            except (OSError, ValueError):
                continue

            pipeline = {}
            if pipeline_path.exists():
                try:
                    pipeline = read_json(pipeline_path)
                except (OSError, ValueError):
                    pass

            agent_metrics = {}
            if metrics_path.exists():
                try:
                    agent_metrics = read_json(metrics_path)
                except (OSError, ValueError):
                    pass

            run_metrics = self.compute_run_metrics(effects, pipeline, agent_metrics)

            # Merge into aggregate
            for skill_key, skill_entry in run_metrics.get("skills", {}).items():
                if skill_key not in aggregate:
                    aggregate[skill_key] = {
                        "name": skill_entry.get("name", ""),
                        "sha256": skill_entry.get("sha256", ""),
                        "selection_count": 0,
                        "influence_count": 0,
                        "policy_accept_count": 0,
                        "policy_reject_count": 0,
                        "verify_pass_count": 0,
                        "harm_count": 0,
                        "llm_helped_count": 0,
                    }

                agg = aggregate[skill_key]
                agg["selection_count"] += skill_entry.get("selection_count", 0)
                agg["influence_count"] += skill_entry.get("influence_count", 0)
                agg["policy_accept_count"] += skill_entry.get("policy_accept_count", 0)
                agg["policy_reject_count"] += skill_entry.get("policy_reject_count", 0)
                agg["verify_pass_count"] += skill_entry.get("verify_pass_count", 0)
                agg["harm_count"] += skill_entry.get("harm_count", 0)
                agg["llm_helped_count"] += skill_entry.get("llm_helped_count", 0)

        # Compute rates for aggregate
        for skill_key, agg in aggregate.items():
            total = agg["policy_accept_count"] + agg["policy_reject_count"]
            agg["policy_accept_rate"] = agg["policy_accept_count"] / total if total > 0 else 1.0
            agg["verify_pass_rate"] = agg["verify_pass_count"] / agg["influence_count"] if agg["influence_count"] > 0 else 0.0
            agg["harm_rate"] = agg["harm_count"] / agg["influence_count"] if agg["influence_count"] > 0 else 0.0

        result = {"skills": aggregate, "computed_at": utc_now_iso()}

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, result)

        return result
