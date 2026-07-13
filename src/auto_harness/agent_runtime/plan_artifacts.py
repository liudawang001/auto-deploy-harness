"""Plan Artifact Writer for LLM plan-first deployment.

Writes all plan-first related artifacts to the run directory:
- project_snapshot.json
- llm_deployment_plan.raw.json
- llm_deployment_plan.parsed.json
- llm_plan_policy.json
- effective_deployment_plan.json
- plan_revisions.jsonl
- llm_contribution_evidence.json
"""
from pathlib import Path
from typing import Any, Dict

from auto_harness.models.base import write_json


class PlanArtifactWriter:
    """Writes plan-first deployment artifacts to the run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.reports_dir = self.run_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_project_snapshot(self, snapshot: Dict) -> Path:
        path = self.reports_dir / "project_snapshot.json"
        write_json(path, snapshot)
        return path

    def write_raw_plan(self, raw_plan: Any) -> Path:
        path = self.reports_dir / "llm_deployment_plan.raw.json"
        write_json(path, raw_plan)
        return path

    def write_parsed_plan(self, parsed_plan: Dict) -> Path:
        path = self.reports_dir / "llm_deployment_plan.parsed.json"
        write_json(path, parsed_plan)
        return path

    def write_policy_result(self, policy_result: Dict) -> Path:
        path = self.reports_dir / "llm_plan_policy.json"
        write_json(path, policy_result)
        return path

    def write_effective_plan(self, effective_plan: Dict) -> Path:
        path = self.reports_dir / "effective_deployment_plan.json"
        write_json(path, effective_plan)
        return path

    def write_plan_revision(self, revision: Dict) -> Path:
        """Append a revision entry to plan_revisions.jsonl."""
        path = self.reports_dir / "plan_revisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(revision, ensure_ascii=False) + "\n")
        return path

    def write_contribution_evidence(self, evidence: Dict) -> Path:
        path = self.reports_dir / "llm_contribution_evidence.json"
        write_json(path, evidence)
        return path

    def write_pipeline_results(self, results: Dict) -> Path:
        path = self.reports_dir / "pipeline_results.json"
        write_json(path, results)
        return path

    def write_skill_effects(self, effects: Dict) -> Path:
        """Write skill effects to reports/skill_effects.json."""
        path = self.reports_dir / "skill_effects.json"
        write_json(path, effects)
        return path
