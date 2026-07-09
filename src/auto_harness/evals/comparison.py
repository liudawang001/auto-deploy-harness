import json
from pathlib import Path
from typing import Dict, List

from auto_harness.evals.metrics import summarize_runs
from auto_harness.models.base import write_json


class AgentComparisonReporter:
    """Builds baseline/off vs planner/gated_actor comparison reports from run summaries.

    Supports two modes:
    - from_manifest(): generates skeleton report from manifest (backward compatible)
    - run_eval(): executes real off vs gated_actor comparison using test servers
    """

    def build(self, eval_id: str, targets: List[Dict], baseline_runs: List[Dict], agent_runs: List[Dict], output_dir: Path) -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        agent_by_target = {item.get("target_id"): item for item in agent_runs}
        baseline_by_target = {item.get("target_id"): item for item in baseline_runs}
        helped = []
        for target_id, agent in agent_by_target.items():
            baseline = baseline_by_target.get(target_id, {})
            if baseline.get("verify_status") not in ("pass", "passed") and agent.get("verify_status") in ("pass", "passed"):
                helped.append({
                    "target_id": target_id,
                    "help_type": agent.get("help_type") or "agent_changed_path",
                    "evidence": agent.get("evidence") or "agent_steps.jsonl",
                })
        report = {
            "eval_id": eval_id,
            "target_count": len(targets),
            "baseline": summarize_runs(baseline_runs),
            "agent": summarize_runs(agent_runs),
            "baseline_failed_agent_passed_count": len(helped),
            "llm_helped_cases": helped,
            "targets": targets,
        }
        write_json(output_dir / "comparison_report.json", report)
        (output_dir / "comparison_report.md").write_text(self._markdown(report), encoding="utf-8")
        return report

    def from_manifest(self, manifest_path: Path, output_dir: Path) -> Dict:
        """Generate skeleton report from manifest (backward compatible)."""
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        targets = manifest.get("targets") or []
        baseline_runs = [{"target_id": item["id"], "verify_status": "unknown"} for item in targets]
        agent_runs = [{"target_id": item["id"], "verify_status": "unknown"} for item in targets]
        return self.build(manifest.get("eval_id", "local-fixture-eval"), targets, baseline_runs, agent_runs, output_dir)

    def run_eval(self, output_dir: Path) -> Dict:
        """Execute real off vs gated_actor comparison using test servers and agent verify loop.

        This is the Phase 7 deliverable per design doc §12.
        """
        from auto_harness.evals.agent_verify_eval import run_agent_verify_eval
        return run_agent_verify_eval(output_dir=output_dir)

    def _markdown(self, report: Dict) -> str:
        lines = [
            "# Agent Comparison Report",
            "",
            "- Eval ID: `%s`" % report.get("eval_id", ""),
            "- Targets: `%s`" % report.get("target_count", 0),
            "- Baseline verify pass: `%s/%s`" % (report["baseline"].get("verify_pass", 0), report["baseline"].get("total", 0)),
            "- Agent verify pass: `%s/%s`" % (report["agent"].get("verify_pass", 0), report["agent"].get("total", 0)),
            "- Baseline failed, agent passed: `%s`" % report.get("baseline_failed_agent_passed_count", 0),
            "",
            "## LLM Helped Cases",
            "",
        ]
        for item in report.get("llm_helped_cases") or []:
            lines.append("- `%s`: %s (%s)" % (item.get("target_id"), item.get("help_type"), item.get("evidence")))
        if not report.get("llm_helped_cases"):
            lines.append("- none recorded yet")
        lines.append("")
        return "\n".join(lines)
