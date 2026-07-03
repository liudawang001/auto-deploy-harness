from pathlib import Path
from typing import Dict

from auto_harness.models.result import StageResult


class ReportGenerator:
    def generate(self, run_dir: Path, task: Dict, results: Dict[str, Dict]) -> StageResult:
        report_path = run_dir / "reports" / "report.md"
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
        if verify:
            lines.extend([
                "## Verify",
                "",
                "- Final status: `%s`" % verify.get("status", ""),
                "- Trace ID: `%s`" % verify.get("trace_id", ""),
                "- Next action: `%s`" % verify.get("next_action", ""),
                "",
            ])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return StageResult("report", "passed", "report generated", {"report_path": str(report_path)}, evidence=[str(report_path)])
