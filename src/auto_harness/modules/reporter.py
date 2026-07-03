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
            lines.extend([
                "### %s" % stage,
                "",
                "- Status: `%s`" % result.get("status", ""),
                "- Summary: %s" % result.get("summary", ""),
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

