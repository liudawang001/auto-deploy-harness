from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json
from auto_harness.models.result import StageResult


class ReportGenerator:
    def generate(self, run_dir: Path, task: Dict, results: Dict[str, Dict]) -> StageResult:
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
        if verify:
            lines.extend([
                "## Verify",
                "",
                "- Final status: `%s`" % verify.get("status", ""),
                "- Trace ID: `%s`" % verify.get("trace_id", ""),
                "- Next action: `%s`" % verify.get("next_action", ""),
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
