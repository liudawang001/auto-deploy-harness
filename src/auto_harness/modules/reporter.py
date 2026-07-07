from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json
from auto_harness.models.result import StageResult


class ReportGenerator:
    def generate(self, run_dir: Path, task: Dict, results: Dict[str, Dict], execution_audit: Dict = None) -> StageResult:
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
        run_selection = self._run_candidate_selection(results)
        if run_selection:
            lines.extend([
                "## Run Candidate Selection",
                "",
                "- Command: `%s`" % " ".join(str(part) for part in run_selection.get("cmd", [])),
                "- Score: `%.2f`" % float(run_selection.get("score") or 0),
                "- Selected by: `%s`" % run_selection.get("selected_by", ""),
                "- Reasons:",
            ])
            for reason in run_selection.get("score_reasons") or []:
                lines.append("  - %s" % reason)
            lines.append("")
        if verify:
            lines.extend([
                "## Verify",
                "",
                "- Final status: `%s`" % verify.get("status", ""),
                "- Trace ID: `%s`" % verify.get("trace_id", ""),
                "- Next action: `%s`" % verify.get("next_action", ""),
                "",
            ])
        execution_audit = execution_audit or self._read_optional(run_dir / "reports" / "execution_audit.json")
        if isinstance(execution_audit, dict) and execution_audit:
            lines.extend([
                "## Execution Audit",
                "",
                "- Requested start stage: `%s`" % execution_audit.get("requested_start_stage", ""),
                "- Effective start stage: `%s`" % execution_audit.get("effective_start_stage", ""),
                "- Fallback applied: `%s`" % str(bool(execution_audit.get("fallback_applied"))).lower(),
                "- Reused stages: %s" % self._format_stage_list(execution_audit.get("reused_stages") or []),
                "- Rerun stages: %s" % self._format_stage_list(execution_audit.get("rerun_stages") or []),
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

    def _run_candidate_selection(self, results: Dict[str, Dict]) -> Dict:
        runner = results.get("runner", {}).get("data", {})
        if isinstance(runner, dict) and isinstance(runner.get("candidate_selection"), dict):
            return runner["candidate_selection"]
        analyze = results.get("analyze", {}).get("data", {})
        candidates = analyze.get("run_candidates") if isinstance(analyze, dict) else []
        if candidates:
            candidate = candidates[0]
            return {
                "cmd": candidate.get("cmd", []),
                "score": float(candidate.get("score") or candidate.get("confidence") or 0),
                "score_reasons": list(candidate.get("score_reasons") or []),
                "selected_by": candidate.get("selected_by") or candidate.get("preferred_by") or "deterministic",
            }
        return {}

    def _format_stage_list(self, stages) -> str:
        names = [stage for stage in stages if isinstance(stage, str) and stage]
        return ", ".join("`%s`" % stage for stage in names) or "`none`"

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
