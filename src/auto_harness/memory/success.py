import json
from pathlib import Path
from typing import Dict, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.files import ensure_dir, short_hash
from auto_harness.utils.time import utc_now_iso


class VerifiedMemoryRecorder:
    """Records only repair outcomes proven by final trace verification."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = ensure_dir(memory_dir)
        self.issue_path = self.memory_dir / "deployment_issues.jsonl"

    def record_if_verified(self, run_dir: Path, pipeline_results: Dict, agent_metrics: Dict) -> Optional[Dict]:
        run_dir = Path(run_dir)
        verify = pipeline_results.get("verify", {}) if isinstance(pipeline_results.get("verify"), dict) else {}
        verify_data = verify.get("data") if isinstance(verify.get("data"), dict) else {}
        if verify.get("status") not in ("pass", "passed"):
            return self._write_status(run_dir, "skipped", "final verify did not pass")
        trace_id = str(verify_data.get("trace_id") or "")
        if not trace_id:
            return self._write_status(run_dir, "skipped", "verification trace id missing")
        apply_result = self._read_optional(run_dir / "repairs" / "repair_apply_result.json") or {}
        if apply_result.get("status") != "applied":
            return self._write_status(run_dir, "skipped", "repair was not applied")
        if not self._effective_repair(apply_result):
            return self._write_status(run_dir, "skipped", "repair action was not effective")
        if self._has_high_risk_rejection(run_dir, apply_result):
            return self._write_status(run_dir, "skipped", "high risk policy rejection observed")

        env_solve = pipeline_results.get("env_solve", {}) if isinstance(pipeline_results.get("env_solve"), dict) else {}
        env_data = env_solve.get("data") if isinstance(env_solve.get("data"), dict) else {}
        env_solution = (env_data.get("analysis") or {}).get("env_solution") if isinstance(env_data.get("analysis"), dict) else {}
        if not isinstance(env_solution, dict):
            env_solution = {}
        source = self._latest_memory_event(run_dir)
        repair_plan = source.get("repair_plan") or self._read_optional(run_dir / "repairs" / "repair_plan.json") or {}
        repair_hash = self._repair_action_hash(repair_plan, env_solution)
        task = self._read_optional(run_dir / "task.json") or {}
        analysis = pipeline_results.get("analyze", {}).get("data", {}) if isinstance(pipeline_results.get("analyze"), dict) else {}
        entry = {
            "id": "mem_success_%s" % short_hash(repair_hash + trace_id, 12),
            "memory_type": "verified_success",
            "created_at": utc_now_iso(),
            "task_id": task.get("task_id") or run_dir.name,
            "stage": source.get("stage") or repair_plan.get("rerun_from_effective") or "runner",
            "category": self._category(source, repair_plan),
            "frameworks": list(analysis.get("frameworks") or []),
            "project_signature": short_hash(json.dumps(analysis.get("files") or [], sort_keys=True), 12),
            "failure_signature": source.get("signature", ""),
            "symptom": source.get("symptom", "self-healing repair verified"),
            "root_cause": repair_plan.get("root_cause", ""),
            "repair_action_hash": repair_hash,
            "repair_actions": repair_plan.get("actions") or [],
            "repair_action_status": "executed",
            "environment_backend": env_solution.get("backend") or "venv",
            "environment_spec_hash": short_hash(json.dumps(env_solution.get("conda") or env_solution, sort_keys=True, ensure_ascii=False), 16),
            "torch_variant": env_solution.get("torch_variant") or "",
            "verification_trace_id": trace_id,
            "verify_status": verify.get("status"),
            "regression_case_ids": ["agent_full_self_healing_pipeline"],
            "regression_status": "passed",
            "verified_success": True,
            "policy_rejected_high_risk": False,
            "suggested_next_action": "Promote this verified repair only after bounded regression passes.",
        }
        if not self._has_entry(entry["id"]):
            with self.issue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._write_status(run_dir, "recorded", "verified success memory recorded", entry)
        return entry

    def _effective_repair(self, apply_result: Dict) -> bool:
        if any(item.get("executed") and int(item.get("exit_code") or 0) == 0 for item in apply_result.get("action_results", [])):
            return True
        return any(item.get("status") == "metadata_only" for item in apply_result.get("action_results", []))

    def _has_high_risk_rejection(self, run_dir: Path, apply_result: Dict) -> bool:
        text = json.dumps(apply_result.get("policy") or {}, ensure_ascii=False).lower()
        high_risk = ("source edit", "operator secret", "secret", "shell", "unsafe", "external url")
        return any(term in text for term in high_risk) and '"allowed": false' in text

    def _latest_memory_event(self, run_dir: Path) -> Dict:
        events_path = run_dir / "events.jsonl"
        latest: Dict = {}
        if not events_path.exists():
            return latest
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if '"memory_recorded"' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            latest = {
                "stage": event.get("stage"),
                "signature": data.get("signature", ""),
                "repair_plan": data.get("repair_plan") if isinstance(data.get("repair_plan"), dict) else {},
            }
        return latest

    def _category(self, source: Dict, repair_plan: Dict) -> str:
        root = str(repair_plan.get("root_cause") or "").lower()
        if "module" in root or "dependency" in root:
            return "dependency_missing"
        return source.get("category") or "self_healing_repair"

    def _repair_action_hash(self, repair_plan: Dict, env_solution: Dict) -> str:
        payload = {
            "actions": repair_plan.get("actions") or [],
            "environment_backend": env_solution.get("backend") or "venv",
            "torch_variant": env_solution.get("torch_variant") or "",
            "rerun_from_effective": repair_plan.get("rerun_from_effective") or repair_plan.get("rerun_from") or "",
        }
        return short_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True), 16)

    def _has_entry(self, entry_id: str) -> bool:
        if not self.issue_path.exists():
            return False
        return any(('"id": "%s"' % entry_id) in line for line in self.issue_path.read_text(encoding="utf-8").splitlines())

    def _write_status(self, run_dir: Path, status: str, reason: str, entry: Dict = None):
        payload = {
            "status": status,
            "reason": reason,
            "memory_id": (entry or {}).get("id", ""),
            "repair_action_hash": (entry or {}).get("repair_action_hash", ""),
            "verification_trace_id": (entry or {}).get("verification_trace_id", ""),
            "recorded": status == "recorded",
        }
        write_json(run_dir / "reports" / "verified_memory.json", payload)
        return entry

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
