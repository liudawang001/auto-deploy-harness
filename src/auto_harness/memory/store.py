import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import to_plain
from auto_harness.models.result import StageResult
from auto_harness.utils.files import ensure_dir, short_hash
from auto_harness.utils.time import utc_now_iso


class MemoryStore:
    """Append-only issue memory used to avoid repeating deployment failures."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = ensure_dir(memory_dir)
        self.issue_path = self.memory_dir / "deployment_issues.jsonl"

    def remember_issue(self, task_id: str, stage: str, result: StageResult, analysis: Dict) -> Optional[Dict]:
        if result.status in ("passed", "pass"):
            return None
        plain = to_plain(result)
        diagnosis = plain.get("data", {}).get("diagnosis", {}) if isinstance(plain.get("data"), dict) else {}
        category = diagnosis.get("category") or self._category_from_result(plain)
        frameworks = list(analysis.get("frameworks") or [])
        symptom = plain.get("error") or plain.get("summary") or "stage did not pass"
        root_cause = diagnosis.get("root_cause") or ""
        signature = short_hash(
            json.dumps(
                {
                    "stage": stage,
                    "category": category,
                    "frameworks": frameworks,
                    "symptom": symptom[-500:],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            12,
        )
        entry = {
            "id": "mem_%s" % signature,
            "created_at": utc_now_iso(),
            "task_id": task_id,
            "stage": stage,
            "category": category,
            "frameworks": frameworks,
            "signature": signature,
            "symptom": symptom[-2000:],
            "root_cause": root_cause,
            "fix_status": "unresolved",
            "suggested_next_action": self._suggest_next_action(stage, category),
            "source_result": {
                "status": plain.get("status"),
                "summary": plain.get("summary"),
                "evidence": plain.get("evidence", []),
            },
        }
        if not self._has_signature(signature):
            with self.issue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def query(self, stage: str, analysis: Dict, limit: int = 5) -> List[Dict]:
        if not self.issue_path.exists():
            return []
        frameworks = set(analysis.get("frameworks") or [])
        hits: List[Dict] = []
        for entry in self._read_entries():
            score = 0
            if entry.get("stage") == stage:
                score += 4
            overlap = frameworks.intersection(set(entry.get("frameworks") or []))
            score += len(overlap) * 2
            if score <= 0:
                continue
            item = dict(entry)
            item["score"] = score
            hits.append(item)
        return sorted(hits, key=lambda item: (-item["score"], item.get("created_at", "")))[:limit]

    def _read_entries(self) -> List[Dict]:
        entries: List[Dict] = []
        if not self.issue_path.exists():
            return entries
        with self.issue_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _has_signature(self, signature: str) -> bool:
        return any(entry.get("signature") == signature for entry in self._read_entries())

    def _category_from_result(self, plain: Dict) -> str:
        error = str(plain.get("error") or plain.get("summary") or "").lower()
        if "dependency" in error or "install" in error or "pip" in error:
            return "dependency_failure"
        if "port" in error or "service" in error or "exited" in error:
            return "service_start_failure"
        if "verify" in error or "evidence" in error:
            return "verification_gap"
        return "unknown"

    def _suggest_next_action(self, stage: str, category: str) -> str:
        if stage == "verify":
            return "Inspect service API shape and add a trace-producing verification request."
        if category == "dependency_failure":
            return "Review dependency logs, pin incompatible packages, then retry install in isolated env."
        if category == "service_start_failure":
            return "Inspect runner log, check port/process readiness, then try the next run candidate."
        return "Diagnose logs and update the matching deployment skill with reusable handling rules."
