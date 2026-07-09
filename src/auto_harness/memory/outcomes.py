"""Skill Outcome Tracking: record and summarize skill selection and execution outcomes.

Every time a skill is selected and used during a deployment run, the outcome
is recorded to memory/skill_outcomes.jsonl. This enables:
- Tracking which skill versions helped or hurt
- Attributing success/failure to specific skill patches
- Feeding back into the shadow evaluation loop
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.utils.files import ensure_dir
from auto_harness.utils.time import utc_now_iso


class SkillOutcomeRecorder:
    """Record and summarize skill outcomes from deployment runs.

    Records are append-only to memory/skill_outcomes.jsonl.
    Each record links run_id → skill_name → skill_sha256 → outcome.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = ensure_dir(Path(memory_dir))
        self.outcomes_path = self.memory_dir / "skill_outcomes.jsonl"

    def record_run(
        self,
        run_id: str,
        stage: str,
        selected_skills: List[Dict],
        result: Dict,
        agent_metadata: Dict = None,
    ) -> Dict:
        """Record skill outcomes for a single run.

        Args:
            run_id: The deployment run ID.
            stage: The pipeline stage (e.g. "verify").
            selected_skills: List of skill dicts with name, path, sha256.
            result: The stage result dict with status, etc.
            agent_metadata: Optional agent metadata (llm_helped, tool_selected, etc.).

        Returns:
            Dict with recorded_count and records.
        """
        agent_metadata = agent_metadata or {}
        records = []

        if not selected_skills:
            # Record that no skill was selected
            record = {
                "created_at": utc_now_iso(),
                "run_id": run_id,
                "stage": stage,
                "skill_name": "",
                "skill_path": "",
                "skill_sha256": "",
                "candidate_id": "",
                "selected": False,
                "status": result.get("status", "unknown"),
                "llm_helped": agent_metadata.get("llm_helped", False),
                "tool_selected": agent_metadata.get("tool_selected", ""),
                "policy_rejected": agent_metadata.get("policy_rejected", False),
                "trace_verified": agent_metadata.get("trace_verified", False),
            }
            self._append(record)
            records.append(record)
        else:
            for skill in selected_skills:
                skill_path = str(skill.get("path", ""))
                skill_content = ""
                if skill_path and Path(skill_path).exists():
                    try:
                        skill_content = Path(skill_path).read_text(encoding="utf-8")
                    except OSError:
                        pass

                skill_sha = skill.get("sha256") or (hashlib.sha256(skill_content.encode("utf-8")).hexdigest() if skill_content else "")

                record = {
                    "created_at": utc_now_iso(),
                    "run_id": run_id,
                    "stage": stage,
                    "skill_name": skill.get("name", ""),
                    "skill_path": skill_path,
                    "skill_sha256": skill_sha,
                    "candidate_id": skill.get("candidate_id", ""),
                    "selected": True,
                    "status": result.get("status", "unknown"),
                    "llm_helped": agent_metadata.get("llm_helped", False),
                    "tool_selected": agent_metadata.get("tool_selected", ""),
                    "policy_rejected": agent_metadata.get("policy_rejected", False),
                    "trace_verified": agent_metadata.get("trace_verified", False),
                }
                self._append(record)
                records.append(record)

        return {
            "recorded_count": len(records),
            "records": records,
        }

    def summarize(self, skill_name: str = None, candidate_id: str = None) -> Dict:
        """Summarize skill outcome records.

        Args:
            skill_name: Filter by skill name (optional).
            candidate_id: Filter by candidate ID (optional).

        Returns:
            Dict with total, passed, failed, llm_helped_count, etc.
        """
        entries = self._read_entries()

        # Apply filters
        if skill_name:
            entries = [e for e in entries if e.get("skill_name") == skill_name]
        if candidate_id:
            entries = [e for e in entries if e.get("candidate_id") == candidate_id]

        total = len(entries)
        passed = sum(1 for e in entries if e.get("status") in ("pass", "passed", "success"))
        failed = sum(1 for e in entries if e.get("status") in ("fail", "failed"))
        uncertain = sum(1 for e in entries if e.get("status") == "uncertain")
        llm_helped = sum(1 for e in entries if e.get("llm_helped"))
        trace_verified = sum(1 for e in entries if e.get("trace_verified"))
        policy_rejected = sum(1 for e in entries if e.get("policy_rejected"))

        # Group by skill_sha256
        by_sha: Dict[str, Dict] = {}
        for entry in entries:
            sha = entry.get("skill_sha256", "")
            if sha not in by_sha:
                by_sha[sha] = {"count": 0, "passed": 0, "failed": 0, "llm_helped": 0}
            by_sha[sha]["count"] += 1
            if entry.get("status") in ("pass", "passed", "success"):
                by_sha[sha]["passed"] += 1
            if entry.get("status") in ("fail", "failed"):
                by_sha[sha]["failed"] += 1
            if entry.get("llm_helped"):
                by_sha[sha]["llm_helped"] += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "uncertain": uncertain,
            "llm_helped_count": llm_helped,
            "trace_verified_count": trace_verified,
            "policy_rejected_count": policy_rejected,
            "by_skill_sha": by_sha,
        }

    def _append(self, record: Dict) -> None:
        """Append a record to the outcomes JSONL file."""
        with self.outcomes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_entries(self) -> List[Dict]:
        """Read all entries from the outcomes JSONL file."""
        if not self.outcomes_path.exists():
            return []
        entries = []
        for line in self.outcomes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
