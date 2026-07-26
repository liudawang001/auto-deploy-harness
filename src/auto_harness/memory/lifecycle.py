"""Tamper-evident lifecycle for memory-derived skill candidates."""

import hashlib
import json
from pathlib import Path
from typing import Dict

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


class SkillCandidateLifecycle:
    ALLOWED_TRANSITIONS = {
        "proposed": {"approved", "rejected"},
        "approved": {"regression_passed", "regression_failed", "rejected"},
        "regression_failed": {"approved", "regression_passed", "rejected"},
        "regression_passed": {"shadow_passed", "shadow_failed", "active", "rejected"},
        "shadow_failed": {"regression_passed", "rejected"},
        "shadow_passed": {"shadow_failed", "active", "rejected"},
        "active": {"rolled_back"},
        "rejected": set(),
        "rolled_back": set(),
    }

    def initialize(self, candidate_path: Path) -> Dict:
        candidate_path = Path(candidate_path)
        candidate = read_json(candidate_path)
        current = self.normalize_status(candidate.get("status"))
        if current not in self.ALLOWED_TRANSITIONS:
            current = "proposed"
        candidate["status"] = current
        if candidate.get("lifecycle", {}).get("last_event_hash"):
            write_json(candidate_path, candidate)
            return {"status": "already_initialized", "candidate_status": current}
        return self._record(candidate_path, candidate, "", current, "system", {"reason": "candidate created"})

    def transition(
        self,
        candidate_path: Path,
        target_status: str,
        actor: str,
        evidence: Dict = None,
        updates: Dict = None,
    ) -> Dict:
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}
        candidate = read_json(candidate_path)
        current = self.normalize_status(candidate.get("status"))
        target = self.normalize_status(target_status)
        if target == current:
            if updates:
                candidate.update(updates)
                write_json(candidate_path, candidate)
            return {
                "status": "unchanged",
                "candidate_id": candidate.get("candidate_id", ""),
                "candidate_status": current,
            }
        if target not in self.ALLOWED_TRANSITIONS.get(current, set()):
            return {
                "status": "failed",
                "candidate_id": candidate.get("candidate_id", ""),
                "candidate_status": current,
                "target_status": target,
                "error": "invalid lifecycle transition: %s -> %s" % (current, target),
            }
        if updates:
            candidate.update(updates)
        return self._record(candidate_path, candidate, current, target, actor, evidence or {})

    def normalize_status(self, status) -> str:
        value = str(status or "").strip().lower()
        return "proposed" if value in ("", "candidate") else value

    def _record(
        self,
        candidate_path: Path,
        candidate: Dict,
        previous_status: str,
        target_status: str,
        actor: str,
        evidence: Dict,
    ) -> Dict:
        lifecycle = candidate.get("lifecycle") if isinstance(candidate.get("lifecycle"), dict) else {}
        previous_hash = str(lifecycle.get("last_event_hash") or "")
        event = {
            "candidate_id": candidate.get("candidate_id", ""),
            "from_status": previous_status,
            "to_status": target_status,
            "actor": actor or "system",
            "occurred_at": utc_now_iso(),
            "evidence": evidence,
            "previous_event_hash": previous_hash,
        }
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        event_hash = hashlib.sha256((previous_hash + canonical).encode("utf-8")).hexdigest()
        event["event_hash"] = event_hash

        audit_path = candidate_path.with_suffix(".lifecycle.jsonl")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

        candidate["status"] = target_status
        candidate["lifecycle"] = {
            "schema_version": 1,
            "current_status": target_status,
            "last_event_hash": event_hash,
            "audit_path": str(audit_path),
        }
        write_json(candidate_path, candidate)
        return {
            "status": "transitioned",
            "candidate_id": candidate.get("candidate_id", ""),
            "from_status": previous_status,
            "candidate_status": target_status,
            "event_hash": event_hash,
            "audit_path": str(audit_path),
        }
