"""Skill Rollback Manager: reverse a promoted skill candidate.

Rollback restores the skill file to its pre-promotion state using the
rollback copy created during SkillPatchApplier.apply_candidate().

Key rules:
- Rollback only works for candidates that were promoted (status=active)
- The rollback_path must exist on disk
- Current skill is saved to history before restoring
- Candidate status is updated to rolled_back
"""
import hashlib
import json
from pathlib import Path
from typing import Dict

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class SkillRollbackManager:
    """Manage rollback of promoted skill candidates."""

    def rollback_candidate(self, candidate_path: Path) -> Dict:
        """Rollback a promoted candidate, restoring the pre-promotion skill.

        Args:
            candidate_path: Path to the candidate JSON file.

        Returns:
            Dict with rollback status.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")
        promotion = candidate.get("promotion", {})

        # Check candidate was promoted
        if candidate.get("status") != "active":
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "candidate status is '%s', not 'active'" % candidate.get("status"),
            }

        # Get rollback path
        rollback_path_str = promotion.get("rollback_path", "")
        if not rollback_path_str:
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "no rollback_path in promotion metadata",
            }

        rollback_path = Path(rollback_path_str)
        if not rollback_path.exists():
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "rollback file not found: %s" % rollback_path_str,
            }

        # Get target skill path
        target_skill_rel = candidate.get("target_skill", "")
        # The target skill is relative to skills_dir, but we need the full path
        # We can derive it from the promotion info or from the marker
        # Use the rollback_path's parent structure to find the target
        # Actually, the target skill path was set during apply_candidate
        # We need the skills_dir. Let's compute it from rollback_path structure.
        # rollback_path is skills/<skill>/history/<timestamp>_<candidate_id>.md
        # target skill is skills/<skill>/SKILL.md

        # Derive target skill path from rollback path
        # rollback_path: skills/<skill-name>/history/20260709_skillcand_xxx.md
        # target:        skills/<skill-name>/SKILL.md
        skill_dir = rollback_path.parent.parent
        target_path = skill_dir / "SKILL.md"

        if not target_path.exists():
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "target skill not found: %s" % str(target_path),
            }

        # Save current skill to history before restoring
        current_content = target_path.read_text(encoding="utf-8")
        history_dir = rollback_path.parent
        history_dir.mkdir(parents=True, exist_ok=True)
        pre_rollback_path = history_dir / ("%s_pre_rollback_%s.md" % (
            utc_now_iso().replace(":", "").replace("-", "").split(".")[0],
            candidate_id,
        ))
        pre_rollback_path.write_text(current_content, encoding="utf-8")

        # Restore from rollback
        rollback_content = rollback_path.read_text(encoding="utf-8")
        target_path.write_text(rollback_content, encoding="utf-8")

        # Compute restored sha
        restored_sha = hashlib.sha256(rollback_content.encode("utf-8")).hexdigest()
        previous_sha = hashlib.sha256(current_content.encode("utf-8")).hexdigest()

        # Update candidate
        candidate["status"] = "rolled_back"
        candidate["rollback"] = {
            "rolled_back_at": utc_now_iso(),
            "restored_sha256": restored_sha,
            "previous_active_sha256": previous_sha,
            "pre_rollback_backup": str(pre_rollback_path),
        }
        write_json(candidate_path, candidate)

        return {
            "status": "rolled_back",
            "candidate_id": candidate_id,
            "target_skill": str(target_path),
            "restored_sha256": restored_sha,
            "previous_active_sha256": previous_sha,
            "pre_rollback_backup": str(pre_rollback_path),
        }

    def rollback_to_history(self, skill_path: Path, history_path: Path) -> Dict:
        """Restore a skill file from a specific history backup.

        Args:
            skill_path: Path to the target SKILL.md.
            history_path: Path to the history backup file.

        Returns:
            Dict with rollback status.
        """
        skill_path = Path(skill_path)
        history_path = Path(history_path)

        if not skill_path.exists():
            return {"status": "failed", "error": "target skill not found"}
        if not history_path.exists():
            return {"status": "failed", "error": "history file not found"}

        # Save current before restoring
        current_content = skill_path.read_text(encoding="utf-8")
        history_dir = skill_path.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        pre_rollback_path = history_dir / ("%s_manual_rollback.md" % (
            utc_now_iso().replace(":", "").replace("-", "").split(".")[0],
        ))
        pre_rollback_path.write_text(current_content, encoding="utf-8")

        # Restore from history
        history_content = history_path.read_text(encoding="utf-8")
        skill_path.write_text(history_content, encoding="utf-8")

        restored_sha = hashlib.sha256(history_content.encode("utf-8")).hexdigest()
        previous_sha = hashlib.sha256(current_content.encode("utf-8")).hexdigest()

        return {
            "status": "rolled_back",
            "target_skill": str(skill_path),
            "restored_sha256": restored_sha,
            "previous_active_sha256": previous_sha,
            "pre_rollback_backup": str(pre_rollback_path),
        }
