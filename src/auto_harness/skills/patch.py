"""Skill patch validation and application.

SkillPatchValidator checks candidate markdown for security violations.
SkillPatchApplier applies a validated candidate to a target skill file,
using marker blocks so changes are traceable and reversible.

Key rules:
- Patches are appended with <!-- auto-harness-skill-evolution:candidate_<id> --> markers
- base_skill_sha256 must match current target skill sha256 before apply
- Secrets, absolute paths, HTTP 200 false success, and privilege escalation are rejected
- The original skill content is never deleted or overwritten
"""
import hashlib
import re
from pathlib import Path
from typing import Dict

from auto_harness.models.base import write_json
from auto_harness.utils.files import ensure_dir
from auto_harness.utils.time import utc_now_iso


# Reuse security patterns from curator — same rules apply to skill patches
_SECRET_PATTERNS = re.compile(
    r"(api_key|token\s*=|password|secret|Authorization:|Bearer\s|private\s+key)",
    re.IGNORECASE,
)

_ABS_PATH_PATTERNS = re.compile(
    r"(/tmp/|/Users/\w|C:\\Users|/var/folders/)",
    re.IGNORECASE,
)

_PRIVILEGE_ESCALATION = re.compile(
    r"(allow\s+arbitrary\s+shell|allow\s+source\s+edit\s+by\s+default|"
    r"disable\s+trace\s+verification|bypass\s+regression)",
    re.IGNORECASE,
)

_HTTP_200_SUCCESS = re.compile(
    r"(HTTP\s*200\s*(is|means|=)\s*enough|"
    r"HTTP\s*200\s+alone\s*(as|is)\s+success|"
    r"mark\s+success\s+on\s+HTTP\s*200\s+alone|"
    r"HTTP\s*200\s*(alone\s*)?(is|means|as)\s+success|"
    r"200\s+(is|means)\s+enough|"
    r"HTTP\s*200\s+means\s+success)",
    re.IGNORECASE,
)

# Maximum allowed markdown length for a skill patch (prevents LLM from dumping huge content)
_MAX_MARKDOWN_LENGTH = 10000


class SkillPatchValidator:
    """Validate a skill patch candidate before it can be applied to a skill file."""

    def validate(self, markdown: str) -> Dict:
        """Validate patch markdown content.

        Returns:
            {
                "valid": bool,
                "reasons": list[str],  # why it's valid
                "reject_reasons": list[str],  # why it's rejected
            }
        """
        reasons = []
        reject_reasons = []

        # Check markdown is non-empty
        if not markdown or not markdown.strip():
            reject_reasons.append("markdown is empty")
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("markdown non-empty")

        # Check length is reasonable
        if len(markdown) > _MAX_MARKDOWN_LENGTH:
            reject_reasons.append("markdown exceeds max length (%d > %d)" % (len(markdown), _MAX_MARKDOWN_LENGTH))
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("markdown length reasonable")

        # Check for secrets
        if _SECRET_PATTERNS.search(markdown):
            reject_reasons.append("contains secret-like content (api_key, token, password, etc.)")
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("no secrets detected")

        # Check for absolute one-off paths
        if _ABS_PATH_PATTERNS.search(markdown):
            reject_reasons.append("contains absolute one-off path (/tmp/, /Users/, C:\\Users)")
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("no absolute paths")

        # Check for HTTP 200 false success
        if _HTTP_200_SUCCESS.search(markdown):
            reject_reasons.append("contains HTTP 200 as success rule")
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("no HTTP 200 false success")

        # Check for privilege escalation
        if _PRIVILEGE_ESCALATION.search(markdown):
            reject_reasons.append("suggests expanding shell or source edit permissions")
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("no privilege escalation")

        # Check for deletion of original content (heuristic: no "delete" or "remove" original sections)
        # This is a soft check — we don't allow "delete the existing" patterns
        if re.search(r"delete\s+(the\s+)?(existing|original|current)\s+(section|content|rule)", markdown, re.IGNORECASE):
            reject_reasons.append("suggests deleting original skill content")
            return {"valid": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("no deletion of original content")

        return {"valid": True, "reasons": reasons, "reject_reasons": reject_reasons}


class SkillPatchApplier:
    """Apply a validated skill patch candidate to a target skill file.

    Uses marker blocks so the patch is traceable and reversible.
    Writes rollback copy before modifying the skill.
    """

    def apply_candidate(self, candidate: Dict, skills_dir: Path) -> Dict:
        """Apply a candidate patch to the target skill.

        Args:
            candidate: Dict with target_skill, base_skill_sha256, patch (section_title, markdown),
                       candidate_id.
            skills_dir: Path to the skills directory.

        Returns:
            Dict with status, target_skill path, previous_sha256, new_sha256, rollback_path.
        """
        candidate_id = candidate.get("candidate_id", "unknown")
        target_skill_rel = candidate.get("target_skill", "")
        base_sha = candidate.get("base_skill_sha256", "")
        patch = candidate.get("patch", {})
        markdown = patch.get("markdown", "")

        target_path = skills_dir / target_skill_rel

        # Check target exists
        if not target_path.exists():
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "target skill does not exist: %s" % str(target_path),
            }

        # Read current content
        raw = target_path.read_text(encoding="utf-8")

        # Check base sha matches
        current_sha = _sha256(raw)
        if base_sha and current_sha != base_sha:
            return {
                "status": "base_changed",
                "candidate_id": candidate_id,
                "current_sha256": current_sha,
                "expected_sha256": base_sha,
                "error": "target skill has been modified since candidate was created",
            }

        # Check marker not already present
        marker = "auto-harness-skill-evolution:%s" % candidate_id
        if marker in raw:
            return {
                "status": "already_applied",
                "candidate_id": candidate_id,
                "target_skill": str(target_path),
            }

        # Write rollback copy
        history_dir = target_path.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        rollback_path = history_dir / ("%s_%s.md" % (
            utc_now_iso().replace(":", "").replace("-", "").split(".")[0],
            candidate_id,
        ))
        rollback_path.write_text(raw, encoding="utf-8")

        # Apply patch with marker
        section_title = patch.get("section_title", "Memory Evolution Patch")
        block = "\n\n<!-- %s -->\n## %s\n%s\n<!-- /%s -->\n" % (
            marker,
            section_title,
            markdown.strip(),
            marker,
        )
        new_raw = raw.rstrip() + block
        target_path.write_text(new_raw, encoding="utf-8")

        new_sha = _sha256(new_raw)

        return {
            "status": "applied",
            "candidate_id": candidate_id,
            "target_skill": str(target_path),
            "previous_sha256": current_sha,
            "new_sha256": new_sha,
            "rollback_path": str(rollback_path),
            "marker": marker,
        }


def _sha256(text: str) -> str:
    """Compute sha256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
