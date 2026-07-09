"""Memory quality gate: classify and filter memory entries for evolution.

Only verified_resolution and above are eligible for skill evolution.
metadata_only repairs, unverified entries, and high-risk rejections are excluded.
"""
import re
from typing import Dict, List


# Patterns that indicate secret-like values in suggested actions
_SECRET_PATTERNS = re.compile(
    r"(api_key|token\s*=|password|secret|Authorization:|Bearer\s)",
    re.IGNORECASE,
)

# Patterns for absolute one-off paths
_ABS_PATH_PATTERNS = re.compile(
    r"(/tmp/|/Users/\w|C:\\Users|/var/folders/)",
    re.IGNORECASE,
)


class MemoryQualityGate:
    """Classify memory entries and determine evolution eligibility.

    Quality levels:
      raw_issue:          failed/uncertain memory, not eligible
      diagnosed_issue:    has root_cause/diagnosis but no verified pass, not eligible
      verified_resolution: final verify passed + trace_id + repair_action_hash + truly executed
      regression_proven:  verified_resolution + regression passed + case_ids non-empty
      production_proven:  multiple active skill outcomes (not fully implemented this round)
    """

    def classify(self, entry: Dict) -> Dict:
        """Classify a memory entry and return its quality level + eligibility.

        Returns:
            {
                "level": str,
                "eligible": bool,
                "reasons": list[str],
                "reject_reasons": list[str],
            }
        """
        reasons: List[str] = []
        reject_reasons: List[str] = []

        # Check verified_success
        if entry.get("verified_success") is not True:
            reject_reasons.append("verified_success != true")
            return {
                "level": "raw_issue" if not entry.get("root_cause") else "diagnosed_issue",
                "eligible": False,
                "reasons": reasons,
                "reject_reasons": reject_reasons,
            }
        reasons.append("verified_success=true")

        # Check verification_trace_id
        trace_id = str(entry.get("verification_trace_id") or "").strip()
        if not trace_id:
            reject_reasons.append("verification_trace_id missing")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("verification_trace_id present")

        # Check repair_action_hash
        repair_hash = str(entry.get("repair_action_hash") or "").strip()
        if not repair_hash:
            reject_reasons.append("repair_action_hash missing")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}
        reasons.append("repair_action_hash present")

        # Check policy_rejected_high_risk
        if entry.get("policy_rejected_high_risk") is True or entry.get("rejected_high_risk_action") is True:
            reject_reasons.append("policy_rejected_high_risk=true")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}

        # Check repair_action_status
        repair_status = str(entry.get("repair_action_status") or "").lower()
        if repair_status and repair_status not in ("executed", "success", "passed", "succeeded"):
            reject_reasons.append("repair_action_status not in executed/success/passed/succeeded")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}

        # Check metadata_only
        if entry.get("metadata_only") is True:
            reject_reasons.append("metadata_only=true")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}

        # Check for secret-like values in suggested_next_action
        suggested = str(entry.get("suggested_next_action") or "")
        if _SECRET_PATTERNS.search(suggested):
            reject_reasons.append("secret-like value present in suggested action")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}

        # Check for absolute tmp paths in suggested action
        if _ABS_PATH_PATTERNS.search(suggested):
            reject_reasons.append("absolute tmp path in suggested action")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}

        # Check regression_status
        regression_status = str(entry.get("regression_status") or "").lower()
        if regression_status == "failed":
            reject_reasons.append("regression_status failed")
            return {"level": "diagnosed_issue", "eligible": False, "reasons": reasons, "reject_reasons": reject_reasons}

        # At this point, we have verified_resolution at minimum
        level = "verified_resolution"

        # Check regression_proven
        regression_case_ids = entry.get("regression_case_ids")
        if (
            regression_status in ("pass", "passed", "success", "succeeded")
            and isinstance(regression_case_ids, list)
            and regression_case_ids
        ):
            level = "regression_proven"
            reasons.append("regression passed with case_ids")

        return {
            "level": level,
            "eligible": True,
            "reasons": reasons,
            "reject_reasons": reject_reasons,
        }

    def eligible_for_evolution(self, entry: Dict) -> bool:
        """Return True if the entry is eligible for skill evolution."""
        return self.classify(entry)["eligible"]

    def filter_verified(self, entries: List[Dict]) -> List[Dict]:
        """Filter a list of entries, returning only those eligible for evolution."""
        return [entry for entry in entries if self.eligible_for_evolution(entry)]
