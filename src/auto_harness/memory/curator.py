"""LLM Memory Curator: call LLM to generalize verified memories into skill patch candidates.

The curator takes a cluster of verified memory entries, sends them to an LLM provider,
and parses the JSON response into a structured candidate draft. It does NOT modify
any official skill files — it only produces candidate drafts for downstream gating.

Security: The curator validates LLM output for secrets, absolute paths, HTTP 200
false-success rules, and privilege escalation suggestions.
"""
import json
import re
from typing import Dict, Optional

from auto_harness.utils.time import utc_now_iso


# Patterns that indicate secret-like content
_SECRET_PATTERNS = re.compile(
    r"(api_key|token\s*=|password|secret|Authorization:|Bearer\s|private\s+key)",
    re.IGNORECASE,
)

# Absolute one-off paths
_ABS_PATH_PATTERNS = re.compile(
    r"(/tmp/|/Users/\w|C:\\Users|/var/folders/)",
    re.IGNORECASE,
)

# Patterns that suggest weakening security
_PRIVILEGE_ESCALATION = re.compile(
    r"(allow\s+arbitrary\s+shell|allow\s+source\s+edit\s+by\s+default|"
    r"disable\s+trace\s+verification|bypass\s+regression)",
    re.IGNORECASE,
)

# HTTP 200 alone treated as success
_HTTP_200_SUCCESS = re.compile(
    r"(HTTP\s*200\s*(is|means|=)\s*enough|"
    r"HTTP\s*200\s+alone\s*(as|is)\s+success|"
    r"mark\s+success\s+on\s+HTTP\s*200\s+alone|"
    r"HTTP\s*200\s*(alone\s*)?(is|means|as)\s+success|"
    r"200\s+(is|means)\s+enough|"
    r"HTTP\s*200\s+means\s+success)",
    re.IGNORECASE,
)

# Phrases that indicate a prohibition — safe, should not trigger detectors
_DO_NOT_PHRASES = re.compile(
    r"(do\s+not\s+mark\s+success\s+on\s+HTTP\s*200\s+alone|"
    r"do\s+not\s+reuse\s+old\s+trace_id|"
    r"do\s+not\s+include\s+secrets?|"
    r"do\s+not\s+(?:allow|use)\s+(?:arbitrary|shell|source\s+edit))",
    re.IGNORECASE,
)


def _strip_do_not_phrases(text: str) -> str:
    """Remove 'do not X' prohibition phrases before security validation."""
    return _DO_NOT_PHRASES.sub("", text)

# Minimum required top-level keys in a valid curator response
_REQUIRED_KEYS = {"status", "pattern", "reusable_rule", "skill_patch"}

# Minimum required keys in skill_patch
_REQUIRED_PATCH_KEYS = {"target_skill", "section_title", "markdown"}


class MemoryCurator:
    """Generalize verified memory clusters into skill patch candidates via LLM.

    The curator:
    1. Takes a verified memory cluster + optional target skill content
    2. Sends a structured prompt to the LLM provider
    3. Parses the JSON response
    4. Validates for security and completeness
    5. Returns a candidate draft dict

    It never writes to skill files directly.
    """

    def __init__(self, provider=None, max_input_chars: int = 20000):
        self.provider = provider
        self.max_input_chars = max_input_chars

    def curate(self, cluster: dict, target_skill_content: str = "") -> dict:
        """Generate a skill patch candidate draft from a verified memory cluster.

        Args:
            cluster: Dict with stage, category, frameworks, memory_ids, symptoms,
                     root_causes, repair_actions, verification_trace_ids, regression_case_ids.
            target_skill_content: Current content of the target skill file (for context).

        Returns:
            Dict with status, candidate_draft (or error), and metadata.
        """
        if not self.provider:
            return {
                "status": "failed",
                "error": "no LLM provider configured",
                "candidate_draft": None,
            }

        prompt = self._build_prompt(cluster, target_skill_content)
        if len(prompt) > self.max_input_chars:
            prompt = prompt[: self.max_input_chars]

        try:
            result = self.provider.complete([{"role": "user", "content": prompt}])
            raw_text = result.text if hasattr(result, "text") else str(result)
        except Exception as exc:
            return {
                "status": "failed",
                "error": "LLM provider error: %s" % str(exc)[:200],
                "candidate_draft": None,
            }

        parsed = self.parse_response(raw_text)
        if parsed.get("status") != "ok":
            return {
                "status": "failed",
                "error": parsed.get("error", "parse failed"),
                "candidate_draft": None,
                "raw_response_hash": _sha256(raw_text),
            }

        # Security validation on parsed output
        validation = self._validate_candidate(parsed)
        if not validation["valid"]:
            return {
                "status": "rejected",
                "error": validation["reason"],
                "candidate_draft": None,
                "raw_response_hash": _sha256(raw_text),
            }

        return {
            "status": "ok",
            "candidate_draft": parsed,
            "raw_response_hash": _sha256(raw_text),
            "curated_at": utc_now_iso(),
        }

    def parse_response(self, text: str) -> dict:
        """Parse LLM response text into a structured dict.

        Returns:
            Dict with status and parsed content, or status=error with reason.
        """
        text = text.strip()

        # Try to extract JSON from the response
        # First, try direct parse
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try to find JSON block in markdown code fence
            json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1).strip())
                except (json.JSONDecodeError, ValueError):
                    return {"status": "error", "error": "invalid JSON in LLM response"}
            else:
                return {"status": "error", "error": "LLM response is not valid JSON"}

        if not isinstance(data, dict):
            return {"status": "error", "error": "LLM response is not a JSON object"}

        # Check top-level status
        if data.get("status") != "ok":
            return {"status": "error", "error": "LLM response status is not 'ok': %s" % data.get("status")}

        # Check required keys
        missing = _REQUIRED_KEYS - set(data.keys())
        if missing:
            return {"status": "error", "error": "missing required keys: %s" % ", ".join(sorted(missing))}

        # Check required skill_patch keys
        patch = data.get("skill_patch", {})
        if not isinstance(patch, dict):
            return {"status": "error", "error": "skill_patch is not a dict"}
        missing_patch = _REQUIRED_PATCH_KEYS - set(patch.keys())
        if missing_patch:
            return {"status": "error", "error": "skill_patch missing keys: %s" % ", ".join(sorted(missing_patch))}

        return data

    def _validate_candidate(self, data: dict) -> dict:
        """Validate a parsed candidate draft for security violations.

        Returns:
            {"valid": bool, "reason": str}
        """
        markdown = data.get("skill_patch", {}).get("markdown", "")

        # Strip "do not X" phrases from markdown — they are prohibitions,
        # not recommendations. Without this, "Do not mark success on HTTP 200
        # alone" would falsely trigger the HTTP 200 success detector.
        filtered = _strip_do_not_phrases(markdown)

        # Also strip reusable_rule.do_not items
        rule = data.get("reusable_rule", {})
        do_not_items = rule.get("do_not", [])
        if isinstance(do_not_items, list):
            for item in do_not_items:
                filtered = filtered.replace(str(item), "")

        # Check for secrets
        if _SECRET_PATTERNS.search(filtered):
            return {"valid": False, "reason": "secret-like content in skill_patch markdown"}

        # Check for absolute paths
        if _ABS_PATH_PATTERNS.search(filtered):
            return {"valid": False, "reason": "absolute one-off path in skill_patch markdown"}

        # Check for HTTP 200 false success rule (only in positive assertions, not do_not)
        if _HTTP_200_SUCCESS.search(filtered):
            return {"valid": False, "reason": "HTTP 200 alone as success rule in skill_patch markdown"}

        # Check for privilege escalation
        if _PRIVILEGE_ESCALATION.search(filtered):
            return {"valid": False, "reason": "privilege escalation suggestion in skill_patch markdown"}

        return {"valid": True, "reason": ""}

    def _build_prompt(self, cluster: dict, target_skill_content: str) -> str:
        """Build the LLM prompt from cluster data and target skill content."""
        prompt_data = {
            "task": "generalize verified deployment memories into a reusable skill patch candidate",
            "cluster": {
                "stage": cluster.get("stage", "unknown"),
                "category": cluster.get("category", "unknown"),
                "frameworks": cluster.get("frameworks", []),
                "memory_ids": cluster.get("memory_ids", []),
                "symptoms": cluster.get("symptoms", [])[:5],
                "root_causes": cluster.get("root_causes", [])[:5],
                "repair_actions": cluster.get("repair_actions", [])[:5],
                "verification_trace_ids": cluster.get("verification_trace_ids", [])[:3],
                "regression_case_ids": cluster.get("regression_case_ids", [])[:5],
            },
            "target_skill_excerpt": target_skill_content[:2000] if target_skill_content else "",
            "constraints": [
                "Do not include secrets.",
                "Do not include one-off absolute paths.",
                "Do not mark HTTP 200 alone as success.",
                "Only propose reusable rules.",
                "Output JSON only.",
                "Do not suggest bypassing trace verification.",
                "Do not suggest expanding shell or source edit permissions.",
            ],
        }
        return json.dumps(prompt_data, ensure_ascii=False, indent=2)


def _sha256(text: str) -> str:
    """Compute sha256 hash of text."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
