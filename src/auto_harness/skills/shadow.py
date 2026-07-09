"""Shadow Skill Evaluation: evaluate candidate skill patches without affecting real deployments.

The shadow evaluator reads the run results and checks whether a candidate's
recommended tools/actions would have helped or harmed the deployment outcome.

Key properties:
- Shadow evaluation does NOT change real deployment behavior
- It only reads and compares: candidate recommendations vs actual outcomes
- Results are written to shadow artifacts for later promotion gating
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class ShadowSkillEvaluator:
    """Evaluate a candidate skill patch in shadow mode against a real run.

    Shadow mode:
    1. Read run_dir/reports/agent_verify_result.json (actual outcome)
    2. Read run_dir/agent_verify_steps.jsonl (actual steps taken)
    3. Read candidate pattern / target skill
    4. Check if candidate matches current stage/framework/failure_signature
    5. If candidate recommended tool matches the helped tool → would_help=true
    6. If candidate recommends bypassing trace or expanding permissions → would_harm=true
    7. Write shadow evaluation artifact
    """

    # Default shadow promotion thresholds
    DEFAULT_HELPED_THRESHOLD = 2
    DEFAULT_HARMFUL_THRESHOLD = 0

    def evaluate_run(self, run_dir: Path, candidate_path: Path, active_context: dict = None) -> Dict:
        """Evaluate a candidate against a single run in shadow mode.

        Args:
            run_dir: Path to the run directory.
            candidate_path: Path to the candidate JSON file.
            active_context: Optional dict with stage, frameworks, etc.

        Returns:
            Shadow evaluation result dict.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")
        pattern = candidate.get("pattern", {})
        reusable_rule = candidate.get("reusable_rule", {})

        # Read actual outcome from the run
        agent_verify_result = self._read_json(run_dir / "reports" / "agent_verify_result.json")
        agent_steps = self._read_jsonl(run_dir / "agent_verify_steps.jsonl")

        # Determine match
        matched = self._check_match(pattern, active_context or agent_verify_result or {})

        # Check if candidate would help
        would_help = False
        would_harm = False
        reason = ""

        if not matched:
            reason = "candidate pattern does not match current run context"
        else:
            # Check if candidate's recommended actions align with successful steps
            recommended_tools = self._extract_recommended_tools(reusable_rule)
            successful_tools = self._extract_successful_tools(agent_steps)

            if recommended_tools and successful_tools:
                overlap = set(recommended_tools) & set(successful_tools)
                if overlap:
                    would_help = True
                    reason = "candidate recommends %s, same as successful agent step(s)" % ", ".join(overlap)

            # Check for harmful recommendations
            do_not = reusable_rule.get("do_not", [])
            do_items = reusable_rule.get("do", [])
            all_items = do_items + do_not

            for item in all_items:
                item_lower = str(item).lower()
                if any(term in item_lower for term in ("bypass trace", "disable verification", "skip regression")):
                    # If the item is in do_not, it's actually safe (prohibiting the bad thing)
                    if item in do_not:
                        continue
                    would_harm = True
                    reason = "candidate suggests bypassing trace verification"
                    break
                if any(term in item_lower for term in ("allow shell", "source edit by default", "arbitrary command")):
                    if item in do_not:
                        continue
                    would_harm = True
                    reason = "candidate suggests expanding permissions"
                    break

            if not reason:
                if would_help:
                    reason = "candidate recommendations align with successful outcomes"
                else:
                    reason = "candidate matched but no overlap with successful tools"

        # Compute active skill sha for audit
        active_skill_sha = ""
        target_skill = candidate.get("target_skill", "")
        if target_skill and active_context:
            active_skill_sha = active_context.get("skill_sha256", "")

        result = {
            "candidate_id": candidate_id,
            "run_id": run_dir.name,
            "matched": matched,
            "would_help": would_help,
            "would_harm": would_harm,
            "reason": reason,
            "active_skill_sha256": active_skill_sha,
            "candidate_base_sha256": candidate.get("base_skill_sha256", ""),
            "evaluated_at": utc_now_iso(),
        }

        return result

    def record(self, candidate_path: Path, result: Dict) -> Dict:
        """Record a shadow evaluation result in the candidate file.

        Updates the candidate's shadow section with accumulated counts.

        Args:
            candidate_path: Path to the candidate JSON file.
            result: Shadow evaluation result from evaluate_run().

        Returns:
            Updated shadow section dict.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        shadow = candidate.get("shadow", {})

        # Initialize shadow if needed
        if not shadow.get("enabled"):
            shadow = {
                "enabled": True,
                "helped_count": 0,
                "harmful_count": 0,
                "evaluations": [],
            }

        # Update counts
        if result.get("would_help"):
            shadow["helped_count"] = shadow.get("helped_count", 0) + 1
        if result.get("would_harm"):
            shadow["harmful_count"] = shadow.get("harmful_count", 0) + 1

        # Append evaluation record
        evaluations = shadow.get("evaluations", [])
        evaluations.append({
            "run_id": result.get("run_id", ""),
            "matched": result.get("matched", False),
            "would_help": result.get("would_help", False),
            "would_harm": result.get("would_harm", False),
            "reason": result.get("reason", ""),
            "evaluated_at": result.get("evaluated_at", ""),
        })
        shadow["evaluations"] = evaluations

        # Update candidate
        candidate["shadow"] = shadow

        # Update status if shadow threshold met
        helped = shadow["helped_count"]
        harmful = shadow["harmful_count"]
        if helped >= self.DEFAULT_HELPED_THRESHOLD and harmful <= self.DEFAULT_HARMFUL_THRESHOLD:
            candidate["status"] = "shadow_passed"
        elif harmful > self.DEFAULT_HARMFUL_THRESHOLD:
            candidate["status"] = "shadow_failed"

        write_json(candidate_path, candidate)

        # Also write shadow artifact alongside candidate
        shadow_artifact_path = candidate_path.with_suffix(".shadow.json")
        existing_shadow = []
        if shadow_artifact_path.exists():
            try:
                existing_shadow = json.loads(shadow_artifact_path.read_text(encoding="utf-8"))
                if not isinstance(existing_shadow, list):
                    existing_shadow = []
            except (json.JSONDecodeError, ValueError):
                existing_shadow = []
        existing_shadow.append(result)
        write_json(shadow_artifact_path, existing_shadow)

        return shadow

    def _check_match(self, pattern: Dict, context: Dict) -> bool:
        """Check if candidate pattern matches the current run context."""
        if not pattern or not context:
            return False

        # Match stage
        pattern_stage = pattern.get("stage", "")
        ctx_stage = context.get("stage", "")
        if pattern_stage and ctx_stage and pattern_stage != ctx_stage:
            return False

        # Match frameworks
        pattern_fw = set(pattern.get("frameworks") or [])
        ctx_fw = set(context.get("frameworks") or [])
        if pattern_fw and ctx_fw and not pattern_fw & ctx_fw:
            return False

        return True

    def _extract_recommended_tools(self, rule: Dict) -> List[str]:
        """Extract tool names from candidate's reusable_rule.do items."""
        tools = []
        for item in rule.get("do", []):
            item_lower = str(item).lower()
            # Map common action descriptions to tool names
            if "probe_http" in item_lower or "http" in item_lower:
                tools.append("probe_http")
            elif "gradio" in item_lower or "config" in item_lower:
                tools.append("discover_gradio_api")
            elif "openapi" in item_lower:
                tools.append("discover_openapi_schema")
            elif "browser" in item_lower or "dom" in item_lower:
                tools.append("probe_browser_dom")
        return tools

    def _extract_successful_tools(self, steps: List[Dict]) -> List[str]:
        """Extract tool names from successful agent steps."""
        tools = []
        if not isinstance(steps, list):
            return tools
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("status") in ("passed", "ok") and step.get("tool_name"):
                tools.append(step["tool_name"])
        return tools

    def _read_json(self, path: Path) -> Optional[Dict]:
        """Read a JSON file, return None if not found."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def _read_jsonl(self, path: Path) -> List[Dict]:
        """Read a JSONL file, return empty list if not found."""
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
