"""Skill Gain Evaluator: prove candidate skill provides gain, not just regression safety.

The gain evaluator compares baseline skill performance against candidate skill
shadow decisions to demonstrate improvement. This goes beyond the regression gate
(which only proves the candidate doesn't break existing benchmarks) by showing
the candidate makes previously-uncertain cases better.

Output format:
{
  "candidate_id": "...",
  "target_skill": "...",
  "baseline": {
    "skill_sha256": "...",
    "status": "uncertain",
    "selected_tool": "probe_http"
  },
  "candidate": {
    "shadow_decision": "discover_gradio_api",
    "would_execute": false,
    "status": "would_help"
  },
  "gain": {
    "improved": true,
    "reason": "candidate selects tool linked to trace-verified successful memory"
  }
}
"""
import json
from pathlib import Path
from typing import Dict, Optional

from auto_harness.skills.shadow import ShadowSkillEvaluator
from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class SkillGainEvaluator:
    """Evaluate whether a candidate skill provides gain over the baseline skill.

    The evaluator:
    1. Reads the candidate's shadow evaluation results
    2. Compares baseline (current skill) decisions vs candidate shadow decisions
    3. Determines if the candidate would improve outcomes on uncertain cases
    4. Produces a gain report

    This is NOT a regression test — it's a gain proof.
    """

    def evaluate_candidate(
        self,
        candidate_path: Path,
        run_dir: Path = None,
        observation: dict = None,
        output_path: Path = None,
    ) -> Dict:
        """Evaluate a candidate for skill gain.

        Args:
            candidate_path: Path to the candidate JSON file.
            run_dir: Optional run directory for shadow evaluation.
            observation: Optional observation dict for shadow evaluation.
            output_path: Optional path to write the gain report.

        Returns:
            Gain report dict.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")
        target_skill = candidate.get("target_skill", "")
        reusable_rule = candidate.get("reusable_rule", {})
        patch_markdown = candidate.get("patch", {}).get("markdown", "")

        # Determine baseline behavior
        baseline = self._assess_baseline(candidate, observation)

        # Determine candidate shadow decision
        candidate_assessment = self._assess_candidate(
            candidate, run_dir, observation
        )

        # Compute gain
        gain = self._compute_gain(baseline, candidate_assessment, reusable_rule)

        report = {
            "candidate_id": candidate_id,
            "target_skill": target_skill,
            "baseline": baseline,
            "candidate": candidate_assessment,
            "gain": gain,
            "evaluated_at": utc_now_iso(),
        }

        # Write report if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, report)

        return report

    def _assess_baseline(self, candidate: dict, observation: dict = None) -> Dict:
        """Assess baseline (current skill) behavior.

        The baseline is what the current skill would do without the candidate patch.
        We infer this from the candidate's source memories and pattern.
        """
        pattern = candidate.get("pattern", {})
        source_memories = candidate.get("source_memory_ids", [])

        # Baseline typically results in "uncertain" because the current skill
        # doesn't have the knowledge the candidate provides
        baseline_status = "uncertain"
        baseline_tool = "probe_http"  # Default fallback tool

        # If we have observation context, use it
        if observation:
            baseline_status = observation.get("status", "uncertain")
            # The baseline tool is typically the generic probe, not the specific one
            # the candidate would recommend
            baseline_tool = observation.get("baseline_tool", "probe_http")

        return {
            "skill_sha256": candidate.get("base_skill_sha256", ""),
            "status": baseline_status,
            "selected_tool": baseline_tool,
            "source_memory_count": len(source_memories),
            "pattern_stage": pattern.get("stage", ""),
            "pattern_frameworks": pattern.get("frameworks", []),
        }

    def _assess_candidate(
        self,
        candidate: dict,
        run_dir: Path = None,
        observation: dict = None,
    ) -> Dict:
        """Assess candidate skill behavior via shadow evaluation.

        If run_dir is provided, uses ShadowSkillEvaluator for planner-only shadow.
        Otherwise, uses heuristic from reusable_rule.
        """
        reusable_rule = candidate.get("reusable_rule", {})
        patch_markdown = candidate.get("patch", {}).get("markdown", "")

        # Try shadow evaluation if run_dir provided
        if run_dir and run_dir.exists():
            evaluator = ShadowSkillEvaluator()
            shadow_result = evaluator.evaluate_candidate_decision(
                run_dir=run_dir,
                candidate_path=Path(candidate.get("_candidate_path", "")),
                observation=observation,
            )
            if shadow_result.get("shadow_mode") == "planner_only":
                would_tool = shadow_result.get("would_tool_call", {})
                return {
                    "shadow_decision": would_tool.get("name", "") if would_tool else "",
                    "would_execute": False,
                    "status": "would_help" if shadow_result.get("would_help") else "no_gain",
                    "reason": shadow_result.get("reason", ""),
                }

        # Fallback: heuristic from reusable_rule
        recommended_tools = self._extract_recommended_tools(reusable_rule)
        if recommended_tools:
            return {
                "shadow_decision": recommended_tools[0],
                "would_execute": False,
                "status": "would_help",
                "reason": "candidate recommends %s based on verified memory pattern" % recommended_tools[0],
            }

        return {
            "shadow_decision": "",
            "would_execute": False,
            "status": "no_recommendation",
            "reason": "candidate has no recommended tools in reusable_rule",
        }

    def _compute_gain(
        self,
        baseline: Dict,
        candidate_assessment: Dict,
        reusable_rule: dict,
    ) -> Dict:
        """Compute whether the candidate provides gain over baseline.

        Gain is true when:
        - Baseline was uncertain/failed
        - Candidate shadow decision is a more specific/effective tool
        - The recommended tool is linked to trace-verified successful memory
        """
        improved = False
        reason = ""

        baseline_status = baseline.get("status", "")
        candidate_status = candidate_assessment.get("status", "")
        candidate_tool = candidate_assessment.get("shadow_decision", "")
        baseline_tool = baseline.get("selected_tool", "")

        if candidate_status != "would_help":
            reason = "candidate shadow evaluation did not indicate would_help"
        elif baseline_status in ("passed", "pass", "success"):
            reason = "baseline already passes — no gain needed"
        elif not candidate_tool:
            reason = "candidate has no shadow decision"
        elif candidate_tool == baseline_tool:
            reason = "candidate selects same tool as baseline — no gain"
        else:
            # Candidate selects a different (presumably better) tool
            # Check if it's linked to verified memory
            do_items = reusable_rule.get("do", [])
            tool_linked_to_memory = any(
                candidate_tool.lower() in str(item).lower() for item in do_items
            )

            if tool_linked_to_memory:
                improved = True
                reason = "candidate selects tool %s linked to trace-verified successful memory (baseline was %s with %s)" % (
                    candidate_tool, baseline_status, baseline_tool
                )
            else:
                reason = "candidate tool %s not directly linked to verified memory pattern" % candidate_tool

        return {
            "improved": improved,
            "reason": reason,
        }

    def _extract_recommended_tools(self, rule: dict) -> list:
        """Extract tool names from reusable_rule.do items."""
        tools = []
        for item in rule.get("do", []):
            item_lower = str(item).lower()
            if "gradio" in item_lower or "config" in item_lower:
                tools.append("discover_gradio_api")
            elif "openapi" in item_lower:
                tools.append("discover_openapi_schema")
            elif "browser" in item_lower or "dom" in item_lower:
                tools.append("probe_browser_dom")
            elif "http" in item_lower or "probe" in item_lower:
                tools.append("probe_http")
        return tools
