"""Shadow Skill Evaluation: evaluate candidate skill patches without affecting real deployments.

The shadow evaluator reads the run results and checks whether a candidate's
recommended tools/actions would have helped or harmed the deployment outcome.

Two evaluation modes:
1. Artifact overlap mode (evaluate_run): checks if candidate recommended tools
   overlap with successful agent steps from a prior run.
2. Planner-only shadow mode (evaluate_candidate_decision): constructs a shadow
   prompt context from the candidate patch + run observation, calls the planner
   to generate a would_tool_call, and compares it with the actual tool call.
   The shadow planner NEVER executes any tool.

Key properties:
- Shadow evaluation does NOT change real deployment behavior
- It only reads and compares: candidate recommendations vs actual outcomes
- Results are written to shadow artifacts for later promotion gating
- helped_count requires evidence trace support (not just text overlap)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.memory.lifecycle import SkillCandidateLifecycle
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

    def __init__(self) -> None:
        self.lifecycle = SkillCandidateLifecycle()

    def evaluate_candidate_decision(
        self,
        run_dir: Path,
        candidate_path: Path,
        observation: dict = None,
        planner=None,
    ) -> Dict:
        """Evaluate a candidate using planner-only shadow decision.

        This is a more rigorous shadow evaluation that:
        1. Reads candidate.patch.markdown
        2. Reads the run's agent observation or agent_verify_result
        3. Constructs a shadow prompt context with candidate skill injected
        4. Calls planner to generate would_tool_call (NO tool execution)
        5. Compares would_tool_call with actual executed tool
        6. Writes candidate decision artifact with executed=false

        Args:
            run_dir: Path to the run directory.
            candidate_path: Path to the candidate JSON file.
            observation: Optional observation dict (stage, frameworks, etc.).
            planner: Optional VerifyPlanner instance. If None, falls back to
                     heuristic comparison.

        Returns:
            Dict with shadow_mode="planner_only", would_tool_call, actual_tool_call,
            would_help, would_harm, reason, executed=False.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")
        patch_markdown = candidate.get("patch", {}).get("markdown", "")
        reusable_rule = candidate.get("reusable_rule", {})

        # Read actual outcome
        agent_verify_result = self._read_json(run_dir / "reports" / "agent_verify_result.json")
        agent_steps = self._read_jsonl(run_dir / "agent_verify_steps.jsonl")

        # Get actual tool call from steps
        actual_tool_call = self._get_actual_tool_call(agent_steps)

        # Get observation context
        obs = observation or agent_verify_result or {}

        # Try planner-based shadow decision
        would_tool_call = None
        if planner is not None:
            would_tool_call = self._planner_shadow_decision(
                planner, obs, patch_markdown, reusable_rule
            )

        # If no planner or planner failed, fall back to heuristic
        if would_tool_call is None:
            would_tool_call = self._heuristic_shadow_decision(
                reusable_rule, obs
            )

        # Compare would vs actual
        would_help = False
        would_harm = False
        reason = ""

        if would_tool_call is None:
            reason = "no shadow decision could be generated"
        elif actual_tool_call is None:
            # No actual tool was called — candidate suggests one
            would_help = True
            reason = "candidate suggests tool %s where no tool was called" % would_tool_call.get("name", "")
        else:
            would_name = would_tool_call.get("name", "")
            actual_name = actual_tool_call.get("name", "")

            if would_name == actual_name:
                would_help = True
                reason = "candidate selects same tool as actual: %s" % would_name
            elif would_name and actual_name:
                # Different tools — check if candidate tool is more specific
                recommended = self._extract_recommended_tools(reusable_rule)
                if would_name in recommended:
                    would_help = True
                    reason = "candidate selects recommended tool %s (actual was %s)" % (would_name, actual_name)
                else:
                    reason = "candidate selects %s, actual was %s" % (would_name, actual_name)

        # Check for harm
        do_items = reusable_rule.get("do", [])
        for item in do_items:
            item_lower = str(item).lower()
            if any(term in item_lower for term in ("bypass trace", "disable verification", "skip regression")):
                would_harm = True
                reason = "candidate suggests bypassing trace verification"
                break
            if any(term in item_lower for term in ("allow shell", "source edit by default", "arbitrary command")):
                would_harm = True
                reason = "candidate suggests expanding permissions"
                break

        # Evidence trace check: would_help requires evidence support
        if would_help and not would_harm:
            if not self._has_evidence_trace_support(obs, agent_verify_result):
                would_help = False
                reason = "would_help demoted: no evidence trace support (need final_status=passed + llm_helped + trace_id)"

        result = {
            "candidate_id": candidate_id,
            "run_id": run_dir.name if hasattr(run_dir, "name") else str(run_dir),
            "shadow_mode": "planner_only",
            "would_tool_call": would_tool_call,
            "actual_tool_call": actual_tool_call,
            "would_help": would_help,
            "would_harm": would_harm,
            "reason": reason,
            "executed": False,
            "evaluated_at": utc_now_iso(),
        }

        # Write shadow artifact
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

        return result

    def _planner_shadow_decision(self, planner, observation: dict, patch_markdown: str, reusable_rule: dict) -> Optional[Dict]:
        """Call planner with candidate skill injected into context.

        The planner generates a would_tool_call but NO tool is executed.
        Returns None if planner fails or returns no_action.
        """
        # Build shadow observation with candidate skill context
        shadow_obs = dict(observation)
        shadow_obs["candidate_skill_patch"] = patch_markdown[:2000]
        shadow_obs["candidate_reusable_rule"] = reusable_rule
        shadow_obs["shadow_mode"] = True

        try:
            from auto_harness.agent_runtime.schemas import VERIFY_TOOLS
            decision = planner.plan_verify(shadow_obs, allowed_tools=list(VERIFY_TOOLS))
            if decision.status == "ok" and decision.tool_call:
                return {
                    "name": decision.tool_call.name,
                    "input": decision.tool_call.input or {},
                }
        except Exception:
            pass

        return None

    def _heuristic_shadow_decision(self, reusable_rule: dict, observation: dict) -> Optional[Dict]:
        """Fallback heuristic: extract recommended tool from reusable_rule.do.

        Returns a would_tool_call dict or None.
        """
        recommended = self._extract_recommended_tools(reusable_rule)
        if recommended:
            return {"name": recommended[0], "input": {}}
        return None

    def _get_actual_tool_call(self, steps: List[Dict]) -> Optional[Dict]:
        """Extract the first successful tool call from agent steps."""
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("tool_name"):
                return {
                    "name": step["tool_name"],
                    "input": step.get("tool_input", {}),
                }
        return None

    def _has_evidence_trace_support(self, observation: dict, agent_verify_result: Optional[Dict]) -> bool:
        """Check if there is evidence trace support for would_help.

        Requirements:
        - agent_verify_result.final_status == passed (or observation has passed status)
        - llm_helped == true
        - evidence contains current trace_id

        This prevents would_help from being true based on text overlap alone.
        """
        result = agent_verify_result or {}

        # Check final status
        final_status = result.get("final_status", observation.get("status", ""))
        if final_status not in ("passed", "pass", "success"):
            return False

        # Check llm_helped
        llm_helped = result.get("llm_helped", False)
        if not llm_helped:
            return False

        # Check for trace_id in evidence
        evidence_paths = result.get("evidence_paths", [])
        trace_id = result.get("trace_id", observation.get("trace_id", ""))
        if trace_id and evidence_paths:
            return True

        # If strong_verify_pass is true, that's also evidence
        if result.get("strong_verify_pass", False):
            return True

        return False

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

        # Evidence trace check: would_help requires evidence support
        # This prevents would_help from being true based on text overlap alone.
        if would_help and not would_harm:
            if not self._has_evidence_trace_support(active_context or {}, agent_verify_result):
                would_help = False
                reason = "would_help demoted: no evidence trace support (need final_status=passed + llm_helped + trace_id)"

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
        current_status = self.lifecycle.normalize_status(candidate.get("status"))
        if current_status not in ("regression_passed", "shadow_passed", "shadow_failed"):
            return {
                "status": "blocked",
                "error": "shadow evaluation requires regression_passed candidate",
                "candidate_status": current_status,
            }
        shadow = candidate.get("shadow", {})

        # Initialize shadow if needed
        if not shadow.get("enabled"):
            shadow = {
                "enabled": True,
                "helped_count": 0,
                "harmful_count": 0,
                "evaluations": [],
            }
        run_id = str(result.get("run_id") or "")
        if run_id and any(
            str(item.get("run_id") or "") == run_id
            for item in shadow.get("evaluations", [])
            if isinstance(item, dict)
        ):
            return {
                "status": "duplicate",
                "run_id": run_id,
                "helped_count": shadow.get("helped_count", 0),
                "harmful_count": shadow.get("harmful_count", 0),
                "error": "shadow run_id already recorded",
            }

        # Update counts
        if result.get("would_help"):
            shadow["helped_count"] = shadow.get("helped_count", 0) + 1
        if result.get("would_harm"):
            shadow["harmful_count"] = shadow.get("harmful_count", 0) + 1

        # Append evaluation record
        evaluations = shadow.get("evaluations", [])
        evaluations.append({
            "run_id": run_id,
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
            target_status = "shadow_passed"
        elif harmful > self.DEFAULT_HARMFUL_THRESHOLD:
            target_status = "shadow_failed"
        else:
            target_status = ""

        write_json(candidate_path, candidate)
        if target_status:
            transition = self.lifecycle.transition(
                candidate_path,
                target_status,
                "shadow_gate",
                evidence={
                    "helped_count": helped,
                    "harmful_count": harmful,
                    "run_id": run_id,
                },
            )
            if transition.get("status") == "failed":
                return transition

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
