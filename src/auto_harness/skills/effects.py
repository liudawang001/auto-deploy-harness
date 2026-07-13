"""Skill Effect Recorder: record which plan/tool fields were influenced by skills.

Records whether each selected skill actually influenced the LLM plan,
compiled plan, or policy result. Outputs to runs/<task-id>/reports/skill_effects.json.

First-version uses rule-based heuristics:
- verification_skill → verify.request field changes
- execution_skill → run.candidates or install_commands field changes
- repair_skill → replan or repair action field changes
- security_skill → policy rejection due to unsafe guidance
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


@dataclass
class SkillEffect:
    """Record of a skill's influence on a deployment plan field."""
    skill_name: str
    skill_sha256: str
    stage: str
    effect_type: str  # verify_hint_generation, runner_candidate_selection, install_command_change, repair_action, policy_rejection
    field_changed: str  # verify.request, run.candidates, environment.install_commands, repair.action
    accepted_by_policy: bool
    evidence: Dict = field(default_factory=dict)


class SkillEffectRecorder:
    """Records skill effects on deployment plan fields.

    Analyzes the relationship between selected skills and plan/compiler
    output to determine which skills influenced which fields.
    """

    def record_effects(
        self,
        *,
        task_id: str,
        routed_skills: List[Dict],
        compiled_plan: Dict,
        policy_result: Dict,
        original_analysis: Dict = None,
    ) -> Dict:
        """Record skill effects for a deployment run.

        Args:
            task_id: The deployment task ID.
            routed_skills: List of routed skill dicts (from SkillContextBuilder output).
            compiled_plan: The compiled effective deployment plan.
            policy_result: The policy gate result.
            original_analysis: The deterministic analysis before LLM merge.

        Returns:
            Dict with task_id, effects list, and created_at.
        """
        effects: List[Dict] = []
        original_analysis = original_analysis or {}

        for skill in routed_skills:
            skill_effects = self._detect_skill_effects(
                skill=skill,
                compiled_plan=compiled_plan,
                policy_result=policy_result,
                original_analysis=original_analysis,
            )
            for effect in skill_effects:
                effects.append({
                    "skill_name": effect.skill_name,
                    "skill_sha256": effect.skill_sha256,
                    "stage": effect.stage,
                    "effect_type": effect.effect_type,
                    "field_changed": effect.field_changed,
                    "accepted_by_policy": effect.accepted_by_policy,
                    "evidence": effect.evidence,
                })

        return {
            "task_id": task_id,
            "effects": effects,
            "created_at": utc_now_iso(),
        }

    def write_effects(self, run_dir: Path, effects_data: Dict) -> Path:
        """Write skill effects to reports/skill_effects.json.

        Args:
            run_dir: The run directory.
            effects_data: The effects dict from record_effects.

        Returns:
            Path to the written file.
        """
        reports_dir = Path(run_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / "skill_effects.json"
        write_json(path, effects_data)
        return path

    def _detect_skill_effects(
        self,
        skill: Dict,
        compiled_plan: Dict,
        policy_result: Dict,
        original_analysis: Dict,
    ) -> List[SkillEffect]:
        """Detect effects for a single skill.

        Uses rule-based heuristics to determine if the skill influenced
        specific plan fields.
        """
        effects: List[SkillEffect] = []
        skill_name = skill.get("name", "")
        skill_sha256 = skill.get("sha256", "")
        skill_type = skill.get("type", "")
        skill_stage = skill.get("stage", "plan_first")
        allowed_tools = skill.get("allowed_tools", skill.get("allowed_plan_effects", []))
        policy_allowed = policy_result.get("allowed", True)

        effective_plan = compiled_plan.get("effective_plan", {})
        analysis = compiled_plan.get("analysis", {})

        # Check verification_skill → verify.request
        if skill_type == "verification_skill":
            verify_hint = analysis.get("verify_hint", {})
            original_verify = original_analysis.get("verify_hint", {})
            if verify_hint and verify_hint != original_verify:
                effects.append(SkillEffect(
                    skill_name=skill_name,
                    skill_sha256=skill_sha256,
                    stage=skill_stage,
                    effect_type="verify_hint_generation",
                    field_changed="verify.request",
                    accepted_by_policy=policy_allowed,
                    evidence={"has_verify_hint": bool(verify_hint)},
                ))

        # Check execution_skill → run.candidates or install_commands
        if skill_type == "execution_skill":
            run_candidates = analysis.get("run_candidates", [])
            original_candidates = original_analysis.get("run_candidates", [])
            if run_candidates and len(run_candidates) != len(original_candidates):
                # LLM added or changed candidates
                llm_candidates = [c for c in run_candidates if c.get("selected_by") in ("llm_plan_first", "llm_runner_gate")]
                if llm_candidates:
                    effects.append(SkillEffect(
                        skill_name=skill_name,
                        skill_sha256=skill_sha256,
                        stage=skill_stage,
                        effect_type="runner_candidate_selection",
                        field_changed="run.candidates",
                        accepted_by_policy=policy_allowed,
                        evidence={"llm_candidate_count": len(llm_candidates)},
                    ))

            install_plan = analysis.get("install_plan", [])
            original_install = original_analysis.get("install_plan", [])
            if install_plan and len(install_plan) != len(original_install):
                effects.append(SkillEffect(
                    skill_name=skill_name,
                    skill_sha256=skill_sha256,
                    stage=skill_stage,
                    effect_type="install_command_change",
                    field_changed="environment.install_commands",
                    accepted_by_policy=policy_allowed,
                    evidence={"install_count": len(install_plan)},
                ))

        # Check repair_skill → repair action
        if skill_type == "repair_skill":
            # Check if repair actions were generated
            repair_actions = effective_plan.get("repair", {}).get("actions", [])
            if repair_actions:
                effects.append(SkillEffect(
                    skill_name=skill_name,
                    skill_sha256=skill_sha256,
                    stage=skill_stage,
                    effect_type="repair_action",
                    field_changed="repair.actions",
                    accepted_by_policy=policy_allowed,
                    evidence={"action_count": len(repair_actions)},
                ))

        # Check security_skill → policy rejection
        if skill_type == "security_skill" and not policy_allowed:
            rejected_items = policy_result.get("rejected_items", [])
            if rejected_items:
                effects.append(SkillEffect(
                    skill_name=skill_name,
                    skill_sha256=skill_sha256,
                    stage=skill_stage,
                    effect_type="policy_rejection",
                    field_changed="policy",
                    accepted_by_policy=False,
                    evidence={"rejected_count": len(rejected_items)},
                ))

        return effects
