from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


class RepairLoopController:
    # Safe stages for repair resume (document Phase 7.4)
    SAFE_RERUN_STAGES = (
        "env_deploy",
        "model_prepare",
        "runner",
        "verify",
    )
    # Legacy stages that were previously allowed but are now forbidden
    FORBIDDEN_RERUN_STAGES = (
        "analyze",
        "resource_plan",
        "report",
    )

    def __init__(self, max_attempts: int = 2) -> None:
        self.max_attempts = max(0, int(max_attempts or 0))

    def gate(self, run_dir: Path, stage: str, memory_entry: Dict, plan: Dict, policy_result: Dict, last_safe_stage: str = None) -> Dict:
        repair_dir = run_dir / "repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_state(repair_dir)
        signature = memory_entry.get("signature") or self._signature(stage, plan)
        attempts = int(state.get("attempts", {}).get(signature, 0))
        requested_rerun = plan.get("rerun_from") or stage
        effective_rerun = self._effective_rerun_from(requested_rerun, stage, last_safe_stage)
        loop_reasons: List[str] = []
        if requested_rerun != effective_rerun:
            loop_reasons.append("rerun_from is not safe; fallback selected")
        if attempts >= self.max_attempts:
            loop_reasons.append("repair attempt limit reached")

        policy_allowed = bool(policy_result.get("allowed"))
        loop_allowed = policy_allowed and attempts < self.max_attempts
        next_attempt = attempts + 1 if loop_allowed else attempts
        plan["rerun_from_effective"] = effective_rerun
        plan["rerun_from_requested"] = requested_rerun

        loop_result = {
            "signature": signature,
            "stage": stage,
            "max_attempts": self.max_attempts,
            "attempts_before": attempts,
            "attempts_after": next_attempt,
            "allowed": loop_allowed,
            "reasons": loop_reasons,
            "rerun_from_requested": requested_rerun,
            "rerun_from_effective": effective_rerun,
            "approval_present": self.load_approval(run_dir).get("approved") is True,
        }
        state.setdefault("attempts", {})[signature] = next_attempt
        state.setdefault("history", []).append({
            "created_at": utc_now_iso(),
            "signature": signature,
            "stage": stage,
            "policy_allowed": policy_allowed,
            "loop_allowed": loop_allowed,
            "attempt": next_attempt,
            "rerun_from_requested": requested_rerun,
            "rerun_from_effective": effective_rerun,
            "reasons": loop_reasons,
        })
        write_json(self._state_path(repair_dir), state)

        effective_policy = {
            "allowed": loop_allowed,
            "decisions": list(policy_result.get("decisions") or []),
            "loop": loop_result,
        }
        if not loop_allowed and loop_reasons:
            effective_policy["decisions"].append({
                "action_type": "repair_loop",
                "allowed": False,
                "reasons": loop_reasons,
            })
        return effective_policy

    def approve_latest(self, run_dir: Path, note: str = "") -> Dict:
        repair_dir = run_dir / "repairs"
        plan = self._read_optional(repair_dir / "repair_plan.json") or {}
        action_types = sorted({action.get("type") for action in plan.get("actions", []) if action.get("type")})
        approval = {
            "approved": True,
            "created_at": utc_now_iso(),
            "note": note,
            "approved_action_types": action_types,
            "plan_root_cause": plan.get("root_cause", ""),
            "plan_rerun_from": plan.get("rerun_from_effective") or plan.get("rerun_from", ""),
            "values_recorded": False,
        }
        repair_dir.mkdir(parents=True, exist_ok=True)
        write_json(repair_dir / "operator_approval.json", approval)
        return approval

    def load_approval(self, run_dir: Path) -> Dict:
        return self._read_optional(run_dir / "repairs" / "operator_approval.json") or {}

    def _effective_rerun_from(self, requested: str, stage: str, last_safe_stage: str = None) -> str:
        """Determine the effective rerun stage.

        Only SAFE_RERUN_STAGES are allowed. Forbidden stages (analyze, resource_plan, report)
        are rejected. Unknown stages default to env_deploy (safest).
        """
        if requested in self.SAFE_RERUN_STAGES:
            return requested
        if requested in self.FORBIDDEN_RERUN_STAGES:
            # Reject forbidden stages, fallback to env_deploy
            return "env_deploy"
        if stage in self.SAFE_RERUN_STAGES:
            return stage
        if last_safe_stage in self.SAFE_RERUN_STAGES:
            return last_safe_stage
        return "env_deploy"

    def _signature(self, stage: str, plan: Dict) -> str:
        return "%s:%s:%s" % (stage, plan.get("root_cause", ""), plan.get("rerun_from", ""))

    def _state_path(self, repair_dir: Path) -> Path:
        return repair_dir / "repair_loop_state.json"

    def _load_state(self, repair_dir: Path) -> Dict:
        state = self._read_optional(self._state_path(repair_dir))
        if isinstance(state, dict):
            state.setdefault("attempts", {})
            state.setdefault("history", [])
            return state
        return {"attempts": {}, "history": []}

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
