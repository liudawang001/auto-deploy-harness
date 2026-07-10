"""Stop conditions for the DeploymentAgentLoop.

Determines when the agent loop should stop based on various criteria:
- verify passed
- max_iterations reached
- policy rejected all actions
- no safe action available
- same failure repeats twice
- required human secret/source edit
"""
from typing import Dict, List, Optional


class StopCondition:
    """Evaluates whether the agent loop should stop.

    Each check returns (should_stop: bool, reason: str).
    """

    def __init__(self, max_iterations: int = 5, stop_on_verify_pass: bool = True) -> None:
        self.max_iterations = max_iterations
        self.stop_on_verify_pass = stop_on_verify_pass
        self._failure_history: List[str] = []

    def check(self, *, iteration: int, verify_status: str, policy_results: List[Dict],
              stage_status: Dict, last_error: str = "") -> tuple:
        """Check all stop conditions.

        Returns:
            (should_stop, reason) tuple
        """
        # 1. Verify passed
        if self.stop_on_verify_pass and verify_status in ("pass", "passed"):
            return True, "verify_passed"

        # 2. Max iterations reached
        if iteration >= self.max_iterations:
            return True, "max_iterations_reached"

        # 3. Policy rejected all actions in this iteration
        if policy_results and all(not p.get("allowed", False) for p in policy_results):
            return True, "all_actions_policy_rejected"

        # 4. Same failure repeats twice
        if last_error:
            self._failure_history.append(last_error)
            if self._failure_history.count(last_error) >= 2:
                return True, "same_failure_repeats_twice"

        # 5. Required human intervention (secret, source edit)
        if self._requires_human_intervention(stage_status):
            return True, "requires_human_intervention"

        return False, ""

    def _requires_human_intervention(self, stage_status: Dict) -> bool:
        """Check if any stage requires human intervention."""
        for stage, info in stage_status.items():
            status = info.get("status", "") if isinstance(info, dict) else ""
            if status in ("requires_secret", "requires_source_edit"):
                return True
        return False

    def reset(self) -> None:
        """Reset failure history for a new run."""
        self._failure_history.clear()
