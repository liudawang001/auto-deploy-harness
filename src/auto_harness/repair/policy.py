from typing import Dict, List

from auto_harness.models.task import RuntimePolicy


class RepairPolicy:
    def check(self, plan: Dict, runtime: RuntimePolicy) -> Dict:
        decisions: List[Dict] = []
        allowed = True
        for action in plan.get("actions", []):
            decision = self._check_action(action, runtime)
            decisions.append(decision)
            if not decision["allowed"]:
                allowed = False
        return {
            "allowed": allowed,
            "decisions": decisions,
        }

    def _check_action(self, action: Dict, runtime: RuntimePolicy) -> Dict:
        requires = action.get("requires") or {}
        reasons = []
        if requires.get("source_edit") and not runtime.allow_source_edit:
            reasons.append("source edit is not allowed")
        if requires.get("dependency_install") and not runtime.allow_dependency_install:
            reasons.append("dependency install is not allowed")
        if requires.get("service_restart") and not runtime.allow_service_start:
            reasons.append("service restart is not allowed")
        if requires.get("operator_secret"):
            reasons.append("operator secret is required")
        if requires.get("operator_approval"):
            reasons.append("operator approval is required")
        return {
            "action_type": action.get("type", ""),
            "allowed": not reasons,
            "reasons": reasons,
        }
