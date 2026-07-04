from typing import Dict, List

from auto_harness.models.task import RuntimePolicy


class RepairPolicy:
    def check(self, plan: Dict, runtime: RuntimePolicy, operator_approval: Dict = None) -> Dict:
        decisions: List[Dict] = []
        allowed = True
        for action in plan.get("actions", []):
            decision = self._check_action(action, runtime, operator_approval or {})
            decisions.append(decision)
            if not decision["allowed"]:
                allowed = False
        return {
            "allowed": allowed,
            "decisions": decisions,
        }

    def _check_action(self, action: Dict, runtime: RuntimePolicy, operator_approval: Dict) -> Dict:
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
        if requires.get("operator_approval") and not self._approved(action, operator_approval):
            reasons.append("operator approval is required")
        return {
            "action_type": action.get("type", ""),
            "allowed": not reasons,
            "reasons": reasons,
        }

    def _approved(self, action: Dict, operator_approval: Dict) -> bool:
        if operator_approval.get("approved") is not True:
            return False
        approved_types = operator_approval.get("approved_action_types") or []
        return not approved_types or action.get("type") in approved_types
