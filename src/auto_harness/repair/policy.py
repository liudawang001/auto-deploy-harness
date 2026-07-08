from typing import Dict, List

from auto_harness.agent.repair_actions import safe_package_spec
from auto_harness.models.task import RuntimePolicy
from auto_harness.repair.actions import RepairActionNormalizer


class RepairPolicy:
    def __init__(self) -> None:
        self.normalizer = RepairActionNormalizer()

    def check(self, plan: Dict, runtime: RuntimePolicy, operator_approval: Dict = None) -> Dict:
        decisions: List[Dict] = []
        allowed = True
        for action in self.normalizer.normalize_many(plan.get("actions", [])):
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
        if action.get("type") in ("install_package", "install_pip_package", "install_conda_package", "pin_dependency"):
            package = str((action.get("payload") or {}).get("package") or "")
            if not safe_package_spec(package):
                reasons.append("unsafe package spec")
        if action.get("type") in ("install_conda_package",):
            for channel in (action.get("payload") or {}).get("channels") or []:
                if channel not in {"defaults", "conda-forge", "pytorch", "nvidia", "fastai"}:
                    reasons.append("unsafe conda channel")
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
