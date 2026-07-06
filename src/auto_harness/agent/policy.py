import json
import re
from typing import Dict, List

from auto_harness.models.task import RuntimePolicy


class AgentActionPolicy:
    PHASE1_ALLOWED = {
        "add_run_candidate",
        "select_run_candidate",
        "update_verify_hint",
        "add_dependency_constraint",
    }
    TIER0_ALLOWED = {
        "update_verify_hint",
        "request_env_var_name_only",
        "rerun_from_stage",
    }
    TIER1_ALLOWED = {
        "install_package",
        "add_dependency_constraint",
        "switch_torch_variant",
        "retry_model_download",
    }
    SHELL_METACHARS = (";", "&&", "|", ">", "<", "`", "$(")
    SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+([<>=!~]=?[A-Za-z0-9_.+*~-]+)?$")

    def validate(self, decision, runtime_policy: RuntimePolicy, mode: str = "planner") -> Dict:
        accepted: List[Dict] = []
        rejected: List[Dict] = []
        for action in getattr(decision, "actions", []) or []:
            plain = self._plain_action(action)
            reason = self._reject_reason(plain, runtime_policy, mode)
            if reason:
                rejected.append({"action_type": plain.get("type", ""), "reason": reason, "action": plain})
            else:
                accepted.append(plain)
        return {
            "allowed": not rejected,
            "accepted_actions": accepted,
            "rejected_actions": rejected,
        }

    def _reject_reason(self, action: Dict, runtime_policy: RuntimePolicy, mode: str) -> str:
        action_type = action.get("type", "")
        payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
        requires = action.get("requires") if isinstance(action.get("requires"), dict) else {}
        if self._contains_secret_value(action):
            return "action payload appears to contain a secret value"
        if action_type in ("add_run_candidate", "select_run_candidate"):
            cmd = payload.get("cmd")
            if not isinstance(cmd, list):
                return "command payload must be a list"
            if any(not isinstance(part, str) for part in cmd):
                return "command parts must be strings"
            if any(self._has_shell_metachar(part) for part in cmd):
                return "command contains shell metacharacters"
        if action_type == "update_verify_hint":
            verify_hint = payload.get("verify_hint") if isinstance(payload.get("verify_hint"), dict) else payload
            if not self._verify_hint_has_trace(verify_hint):
                return "verify hint must contain {{trace_id}}"
        if action_type == "add_dependency_constraint":
            package = payload.get("package", "")
            constraint = payload.get("constraint", "")
            if not package or self._has_shell_metachar(str(package) + str(constraint)):
                return "dependency constraint is unsafe"
        if action_type == "install_package":
            if mode != "gated_actor":
                return "install_package requires gated_actor mode"
            if not runtime_policy.allow_dependency_install:
                return "dependency install is not allowed"
            package = str(payload.get("package") or "")
            if not self.safe_package_spec(package):
                return "unsafe package spec"
        if requires.get("source_edit") or action_type == "propose_source_patch":
            if not runtime_policy.allow_source_edit:
                return "source edit is not allowed"
        allowed = set(self.PHASE1_ALLOWED)
        if mode == "gated_actor":
            allowed = allowed | self.TIER0_ALLOWED | self.TIER1_ALLOWED
        if action_type not in allowed:
            return "action type is not allowed in %s mode" % mode
        return ""

    def safe_package_spec(self, package: str) -> bool:
        if not package or not self.SAFE_PACKAGE_RE.match(package):
            return False
        lowered = package.lower()
        forbidden = ("--extra-index-url", "--trusted-host", " -e ", "git+", "http://", "https://", "/", "\\")
        return not any(token in lowered for token in forbidden)

    def _has_shell_metachar(self, value: str) -> bool:
        return any(token in value for token in self.SHELL_METACHARS)

    def _contains_secret_value(self, value) -> bool:
        text = json.dumps(value, ensure_ascii=False).lower()
        secret_markers = ("api_secret", "api_key=", "token=", "bearer ", "sk-", "password", "xunfei_api_secret")
        return any(marker in text for marker in secret_markers)

    def _verify_hint_has_trace(self, verify_hint: Dict) -> bool:
        if not isinstance(verify_hint, dict):
            return False
        return "{{trace_id}}" in json.dumps(verify_hint, ensure_ascii=False)

    def _plain_action(self, action) -> Dict:
        if isinstance(action, dict):
            return {
                "type": action.get("type", ""),
                "reason": action.get("reason", ""),
                "confidence": float(action.get("confidence") or 0),
                "payload": action.get("payload") if isinstance(action.get("payload"), dict) else {},
                "requires": action.get("requires") if isinstance(action.get("requires"), dict) else {},
            }
        return {
            "type": getattr(action, "type", ""),
            "reason": getattr(action, "reason", ""),
            "confidence": float(getattr(action, "confidence", 0) or 0),
            "payload": getattr(action, "payload", {}) or {},
            "requires": getattr(action, "requires", {}) or {},
        }
