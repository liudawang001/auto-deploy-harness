import re
from typing import Dict, List


class RepairActionRegistry:
    """Authoritative contract for repair actions consumed by RepairApplier."""

    SPECS = {
        "install_package": {"kind": "command", "required_payload": ("package",)},
        "install_pip_package": {"kind": "command", "required_payload": ("package",)},
        "pin_dependency": {"kind": "command", "required_payload": ("package",)},
        "install_conda_package": {"kind": "command", "required_payload": ("package",)},
        "set_env_var_name_only": {"kind": "operator_input", "required_payload": ("env_vars",)},
        "update_verify_hint": {"kind": "metadata", "required_payload": ("verify_hint",)},
        "rerun_from_stage": {"kind": "control", "required_payload": ("stage",)},
    }
    SAFE_RERUN_STAGES = {"env_deploy", "model_prepare", "runner", "verify"}
    ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")

    def supported_types(self) -> List[str]:
        return sorted(self.SPECS)

    def validate(self, action: Dict) -> Dict:
        action_type = str(action.get("type") or "")
        spec = self.SPECS.get(action_type)
        if not spec:
            return {
                "allowed": False,
                "action_type": action_type,
                "kind": "unsupported",
                "reasons": ["unsupported repair action type"],
            }
        payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
        reasons = []
        for field in spec["required_payload"]:
            value = payload.get(field)
            if value is None or value == "" or value == [] or value == {}:
                reasons.append("missing required payload field: %s" % field)
        if action_type == "set_env_var_name_only":
            env_vars = payload.get("env_vars")
            if not isinstance(env_vars, list) or any(
                not isinstance(name, str) or not self.ENV_NAME_RE.match(name)
                for name in (env_vars or [])
            ):
                reasons.append("env_vars must contain safe environment variable names only")
        if action_type == "rerun_from_stage" and payload.get("stage") not in self.SAFE_RERUN_STAGES:
            reasons.append("rerun stage is not allowed")
        if action_type == "update_verify_hint" and not isinstance(payload.get("verify_hint"), dict):
            reasons.append("verify_hint must be an object")
        return {
            "allowed": not reasons,
            "action_type": action_type,
            "kind": spec["kind"],
            "reasons": reasons,
        }


class RepairActionNormalizer:
    def normalize_many(self, actions: List[Dict]) -> List[Dict]:
        normalized: List[Dict] = []
        for action in actions or []:
            item = self.normalize(action)
            if isinstance(item, list):
                normalized.extend(item)
            elif item:
                normalized.append(item)
        return normalized

    def normalize(self, action: Dict):
        if not isinstance(action, dict):
            return {}
        action_type = action.get("type", "")
        payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
        if action_type == "request_env_var_name_only":
            action_type = "set_env_var_name_only"
        base = {
            "type": action_type,
            "reason": action.get("reason", ""),
            "confidence": float(action.get("confidence") or 0),
            "requires": action.get("requires") if isinstance(action.get("requires"), dict) else {},
        }
        if action_type in ("install_package", "install_pip_package", "pin_dependency", "install_conda_package"):
            packages = self._packages(payload)
            common_payload = dict(payload)
            common_payload.pop("packages", None)
            common_payload.pop("package", None)
            return [
                dict(base, type=action_type, payload=dict(common_payload, package=package))
                for package in packages
            ]
        if action_type == "rerun_from_stage":
            stage = payload.get("stage") or action.get("rerun_from")
            return dict(base, payload={"stage": stage} if stage else payload)
        if action_type == "update_verify_hint":
            verify_hint = payload.get("verify_hint") if isinstance(payload.get("verify_hint"), dict) else payload
            return dict(base, payload={"verify_hint": verify_hint})
        return dict(base, payload=payload)

    def _packages(self, payload: Dict) -> List[str]:
        if isinstance(payload.get("packages"), list):
            return [str(item).strip() for item in payload["packages"] if str(item).strip()]
        package = str(payload.get("package") or "").strip()
        return [package] if package else []
