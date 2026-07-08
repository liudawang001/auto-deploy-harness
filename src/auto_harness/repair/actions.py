from typing import Dict, List


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
        base = {
            "type": action_type,
            "reason": action.get("reason", ""),
            "confidence": float(action.get("confidence") or 0),
            "requires": action.get("requires") if isinstance(action.get("requires"), dict) else {},
        }
        if action_type in ("install_package", "install_pip_package", "install_conda_package"):
            packages = self._packages(payload)
            return [dict(base, type=action_type, payload={"package": package}) for package in packages]
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
