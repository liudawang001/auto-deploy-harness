from typing import Dict, List

from auto_harness.models.base import to_plain
from auto_harness.models.result import StageResult
from auto_harness.repair.schema import RepairAction, RepairPlan


class RepairPlanner:
    def propose(self, stage: str, result: StageResult, analysis: Dict = None) -> Dict:
        analysis = analysis or {}
        plain = to_plain(result)
        data = plain.get("data") if isinstance(plain.get("data"), dict) else {}
        agent_diagnosis = data.get("agent_diagnosis") if isinstance(data.get("agent_diagnosis"), dict) else {}
        if agent_diagnosis.get("status") == "ok":
            data = dict(data)
            diagnosis = dict(agent_diagnosis.get("diagnosis") or {})
            if agent_diagnosis.get("actions"):
                diagnosis["recommended_actions"] = agent_diagnosis.get("actions")
            if agent_diagnosis.get("rerun_from"):
                diagnosis["rerun_from"] = agent_diagnosis.get("rerun_from")
            data["diagnosis"] = diagnosis
        diagnosis = data.get("diagnosis") or {}
        category = diagnosis.get("category") or self._category_from_result(plain)
        root_cause = diagnosis.get("root_cause") or diagnosis.get("signal") or plain.get("error") or plain.get("summary") or "unknown"
        confidence = float(diagnosis.get("confidence") or 0.5)
        actions = self._actions_for(stage, category, data, analysis)
        plan = RepairPlan(
            root_cause=root_cause,
            confidence=confidence,
            actions=actions,
            rollback={"type": "rerun_from_last_safe_stage"},
            rerun_from=diagnosis.get("rerun_from") or self._rerun_from(stage, category),
            verification_required=True,
            status="proposed" if actions else "needs_manual_review",
        )
        return to_plain(plan)

    def _actions_for(self, stage: str, category: str, data: Dict, analysis: Dict) -> List[RepairAction]:
        diagnosis = data.get("diagnosis") or {}
        structured = self._structured_actions(diagnosis)
        if structured:
            return structured
        if category == "dependency_missing":
            package = data.get("diagnosis", {}).get("signal") or ""
            return [
                RepairAction(
                    type="install_package",
                    reason="missing Python package detected",
                    requires={"dependency_install": True, "network": True, "source_edit": False},
                    payload={"package": package},
                )
            ]
        if category in ("cuda_oom", "torch_cuda_unavailable"):
            return [
                RepairAction(
                    type="adjust_runtime",
                    reason="CUDA runtime issue detected",
                    requires={"dependency_install": True, "service_restart": True},
                    payload={"strategy": "select compatible torch wheel or CPU fallback"},
                )
            ]
        if category == "disk_full":
            return [
                RepairAction(
                    type="change_cache_dir",
                    reason="disk space exhausted",
                    requires={"operator_approval": True},
                    payload={"config": "model_cache_dir"},
                )
            ]
        if category == "auth_required":
            env_vars = data.get("diagnosis", {}).get("required_env_vars") or ["HF_TOKEN", "MODELSCOPE_TOKEN"]
            return [
                RepairAction(
                    type="set_env_var_name_only",
                    reason="model repository requires token",
                    requires={"operator_secret": True},
                    payload={"env_vars": sorted(set(env_vars)), "values_recorded": False},
                )
            ]
        if stage == "verify":
            return [
                RepairAction(
                    type="update_verify_hint",
                    reason="verification could not prove trace handling",
                    requires={"source_edit": False},
                    payload={"service_type": analysis.get("verify_hint", {}).get("service_type", "unknown")},
                )
            ]
        return []

    def _structured_actions(self, diagnosis: Dict) -> List[RepairAction]:
        actions = []
        for raw in diagnosis.get("recommended_actions") or []:
            action_type = raw.get("type")
            if not action_type:
                continue
            actions.append(RepairAction(
                type=action_type,
                reason=raw.get("reason") or diagnosis.get("suggested_fix") or "diagnosed repair action",
                requires=raw.get("requires") or diagnosis.get("requires") or {},
                payload=raw.get("payload") or {},
            ))
        return actions

    def _category_from_result(self, plain: Dict) -> str:
        error = str(plain.get("error") or plain.get("summary") or "").lower()
        if "token" in error or "401" in error or "unauthorized" in error:
            return "auth_required"
        if "space" in error or "disk" in error:
            return "disk_full"
        if "dependency" in error or "module" in error:
            return "dependency_missing"
        return "unknown"

    def _rerun_from(self, stage: str, category: str) -> str:
        if category in ("dependency_missing", "cuda_oom", "torch_cuda_unavailable"):
            return "env_deploy"
        if category in ("numpy_abi_conflict", "pydantic_conflict", "protobuf_conflict"):
            return "env_deploy"
        if category == "wheel_build_failed":
            return "env_solve"
        if category in ("auth_required", "disk_full"):
            return "model_prepare"
        return stage
