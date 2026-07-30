from typing import Dict, List

from auto_harness.models.base import to_plain
from auto_harness.models.result import StageResult
from auto_harness.repair.actions import RepairActionNormalizer, RepairActionRegistry
from auto_harness.repair.schema import RepairAction, RepairPlan


class RepairPlanner:
    RERUN_STAGES = ("analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy", "model_prepare", "runner", "verify")

    def __init__(self) -> None:
        self.normalizer = RepairActionNormalizer()
        self.registry = RepairActionRegistry()

    def propose(self, stage: str, result: StageResult, analysis: Dict = None, skill_context: Dict = None) -> Dict:
        analysis = analysis or {}
        plain = to_plain(result)
        data = plain.get("data") if isinstance(plain.get("data"), dict) else {}
        agent_diagnosis = data.get("agent_diagnosis") if isinstance(data.get("agent_diagnosis"), dict) else {}
        accepted_agent_actions = agent_diagnosis.get("accepted_actions") or []
        if agent_diagnosis.get("status") in ("ok", "failed") or accepted_agent_actions:
            data = dict(data)
            diagnosis = dict(agent_diagnosis.get("diagnosis") or {})
            if accepted_agent_actions:
                diagnosis["recommended_actions"] = accepted_agent_actions
            elif agent_diagnosis.get("status") == "ok" and agent_diagnosis.get("actions"):
                diagnosis["recommended_actions"] = agent_diagnosis.get("actions")
            if agent_diagnosis.get("rerun_from"):
                diagnosis["rerun_from"] = agent_diagnosis.get("rerun_from")
                diagnosis["rerun_from_proposed"] = agent_diagnosis.get("rerun_from")
            if agent_diagnosis.get("rerun_reason"):
                diagnosis["rerun_reason"] = agent_diagnosis.get("rerun_reason")
            data["diagnosis"] = diagnosis
        diagnosis = data.get("diagnosis") or {}
        category = diagnosis.get("category") or self._category_from_result(plain)
        root_cause = diagnosis.get("root_cause") or diagnosis.get("signal") or plain.get("error") or plain.get("summary") or "unknown"
        confidence = float(diagnosis.get("confidence") or 0.5)
        actions = self._actions_for(stage, category, data, analysis)
        required_rerun = self._rerun_from(stage, category)
        proposed_rerun = diagnosis.get("rerun_from_proposed") or diagnosis.get("rerun_from")
        effective_rerun = self._safe_rerun_from(proposed_rerun, required_rerun)
        plan = RepairPlan(
            root_cause=root_cause,
            confidence=confidence,
            actions=actions,
            rollback={"type": "rerun_from_last_safe_stage"},
            rerun_from=effective_rerun,
            verification_required=True,
            status="proposed" if actions else "needs_manual_review",
        )
        plain_plan = to_plain(plan)
        plain_plan["rerun_from_required"] = required_rerun
        plain_plan["rerun_from_proposed"] = proposed_rerun or ""
        plain_plan["rerun_from_effective"] = effective_rerun
        plain_plan["rerun_reason"] = diagnosis.get("rerun_reason", "")
        plain_plan["rerun_from_source"] = "llm" if proposed_rerun else "deterministic"
        plain_plan["actions"] = self.normalizer.normalize_many(plain_plan.get("actions", []))
        contract_decisions = [self.registry.validate(action) for action in plain_plan["actions"]]
        unsupported = [item for item in contract_decisions if not item["allowed"]]
        plain_plan["action_contract"] = {
            "supported_types": self.registry.supported_types(),
            "decisions": contract_decisions,
            "valid": not unsupported,
        }
        if unsupported:
            plain_plan["status"] = "needs_manual_review"
            plain_plan["contract_rejection_reasons"] = [
                "%s: %s" % (item["action_type"] or "<missing>", "; ".join(item["reasons"]))
                for item in unsupported
            ]
        plain_plan["failure_hypothesis"] = root_cause
        plain_plan["hypothesis_confidence"] = confidence
        plain_plan["evidence"] = self._evidence(stage, plain, diagnosis)
        plain_plan["expected_effect"] = self._expected_effect(category, plain_plan["actions"])
        plain_plan["verification_plan"] = "rerun from %s, then require trace-based verify pass" % effective_rerun
        plain_plan["rollback_plan"] = "discard current run workspace or resume from previous safe stage"
        plain_plan["risk"] = self._risk(plain_plan["actions"])
        plain_plan["repair_effectiveness_criteria"] = {
            "repair_proposed": True,
            "policy_accepted_required": True,
            "action_executed_or_metadata_applied_required": True,
            "rerun_performed_required": True,
            "final_verify_pass_required": True,
            "metadata_only_counts_as_executed": False,
        }
        if proposed_rerun and proposed_rerun != effective_rerun:
            plain_plan["rerun_from_adjustment_reason"] = "proposed rerun_from is not safe or is later than required safe stage"
        return plain_plan

    def _safe_rerun_from(self, proposed: str, required: str) -> str:
        if required not in self.RERUN_STAGES:
            required = "analyze"
        if proposed not in self.RERUN_STAGES:
            return required
        if self.RERUN_STAGES.index(proposed) <= self.RERUN_STAGES.index(required):
            return proposed
        return required

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

    def _evidence(self, stage: str, plain: Dict, diagnosis: Dict) -> List[str]:
        evidence = []
        if diagnosis.get("signal"):
            evidence.append("%s diagnosis signal: %s" % (stage, diagnosis["signal"]))
        if diagnosis.get("root_cause"):
            evidence.append("%s root cause: %s" % (stage, diagnosis["root_cause"]))
        if plain.get("error"):
            evidence.append("%s error: %s" % (stage, str(plain["error"])[-500:]))
        if not evidence and plain.get("summary"):
            evidence.append("%s summary: %s" % (stage, plain["summary"]))
        return evidence

    def _expected_effect(self, category: str, actions: List[Dict]) -> str:
        if category == "dependency_missing":
            packages = [((action.get("payload") or {}).get("package") or "") for action in actions]
            return "missing imports resolve after installing %s and service stays alive" % ", ".join(pkg for pkg in packages if pkg)
        if category in ("numpy_abi_conflict", "pydantic_conflict", "protobuf_conflict"):
            return "dependency ABI/API conflict is pinned away and environment deploy succeeds"
        if category == "auth_required":
            return "operator provides required token via environment variable name only"
        if actions:
            return "policy-approved action changes the failing observation before verify"
        return "manual review identifies a lower-risk next action"

    def _risk(self, actions: List[Dict]) -> str:
        if any((action.get("requires") or {}).get("operator_secret") or (action.get("requires") or {}).get("source_edit") for action in actions):
            return "high"
        if any((action.get("requires") or {}).get("dependency_install") or (action.get("requires") or {}).get("network") for action in actions):
            return "medium"
        return "low"
