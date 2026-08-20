"""Policy-gated executor for internal state deltas used by native tools."""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from auto_harness.agent_runtime.decision_gate import GateCritic, StagePolicyValidator
from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.providers.protocols.schemas import canonical_json_hash
from auto_harness.utils.atomic import FileLock
from auto_harness.utils.time import utc_now_iso


class StateDeltaToolExecutor:
    """Apply allowlisted internal-state changes; never execute external effects."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self.policy = StagePolicyValidator()
        self.critic = GateCritic()

    def execute(self, tool_call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        schema = self.registry.tools.get(tool_call.name)
        if schema is None or not schema.implemented or schema.executor != "state_delta":
            return self._reject(tool_call.name, "tool is not an implemented state-delta tool")
        if str(context.get("agent_mode", "")) != "gated_actor":
            return self._reject(tool_call.name, "state delta requires gated_actor mode")
        state = context.get("state")
        if not isinstance(state, dict):
            return self._reject(tool_call.name, "mutable state context is unavailable")
        tool_input = tool_call.input or {}
        if self._contains_status_override(tool_input):
            return self._reject(tool_call.name, "tool input cannot set final or stage status")
        stage = str(context.get("stage") or "")
        policy_stage = "plan" if tool_call.name == "set_stage_hint" else stage
        raw_call = {"name": tool_call.name, "input": tool_input}
        critic = self.critic.evaluate(raw_call, policy_stage, state)
        if not critic.get("allowed"):
            return self._reject(tool_call.name, critic.get("reason", "critic rejected"))
        policy = self.policy.validate(raw_call, policy_stage, state)
        if not policy.get("allowed"):
            return self._reject(tool_call.name, policy.get("reason", "policy rejected"))

        before = copy.deepcopy(state)
        changed_fields = self._apply(tool_call.name, tool_input, state)
        if not changed_fields:
            return self._reject(tool_call.name, "state delta produced no allowed change")
        before_hash = canonical_json_hash(before)
        after_hash = canonical_json_hash(state)
        evidence = {
            "changed": before_hash != after_hash,
            "changed_fields": changed_fields,
            "before_state_hash": before_hash,
            "after_state_hash": after_hash,
            "tool_call_id": tool_call.call_id,
            "provider_protocol": tool_call.provider_protocol,
            "stage": str(context.get("stage", "")),
            "policy_allowed": True,
            "executed": False,
        }
        self._append_contribution(context.get("run_dir"), tool_call, evidence)
        return ToolResult(
            status="passed",
            tool_name=tool_call.name,
            category="state_delta",
            policy_allowed=True,
            executed=False,
            applied=before_hash != after_hash,
            metadata_only=False,
            evidence=evidence,
        )

    def _apply(self, name: str, value: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
        if name == "select_runner_candidate":
            candidate_id = str(value.get("candidate_id", ""))
            candidates = list(state.get("run_candidates") or [])
            state["run_candidates"] = (
                [item for item in candidates if item.get("id") == candidate_id]
                + [item for item in candidates if item.get("id") != candidate_id]
            )
            state["selected_runner_candidate_id"] = candidate_id
            return ["run_candidates", "selected_runner_candidate_id"]
        if name == "set_stage_hint":
            stage = str(value.get("stage", ""))
            hints = copy.deepcopy(value.get("hints") or {})
            state.setdefault("stage_hints", {})[stage] = hints
            return ["stage_hints.%s" % stage]
        if name == "select_environment_backend":
            state.setdefault("environment", {})["backend"] = str(value.get("backend", ""))
            return ["environment.backend"]
        if name == "apply_dependency_constraint":
            constraint = {
                "package": str(value.get("package", "")),
                "version_spec": str(value.get("version_spec", "")),
                "scope": str(value.get("scope", "temporary_overlay")),
                "reason": str(value.get("reason", "")),
            }
            constraints = state.setdefault("environment", {}).setdefault(
                "dependency_constraints", []
            )
            if constraint not in constraints:
                constraints.append(constraint)
            return ["environment.dependency_constraints"]
        if name == "select_torch_variant":
            state.setdefault("environment", {})["torch_variant"] = copy.deepcopy(value)
            return ["environment.torch_variant"]
        if name == "select_model_source":
            assets = state.setdefault("model_assets", {})
            for key in ("source", "repo_id", "target_path", "fallback"):
                if key in value:
                    assets[key] = copy.deepcopy(value[key])
            return ["model_assets.%s" % key for key in value if key in {
                "source", "repo_id", "target_path", "fallback",
            }]
        if name == "select_model_asset_strategy":
            assets = state.setdefault("model_assets", {})
            assets["strategy"] = str(value.get("strategy", ""))
            if "fallback" in value:
                assets["fallback"] = copy.deepcopy(value["fallback"])
            return ["model_assets.strategy"] + (
                ["model_assets.fallback"] if "fallback" in value else []
            )
        return []

    @staticmethod
    def _contains_status_override(value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"status", "final_status", "stage_status"}:
                    return True
                if StateDeltaToolExecutor._contains_status_override(item):
                    return True
        elif isinstance(value, list):
            return any(StateDeltaToolExecutor._contains_status_override(item) for item in value)
        return False

    @staticmethod
    def _append_contribution(run_dir: Any, tool_call: ToolCall, evidence: Dict[str, Any]) -> None:
        if not run_dir:
            return
        path = Path(run_dir) / "reports" / "native_state_deltas.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "call_id": tool_call.call_id,
            "tool_name": tool_call.name,
            "arguments_hash": tool_call.arguments_hash,
            "provider_protocol": tool_call.provider_protocol,
            "changed_fields": evidence["changed_fields"],
            "before_state_hash": evidence["before_state_hash"],
            "after_state_hash": evidence["after_state_hash"],
            "applied": evidence["changed"],
            "recorded_at": utc_now_iso(),
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with FileLock(path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    @staticmethod
    def _reject(name: str, reason: str) -> ToolResult:
        return ToolResult(
            status="rejected",
            tool_name=name,
            category="state_delta",
            policy_allowed=False,
            executed=False,
            applied=False,
            error=str(reason)[:300],
        )


class NativeToolExecutorRouter:
    """Route native calls to existing executors by registry contract."""

    def __init__(self, config: Any = None, registry: Any = None) -> None:
        from auto_harness.tools.registry import ToolRegistry
        from auto_harness.tools.repository_executor import RepositoryToolExecutor
        from auto_harness.tools.retrieval_executor import RetrievalToolExecutor

        self.registry = registry or ToolRegistry()
        self.repository = RepositoryToolExecutor(config=config, registry=self.registry)
        self.retrieval = RetrievalToolExecutor(config=config, registry=self.registry)
        self.state_delta = StateDeltaToolExecutor(self.registry)

    def execute(self, tool_call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        schema = self.registry.tools.get(tool_call.name)
        executor = str(getattr(schema, "executor", ""))
        if executor == "repository":
            return self.repository.execute(tool_call, context)
        if executor == "retrieval":
            return self.retrieval.execute(tool_call, context)
        if executor == "state_delta":
            return self.state_delta.execute(tool_call, context)
        return ToolResult(
            status="rejected",
            tool_name=tool_call.name,
            category=str(getattr(schema, "category", "read_only")),
            policy_allowed=False,
            error="native executor is unavailable for this tool",
        )


__all__ = ["NativeToolExecutorRouter", "StateDeltaToolExecutor"]
