"""Bounded provider-native tool loop with local policy enforcement."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.models.base import to_plain
from auto_harness.context.capabilities import resolve_provider_capabilities
from auto_harness.context.tokens import ConservativeTokenEstimator
from auto_harness.context.executor import execute_native_tool_turn
from auto_harness.providers.base import Message, ProviderRequestContext
from auto_harness.providers.protocols import (
    NormalizedToolResult,
    ToolCallConflictError,
    ToolCallProtocolError,
    assistant_tool_call_message,
    canonical_json_hash,
    normalize_provider_tool_call,
    project_provider_tools,
    redact_tool_payload,
    validate_tool_arguments,
    tool_operation_id,
    tool_result_message,
)
from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.state_delta_executor import NativeToolExecutorRouter
from auto_harness.recovery.journal import OperationJournal
from auto_harness.recovery.tool_call_ledger import ToolCallLedger


@dataclass
class NativeToolLoopResult:
    status: str
    final_text: str = ""
    stop_reason: str = ""
    turn_count: int = 0
    executed_tool_count: int = 0
    rejected_tool_count: int = 0
    reused_tool_count: int = 0
    conflict_count: int = 0
    truncated_result_count: int = 0
    loop_limit_count: int = 0
    tool_schema_estimated_tokens: int = 0
    max_input_tokens: int = 0
    provider_request_ids: List[str] = field(default_factory=list)
    tool_schema_hash: str = ""
    visible_tool_names: List[str] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    tool_results: List[NormalizedToolResult] = field(default_factory=list)


class NativeToolTurnLoop:
    """Run a serial, bounded native-tool conversation.

    v0.3 defaults to repository read-only tools. Tool schemas are projected
    locally and every returned call is normalized, checked against that exact
    allowlist, and revalidated by the repository executor policy.
    """

    def __init__(
        self,
        provider: Any,
        *,
        registry: Optional[ToolRegistry] = None,
        executor: Optional[Any] = None,
        stage: str = "plan",
        agent_mode: str = "planner",
        allowed_categories: Sequence[str] = ("read_only",),
        max_turns: int = 6,
        max_calls_per_turn: int = 1,
        max_tool_result_chars: int = 12000,
        run_dir: Optional[Path] = None,
        fault_injector: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        if max_turns < 1 or max_calls_per_turn < 1:
            raise ValueError("native tool loop bounds must be positive")
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.executor = executor or NativeToolExecutorRouter(registry=self.registry)
        self.stage = stage
        self.agent_mode = agent_mode
        self.allowed_categories = tuple(allowed_categories)
        self.max_turns = int(max_turns)
        self.max_calls_per_turn = int(max_calls_per_turn)
        self.max_tool_result_chars = int(max_tool_result_chars)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.fault_injector = fault_injector
        self.config = config
        self.ledger = ToolCallLedger(self.run_dir) if self.run_dir else None
        self.operation_journal = OperationJournal(self.run_dir) if self.run_dir else None

    def run(
        self,
        messages: List[Message],
        *,
        context: Dict[str, Any],
        task_id: str = "",
        repository_fingerprint: str = "",
        runtime_policy_fingerprint: str = "",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        request_context: Optional[ProviderRequestContext] = None,
    ) -> NativeToolLoopResult:
        projection = project_provider_tools(
            self.registry,
            stage=self.stage,
            agent_mode=self.agent_mode,
            allowed_categories=self.allowed_categories,
        )
        conversation = list(messages)
        seen_call_hashes: Dict[str, str] = {}
        results_by_fingerprint: Dict[str, NormalizedToolResult] = {}
        outcome = NativeToolLoopResult(
            status="stopped",
            stop_reason="max_turns",
            tool_schema_hash=projection.schema_hash,
            visible_tool_names=list(projection.tool_names),
        )
        estimator = ConservativeTokenEstimator()
        capabilities = resolve_provider_capabilities(self.provider, self.config)
        reserved_output = min(
            int(max_output_tokens or getattr(self.provider, "max_tokens", 4096) or 4096),
            capabilities.max_output_tokens,
        )
        safety_margin = int(
            getattr(self.config, "agent_context_safety_margin_tokens", 2048)
            if self.config is not None and not isinstance(self.config, dict)
            else (self.config or {}).get("agent_context_safety_margin_tokens", 2048)
        )
        outcome.max_input_tokens = max(
            0, capabilities.context_window_tokens - reserved_output - safety_margin,
        )
        outcome.tool_schema_estimated_tokens = estimator.estimate_text(
            json.dumps(projection.tools, ensure_ascii=False, sort_keys=True)
        )
        visible_names = set(projection.tool_names)

        for turn_index in range(self.max_turns):
            request_tokens = estimator.estimate_request(
                conversation, capabilities, tools=projection.tools,
            )
            if request_tokens > outcome.max_input_tokens:
                outcome.stop_reason = "context_budget_exceeded"
                break
            response = execute_native_tool_turn(
                self.provider,
                conversation,
                projection.tools,
                tool_choice="auto",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                request_context=request_context,
            )
            outcome.turn_count += 1
            if response.request_id:
                outcome.provider_request_ids.append(response.request_id)
            if response.protocol != "native_tools":
                outcome.stop_reason = "provider_protocol_mismatch"
                break
            raw_calls = list(response.tool_calls or [])
            conversation.append(assistant_tool_call_message(response))
            if not raw_calls:
                outcome.status = "completed"
                outcome.stop_reason = "assistant_final"
                outcome.final_text = str(response.text or "")
                if self.ledger:
                    self.ledger.record_turn(
                        turn_index,
                        call_ids=[],
                        provider_request_id=response.request_id,
                        finish_reason=response.finish_reason,
                    )
                break
            if len(raw_calls) > self.max_calls_per_turn:
                outcome.stop_reason = "too_many_tool_calls"
                break

            turn_call_ids: List[str] = []
            for call_index, raw_call in enumerate(raw_calls):
                try:
                    normalized = normalize_provider_tool_call(
                        raw_call,
                        provider_name=str(getattr(self.provider, "provider_name", "")),
                        provider_model=str(getattr(self.provider, "model", "")),
                        turn_index=turn_index,
                        call_index=call_index,
                        seen_call_hashes=seen_call_hashes,
                    )
                except ToolCallProtocolError as exc:
                    call_id = "invalid_%d_%d" % (turn_index, call_index)
                    rejected = self._rejected(call_id, "", str(exc), task_id, {})
                    outcome.rejected_tool_count += 1
                    outcome.tool_results.append(rejected)
                    conversation.append(tool_result_message(rejected, max_chars=self.max_tool_result_chars))
                    continue
                seen_call_hashes[normalized.call_id] = normalized.arguments_hash
                operation_id = tool_operation_id(
                    task_id=task_id,
                    stage=self.stage,
                    tool_name=normalized.tool_name,
                    arguments=normalized.arguments,
                    repository_fingerprint=repository_fingerprint,
                    runtime_policy_fingerprint=runtime_policy_fingerprint,
                )
                self._raise_fault(
                    "after_provider_before_call_ledger", task_id, operation_id,
                )
                if self.ledger:
                    try:
                        self.ledger.record_call(
                            normalized,
                            task_id=task_id,
                            operation_id=operation_id,
                            tool_schema_hash=projection.schema_hash,
                        )
                    except ToolCallConflictError:
                        outcome.stop_reason = "tool_call_conflict"
                        outcome.conflict_count += 1
                        outcome.messages = conversation
                        return outcome
                turn_call_ids.append(normalized.call_id)
                self._raise_fault(
                    "after_call_ledger_before_policy", task_id, operation_id,
                )
                if normalized.tool_name not in visible_names:
                    rejected = self._rejected(
                        normalized.call_id, normalized.tool_name,
                        "tool is not visible in the current stage", task_id,
                        normalized.arguments,
                    )
                    outcome.rejected_tool_count += 1
                    outcome.tool_results.append(rejected)
                    self._persist_result(normalized.call_id, rejected)
                    self._raise_fault(
                        "after_tool_result_before_provider_feedback",
                        task_id, operation_id,
                    )
                    feedback = tool_result_message(rejected, max_chars=self.max_tool_result_chars)
                    if '"truncated":true' in feedback.content:
                        outcome.truncated_result_count += 1
                    conversation.append(feedback)
                    continue
                projected_tool = next(
                    item for item in projection.tools
                    if item["function"]["name"] == normalized.tool_name
                )
                try:
                    validate_tool_arguments(
                        projected_tool["function"]["parameters"],
                        normalized.arguments,
                    )
                except ValueError as exc:
                    rejected = self._rejected(
                        normalized.call_id,
                        normalized.tool_name,
                        "tool arguments rejected: %s" % str(exc),
                        task_id,
                        normalized.arguments,
                        operation_id,
                    )
                    outcome.rejected_tool_count += 1
                    outcome.tool_results.append(rejected)
                    self._persist_result(normalized.call_id, rejected)
                    feedback = tool_result_message(
                        rejected, max_chars=self.max_tool_result_chars,
                    )
                    conversation.append(feedback)
                    continue

                persisted = (
                    self.ledger.load_result(
                        call_id=normalized.call_id,
                        operation_id=operation_id,
                    )
                    if self.ledger else None
                )
                if persisted is not None:
                    result = NormalizedToolResult(**{
                        **persisted.to_dict(),
                        "call_id": normalized.call_id,
                        "reused": True,
                    })
                    outcome.reused_tool_count += 1
                elif normalized.fingerprint in results_by_fingerprint:
                    previous = results_by_fingerprint[normalized.fingerprint]
                    result = NormalizedToolResult(
                        **{**previous.to_dict(), "call_id": normalized.call_id, "reused": True}
                    )
                    outcome.reused_tool_count += 1
                else:
                    call = ToolCall(
                        name=normalized.tool_name,
                        input=normalized.arguments,
                        idempotency_key=normalized.fingerprint,
                        call_id=normalized.call_id,
                        arguments_hash=normalized.arguments_hash,
                        provider_protocol="native_tools",
                    )
                    execution_context = {
                        **context,
                        "agent_mode": self.agent_mode,
                        "stage": self.stage,
                    }
                    if self.run_dir is not None:
                        execution_context["run_dir"] = str(self.run_dir)
                    result = self._execute_with_recovery(
                        call,
                        normalized=normalized,
                        operation_id=operation_id,
                        context=execution_context,
                        task_id=task_id,
                        repository_fingerprint=repository_fingerprint,
                        runtime_policy_fingerprint=runtime_policy_fingerprint,
                    )
                    if result.reused:
                        outcome.reused_tool_count += 1
                    results_by_fingerprint[normalized.fingerprint] = result
                    if result.executed:
                        outcome.executed_tool_count += 1
                    if not result.policy_allowed or result.status == "rejected":
                        outcome.rejected_tool_count += 1
                outcome.tool_results.append(result)
                self._persist_result(normalized.call_id, result)
                self._raise_fault(
                    "after_tool_result_before_provider_feedback",
                    task_id, operation_id,
                )
                feedback = tool_result_message(result, max_chars=self.max_tool_result_chars)
                if '"truncated":true' in feedback.content:
                    outcome.truncated_result_count += 1
                conversation.append(feedback)

            if turn_call_ids:
                self._raise_fault(
                    "after_provider_feedback_before_checkpoint",
                    task_id,
                    outcome.tool_results[-1].operation_id,
                )
            if self.ledger:
                self.ledger.record_turn(
                    turn_index,
                    call_ids=turn_call_ids,
                    provider_request_id=response.request_id,
                    finish_reason=response.finish_reason,
                )

        if outcome.stop_reason in {"max_turns", "too_many_tool_calls"}:
            outcome.loop_limit_count += 1
        outcome.messages = conversation
        return outcome

    def _execute_with_recovery(
        self,
        call: ToolCall,
        *,
        normalized: Any,
        operation_id: str,
        context: Dict[str, Any],
        task_id: str,
        repository_fingerprint: str,
        runtime_policy_fingerprint: str,
    ) -> NormalizedToolResult:
        schema = self.registry.tools.get(call.name)
        category = str(getattr(schema, "category", "read_only"))
        journal = self.operation_journal if category == "side_effect" else None
        if journal is not None:
            existing = journal.load(operation_id)
            if existing and existing.get("status") == "committed":
                stored = existing.get("tool_result")
                if isinstance(stored, dict):
                    return NormalizedToolResult(**{
                        **stored, "call_id": call.call_id, "reused": True,
                    })
            if existing and existing.get("status") == "running":
                journal.recover_running(operation_id)
                return self._rejected(
                    call.call_id, call.name,
                    "side-effect reconciliation is required after an interrupted operation",
                    task_id, call.input, operation_id,
                )
            if existing and existing.get("status") in {"unknown", "manual", "conflict"}:
                return self._rejected(
                    call.call_id, call.name,
                    "side-effect operation requires reconciliation or approval",
                    task_id, call.input, operation_id,
                )
            journal.begin({
                "schema_version": 1,
                "operation_id": operation_id,
                "idempotency_key": normalized.fingerprint,
                "task_id": str(task_id),
                "stage": self.stage,
                "action": call.name,
                "resource_type": "agent_tool",
                "resource_identity": {"tool_name": call.name},
                "normalized_input_hash": call.arguments_hash,
            })
            self._raise_fault(
                "after_journal_begin_before_side_effect", task_id, operation_id,
            )

        executed = self.executor.execute(call, context)
        if journal is not None:
            self._raise_fault(
                "after_side_effect_before_journal_commit", task_id, operation_id,
            )
        result = self._normalize_result(
            normalized.call_id,
            normalized.tool_name,
            normalized.arguments,
            executed,
            task_id=task_id,
            repository_fingerprint=repository_fingerprint,
            runtime_policy_fingerprint=runtime_policy_fingerprint,
        )
        if journal is not None:
            journal.transition(
                operation_id,
                "committed" if result.status in {"passed", "completed", "ok"} else "failed",
                committed_at="",
                result_artifacts=list(result.evidence_paths),
                tool_result=result.to_dict(),
            )
            self._raise_fault(
                "after_journal_commit_before_tool_result", task_id, operation_id,
            )
        return result

    def _persist_result(self, call_id: str, result: NormalizedToolResult) -> None:
        if not self.ledger:
            return
        self.ledger.persist_result(result)
        self.ledger.finalize_call(call_id, result)

    def _raise_fault(self, point: str, task_id: str, operation_id: str) -> None:
        if self.fault_injector is None:
            return
        if callable(self.fault_injector):
            self.fault_injector(point, operation_id)
            return
        if self.run_dir is None:
            raise ValueError("fault injection requires run_dir")
        self.fault_injector.raise_if_configured(
            self.run_dir, task_id, self.stage, point, operation_id,
        )

    def _rejected(
        self,
        call_id: str,
        tool_name: str,
        error: str,
        task_id: str,
        arguments: Dict[str, Any],
        operation_id: str = "",
    ) -> NormalizedToolResult:
        schema = self.registry.tools.get(tool_name)
        return NormalizedToolResult(
            call_id=call_id,
            operation_id=operation_id or tool_operation_id(
                task_id=task_id,
                stage=self.stage,
                tool_name=tool_name or "invalid_tool_call",
                arguments=arguments,
            ),
            tool_name=tool_name,
            status="rejected",
            category=str(getattr(schema, "category", "read_only")),
            policy_allowed=False,
            executed=False,
            applied=False,
            error=error[:300],
        )

    def _normalize_result(
        self,
        call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: ToolResult,
        *,
        task_id: str,
        repository_fingerprint: str,
        runtime_policy_fingerprint: str,
    ) -> NormalizedToolResult:
        evidence = redact_tool_payload(to_plain(result.evidence or {}))
        evidence_paths = [result.evidence_path] if result.evidence_path else []
        payload = {
            "evidence": evidence,
            "metadata_only": bool(result.metadata_only),
            "strong_verify_pass": bool(result.strong_verify_pass),
        }
        return NormalizedToolResult(
            call_id=call_id,
            operation_id=tool_operation_id(
                task_id=task_id,
                stage=self.stage,
                tool_name=tool_name,
                arguments=arguments,
                repository_fingerprint=repository_fingerprint,
                runtime_policy_fingerprint=runtime_policy_fingerprint,
            ),
            tool_name=tool_name,
            status=result.status,
            category=result.category,
            policy_allowed=bool(result.policy_allowed),
            executed=bool(result.executed),
            applied=bool(result.applied),
            result=payload,
            result_hash=canonical_json_hash(payload),
            evidence_paths=evidence_paths,
            error=str(result.error or ""),
        )


__all__ = ["NativeToolLoopResult", "NativeToolTurnLoop"]
