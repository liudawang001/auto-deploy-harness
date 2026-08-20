"""Crash-safe ledger for protocol-independent agent tool calls."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from auto_harness.models.base import read_json, to_plain
from auto_harness.providers.base import Message
from auto_harness.providers.protocols.normalizer import ToolCallConflictError
from auto_harness.providers.protocols.schemas import (
    NormalizedToolCall,
    NormalizedToolResult,
)
from auto_harness.providers.protocols.tool_messages import tool_result_message
from auto_harness.providers.protocols.tool_messages import redact_tool_payload
from auto_harness.utils.atomic import FileLock, atomic_write_text
from auto_harness.utils.time import utc_now_iso


class ToolCallLedger:
    """Persist calls, results and turn checkpoints under one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.root = Path(run_dir) / "agent_tool_calls"
        self.calls_dir = self.root / "calls"
        self.results_dir = self.root / "results"
        self.operation_results_dir = self.results_dir / "by_operation"
        self.turns_dir = self.root / "turns"
        self.calls_path = self.root / "calls.jsonl"
        for path in (
            self.calls_dir, self.results_dir,
            self.operation_results_dir, self.turns_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: str) -> str:
        value = str(value or "")
        if not value or not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError("invalid tool ledger identifier")
        return value

    def call_path(self, call_id: str) -> Path:
        return self.calls_dir / (self._safe_id(call_id) + ".json")

    def result_path(self, call_id: str) -> Path:
        return self.results_dir / (self._safe_id(call_id) + ".json")

    def operation_result_path(self, operation_id: str) -> Path:
        return self.operation_results_dir / (self._safe_id(operation_id) + ".json")

    def record_call(
        self,
        call: NormalizedToolCall,
        *,
        task_id: str,
        operation_id: str,
        tool_schema_hash: str,
    ) -> Dict[str, Any]:
        """Atomically record a received call before policy or execution."""
        path = self.call_path(call.call_id)
        with FileLock(path):
            existing = self._load(path)
            if existing:
                if existing.get("arguments_hash") != call.arguments_hash:
                    raise ToolCallConflictError(
                        "tool call id reused with different arguments"
                    )
                return existing
            record = {
                "schema_version": 1,
                "task_id": str(task_id),
                "turn_index": call.turn_index,
                "call_index": call.call_index,
                "call_id": call.call_id,
                "operation_id": operation_id,
                "tool_name": call.tool_name,
                "arguments": redact_tool_payload(to_plain(call.arguments)),
                "arguments_hash": call.arguments_hash,
                "tool_schema_hash": tool_schema_hash,
                "provider_protocol": call.provider_protocol,
                "provider_name": call.provider_name,
                "provider_model": call.provider_model,
                "policy_verdict": "pending",
                "result_status": "pending",
                "result_hash": "",
                "evidence_paths": [],
                "received_at": utc_now_iso(),
            }
            self._write(path, record)
            return record

    def finalize_call(
        self,
        call_id: str,
        result: NormalizedToolResult,
    ) -> Dict[str, Any]:
        path = self.call_path(call_id)
        with FileLock(path):
            record = self._load(path)
            if not record:
                raise KeyError("tool call is not recorded: %s" % call_id)
            record.update({
                "policy_verdict": "allowed" if result.policy_allowed else "rejected",
                "result_status": result.status,
                "result_hash": result.result_hash,
                "evidence_paths": list(result.evidence_paths),
                "finalized_at": utc_now_iso(),
            })
            self._write(path, record)
        self._append_call(record)
        return record

    def persist_result(self, result: NormalizedToolResult) -> None:
        """Write both provider-call and semantic-operation result indexes."""
        payload = result.to_dict()
        for path in (
            self.result_path(result.call_id),
            self.operation_result_path(result.operation_id),
        ):
            with FileLock(path):
                existing = self._load(path)
                if existing:
                    existing_hash = existing.get("result_hash")
                    if existing_hash and result.result_hash and existing_hash != result.result_hash:
                        raise ValueError("tool result identity collision")
                    continue
                self._write(path, payload)

    def load_result(
        self,
        *,
        call_id: str = "",
        operation_id: str = "",
    ) -> Optional[NormalizedToolResult]:
        paths: List[Path] = []
        if call_id:
            paths.append(self.result_path(call_id))
        if operation_id:
            paths.append(self.operation_result_path(operation_id))
        for path in paths:
            payload = self._load(path)
            if payload:
                return NormalizedToolResult(**payload)
        return None

    def record_turn(
        self,
        turn_index: int,
        *,
        call_ids: Iterable[str],
        provider_request_id: str = "",
        finish_reason: str = "",
        checkpoint_status: str = "complete",
    ) -> Path:
        path = self.turns_dir / ("turn_%04d.json" % (int(turn_index) + 1))
        payload = {
            "schema_version": 1,
            "turn_index": int(turn_index),
            "call_ids": list(call_ids),
            "provider_request_id": str(provider_request_id),
            "finish_reason": str(finish_reason),
            "checkpoint_status": str(checkpoint_status),
            "written_at": utc_now_iso(),
        }
        with FileLock(path):
            self._write(path, payload)
        return path

    def rebuild_exchange(self, call_id: str, *, max_result_chars: int = 12000) -> List[Message]:
        """Rebuild the canonical assistant-call/tool-result message pair."""
        record = self._load(self.call_path(call_id))
        result = self.load_result(call_id=call_id)
        if not record or not result:
            return []
        raw_call = {
            "id": record["call_id"],
            "type": "function",
            "function": {
                "name": record["tool_name"],
                "arguments": json.dumps(
                    record.get("arguments", {}),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }
        return [
            Message(role="assistant", content="", tool_calls=[raw_call]),
            tool_result_message(result, max_chars=max_result_chars),
        ]

    @staticmethod
    def _load(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write(path: Path, payload: Dict[str, Any]) -> None:
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    def _append_call(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with FileLock(self.calls_path):
            with self.calls_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())


__all__ = ["ToolCallLedger"]
