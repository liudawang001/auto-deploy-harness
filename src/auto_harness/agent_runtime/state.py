"""Agent verify state and step persistence.

Manages the internal state of the act_verify loop and writes artifacts.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class AgentVerifyState:
    """Tracks the state of the act_verify loop across steps."""

    def __init__(self, trace_id: str, initial_status: str = "uncertain") -> None:
        self.trace_id = trace_id
        self.verify_status = initial_status
        self.steps: List[Dict] = []
        self.accepted_tool_count = 0
        self.rejected_tool_count = 0
        self.evidence_paths: List[str] = []
        self.strong_verify_pass = False
        self.stop_reason = ""

    def record_reject(self, reason: str) -> None:
        self.rejected_tool_count += 1
        self.stop_reason = reason

    def record_accepted_tool(self) -> None:
        self.accepted_tool_count += 1

    def apply_tool_result(self, tool_result_dict: Dict) -> None:
        """Update state based on tool execution result."""
        if tool_result_dict.get("strong_verify_pass"):
            self.strong_verify_pass = True
            self.verify_status = "passed"
        if tool_result_dict.get("evidence_path"):
            self.evidence_paths.append(tool_result_dict["evidence_path"])

    def to_result(self, final_status: str = None, stop_reason: str = "", mode: str = "", llm_helped: bool = False) -> Dict:
        return {
            "triggered": True,
            "final_status": final_status or self.verify_status,
            "llm_helped": llm_helped,
            "step_count": len(self.steps),
            "accepted_tool_count": self.accepted_tool_count,
            "rejected_tool_count": self.rejected_tool_count,
            "strong_verify_pass": self.strong_verify_pass,
            "evidence_paths": self.evidence_paths,
            "stop_reason": stop_reason or self.stop_reason,
            "mode": mode,
        }


class AgentStepWriter:
    """Writes agent verify loop artifacts to disk."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._steps_buffer: List[Dict] = []

    def write_step(self, step: Dict) -> None:
        """Append a step record and flush the steps JSONL."""
        self._steps_buffer.append(step)
        self._flush_steps()

    def write_rejected(self, step_index: int, trace_id: str, decision: Dict, critic_result: Dict = None, policy_result: Dict = None, reason: str = "") -> None:
        step = {
            "step_index": step_index,
            "stage": "verify",
            "trace_id": trace_id,
            "decision": decision,
            "critic": critic_result or {},
            "policy": policy_result or {},
            "execution": {"executed": False, "status": "rejected", "reason": reason},
            "state_delta": {},
            "recorded_at": utc_now_iso(),
        }
        self.write_step(step)

    def write_executed(self, step_index: int, trace_id: str, decision: Dict, critic_result: Dict, policy_result: Dict, tool_result: Dict, state_delta: Dict) -> None:
        step = {
            "step_index": step_index,
            "stage": "verify",
            "trace_id": trace_id,
            "decision": decision,
            "critic": critic_result,
            "policy": policy_result,
            "execution": {
                "executed": True,
                "status": tool_result.get("status", "error"),
                "strong_verify_pass": tool_result.get("strong_verify_pass", False),
                "evidence_path": tool_result.get("evidence_path"),
                "error": tool_result.get("error"),
            },
            "state_delta": state_delta,
            "recorded_at": utc_now_iso(),
        }
        self.write_step(step)

    def _flush_steps(self) -> None:
        path = self.run_dir / "agent_verify_steps.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for step in self._steps_buffer:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")
        self._steps_buffer.clear()

    def write_state(self, state: AgentVerifyState, mode: str = "") -> None:
        data = {
            "trace_id": state.trace_id,
            "verify_status": state.verify_status,
            "step_count": len(state.steps),
            "accepted_tool_count": state.accepted_tool_count,
            "rejected_tool_count": state.rejected_tool_count,
            "strong_verify_pass": state.strong_verify_pass,
            "evidence_paths": state.evidence_paths,
            "stop_reason": state.stop_reason,
            "mode": mode,
            "updated_at": utc_now_iso(),
        }
        write_json(self.run_dir / "agent_state.json", data)

    def write_result(self, result: Dict) -> None:
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / "agent_verify_result.json", result)


def compute_idempotency_key(run_id: str, step_index: int, tool_name: str, tool_input: Dict) -> str:
    """Compute a deterministic idempotency key for a tool call."""
    canonical = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
    raw = "%s:%s:%s:%s" % (run_id, step_index, tool_name, canonical)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
