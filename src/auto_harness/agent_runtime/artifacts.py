"""Agent loop artifact writer.

Writes structured artifacts for each step of the AgentLoop:
- agent_steps.jsonl: step-by-step log
- agent_state.json: current state snapshot
- agent_plan.json: deployment plan
- agent_plan_revisions.jsonl: plan revision history
- agent_decisions/<stage>_<NNN>.json: per-decision details
- agent_policy/<stage>_<NNN>.json: per-policy check details
- agent_tools/<stage>_<NNN>.json: per-tool-result details
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class AgentArtifactWriter:
    """Writes agent loop artifacts to the run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._steps_buffer: List[Dict] = []

        # Ensure directories exist
        for subdir in ["agent_decisions", "agent_policy", "agent_tools", "repairs"]:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def write_step(self, step: Dict) -> Path:
        """Append a step to agent_steps.jsonl."""
        self._steps_buffer.append(step)
        path = self.run_dir / "agent_steps.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for s in self._steps_buffer:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        self._steps_buffer.clear()
        return path

    def write_state(self, state_dict: Dict) -> Path:
        """Write current state to agent_state.json."""
        path = self.run_dir / "agent_state.json"
        write_json(path, state_dict)
        return path

    def write_plan(self, plan: Dict) -> Path:
        """Write deployment plan to agent_plan.json."""
        path = self.run_dir / "agent_plan.json"
        write_json(path, plan)
        return path

    def write_plan_revision(self, revision: Dict) -> Path:
        """Append a plan revision to agent_plan_revisions.jsonl."""
        path = self.run_dir / "agent_plan_revisions.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(revision, ensure_ascii=False) + "\n")
        return path

    def write_decision(self, stage: str, step_index: int, decision: Dict) -> Path:
        """Write a decision artifact."""
        filename = "%s_%03d.json" % (stage, step_index)
        path = self.run_dir / "agent_decisions" / filename
        write_json(path, decision)
        return path

    def write_policy_check(self, stage: str, step_index: int, policy: Dict) -> Path:
        """Write a policy check artifact."""
        filename = "%s_%03d.json" % (stage, step_index)
        path = self.run_dir / "agent_policy" / filename
        write_json(path, policy)
        return path

    def write_tool_result(self, stage: str, step_index: int, tool_result: Dict) -> Path:
        """Write a tool result artifact."""
        filename = "%s_%03d.json" % (stage, step_index)
        path = self.run_dir / "agent_tools" / filename
        write_json(path, tool_result)
        return path

    def write_repair_artifact(self, name: str, data: Dict) -> Path:
        """Write a repair artifact."""
        path = self.run_dir / "repairs" / ("%s.json" % name)
        write_json(path, data)
        return path
