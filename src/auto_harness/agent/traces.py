import hashlib
import json
import time
from pathlib import Path
from typing import Dict

from auto_harness.models.base import to_plain, write_json
from auto_harness.utils.time import compact_timestamp


class AgentTraceWriter:
    def __init__(self, trace_dir: Path = None) -> None:
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def write(
        self,
        stage: str,
        provider: str,
        model: str,
        prompt: str,
        observation_summary: Dict,
        raw_output: str,
        parsed_decision,
        policy_result: Dict = None,
        latency_ms: int = 0,
    ) -> str:
        if not self.trace_dir:
            return ""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage": stage,
            "provider": provider,
            "model": model,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "observation_summary": observation_summary,
            "raw_output_tail": raw_output[-4000:],
            "parsed_decision": to_plain(parsed_decision),
            "policy_result": policy_result or {},
            "latency_ms": latency_ms,
            "created_at_ms": int(time.time() * 1000),
        }
        path = self.trace_dir / ("%s_%s.json" % (stage, compact_timestamp()))
        write_json(path, payload)
        return str(path)

    def update_policy_result(self, trace_path: str, policy_result: Dict) -> None:
        if not trace_path:
            return
        path = Path(trace_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        payload["policy_result"] = policy_result or {}
        payload["policy_updated_at_ms"] = int(time.time() * 1000)
        write_json(path, payload)


def observation_summary(observation) -> Dict:
    return {
        "task_id": getattr(observation, "task_id", ""),
        "stage": getattr(observation, "stage", ""),
        "file_count": len(getattr(observation, "file_tree", []) or []),
        "selected_file_paths": sorted((getattr(observation, "selected_files", {}) or {}).keys()),
        "allowed_action_types": list(getattr(observation, "allowed_action_types", []) or []),
        "memory_hit_count": len(getattr(observation, "memory_hits", []) or []),
    }
