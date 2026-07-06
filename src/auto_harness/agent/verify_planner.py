import json
import urllib.parse
from typing import Dict

from auto_harness.agent.engine import AgentDecisionEngine
from auto_harness.agent.prompts import verify_prompt
from auto_harness.agent.schemas import AgentObservation


class AgentVerifyPlanner:
    def __init__(self, provider, config=None, trace_writer=None) -> None:
        self.engine = AgentDecisionEngine(provider, config=config, trace_writer=trace_writer, prompt_builder=verify_prompt)

    def plan(self, observation: AgentObservation) -> Dict:
        decision = self.engine.decide(observation)
        hint = decision.verify_hint
        valid, reason = self.validate_hint(hint)
        return {
            "status": "ok" if decision.status == "ok" and valid else "rejected",
            "decision_status": decision.status,
            "confidence": decision.confidence,
            "reason": decision.summary or decision.rationale or reason,
            "verify_hint": hint if valid else {},
            "reject_reason": "" if valid else reason,
        }

    def validate_hint(self, verify_hint: Dict) -> tuple:
        if not isinstance(verify_hint, dict):
            return False, "verify_hint must be an object"
        request = verify_hint.get("request") if isinstance(verify_hint.get("request"), dict) else {}
        method = str(request.get("method") or "").upper()
        if method not in ("GET", "POST"):
            return False, "method must be GET or POST"
        path = request.get("path")
        if path:
            if not isinstance(path, str) or not path.startswith("/"):
                return False, "path must start with /"
            parsed = urllib.parse.urlparse(path)
            if parsed.scheme or parsed.netloc:
                return False, "path must not contain external URL"
        if "{{trace_id}}" not in json.dumps(request, ensure_ascii=False):
            return False, "request must contain {{trace_id}}"
        if "token" in json.dumps(request, ensure_ascii=False).lower():
            return False, "request must not contain token values"
        return True, ""
