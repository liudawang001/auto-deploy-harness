from typing import Dict

from auto_harness.agent.engine import AgentDecisionEngine
from auto_harness.agent.policy import AgentActionPolicy
from auto_harness.agent.prompts import diagnosis_prompt
from auto_harness.agent.schemas import AgentObservation


class AgentDiagnoser:
    def __init__(self, provider, config=None, trace_writer=None) -> None:
        self.engine = AgentDecisionEngine(provider, config=config, trace_writer=trace_writer, prompt_builder=diagnosis_prompt)
        self.policy = AgentActionPolicy()

    def diagnose(self, observation: AgentObservation) -> Dict:
        decision = self.engine.decide(observation)
        policy = self.policy.validate(decision, observation.runtime_policy or {}, mode="gated_actor")
        self.engine.trace_writer.update_policy_result(decision.trace_path, policy)
        return {
            "status": decision.status,
            "summary": decision.summary,
            "confidence": decision.confidence,
            "diagnosis": decision.diagnosis,
            "actions": [
                {
                    "type": action.type,
                    "reason": action.reason,
                    "confidence": action.confidence,
                    "payload": action.payload,
                    "requires": action.requires,
                }
                for action in decision.actions
            ],
            "rerun_from": decision.plan_delta.get("rerun_from") or decision.diagnosis.get("rerun_from"),
            "rerun_reason": decision.plan_delta.get("rerun_reason") or decision.diagnosis.get("rerun_reason") or "",
            "plan_change_required": bool(
                decision.plan_delta.get("plan_change_required")
                or decision.diagnosis.get("plan_change_required")
            ),
            "raw_text": decision.raw_text,
            "accepted_actions": policy.get("accepted_actions", []),
            "rejected_actions": policy.get("rejected_actions", []),
        }
