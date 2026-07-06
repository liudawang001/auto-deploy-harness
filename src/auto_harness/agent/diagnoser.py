from typing import Dict

from auto_harness.agent.engine import AgentDecisionEngine
from auto_harness.agent.prompts import diagnosis_prompt
from auto_harness.agent.schemas import AgentObservation


class AgentDiagnoser:
    def __init__(self, provider, config=None, trace_writer=None) -> None:
        self.engine = AgentDecisionEngine(provider, config=config, trace_writer=trace_writer, prompt_builder=diagnosis_prompt)

    def diagnose(self, observation: AgentObservation) -> Dict:
        decision = self.engine.decide(observation)
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
            "raw_text": decision.raw_text,
        }
