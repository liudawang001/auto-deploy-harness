import time
from typing import Callable

from auto_harness.agent.prompts import decision_prompt
from auto_harness.agent.schemas import AgentAction, AgentDecision, AgentObservation
from auto_harness.agent.traces import AgentTraceWriter, observation_summary
from auto_harness.providers import Message
from auto_harness.providers.json_utils import parse_json_object


class AgentDecisionEngine:
    def __init__(self, provider, config=None, trace_writer: AgentTraceWriter = None, prompt_builder: Callable = None) -> None:
        self.provider = provider
        self.config = config
        self.trace_writer = trace_writer or AgentTraceWriter()
        self.prompt_builder = prompt_builder or decision_prompt

    def decide(self, observation: AgentObservation) -> AgentDecision:
        prompt = self.prompt_builder(observation)
        started = time.time()
        provider_name = self.provider.__class__.__name__ if self.provider else ""
        model = getattr(self.provider, "model", "") or getattr(self.provider, "model_name", "")
        try:
            result = self.provider.complete([Message(role="user", content=prompt)], temperature=0.0)
            latency_ms = result.latency_ms or int((time.time() - started) * 1000)
            decision = self._parse_decision(observation.stage, result.text, provider_name, model)
            decision.trace_path = self.trace_writer.write(
                observation.stage,
                provider_name,
                model,
                prompt,
                observation_summary(observation),
                result.text,
                decision,
                latency_ms=latency_ms,
            )
            return decision
        except Exception as exc:  # noqa: BLE001 - agent must not break deterministic pipeline
            decision = AgentDecision(
                stage=observation.stage,
                status="failed",
                summary="agent decision failed",
                confidence=0.0,
                raw_text="",
                provider=provider_name,
                model=model,
                rationale=str(exc),
            )
            decision.trace_path = self.trace_writer.write(
                observation.stage,
                provider_name,
                model,
                prompt,
                observation_summary(observation),
                "",
                decision,
                latency_ms=int((time.time() - started) * 1000),
            )
            return decision

    def _parse_decision(self, stage: str, text: str, provider: str, model: str) -> AgentDecision:
        try:
            parsed = parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            return AgentDecision(
                stage=stage,
                status="invalid",
                summary="agent returned invalid JSON",
                confidence=0.0,
                raw_text=text[-4000:],
                provider=provider,
                model=model,
                rationale=str(exc),
            )
        if not isinstance(parsed, dict):
            return AgentDecision(stage=stage, status="invalid", summary="agent JSON was not an object", raw_text=text[-4000:], provider=provider, model=model)
        actions = []
        for item in parsed.get("actions") or []:
            if isinstance(item, dict):
                actions.append(AgentAction(
                    type=str(item.get("type") or ""),
                    reason=str(item.get("reason") or ""),
                    confidence=float(item.get("confidence") or 0),
                    payload=item.get("payload") if isinstance(item.get("payload"), dict) else {},
                    requires=item.get("requires") if isinstance(item.get("requires"), dict) else {},
                ))
        plan_delta = parsed.get("plan_delta") if isinstance(parsed.get("plan_delta"), dict) else {}
        if parsed.get("rerun_from"):
            plan_delta = dict(plan_delta)
            plan_delta["rerun_from"] = parsed.get("rerun_from")
        return AgentDecision(
            stage=str(parsed.get("stage") or stage),
            status=str(parsed.get("status") or "ok"),
            summary=str(parsed.get("summary") or parsed.get("reason") or ""),
            confidence=float(parsed.get("confidence") or 0),
            actions=actions,
            plan_delta=plan_delta,
            risks=parsed.get("risks") if isinstance(parsed.get("risks"), list) else [],
            rationale=str(parsed.get("rationale") or parsed.get("reason") or ""),
            raw_text=text[-4000:],
            provider=provider,
            model=model,
            diagnosis=parsed.get("diagnosis") if isinstance(parsed.get("diagnosis"), dict) else {},
            verify_hint=parsed.get("verify_hint") if isinstance(parsed.get("verify_hint"), dict) else {},
        )
