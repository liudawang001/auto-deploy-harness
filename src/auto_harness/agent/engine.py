import time
from typing import Callable

from auto_harness.agent.prompts import SYSTEM_GUARDRAILS, decision_prompt
from auto_harness.agent.schemas import AgentAction, AgentDecision, AgentObservation
from auto_harness.agent.traces import AgentTraceWriter, observation_summary
from auto_harness.context import (
    ContextPriority,
    ContextSection,
    LLMCallExecutor,
    PromptEnvelope,
    TrustLevel,
    compact_agent_observation,
    get_context_profile,
    safe_context_telemetry,
)
from auto_harness.providers import Message
from auto_harness.providers.json_utils import parse_json_object


class AgentDecisionEngine:
    def __init__(
        self,
        provider,
        config=None,
        trace_writer: AgentTraceWriter = None,
        prompt_builder: Callable = None,
        call_executor: LLMCallExecutor = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.trace_writer = trace_writer or AgentTraceWriter()
        self.prompt_builder = prompt_builder or decision_prompt
        self.call_executor = call_executor or LLMCallExecutor(config=config)

    def decide(self, observation: AgentObservation) -> AgentDecision:
        prompt = self.prompt_builder(observation)
        profile_name, call_site = self._context_route(observation.stage)
        compacted = compact_agent_observation(
            observation,
            profile=profile_name,
            aggressive=False,
            skill_budget_tokens=_config_get(
                self.config, "agent_context_skill_budget_tokens", 2000
            ),
            memory_budget_tokens=_config_get(
                self.config, "agent_context_memory_budget_tokens", 2000
            ),
        )
        retry_observation = compact_agent_observation(
            observation,
            profile=profile_name,
            aggressive=True,
            skill_budget_tokens=_config_get(
                self.config, "agent_context_skill_budget_tokens", 2000
            ),
            memory_budget_tokens=_config_get(
                self.config, "agent_context_memory_budget_tokens", 2000
            ),
        )
        candidate_prompt = self.prompt_builder(compacted)
        retry_prompt = self.prompt_builder(retry_observation)
        envelope = PromptEnvelope(
            messages=[
                Message(role="system", content=SYSTEM_GUARDRAILS),
                Message(role="user", content=prompt),
            ],
            candidate_messages=[
                Message(role="system", content=SYSTEM_GUARDRAILS),
                Message(role="user", content=candidate_prompt),
            ],
            retry_messages=[
                Message(role="system", content=SYSTEM_GUARDRAILS),
                Message(role="user", content=retry_prompt),
            ],
            sections=_observation_sections(observation),
            candidate_sections=_observation_sections(compacted),
            retry_sections=_observation_sections(retry_observation),
            requested_output_tokens=_config_get(
                self.config, "agent_context_reserved_output_tokens", 2048
            ),
        )
        started = time.time()
        provider_name = self.provider.__class__.__name__ if self.provider else ""
        model = getattr(self.provider, "model", "") or getattr(self.provider, "model_name", "")
        context_telemetry = {}
        used_prompt = SYSTEM_GUARDRAILS + "\n" + prompt
        try:
            call_result = self.call_executor.execute(
                call_site=call_site,
                stage=profile_name,
                provider=self.provider,
                envelope=envelope,
                profile=get_context_profile(
                    profile_name,
                    _config_get(
                        self.config,
                        "agent_context_reserved_output_tokens",
                        2048,
                    ),
                ),
                temperature=0.0,
            )
            result = call_result.provider_result
            context_telemetry = safe_context_telemetry(
                getattr(result, "context", {})
            )
            if context_telemetry.get("mode") == "enforce":
                used_prompt = {
                    "retry": SYSTEM_GUARDRAILS + "\n" + retry_prompt,
                    "candidate": SYSTEM_GUARDRAILS + "\n" + candidate_prompt,
                }.get(
                    context_telemetry.get("selected_variant"),
                    SYSTEM_GUARDRAILS + "\n" + prompt,
                )
            latency_ms = result.latency_ms or int((time.time() - started) * 1000)
            decision = self._parse_decision(observation.stage, result.text, provider_name, model)
            decision.trace_path = self.trace_writer.write(
                observation.stage,
                provider_name,
                model,
                used_prompt,
                observation_summary(observation),
                result.text,
                decision,
                latency_ms=latency_ms,
                context=context_telemetry,
            )
            return decision
        except Exception as exc:  # noqa: BLE001 - agent must not break deterministic pipeline
            error_context = safe_context_telemetry(
                getattr(exc, "context", {})
            )
            error_context.setdefault("call_site", call_site)
            error_context.setdefault("profile", {"name": profile_name})
            error_context.setdefault(
                "stop_reason",
                getattr(exc, "stop_reason", "provider_error"),
            )
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
                context=error_context,
            )
            return decision

    def _context_route(self, stage: str):
        name = getattr(self.prompt_builder, "__name__", "")
        if name == "diagnosis_prompt":
            return "diagnose", "agent.diagnose"
        if name == "verify_prompt":
            return "verify", "agent.verify"
        normalized = str(stage or "analyze")
        return normalized, "agent.decision"

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
        if parsed.get("rerun_reason"):
            plan_delta = dict(plan_delta)
            plan_delta["rerun_reason"] = parsed.get("rerun_reason")
        if parsed.get("plan_change_required"):
            plan_delta = dict(plan_delta)
            plan_delta["plan_change_required"] = True
        if isinstance(parsed.get("verify_candidates"), list):
            plan_delta = dict(plan_delta)
            plan_delta["verify_candidates"] = parsed.get("verify_candidates")
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


def _config_get(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _observation_sections(observation):
    selected_files = observation.selected_files or {}
    memories = observation.memory_hits or []
    skills = observation.selected_skills or []
    return [
        ContextSection(
            name="instructions",
            content=SYSTEM_GUARDRAILS,
            priority=ContextPriority.REQUIRED,
            trust_level=TrustLevel.TRUSTED_INSTRUCTION,
            content_type="instruction",
            required=True,
            source="agent.prompts",
        ),
        ContextSection(
            name="repository_files",
            content=selected_files,
            priority=ContextPriority.RELEVANT_EVIDENCE,
            trust_level=TrustLevel.UNTRUSTED_REPOSITORY,
            content_type="repository_snippets",
            source="agent_observation",
            metadata={"included_files": list(selected_files)},
        ),
        ContextSection(
            name="memory",
            content=memories,
            priority=ContextPriority.EXPERIENCE,
            trust_level=TrustLevel.UNTRUSTED_MEMORY,
            content_type="verified_memory",
            source="memory_store",
            metadata={"memory_count": len(memories)},
        ),
        ContextSection(
            name="skills",
            content=skills,
            priority=ContextPriority.EXPERIENCE,
            trust_level=TrustLevel.RUNTIME_FACT,
            content_type="skill_context",
            source="skill_router",
            metadata={"skill_count": len(skills)},
        ),
    ]
