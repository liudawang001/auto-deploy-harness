import inspect
import uuid
from typing import Any, Optional

from auto_harness.context.budget import ContextBudgetManager, context_telemetry
from auto_harness.context.capabilities import resolve_provider_capabilities
from auto_harness.context.models import (
    ContextPriority,
    ContextSection,
    LLMCallResult,
    PromptEnvelope,
    TrustLevel,
)
from auto_harness.context.profiles import get_context_profile
from auto_harness.context.tokens import normalize_usage


class ContextGovernanceError(RuntimeError):
    def __init__(
        self,
        stop_reason: str,
        message: str,
        context: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.context = context or {}


class LLMCallExecutor:
    def __init__(self, config: Any = None, budget_manager=None) -> None:
        self.config = config or _DefaultContextConfig()
        self.budget_manager = budget_manager or ContextBudgetManager()

    def execute(
        self,
        *,
        call_site: str,
        stage: str,
        provider: Any,
        envelope: PromptEnvelope,
        profile=None,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        call_id = uuid.uuid4().hex
        profile = profile or get_context_profile(
            stage,
            _config_get(
                self.config, "agent_context_reserved_output_tokens", 4096
            ),
        )
        if not envelope.sections:
            envelope.sections = _default_sections(envelope, profile.name)
        else:
            for section in envelope.sections:
                section.metadata.setdefault("profile", profile.name)
        capabilities = resolve_provider_capabilities(provider, self.config)
        budget, build = self.budget_manager.build(
            call_site=call_site,
            envelope=envelope,
            profile=profile,
            capabilities=capabilities,
            config=self.config,
        )
        mode = build.mode
        if mode not in {"observe", "shadow", "enforce"}:
            raise self._governance_error(
                "context_invalid_mode",
                "invalid context mode: %s" % mode,
                budget,
                build,
                capabilities,
                call_id=call_id,
            )
        if mode == "enforce" and capabilities.source == "fallback":
            raise self._governance_error(
                "context_capability_unknown",
                "context window is unknown for provider/model",
                budget,
                build,
                capabilities,
                call_id=call_id,
            )
        messages = (
            envelope.candidate_messages or envelope.messages
            if build.selected_variant == "candidate"
            else envelope.messages
        )
        selected_tokens = self.budget_manager.estimate_messages(
            messages, envelope, capabilities
        )
        build.estimated_input_tokens = selected_tokens
        if mode == "enforce" and (
            budget.max_input_tokens <= 0 or selected_tokens > budget.max_input_tokens
        ):
            retry_messages = envelope.retry_messages
            retry_tokens = (
                self.budget_manager.estimate_messages(
                    retry_messages, envelope, capabilities
                )
                if retry_messages
                else None
            )
            if (
                retry_messages
                and budget.max_input_tokens > 0
                and retry_tokens <= budget.max_input_tokens
            ):
                messages = retry_messages
                build.estimated_input_tokens = retry_tokens
                build.selected_variant = "retry"
                build.truncated = True
                self.budget_manager.apply_selected_sections(
                    build,
                    "retry",
                    self.config,
                )
                build.shrink_events.append(
                    {
                        "section": "full_request",
                        "before_tokens": selected_tokens,
                        "after_tokens": retry_tokens,
                        "strategy": "preflight_budget_retry",
                    }
                )
            else:
                build.stop_reason = "context_budget_exceeded"
                raise self._governance_error(
                    "context_budget_exceeded",
                    "estimated input tokens %s exceed budget %s"
                    % (selected_tokens, budget.max_input_tokens),
                    budget,
                    build,
                    capabilities,
                    call_id=call_id,
                )
        if mode == "enforce" and not _preserves_required_content(
            envelope, messages
        ):
            build.stop_reason = "context_required_section_missing"
            raise self._governance_error(
                "context_required_section_missing",
                "compacted request omitted required context",
                budget,
                build,
                capabilities,
                call_id=call_id,
            )
        if mode == "enforce" and not _preserves_profile_requirements(
            profile,
            envelope,
            messages,
            build.selected_variant,
        ):
            build.stop_reason = "context_required_section_missing"
            raise self._governance_error(
                "context_required_section_missing",
                "request omitted a section required by the context profile",
                budget,
                build,
                capabilities,
                call_id=call_id,
            )

        attempts = 0
        try:
            attempts += 1
            provider_result = self._complete(
                provider,
                messages,
                temperature,
                min(
                    int(envelope.requested_output_tokens or profile.reserved_output_tokens),
                    capabilities.max_output_tokens,
                ),
            )
        except Exception as exc:
            retry_messages = envelope.retry_messages
            max_retries = int(
                _config_get(
                    self.config, "agent_context_max_overflow_retries", 1
                )
            )
            if (
                not retry_messages
                or build.selected_variant == "retry"
                or max_retries < 1
                or not is_context_overflow_error(exc)
            ):
                if is_context_overflow_error(exc):
                    raise self._governance_error(
                        "provider_context_limit_exceeded",
                        str(exc),
                        budget,
                        build,
                        capabilities,
                        attempts=attempts,
                        call_id=call_id,
                    ) from exc
                raise
            retry_tokens = self.budget_manager.estimate_messages(
                retry_messages, envelope, capabilities
            )
            if retry_tokens > budget.max_input_tokens:
                raise self._governance_error(
                    "context_budget_exceeded",
                    "retry input exceeds context budget",
                    budget,
                    build,
                    capabilities,
                    attempts=attempts,
                    call_id=call_id,
                )
            if not _preserves_required_content(envelope, retry_messages):
                raise self._governance_error(
                    "context_required_section_missing",
                    "retry request omitted required context",
                    budget,
                    build,
                    capabilities,
                    attempts=attempts,
                    call_id=call_id,
                )
            if not _preserves_profile_requirements(
                profile,
                envelope,
                retry_messages,
                "retry",
            ):
                raise self._governance_error(
                    "context_required_section_missing",
                    "retry request omitted a section required by the context profile",
                    budget,
                    build,
                    capabilities,
                    attempts=attempts,
                    call_id=call_id,
                )
            attempts += 1
            try:
                provider_result = self._complete(
                    provider,
                    retry_messages,
                    temperature,
                    min(
                        int(envelope.requested_output_tokens or profile.reserved_output_tokens),
                        capabilities.max_output_tokens,
                    ),
                )
            except Exception as retry_exc:
                if is_context_overflow_error(retry_exc):
                    build.estimated_input_tokens = retry_tokens
                    build.selected_variant = "retry"
                    build.truncated = True
                    self.budget_manager.apply_selected_sections(
                        build,
                        "retry",
                        self.config,
                    )
                    build.shrink_events.append(
                        {
                            "section": "full_request",
                            "before_tokens": selected_tokens,
                            "after_tokens": retry_tokens,
                            "strategy": "provider_overflow_retry",
                        }
                    )
                    raise self._governance_error(
                        "provider_context_limit_exceeded",
                        str(retry_exc),
                        budget,
                        build,
                        capabilities,
                        attempts=attempts,
                        call_id=call_id,
                    ) from retry_exc
                raise
            build.estimated_input_tokens = retry_tokens
            build.selected_variant = "retry"
            build.truncated = True
            self.budget_manager.apply_selected_sections(
                build,
                "retry",
                self.config,
            )
            build.shrink_events.append(
                {
                    "section": "full_request",
                    "before_tokens": selected_tokens,
                    "after_tokens": retry_tokens,
                    "strategy": "provider_overflow_retry",
                }
            )

        usage = normalize_usage(
            getattr(provider_result, "usage", None),
            build.estimated_input_tokens,
        )
        telemetry = context_telemetry(
            call_id,
            budget,
            build,
            capabilities,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "source": usage.source,
            },
            attempts=attempts,
        )
        provider_result.context = telemetry
        return LLMCallResult(
            call_id=call_id,
            provider_result=provider_result,
            context_result=build,
            usage=usage,
            attempts=attempts,
        )

    @staticmethod
    def _governance_error(
        stop_reason,
        message,
        budget,
        build,
        capabilities,
        attempts=0,
        call_id="",
    ):
        build.stop_reason = stop_reason
        telemetry = context_telemetry(
            call_id,
            budget,
            build,
            capabilities,
            attempts=attempts,
            stop_reason=stop_reason,
        )
        return ContextGovernanceError(
            stop_reason,
            message,
            context=telemetry,
        )

    @staticmethod
    def _complete(provider, messages, temperature: float, max_output_tokens: int):
        complete = provider.complete
        try:
            parameters = inspect.signature(complete).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs = {}
        if "temperature" in parameters or accepts_kwargs:
            kwargs["temperature"] = temperature
        if "max_output_tokens" in parameters or accepts_kwargs:
            kwargs["max_output_tokens"] = max_output_tokens
        return complete(messages, **kwargs)


def is_context_overflow_error(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "").lower()
    message = str(exc).lower()
    return code in {"context_length_exceeded", "context_window_exceeded"} or any(
        marker in message
        for marker in (
            "context length",
            "context_length",
            "context window",
            "too many tokens",
            "prompt is too long",
            "request too large",
        )
    )


class _DefaultContextConfig:
    agent_context_mode = "observe"
    agent_context_window_tokens: Optional[int] = None
    agent_context_reserved_output_tokens = 4096
    agent_context_safety_margin_tokens = 2048
    agent_context_unknown_model_fallback_tokens = 8192
    agent_context_max_overflow_retries = 1


def _default_sections(envelope: PromptEnvelope, profile_name: str):
    sections = []
    for index, message in enumerate(envelope.messages):
        trusted = getattr(message, "role", "") == "system"
        sections.append(
            ContextSection(
                name="instructions" if trusted else "task_%s" % index,
                content=getattr(message, "content", ""),
                priority=ContextPriority.REQUIRED
                if trusted
                else ContextPriority.RELEVANT_EVIDENCE,
                trust_level=TrustLevel.TRUSTED_INSTRUCTION
                if trusted
                else TrustLevel.UNTRUSTED_REPOSITORY,
                content_type="instruction" if trusted else "request",
                required=trusted,
                source="message",
                metadata={"profile": profile_name},
            )
        )
    return sections


def _config_get(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _preserves_required_content(envelope, selected) -> bool:
    original = envelope.messages
    required = [
        (getattr(message, "role", ""), getattr(message, "content", ""))
        for message in original
        if getattr(message, "role", "") == "system"
    ]
    available = {
        (getattr(message, "role", ""), getattr(message, "content", ""))
        for message in selected
    }
    if not all(message in available for message in required):
        return False

    fragments = [
        str(fragment)
        for fragment in (envelope.required_fragments or [])
        if str(fragment)
    ]
    for section in envelope.sections or []:
        if not section.required:
            continue
        fragment = section.metadata.get("required_fragment")
        if fragment:
            fragments.append(str(fragment))
        elif isinstance(section.content, str) and section.content:
            fragments.append(section.content)
    selected_text = "\n".join(
        str(getattr(message, "content", "")) for message in selected
    )
    return all(fragment in selected_text for fragment in fragments)


def _preserves_profile_requirements(
    profile,
    envelope,
    selected,
    variant: str,
) -> bool:
    required = set(profile.required_sections or ())
    roles = {str(getattr(message, "role", "")) for message in selected}
    if "instructions" in required and "system" not in roles:
        return False
    if "task" in required and not any(
        str(getattr(message, "role", "")) != "system"
        and str(getattr(message, "content", ""))
        for message in selected
    ):
        return False
    sections = _sections_for_variant(envelope, variant)
    section_names = {section.name for section in sections}
    return all(
        name in {"instructions", "task"} or name in section_names
        for name in required
    )


def _sections_for_variant(envelope, variant: str):
    if variant == "candidate" and envelope.candidate_sections is not None:
        return envelope.candidate_sections
    if variant == "retry" and envelope.retry_sections is not None:
        return envelope.retry_sections
    return envelope.sections
