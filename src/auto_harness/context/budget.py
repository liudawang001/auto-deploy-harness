import hashlib
import json
from typing import Any, Dict, List

from auto_harness.context.models import (
    ContextBudget,
    ContextBuildResult,
    ContextProfile,
    PromptEnvelope,
    ProviderCapabilities,
)
from auto_harness.context.tokens import ConservativeTokenEstimator


class ContextBudgetManager:
    def __init__(self, estimator=None) -> None:
        self.estimator = estimator or ConservativeTokenEstimator()

    def build(
        self,
        *,
        call_site: str,
        envelope: PromptEnvelope,
        profile: ContextProfile,
        capabilities: ProviderCapabilities,
        config: Any,
    ):
        reserved_output = min(
            int(envelope.requested_output_tokens or profile.reserved_output_tokens),
            capabilities.max_output_tokens,
        )
        safety_margin = max(
            0, int(_config_get(config, "agent_context_safety_margin_tokens", 2048))
        )
        provider_input_limit = (
            capabilities.context_window_tokens - reserved_output - safety_margin
        )
        max_input_tokens = min(provider_input_limit, profile.total_input_cap_tokens)
        budget = ContextBudget(
            call_site=call_site,
            provider_name=capabilities.provider_name,
            model=capabilities.model,
            context_window_tokens=capabilities.context_window_tokens,
            reserved_output_tokens=reserved_output,
            safety_margin_tokens=safety_margin,
            max_input_tokens=max(0, max_input_tokens),
        )
        original_tokens = self._estimate(envelope.messages, envelope, capabilities)
        candidate_messages = envelope.candidate_messages or envelope.messages
        candidate_tokens = self._estimate(candidate_messages, envelope, capabilities)
        mode = str(_config_get(config, "agent_context_mode", "observe") or "observe")
        warn_ratio = float(_config_get(config, "agent_context_warn_ratio", 0.70))
        compact_ratio = float(
            _config_get(config, "agent_context_compact_ratio", 0.85)
        )
        warnings = []
        if max_input_tokens > 0 and original_tokens > max_input_tokens * warn_ratio:
            warnings.append("context_input_above_warning_threshold")

        selected_variant = "original"
        selected_messages = envelope.messages
        if mode == "enforce" and max_input_tokens > 0:
            compact_threshold = int(max_input_tokens * compact_ratio)
            if (
                original_tokens > compact_threshold
                and candidate_tokens < original_tokens
            ):
                selected_variant = "candidate"
                selected_messages = candidate_messages
        selected_tokens = self._estimate(selected_messages, envelope, capabilities)
        selected_sections = _variant_sections(envelope, selected_variant)
        section_metrics = self._metrics_for_config(selected_sections, config)
        dropped = [
            str(item.get("name"))
            for item in section_metrics
            if item.get("dropped")
        ]
        shrink_events = []
        if selected_variant == "candidate":
            shrink_events.append(
                {
                    "section": "full_request",
                    "before_tokens": original_tokens,
                    "after_tokens": candidate_tokens,
                    "strategy": "stage_context_profile",
                }
            )
        return budget, ContextBuildResult(
            envelope=envelope,
            estimated_input_tokens=selected_tokens,
            original_estimated_input_tokens=original_tokens,
            candidate_estimated_input_tokens=candidate_tokens,
            section_metrics=section_metrics,
            truncated=selected_variant != "original",
            dropped_sections=dropped,
            shrink_events=shrink_events,
            included_files=_included_files(selected_sections),
            skill_count=_metadata_count(selected_sections, "skill_count"),
            memory_count=_metadata_count(selected_sections, "memory_count"),
            mode=mode,
            selected_variant=selected_variant,
            warnings=warnings,
        )

    def apply_selected_sections(
        self,
        build: ContextBuildResult,
        variant: str,
        config: Any,
    ) -> None:
        sections = _variant_sections(build.envelope, variant)
        build.section_metrics = self._metrics_for_config(sections, config)
        build.dropped_sections = [
            str(item.get("name"))
            for item in build.section_metrics
            if item.get("dropped")
        ]
        build.included_files = _included_files(sections)
        build.skill_count = _metadata_count(sections, "skill_count")
        build.memory_count = _metadata_count(sections, "memory_count")

    def estimate_messages(
        self,
        messages,
        envelope: PromptEnvelope,
        capabilities: ProviderCapabilities,
    ) -> int:
        return self._estimate(messages, envelope, capabilities)

    def _estimate(self, messages, envelope, capabilities) -> int:
        return self.estimator.estimate_request(
            messages,
            capabilities,
            tools=envelope.tools,
            output_schema=envelope.output_schema,
        )

    def _section_metrics(self, sections) -> List[Dict[str, Any]]:
        metrics = []
        for section in sections:
            serialized = json.dumps(
                section.content, ensure_ascii=False, sort_keys=True, default=str
            )
            metrics.append(
                {
                    "name": section.name,
                    "priority": int(section.priority),
                    "trust_level": section.trust_level.value,
                    "content_type": section.content_type,
                    "source": section.source,
                    "source_hash": section.source_hash
                    or hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                    "estimated_tokens": self.estimator.estimate_text(serialized),
                    "required": section.required,
                    "dropped": bool(section.metadata.get("dropped")),
                    "truncated": bool(section.metadata.get("truncated")),
                }
            )
        return metrics

    def _metrics_for_config(self, sections, config) -> List[Dict[str, Any]]:
        if not bool(
            _config_get(
                config,
                "agent_context_trace_section_details",
                True,
            )
        ):
            return []
        return self._section_metrics(sections)


def context_telemetry(
    call_id: str,
    budget: ContextBudget,
    build: ContextBuildResult,
    capabilities: ProviderCapabilities,
    usage=None,
    attempts: int = 0,
    stop_reason: str = "",
) -> Dict[str, Any]:
    actual = usage or {}
    max_input = max(1, budget.max_input_tokens)
    return {
        "call_id": call_id,
        "call_site": budget.call_site,
        "profile": {
            "name": build.envelope.sections[0].metadata.get("profile", "")
            if build.envelope.sections
            else "",
        },
        "mode": build.mode,
        "selected_variant": build.selected_variant,
        "provider": capabilities.provider_name,
        "model": capabilities.model,
        "capability_source": capabilities.source,
        "context_window_tokens": budget.context_window_tokens,
        "reserved_output_tokens": budget.reserved_output_tokens,
        "safety_margin_tokens": budget.safety_margin_tokens,
        "max_input_tokens": budget.max_input_tokens,
        "original_estimated_input_tokens": build.original_estimated_input_tokens,
        "candidate_estimated_input_tokens": build.candidate_estimated_input_tokens,
        "estimated_input_tokens": build.estimated_input_tokens,
        "usage_ratio": round(build.estimated_input_tokens / float(max_input), 6),
        "token_estimator": "utf8_bytes_upper_bound",
        "truncated": build.truncated,
        "dropped_sections": build.dropped_sections,
        "included_files": build.included_files,
        "skill_count": build.skill_count,
        "memory_count": build.memory_count,
        "shrink_events": build.shrink_events,
        "warnings": build.warnings,
        "sections": build.section_metrics,
        "usage": actual,
        "attempts": attempts,
        "stop_reason": stop_reason,
    }


def _metadata_count(sections, key: str) -> int:
    return sum(int(section.metadata.get(key, 0) or 0) for section in sections)


def _included_files(sections) -> List[str]:
    result = []
    for section in sections:
        for path in section.metadata.get("included_files", []) or []:
            value = str(path)
            if value and not value.startswith("/") and value not in result:
                result.append(value)
    return result


def _variant_sections(envelope: PromptEnvelope, variant: str):
    if variant == "candidate" and envelope.candidate_sections is not None:
        return envelope.candidate_sections
    if variant == "retry" and envelope.retry_sections is not None:
        return envelope.retry_sections
    return envelope.sections


def _config_get(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)
