"""Memory Evolution Mock Provider: returns valid curator JSON for memory-evolve CLI.

This is a specialized mock provider that returns the exact JSON schema
required by MemoryCurator, unlike the generic MockLLMProvider which
returns a simple status response.

Used by: memory-evolve --provider mock
NOT used by: llm-test --provider mock (which uses the original MockLLMProvider)
"""
import json
from typing import List

from auto_harness.providers.base import LLMResult, Message


class MemoryEvolutionMockProvider:
    """Mock LLM provider that returns valid MemoryCurator candidate JSON.

    The response follows the curator output schema with pattern,
    reusable_rule, skill_patch, regression_proposal, and risk fields.
    """

    provider_name = "mock"

    def complete(self, messages: List[Message], temperature: float = 0.2) -> LLMResult:
        content = {
            "status": "ok",
            "pattern": {
                "stage": "verify",
                "frameworks": ["gradio"],
                "failure_signature": "HTTP 200 but current trace_id absent",
                "root_cause_generalized": "The app exposes a non-default Gradio API shape.",
            },
            "reusable_rule": {
                "when": "verify uncertain and framework_hint=gradio",
                "do": [
                    "discover /config with discover_gradio_api",
                    "send current trace_id through inferred API",
                ],
                "do_not": [
                    "do not mark success on HTTP 200 alone",
                    "do not reuse old trace_id",
                ],
            },
            "skill_patch": {
                "target_skill": "verify-evidence/SKILL.md",
                "section_title": "Gradio API shape discovery",
                "markdown": (
                    "When Gradio verify is uncertain, inspect /config and probe "
                    "the inferred callable endpoint with the current trace_id. "
                    "Do not mark success on HTTP 200 alone."
                ),
            },
            "regression_proposal": {
                "case_ids": ["gradio_config_discovery", "gradio_api_shape_variation"],
                "new_case_suggestions": [],
            },
            "risk": {
                "level": "low",
                "overfit_risk": "medium",
                "failure_modes": ["wrong endpoint inference", "trace_id not checked"],
            },
        }
        return LLMResult(text=json.dumps(content, ensure_ascii=False), raw=content, usage={})
