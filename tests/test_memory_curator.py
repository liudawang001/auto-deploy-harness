"""Tests for MemoryCurator (Phase 2).

Verifies:
- Valid curator JSON is parsed correctly
- Invalid JSON is rejected
- Secret-like markdown is rejected
- Absolute local paths are rejected
- HTTP 200 alone success rule is rejected
- Privilege escalation suggestions are rejected
- Mock provider can generate candidate drafts
- Missing required keys are rejected
"""
import json
import unittest
from unittest.mock import MagicMock

from auto_harness.memory.curator import MemoryCurator


def _make_valid_response() -> dict:
    """Create a valid LLM response dict."""
    return {
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
                "discover /config",
                "infer callable dependency",
                "send current trace_id through the inferred API",
            ],
            "do_not": [
                "do not mark success on HTTP 200 alone",
                "do not reuse old trace_id",
            ],
        },
        "skill_patch": {
            "target_skill": "verify-evidence/SKILL.md",
            "section_title": "Gradio API shape discovery",
            "markdown": "## Memory Evolution: Gradio API shape discovery\n\nWhen verify is uncertain and framework=gradio, discover /config endpoint.",
        },
        "regression_proposal": {
            "case_ids": ["gradio_api_shape_variation"],
            "new_case_suggestions": [],
        },
        "risk": {
            "level": "low",
            "overfit_risk": "medium",
            "failure_modes": [
                "wrong endpoint inference",
                "false pass if trace_id is not checked",
            ],
        },
    }


class _FakeProvider:
    """LLM provider that returns a fixed JSON string."""

    def __init__(self, response_dict: dict):
        self._text = json.dumps(response_dict, ensure_ascii=False)

    def complete(self, messages):
        return MagicMock(text=self._text)


class _FakeProviderRaw:
    """LLM provider that returns raw text."""

    def __init__(self, text: str):
        self._text = text

    def complete(self, messages):
        return MagicMock(text=self._text)


class TestMemoryCuratorParse(unittest.TestCase):
    """Test parse_response() method."""

    def setUp(self):
        self.curator = MemoryCurator()

    def test_valid_json_parsed(self):
        """Valid JSON response is parsed correctly."""
        response = _make_valid_response()
        text = json.dumps(response, ensure_ascii=False)
        result = self.curator.parse_response(text)
        self.assertEqual(result["status"], "ok")
        self.assertIn("pattern", result)
        self.assertIn("reusable_rule", result)
        self.assertIn("skill_patch", result)

    def test_invalid_json_rejected(self):
        """Non-JSON text is rejected."""
        result = self.curator.parse_response("I think we should probe the service endpoint.")
        self.assertNotEqual(result["status"], "ok")
        self.assertIn("not valid json", result.get("error", "").lower())

    def test_status_not_ok_rejected(self):
        """Response with status != ok is rejected."""
        response = _make_valid_response()
        response["status"] = "error"
        text = json.dumps(response)
        result = self.curator.parse_response(text)
        self.assertNotEqual(result["status"], "ok")

    def test_missing_pattern_rejected(self):
        """Missing required 'pattern' key is rejected."""
        response = _make_valid_response()
        del response["pattern"]
        text = json.dumps(response)
        result = self.curator.parse_response(text)
        self.assertNotEqual(result["status"], "ok")
        self.assertIn("pattern", result.get("error", ""))

    def test_missing_skill_patch_markdown_rejected(self):
        """Missing skill_patch.markdown is rejected."""
        response = _make_valid_response()
        del response["skill_patch"]["markdown"]
        text = json.dumps(response)
        result = self.curator.parse_response(text)
        self.assertNotEqual(result["status"], "ok")
        self.assertIn("markdown", result.get("error", ""))

    def test_json_in_code_fence_parsed(self):
        """JSON wrapped in markdown code fence is parsed."""
        response = _make_valid_response()
        text = "```json\n%s\n```" % json.dumps(response, ensure_ascii=False)
        result = self.curator.parse_response(text)
        self.assertEqual(result["status"], "ok")

    def test_non_dict_json_rejected(self):
        """JSON array (not object) is rejected."""
        result = self.curator.parse_response("[1, 2, 3]")
        self.assertNotEqual(result["status"], "ok")


class TestMemoryCuratorValidate(unittest.TestCase):
    """Test security validation of parsed candidates."""

    def setUp(self):
        self.curator = MemoryCurator()

    def test_secret_in_markdown_rejected(self):
        """Secret-like content in skill_patch.markdown is rejected."""
        response = _make_valid_response()
        response["skill_patch"]["markdown"] = "Set api_key=sk-xxx in the config"
        validation = self.curator._validate_candidate(response)
        self.assertFalse(validation["valid"])
        self.assertIn("secret", validation["reason"].lower())

    def test_absolute_path_in_markdown_rejected(self):
        """Absolute /tmp/ or /Users/ path in markdown is rejected."""
        response = _make_valid_response()
        response["skill_patch"]["markdown"] = "Copy weights from /tmp/model_cache to project"
        validation = self.curator._validate_candidate(response)
        self.assertFalse(validation["valid"])
        self.assertIn("path", validation["reason"].lower())

    def test_http_200_success_rule_rejected(self):
        """HTTP 200 alone as success rule is rejected."""
        response = _make_valid_response()
        response["skill_patch"]["markdown"] = "HTTP 200 is enough to confirm the service is working"
        validation = self.curator._validate_candidate(response)
        self.assertFalse(validation["valid"])
        self.assertIn("HTTP 200", validation["reason"])

    def test_privilege_escalation_rejected(self):
        """Suggestions to expand shell/source edit permissions are rejected."""
        response = _make_valid_response()
        response["skill_patch"]["markdown"] = "allow arbitrary shell commands for debugging"
        validation = self.curator._validate_candidate(response)
        self.assertFalse(validation["valid"])
        self.assertIn("privilege", validation["reason"].lower())

    def test_bearer_token_rejected(self):
        """Bearer token in markdown is rejected."""
        response = _make_valid_response()
        response["skill_patch"]["markdown"] = "Use Bearer xyz for authentication"
        validation = self.curator._validate_candidate(response)
        self.assertFalse(validation["valid"])
        self.assertIn("secret", validation["reason"].lower())

    def test_valid_candidate_passes(self):
        """Valid candidate passes validation."""
        response = _make_valid_response()
        validation = self.curator._validate_candidate(response)
        self.assertTrue(validation["valid"])


class TestMemoryCuratorCurate(unittest.TestCase):
    """Test the full curate() flow with mock provider."""

    def test_no_provider_returns_failed(self):
        """No provider configured returns failed status."""
        curator = MemoryCurator(provider=None)
        result = curator.curate({})
        self.assertEqual(result["status"], "failed")
        self.assertIn("no LLM provider", result["error"])

    def test_mock_provider_generates_candidate(self):
        """Mock provider can generate a valid candidate draft."""
        response = _make_valid_response()
        provider = _FakeProvider(response)
        curator = MemoryCurator(provider=provider)
        cluster = {
            "stage": "verify",
            "category": "verification_gap",
            "frameworks": ["gradio"],
            "memory_ids": ["mem_001", "mem_002"],
            "symptoms": ["HTTP 200 but no trace_id"],
            "root_causes": ["non-default Gradio API shape"],
            "repair_actions": ["discover /config"],
            "verification_trace_ids": ["trace-001"],
            "regression_case_ids": ["gradio_config_discovery"],
        }
        result = curator.curate(cluster)
        self.assertEqual(result["status"], "ok")
        self.assertIsNotNone(result["candidate_draft"])
        self.assertIn("raw_response_hash", result)

    def test_invalid_json_from_provider_rejected(self):
        """Provider returning non-JSON is rejected."""
        provider = _FakeProviderRaw("I think you should probe the endpoint.")
        curator = MemoryCurator(provider=provider)
        result = curator.curate({})
        self.assertEqual(result["status"], "failed")
        self.assertIn("not valid json", result.get("error", "").lower())

    def test_secret_candidate_rejected(self):
        """Candidate with secret in markdown is rejected at curation level."""
        response = _make_valid_response()
        response["skill_patch"]["markdown"] = "Set api_key=sk-xxx and retry"
        provider = _FakeProvider(response)
        curator = MemoryCurator(provider=provider)
        result = curator.curate({})
        self.assertEqual(result["status"], "rejected")

    def test_provider_exception_handled(self):
        """Provider throwing an exception is handled gracefully."""
        class FailingProvider:
            def complete(self, messages):
                raise RuntimeError("API timeout")
        curator = MemoryCurator(provider=FailingProvider())
        result = curator.curate({})
        self.assertEqual(result["status"], "failed")
        self.assertIn("API timeout", result["error"])


if __name__ == "__main__":
    unittest.main()
