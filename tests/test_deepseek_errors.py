"""Tests for ProviderError and error utilities.

Validates:
- Structured error classification
- Convenience constructors
- sanitize_error_body for secret redaction
- Error serialization safety
"""
import json
import unittest

from auto_harness.providers.errors import (
    NON_RETRYABLE_CATEGORIES,
    CONTEXT_OVERFLOW_CATEGORIES,
    ProviderError,
    ErrorCategory,
    authentication_error,
    insufficient_balance_error,
    rate_limited_error,
    server_error,
    network_timeout_error,
    empty_content_error,
    invalid_response_error,
    context_overflow_error,
    configuration_error,
    sanitize_error_body,
)


class ProviderErrorTests(unittest.TestCase):
    """Tests for ProviderError and convenience constructors."""

    # --- Core ProviderError ---

    def test_basic_error(self):
        err = ProviderError("test message", provider_name="deepseek")
        self.assertEqual(str(err), "test message")
        self.assertEqual(err.provider_name, "deepseek")
        self.assertEqual(err.category, ErrorCategory.PROVIDER_UNAVAILABLE)
        self.assertTrue(err.retryable)

    def test_status_code_stored(self):
        err = ProviderError("msg", provider_name="dp", status_code=401)
        self.assertEqual(err.status_code, 401)

    def test_error_code_stored(self):
        err = ProviderError("msg", provider_name="dp", error_code="invalid_api_key")
        self.assertEqual(err.error_code, "invalid_api_key")

    def test_safe_detail_truncated(self):
        long_detail = "x" * 1000
        err = ProviderError("msg", provider_name="dp", safe_detail=long_detail)
        self.assertLessEqual(len(err.safe_detail), 500)

    def test_to_dict_no_secrets(self):
        err = ProviderError(
            "auth failed",
            provider_name="deepseek",
            status_code=401,
            category=ErrorCategory.AUTHENTICATION_FAILED,
        )
        d = err.to_dict()
        for key in d:
            self.assertNotIn("secret", str(d[key]).lower())
            self.assertNotIn("sk-", str(d[key]))

    # --- Convenience constructors ---

    def test_authentication_error(self):
        err = authentication_error(
            "deepseek", status_code=401, detail="Invalid API Key"
        )
        self.assertEqual(err.category, ErrorCategory.AUTHENTICATION_FAILED)
        self.assertEqual(err.status_code, 401)
        self.assertFalse(err.retryable)

    def test_insufficient_balance_error(self):
        err = insufficient_balance_error("deepseek")
        self.assertEqual(err.category, ErrorCategory.INSUFFICIENT_BALANCE)
        self.assertEqual(err.status_code, 402)
        self.assertFalse(err.retryable)

    def test_rate_limited_error(self):
        err = rate_limited_error("deepseek", retry_after=30)
        self.assertEqual(err.category, ErrorCategory.RATE_LIMITED)
        self.assertEqual(err.status_code, 429)
        self.assertTrue(err.retryable)
        self.assertIn("30", str(err))

    def test_server_error(self):
        err = server_error("deepseek", status_code=500)
        self.assertEqual(err.category, ErrorCategory.SERVER_ERROR)
        self.assertEqual(err.status_code, 500)
        self.assertTrue(err.retryable)

    def test_network_timeout_error(self):
        err = network_timeout_error("deepseek", timeout_seconds=30)
        self.assertEqual(err.category, ErrorCategory.NETWORK_TIMEOUT)
        self.assertTrue(err.retryable)

    def test_empty_content_error(self):
        err = empty_content_error("deepseek", request_id="req-1")
        self.assertEqual(err.category, ErrorCategory.EMPTY_CONTENT)
        self.assertFalse(err.retryable)

    def test_invalid_response_error(self):
        err = invalid_response_error("deepseek", detail="not json")
        self.assertEqual(err.category, ErrorCategory.INVALID_RESPONSE)
        self.assertFalse(err.retryable)

    def test_context_overflow_error(self):
        err = context_overflow_error("deepseek", detail="too many tokens")
        self.assertEqual(err.category, ErrorCategory.CONTEXT_OVERFLOW)
        self.assertFalse(
            err.retryable,
            "context_overflow should be non-retryable (Context Executor handles retries)",
        )

    def test_configuration_error(self):
        err = configuration_error("deepseek", detail="model is retired")
        self.assertEqual(err.category, ErrorCategory.CONFIGURATION_ERROR)
        self.assertFalse(err.retryable)

    # --- Category sets ---

    def test_non_retryable_categories_well_defined(self):
        """Non-retryable categories include auth, balance, invalid, config, context_overflow."""
        for cat in ("authentication_failed", "insufficient_balance",
                     "invalid_request", "invalid_parameter",
                     "configuration_error", "context_overflow",
                     "empty_content", "invalid_response",
                     "deadline_exceeded"):
            self.assertIn(cat, NON_RETRYABLE_CATEGORIES)

    def test_context_overflow_in_own_set(self):
        """Context overflow is in its own category set."""
        self.assertIn(ErrorCategory.CONTEXT_OVERFLOW, CONTEXT_OVERFLOW_CATEGORIES)


class SanitizeErrorBodyTests(unittest.TestCase):
    """Tests for sanitize_error_body."""

    def test_redacts_authorization_header(self):
        body = "Authorization: Bearer sk-1234567890abcdef\nContent-Type: application/json"
        result = sanitize_error_body(body)
        self.assertNotIn("sk-1234567890abcdef", result)
        self.assertIn("[REDACTED]", result)
        self.assertIn("Content-Type", result)

    def test_redacts_api_key_header(self):
        body = "x-api-key: secret-key-value"
        result = sanitize_error_body(body)
        self.assertNotIn("secret-key-value", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_token_in_json(self):
        body = '{"error":{"message":"ok"},"token":"sk-secret"}'
        result = sanitize_error_body(body)
        self.assertNotIn("sk-secret", result)
        self.assertIn("[REDACTED]", result)

    def test_truncates_long_body(self):
        body = "x" * 1000
        result = sanitize_error_body(body)
        self.assertLessEqual(len(result), 500)

    def test_redacts_bare_sk_token(self):
        result = sanitize_error_body("upstream echoed sk-1234567890abcdef")
        self.assertNotIn("sk-1234567890abcdef", result)

    def test_empty_body_returns_empty(self):
        self.assertEqual(sanitize_error_body(""), "")
        self.assertEqual(sanitize_error_body(None), "")

    def test_preserves_json_structure_where_possible(self):
        body = '{"key":"value","info":"safe"}'
        result = sanitize_error_body(body)
        self.assertIn("key", result)
        self.assertIn("safe", result)


if __name__ == "__main__":
    unittest.main()
