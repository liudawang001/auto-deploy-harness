"""Structured provider errors for classification, retry decisions, and safe logging.

All errors produced by providers MUST use ProviderError (or a subclass)
so that upper layers can distinguish auth failures from rate limits from
server errors without string-matching on error messages.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Error categories — stable strings that callers can branch on
# ---------------------------------------------------------------------------

class ErrorCategory:
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_FAILED = "authentication_failed"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    INVALID_PARAMETER = "invalid_parameter"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    SERVER_OVERLOADED = "server_overloaded"
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_ERROR = "network_error"
    CONTEXT_OVERFLOW = "context_overflow"
    EMPTY_CONTENT = "empty_content"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CONFIGURATION_ERROR = "configuration_error"
    DEADLINE_EXCEEDED = "deadline_exceeded"


# Categories that should NEVER be retried (they will not succeed on retry)
NON_RETRYABLE_CATEGORIES = frozenset({
    ErrorCategory.INVALID_REQUEST,
    ErrorCategory.AUTHENTICATION_FAILED,
    ErrorCategory.INSUFFICIENT_BALANCE,
    ErrorCategory.INVALID_PARAMETER,
    ErrorCategory.CONFIGURATION_ERROR,
    ErrorCategory.INVALID_RESPONSE,
    ErrorCategory.EMPTY_CONTENT,
    ErrorCategory.CONTEXT_OVERFLOW,
    ErrorCategory.DEADLINE_EXCEEDED,
})

# Categories that indicate the context is too large
CONTEXT_OVERFLOW_CATEGORIES = frozenset({
    ErrorCategory.CONTEXT_OVERFLOW,
})


class ProviderError(RuntimeError):
    """Structured error from any LLM provider.

    Key design rules:
    - ``safe_detail`` MUST be truncated and MUST NOT contain API keys,
      Authorization headers, or raw HTTP bodies with secrets.
    - ``category`` is a stable string from ``ErrorCategory``.
    - ``retryable`` is determined from the category, not guessed.
    """

    def __init__(
        self,
        message: str,
        *,
        provider_name: str = "",
        status_code: Optional[int] = None,
        error_code: str = "",
        category: str = ErrorCategory.PROVIDER_UNAVAILABLE,
        request_id: str = "",
        safe_detail: str = "",
        retry_count: int = 0,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(sanitize_error_body(message, max_chars=500))
        self.provider_name = provider_name
        self.status_code = status_code
        self.error_code = error_code
        self.category = category
        self.retryable = category not in NON_RETRYABLE_CATEGORIES
        self.request_id = request_id
        self.safe_detail = _truncate_safe_detail(safe_detail)
        self.retry_count = retry_count
        self.retry_after_seconds = (
            max(0.0, float(retry_after_seconds))
            if retry_after_seconds is not None
            else None
        )

    def to_dict(self) -> dict:
        """Safe serialisation — no secrets included."""
        return {
            "provider_name": self.provider_name,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "category": self.category,
            "retryable": self.retryable,
            "request_id": self.request_id,
            "safe_detail": self.safe_detail,
            "retry_count": self.retry_count,
            "retry_after_seconds": self.retry_after_seconds,
        }


# ---------------------------------------------------------------------------
# Convenience constructors so callers don't need to know internal categories
# ---------------------------------------------------------------------------

def authentication_error(
    provider_name: str,
    status_code: int = 401,
    detail: str = "",
    request_id: str = "",
) -> ProviderError:
    return ProviderError(
        "%s authentication failed (HTTP %s)" % (provider_name, status_code),
        provider_name=provider_name,
        status_code=status_code,
        category=ErrorCategory.AUTHENTICATION_FAILED,
        request_id=request_id,
        safe_detail=detail,
    )


def insufficient_balance_error(
    provider_name: str,
    detail: str = "",
    request_id: str = "",
) -> ProviderError:
    return ProviderError(
        "%s: insufficient balance" % provider_name,
        provider_name=provider_name,
        status_code=402,
        category=ErrorCategory.INSUFFICIENT_BALANCE,
        request_id=request_id,
        safe_detail=detail,
    )


def rate_limited_error(
    provider_name: str,
    retry_after: Optional[int] = None,
    detail: str = "",
    request_id: str = "",
) -> ProviderError:
    return ProviderError(
        "%s rate limited (retry-after: %s)" % (provider_name, retry_after or "unknown"),
        provider_name=provider_name,
        status_code=429,
        category=ErrorCategory.RATE_LIMITED,
        request_id=request_id,
        safe_detail=detail,
        retry_after_seconds=retry_after,
    )


def server_error(
    provider_name: str,
    status_code: int = 500,
    detail: str = "",
    request_id: str = "",
) -> ProviderError:
    return ProviderError(
        "%s server error (HTTP %s)" % (provider_name, status_code),
        provider_name=provider_name,
        status_code=status_code,
        category=ErrorCategory.SERVER_ERROR,
        request_id=request_id,
        safe_detail=detail,
    )


def network_timeout_error(
    provider_name: str,
    timeout_seconds: int = 0,
    request_id: str = "",
) -> ProviderError:
    return ProviderError(
        "%s network timeout after %ss" % (provider_name, timeout_seconds),
        provider_name=provider_name,
        category=ErrorCategory.NETWORK_TIMEOUT,
        request_id=request_id,
    )


def empty_content_error(
    provider_name: str,
    request_id: str = "",
) -> ProviderError:
    return ProviderError(
        "%s returned empty content" % provider_name,
        provider_name=provider_name,
        category=ErrorCategory.EMPTY_CONTENT,
        request_id=request_id,
    )


def invalid_response_error(
    provider_name: str,
    detail: str = "",
    request_id: str = "",
) -> ProviderError:
    safe = sanitize_error_body(detail)
    return ProviderError(
        "%s returned invalid response: %s" % (provider_name, safe[:200]),
        provider_name=provider_name,
        category=ErrorCategory.INVALID_RESPONSE,
        request_id=request_id,
        safe_detail=safe,
    )


def context_overflow_error(
    provider_name: str,
    detail: str = "",
    request_id: str = "",
) -> ProviderError:
    safe = sanitize_error_body(detail)
    return ProviderError(
        "%s context overflow: %s" % (provider_name, safe[:200]),
        provider_name=provider_name,
        category=ErrorCategory.CONTEXT_OVERFLOW,
        request_id=request_id,
        safe_detail=safe,
    )


def configuration_error(
    provider_name: str,
    detail: str = "",
) -> ProviderError:
    return ProviderError(
        "%s configuration error: %s" % (provider_name, detail),
        provider_name=provider_name,
        category=ErrorCategory.CONFIGURATION_ERROR,
        safe_detail=detail,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENSITIVE_HEADERS = frozenset({
    "authorization", "x-api-key", "api-key", "cookie",
    "set-cookie", "proxy-authorization",
})

_SENSITIVE_PARAMS = frozenset({
    "api_key", "api-key", "token", "secret", "password",
})


def sanitize_error_body(raw_body: str, max_chars: int = 500) -> str:
    """Remove sensitive fields from an error body before logging.

    This is a best-effort strip for JSON bodies and common key=value
    patterns. It is NOT a cryptographic guarantee — do not pass secrets
    through this function and assume they are gone.
    """
    if not raw_body:
        return ""
    text = str(raw_body)
    for header in _SENSITIVE_HEADERS:
        # Try to strip "header: value" lines
        text = _redact_header_line(text, header)
    for param in _SENSITIVE_PARAMS:
        # Try to strip "param=value" patterns
        text = _redact_param(text, param)
    # Redact common API-key shapes even when an upstream exception includes a
    # bare token without a field name.
    import re
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-[REDACTED]", text)
    return text[:max_chars]


def _truncate_safe_detail(detail: str, max_chars: int = 500) -> str:
    if not detail:
        return ""
    return sanitize_error_body(str(detail), max_chars)


def _redact_header_line(text: str, header: str) -> str:
    """Redact a header line like 'Authorization: Bearer sk-xxx'."""
    import re
    pattern = re.compile(
        r'(' + re.escape(header) + r')\s*:\s*.+',
        re.IGNORECASE,
    )
    return pattern.sub(r'\1: [REDACTED]', text)


def _redact_param(text: str, param: str) -> str:
    """Redact key=value patterns for sensitive param names."""
    import re
    pattern = re.compile(
        r'(["\']?' + re.escape(param) + r'["\']?\s*[:=]\s*)\S+',
        re.IGNORECASE,
    )
    return pattern.sub(r'\1[REDACTED]', text)
