"""Controller outcome normalization for automation and CLI callers."""


SUCCESS_STATUSES = frozenset({"completed", "completed_dry_run", "success"})
PARTIAL_STATUSES = frozenset({
    "blocked", "interrupted", "partial", "stopped", "uncertain"
})
FAILURE_STATUSES = frozenset({"error", "failed", "internal_error"})


def controller_exit_code(status: str) -> int:
    """Map a persisted controller outcome to the public CLI contract."""
    normalized = str(status or "").strip().lower()
    if normalized in SUCCESS_STATUSES:
        return 0
    if normalized in PARTIAL_STATUSES:
        return 1
    if normalized in FAILURE_STATUSES:
        return 3
    return 1
