"""Recovery module: operation journal, reconcilers, and recovery service.

Provides stable operation IDs, journal-based side-effect tracking,
and reconciler-based external state detection for safe resume.
"""
from auto_harness.recovery.schemas import (
    OperationRecord,
    ReconcileResult,
    canonical_json,
    compute_operation_id,
    OPERATION_STATUSES,
    RECONCILE_DECISIONS,
)
from auto_harness.recovery.journal import OperationJournal
from auto_harness.recovery.service import RecoveryService

__all__ = [
    "OperationRecord",
    "ReconcileResult",
    "canonical_json",
    "compute_operation_id",
    "OPERATION_STATUSES",
    "RECONCILE_DECISIONS",
    "OperationJournal",
    "RecoveryService",
]
