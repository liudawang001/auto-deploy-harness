"""Recovery Service: orchestrates prepare/reconcile/apply_decision.

The RecoveryService is the main entry point for side-effect node
recovery. It coordinates the OperationJournal and registered
reconcilers to determine whether an operation can be reused,
continued, retried, or needs manual intervention.
"""
from auto_harness.recovery.schemas import DEFAULT_DECISION, RECONCILE_DECISIONS
from auto_harness.utils.time import utc_now_iso


class RecoveryService:
    """Orchestrates operation lifecycle with journal and reconcilers.

    Usage:
        service = RecoveryService(journal, {
            "model_download": DownloadReconciler(),
            "local_process": ProcessReconciler(probe, port_probe),
            "docker_service": DockerReconciler(command_runner),
        })
        record = service.prepare(spec.to_record())
        result = service.reconcile(record)
        updated = service.apply_decision(record, result)
    """

    def __init__(self, journal, reconcilers) -> None:
        self.journal = journal
        self.reconcilers = dict(reconcilers)

    def prepare(self, record):
        """Prepare an operation for execution.

        Creates the journal record if new, or returns the existing
        record if the operation_id already exists with matching hash.
        """
        return self.journal.create(record)

    def reconcile(self, operation):
        """Reconcile an operation against external state.

        Dispatches to the reconciler registered for the operation's
        resource_type. If no reconciler is registered, returns
        a manual decision (safest default).
        """
        reconciler = self.reconcilers.get(operation["resource_type"])
        if reconciler is None:
            return {
                "decision": "manual",
                "observed_state": {},
                "reason": "reconciler_not_registered",
                "evidence_paths": [],
            }
        return reconciler.reconcile(operation)

    def apply_decision(self, operation, result):
        """Apply a reconcile decision to the operation journal.

        Handles each decision type:
        - reuse: mark committed (skip execution)
        - continue/retry: transition to running (will execute)
        - cleanup_then_retry: mark manual (needs approval)
        - conflict: mark conflict (terminal, stops deployment)
        - manual: mark manual (needs approval)
        - unknown decision: treat as manual

        For continue/retry, if the operation is in 'unknown' status,
        it first transitions through 'retryable' before going to
        'running'. This ensures the state machine is never skipped.
        """
        decision = result.get("decision", DEFAULT_DECISION)

        if decision == "reuse":
            return self.journal.transition(
                operation["operation_id"],
                "committed",
                reconcile_result=result,
                last_checked_at=utc_now_iso(),
            )

        if decision in ("continue", "retry"):
            current = self.journal.load(operation["operation_id"])
            if current["status"] == "unknown":
                self.journal.transition(operation["operation_id"], "retryable")
            current = self.journal.load(operation["operation_id"])
            return self.journal.transition(
                operation["operation_id"],
                "running",
                attempt=int(current.get("attempt", 0)) + 1,
                started_at=utc_now_iso(),
                last_checked_at=utc_now_iso(),
                reconcile_result=result,
            )

        if decision == "cleanup_then_retry":
            return self.journal.transition(
                operation["operation_id"],
                "manual",
                reconcile_result=result,
                last_checked_at=utc_now_iso(),
            )

        if decision == "conflict":
            return self.journal.transition(
                operation["operation_id"],
                "conflict",
                reconcile_result=result,
                last_checked_at=utc_now_iso(),
            )

        # Unknown decision or "manual" → safest default
        return self.journal.transition(
            operation["operation_id"],
            "manual",
            reconcile_result=result,
            last_checked_at=utc_now_iso(),
        )
