"""GraphRecoveryAdapter: bridges LangGraph graph state to the Recovery subsystem.

For side-effect stages (env_deploy, model_prepare, runner), this adapter:
- Creates stable operation IDs before execution
- Prepares/reconciles external state before execution
- Commits or fails journal entries after execution
- Hydrates committed results on resume (no duplicate execution)

Key invariants:
- checkpoint says next stage != external resource is safe
- unknown side effect -> reconcile before retry
- conflict -> stop
- manual/cleanup -> interrupt before side effect
- committed operation -> reuse and hydrate prior result; never duplicate execute
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from auto_harness.recovery.journal import OperationJournal
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


@dataclass(frozen=True)
class RecoveryDecision:
    """Result of a recovery gate check."""
    decision: str  # execute|reuse|continue|retry|approval|stop
    operation: Dict[str, Any]
    reconcile_result: Dict[str, Any]
    hydrated_stage_result: Dict[str, Any]
    stop_reason: str = ""


class GraphRecoveryAdapter:
    """Adapter between LangGraph graph state and Recovery subsystem.

    SIDE_EFFECT_STAGES maps each stage to its recovery resource type.
    """

    SIDE_EFFECT_STAGE_TYPES = {
        "env_deploy": "dependency_install",
        "model_prepare": "model_download",
        "runner": "dynamic",  # resolved at runtime: local_process or docker_service
    }

    def __init__(self, reconcilers=None) -> None:
        """Initialize with optional reconciler instances.

        Args:
            reconcilers: Dict mapping resource_type to reconciler instance.
                         If None, default reconcilers are created lazily.
        """
        self.reconcilers = reconcilers or {}

    def capabilities(self, state: dict) -> dict:
        """Determine recovery capabilities from available reconcilers.

        Returns a dict like {"download": True, "local_process": True, ...}.
        """
        caps = {
            "download": "model_download" in self.reconcilers,
            "local_process": "local_process" in self.reconcilers,
            "docker_service": "docker_service" in self.reconcilers,
            "dependency_install": "dependency_install" in self.reconcilers,
        }
        return caps

    def build_operation(self, state: dict, stage: str) -> dict:
        """Build an operation record for a side-effect stage.

        Generates a stable operation ID from deterministic inputs.
        Never includes: token/secret values, PIDs, container IDs,
        process start time, current time, replan step index.
        """
        resource_type = self._resolve_resource_type(state, stage)
        normalized_input = self._build_normalized_input(state, stage)
        resource_identity = self._build_resource_identity(state, stage)

        operation_id = compute_operation_id(
            task_id=state.get("task_id", ""),
            stage=stage,
            action=self._stage_action(stage),
            normalized_input=normalized_input,
            resource_identity=resource_identity,
        )

        return {
            "operation_id": operation_id,
            "task_id": state.get("task_id", ""),
            "stage": stage,
            "action": self._stage_action(stage),
            "resource_type": resource_type,
            "normalized_input": normalized_input,
            "normalized_input_hash": canonical_json(normalized_input),
            "resource_identity": resource_identity,
            "status": "planned",
        }

    def prepare_or_reconcile(self, state: dict, stage: str) -> RecoveryDecision:
        """Prepare a new operation or reconcile an existing one.

        Decision rules:
        - New operation/planned -> transition to running, execute stage
        - committed -> hydrate result, skip execution (reuse)
        - running after process restart -> transition to unknown, reconcile
        - reconcile reuse -> commit/reuse, hydrate result, skip execution
        - reconcile continue -> continue resumable action
        - reconcile retry -> safe retry
        - cleanup_then_retry -> approval, interrupt
        - conflict -> stop
        - manual/unknown/no reconciler -> approval or stop
        """
        run_dir = Path(state["run_dir"])
        journal = OperationJournal(run_dir)
        service = self._create_service(journal)

        operation = self.build_operation(state, stage)
        operation_id = operation["operation_id"]

        # Check if operation already exists in journal
        existing = journal.load(operation_id)

        if existing is None:
            # New operation -- prepare and transition to running
            record = journal.create(operation)
            return RecoveryDecision(
                decision="execute",
                operation=record,
                reconcile_result={},
                hydrated_stage_result={},
            )

        status = existing.get("status", "planned")

        if status == "committed":
            # Already committed -- hydrate result, skip execution
            hydrated = self._hydrate_committed_result(existing)
            return RecoveryDecision(
                decision="reuse",
                operation=existing,
                reconcile_result={"decision": "reuse"},
                hydrated_stage_result=hydrated,
            )

        if status == "running":
            # Running after potential crash -- reconcile
            journal.transition(operation_id, "unknown")
            reconciler = self._get_reconciler(operation.get("resource_type", ""))
            if reconciler:
                try:
                    reconcile_result = service.reconcile(existing)
                    decision = reconcile_result.get("decision", "manual")
                except Exception:
                    reconcile_result = {"decision": "manual", "reason": "reconcile_failed"}
                    decision = "manual"
            else:
                reconcile_result = {"decision": "manual", "reason": "no_reconciler"}
                decision = "manual"

            if decision == "reuse":
                # Reconciler says it's done -- commit and hydrate
                hydrated = self._hydrate_committed_result(existing)
                return RecoveryDecision(
                    decision="reuse",
                    operation=existing,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result=hydrated,
                )
            elif decision in ("continue", "retry"):
                return RecoveryDecision(
                    decision=decision,
                    operation=existing,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                )
            elif decision == "cleanup_then_retry":
                return RecoveryDecision(
                    decision="approval",
                    operation=existing,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                    stop_reason="cleanup_required",
                )
            elif decision == "conflict":
                return RecoveryDecision(
                    decision="stop",
                    operation=existing,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                    stop_reason="recovery_conflict",
                )
            else:
                # manual, unknown
                return RecoveryDecision(
                    decision="approval",
                    operation=existing,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                    stop_reason="manual_recovery_required",
                )

        if status in ("failed", "unknown"):
            # Failed or unknown -- attempt retry if reconciler says safe
            return RecoveryDecision(
                decision="retry",
                operation=existing,
                reconcile_result={"decision": "retry"},
                hydrated_stage_result={},
            )

        # Default: stop for unknown states
        return RecoveryDecision(
            decision="stop",
            operation=existing or operation,
            reconcile_result={},
            hydrated_stage_result={},
            stop_reason="unexpected_operation_status:%s" % status,
        )

    def commit(self, state: dict, stage: str, executed_result: dict) -> dict:
        """Commit a successfully executed side-effect operation.

        Writes the stage result artifact and records it in the journal.
        """
        run_dir = Path(state["run_dir"])
        journal = OperationJournal(run_dir)

        operation = self.build_operation(state, stage)
        operation_id = operation["operation_id"]

        # Write stage result artifact for hydration
        artifacts_dir = run_dir / "operations"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        from auto_harness.models.base import write_json
        artifact_path = artifacts_dir / ("%s_result.json" % operation_id)
        write_json(artifact_path, executed_result)

        # Commit in journal
        record = journal.load(operation_id)
        if record:
            journal.transition(operation_id, "committed", result_artifacts=[str(artifact_path)])

        return {"status": "committed", "operation_id": operation_id, "artifact_path": str(artifact_path)}

    def fail(self, state: dict, stage: str, error: str) -> dict:
        """Mark a side-effect operation as failed."""
        run_dir = Path(state["run_dir"])
        journal = OperationJournal(run_dir)

        operation = self.build_operation(state, stage)
        operation_id = operation["operation_id"]

        record = journal.load(operation_id)
        if record:
            journal.transition(operation_id, "failed", error=error[:2000])

        return {"status": "failed", "operation_id": operation_id}

    def hydrate_committed_result(self, operation: dict) -> dict:
        """Hydrate a committed operation's result from artifact.

        Used when resuming after a crash where journal says committed
        but graph checkpoint hasn't been updated.
        """
        return self._hydrate_committed_result(operation)

    # --- Private helpers ---

    def _resolve_resource_type(self, state: dict, stage: str) -> str:
        """Resolve the actual resource type for a stage.

        For 'runner', checks if Docker backend is used.
        """
        base_type = self.SIDE_EFFECT_STAGE_TYPES.get(stage, "unknown")
        if stage == "runner":
            runtime_policy = state.get("runtime_policy", {})
            if runtime_policy.get("execution_backend") == "docker":
                return "docker_service"
            return "local_process"
        return base_type

    def _stage_action(self, stage: str) -> str:
        """Map stage to action name."""
        actions = {
            "env_deploy": "install_dependencies",
            "model_prepare": "prepare_model_assets",
            "runner": "start_service",
        }
        return actions.get(stage, "unknown")

    def _build_normalized_input(self, state: dict, stage: str) -> dict:
        """Build deterministic normalized input for operation ID.

        Never includes: tokens, PIDs, container IDs, timestamps.
        """
        if stage == "env_deploy":
            compiled = state.get("compiled_analysis", {})
            return {
                "backend": state.get("runtime_policy", {}).get("env_backend", "auto"),
                "install_plan": [str(cmd)[:200] for cmd in compiled.get("install_plan", [])[:20]],
            }
        elif stage == "model_prepare":
            compiled = state.get("compiled_analysis", {})
            assets = compiled.get("model_assets", {})
            return {
                "source": assets.get("source", "unknown"),
                "repo_id": assets.get("repo_id", ""),
                "revision": assets.get("revision", "main"),
            }
        elif stage == "runner":
            compiled = state.get("compiled_analysis", {})
            candidates = compiled.get("run_candidates", [])
            selected = ""
            for c in candidates:
                if isinstance(c, dict) and c.get("selected"):
                    selected = c.get("id", "")
                    break
            return {
                "argv": [str(a)[:100] for a in compiled.get("run_command", [])[:10]],
                "env_names": list(compiled.get("env_names", [])),
                "selected_candidate_id": selected,
            }
        return {}

    def _build_resource_identity(self, state: dict, stage: str) -> dict:
        """Build resource identity for operation ID.

        Stable identifiers that won't change across replans.
        """
        if stage == "env_deploy":
            return {
                "environment_prefix": str(Path(state["run_dir"]) / "workspace"),
                "repo_path": state.get("repo_dir", ""),
            }
        elif stage == "model_prepare":
            compiled = state.get("compiled_analysis", {})
            assets = compiled.get("model_assets", {})
            return {
                "target_path": str(Path(state["run_dir"]) / "workspace" / "models"),
                "cache_key": assets.get("cache_key", ""),
            }
        elif stage == "runner":
            return {
                "repo_path": state.get("repo_dir", ""),
                "expected_port": str(state.get("compiled_analysis", {}).get("expected_port", "7860")),
            }
        return {}

    def _get_reconciler(self, resource_type: str):
        """Get reconciler for a resource type."""
        return self.reconcilers.get(resource_type)

    def _create_service(self, journal):
        """Create RecoveryService with available reconcilers."""
        from auto_harness.recovery.service import RecoveryService
        return RecoveryService(journal, self.reconcilers)

    def _hydrate_committed_result(self, operation: dict) -> dict:
        """Hydrate stage result from committed operation artifact."""
        artifacts = operation.get("result_artifacts", [])
        if artifacts:
            artifact_path = artifacts[0] if isinstance(artifacts, list) else artifacts
            try:
                from auto_harness.models.base import read_json
                result = read_json(Path(artifact_path))
                return result
            except (OSError, ValueError):
                pass
        return {}
