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
import hashlib
from pathlib import Path
import sys
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
            "idempotency_key": operation_id,
            "task_id": state.get("task_id", ""),
            "run_dir": state.get("run_dir", ""),
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
        - New operation -> begin (atomically set running), execute stage
        - planned (no observed resource) -> begin -> execute
        - planned (with observed resource) -> approval/manual
        - committed -> hydrate result, skip execution (reuse)
        - running after process restart -> transition to unknown, reconcile
        - reconcile reuse -> apply_decision -> commit/reuse, hydrate result
        - reconcile continue/retry -> apply_decision -> continue/retry
        - cleanup_then_retry -> apply_decision -> manual -> approval
        - conflict -> apply_decision -> conflict -> stop
        - failed -> approval (fail-closed, no auto-retry)
        - unknown -> reconcile (must not blindly retry)
        - manual/unknown with no reconciler -> approval or stop
        """
        run_dir = Path(state["run_dir"])
        journal = OperationJournal(run_dir)
        service = self._create_service(journal)

        operation = self.build_operation(state, stage)
        operation_id = operation["operation_id"]

        # Check if operation already exists in journal
        existing = journal.load(operation_id)

        if existing is None:
            # New operation -- atomically begin as running before side effect
            record = journal.begin(operation)
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

        if status == "planned":
            # Planned but not yet executing.
            # If observed resource already exists, require manual review.
            if existing.get("observed_resource"):
                existing = journal.transition(
                    operation_id,
                    "manual",
                    reconcile_result={
                        "decision": "manual",
                        "reason": "planned_with_observed_resource",
                    },
                )
                return RecoveryDecision(
                    decision="approval",
                    operation=existing,
                    reconcile_result={"decision": "manual", "reason": "planned_with_observed_resource"},
                    hydrated_stage_result={},
                    stop_reason="manual_recovery_required",
                )
            # No observed resource -> safe to begin and execute
            started = journal.begin(operation)
            return RecoveryDecision(
                decision="execute",
                operation=started,
                reconcile_result={},
                hydrated_stage_result={},
            )

        if status == "running":
            # Running after potential crash -- recover and reconcile
            existing = journal.recover_running(operation_id)
            status = existing["status"]

        if status == "unknown":
            # Unknown status requires reconciliation -- never blindly retry
            reconciler = self._get_reconciler(operation.get("resource_type", ""))
            if reconciler:
                try:
                    reconcile_result = service.reconcile(existing)
                except Exception:
                    reconcile_result = {"decision": "manual", "reason": "reconcile_failed"}
            else:
                reconcile_result = {"decision": "manual", "reason": "no_reconciler"}

            decision = reconcile_result.get("decision", "manual")

            # Apply the decision to persist state transition
            updated = service.apply_decision(existing, reconcile_result)

            if decision == "reuse":
                hydrated = self._hydrate_committed_result(updated)
                return RecoveryDecision(
                    decision="reuse",
                    operation=updated,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result=hydrated,
                )

            if decision in ("continue", "retry"):
                return RecoveryDecision(
                    decision=decision,
                    operation=updated,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                )

            if decision == "cleanup_then_retry":
                return RecoveryDecision(
                    decision="approval",
                    operation=updated,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                    stop_reason="cleanup_required",
                )

            if decision == "conflict":
                return RecoveryDecision(
                    decision="stop",
                    operation=updated,
                    reconcile_result=reconcile_result,
                    hydrated_stage_result={},
                    stop_reason="recovery_conflict",
                )

            # manual or any other decision
            return RecoveryDecision(
                decision="approval",
                operation=updated,
                reconcile_result=reconcile_result,
                hydrated_stage_result={},
                stop_reason="manual_recovery_required",
            )

        if status == "failed":
            repair_apply = state.get("repair_apply_result") or {}
            authorized_retry = (
                bool(state.get("repair_resume_executed"))
                and int(state.get("repair_count", 0)) > 0
                and state.get("failed_stage") == stage
                and int(repair_apply.get("effective_action_count", 0)) > 0
                and not existing.get("observed_resource")
            )
            if authorized_retry:
                retryable = journal.transition(
                    operation_id,
                    "retryable",
                    reconcile_result={
                        "decision": "retry",
                        "reason": "effective_repair_authorized_retry",
                    },
                )
                return RecoveryDecision(
                    decision="retry",
                    operation=retryable,
                    reconcile_result=retryable.get("reconcile_result", {}),
                    hydrated_stage_result={},
                )
            # Fail-closed: failed operations require operator decision.
            # Auto-retry is not allowed without explicit retry policy.
            return RecoveryDecision(
                decision="approval",
                operation=existing,
                reconcile_result={
                    "decision": "manual",
                    "reason": "failed_operation_requires_operator_decision",
                },
                hydrated_stage_result={},
                stop_reason="failed_operation_requires_operator_decision",
            )

        if status == "retryable":
            # Safe to re-enter running via begin, then execute
            started = journal.begin(operation)
            return RecoveryDecision(
                decision="execute",
                operation=started,
                reconcile_result={},
                hydrated_stage_result={},
            )

        # Default: stop for unexpected states
        return RecoveryDecision(
            decision="stop",
            operation=existing or operation,
            reconcile_result={},
            hydrated_stage_result={},
            stop_reason="unexpected_operation_status:%s" % status,
        )

    def commit(
        self,
        state: dict,
        stage: str,
        executed_result: dict,
        artifact_path: Optional[Path] = None,
    ) -> dict:
        """Commit a successfully executed side-effect operation.

        Writes the stage result artifact and records it in the journal.
        """
        run_dir = Path(state["run_dir"])
        journal = OperationJournal(run_dir)

        operation = self.build_operation(state, stage)
        operation_id = operation["operation_id"]
        artifact_path = Path(artifact_path) if artifact_path else self.persist_result(
            state, stage, executed_result
        )

        # Commit in journal
        record = journal.load(operation_id)
        if record:
            journal.transition(operation_id, "committed", result_artifacts=[str(artifact_path)])

        return {"status": "committed", "operation_id": operation_id, "artifact_path": str(artifact_path)}

    def persist_result(self, state: dict, stage: str, executed_result: dict) -> Path:
        """Persist a deterministic hydration artifact before journal commit."""
        run_dir = Path(state["run_dir"])
        operation_id = self.build_operation(state, stage)["operation_id"]
        artifacts_dir = run_dir / "operations"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        import json
        from auto_harness.models.base import to_plain
        from auto_harness.utils.atomic import atomic_write_text
        artifact_path = artifacts_dir / ("%s_result.json" % operation_id)
        atomic_write_text(
            artifact_path,
            json.dumps(
                to_plain(executed_result),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ) + "\n",
        )
        return artifact_path

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
            solution = self._environment_solution(state)
            conda = solution.get("conda") or {}
            spec = conda.get("spec") or {}
            install_plan = self._environment_install_plan(state)
            normalized = {
                "backend": solution.get("backend") or state.get("runtime_policy", {}).get("env_backend", "auto"),
                "action": conda.get("action") or (solution.get("compatibility_decision") or {}).get("action", ""),
                "package_specs": list(spec.get("conda_dependencies") or []) + list(spec.get("pip_dependencies") or []),
                "python_version": str(spec.get("python") or solution.get("python") or ""),
                "spec_hash": str(spec.get("spec_hash") or (solution.get("compatibility_decision") or {}).get("spec_hash", "")),
            }
            if solution.get("backend", "venv") == "venv":
                # Bind venv recovery identity to the accepted command plan
                # without persisting command arguments that could contain
                # secrets. Conda already has a canonical spec hash shared
                # with EnvDeployModule.
                normalized["install_plan_hash"] = hashlib.sha256(
                    canonical_json(install_plan).encode("utf-8")
                ).hexdigest()
            return normalized
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
            solution = self._environment_solution(state)
            conda = solution.get("conda") or {}
            decision = solution.get("compatibility_decision") or {}
            backend = solution.get("backend", "venv")
            if backend == "venv":
                environment_path = Path(state.get("repo_dir", "")) / ".venv"
                python_version = "%s.%s" % (
                    sys.version_info.major, sys.version_info.minor,
                )
            else:
                environment_path = (
                    decision.get("target_prefix")
                    or conda.get("environment_prefix")
                    or solution.get("environment_prefix")
                    or (Path(state["run_dir"]) / "workspace")
                )
                python_version = str(
                    decision.get("python") or solution.get("python") or ""
                )
            return {
                "environment_path": str(environment_path),
                "backend": backend,
                "tool": decision.get("tool") or conda.get("tool", ""),
                "python_version": python_version,
                "project_id": decision.get("project_id", ""),
                "repo_fingerprint": decision.get("repo_fingerprint", ""),
                "spec_hash": decision.get("spec_hash", ""),
                "gpu_required": bool(solution.get("gpu_required")),
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

    @staticmethod
    def _environment_solution(state: dict) -> dict:
        result = (state.get("stage_results") or {}).get("env_solve") or {}
        data = result.get("data") or {}
        analysis = data.get("analysis") or {}
        solution = analysis.get("env_solution") or {}
        if solution:
            return solution
        return (state.get("compiled_analysis") or {}).get("env_solution") or {}

    @staticmethod
    def _environment_install_plan(state: dict) -> list:
        result = (state.get("stage_results") or {}).get("env_solve") or {}
        data = result.get("data") or {}
        analysis = data.get("analysis") or {}
        plan = analysis.get("install_plan") or data.get("install_plan") or []
        return [list(command) for command in plan if isinstance(command, list)]

    def _get_reconciler(self, resource_type: str):
        """Get reconciler for a resource type."""
        return self.reconcilers.get(resource_type)

    def _create_service(self, journal):
        """Create RecoveryService with available reconcilers."""
        from auto_harness.recovery.service import RecoveryService
        return RecoveryService(journal, self.reconcilers)

    def _hydrate_committed_result(self, operation: dict) -> dict:
        """Hydrate stage result from committed operation artifact."""
        artifacts = list(operation.get("result_artifacts", []) or [])
        if not artifacts:
            operation_id = operation.get("operation_id", "")
            task_run_dir = operation.get("run_dir", "")
            if task_run_dir and operation_id:
                artifacts.append(
                    str(Path(task_run_dir) / "operations" / ("%s_result.json" % operation_id))
                )
        if artifacts:
            artifact_path = artifacts[0] if isinstance(artifacts, list) else artifacts
            try:
                from auto_harness.models.base import read_json
                result = read_json(Path(artifact_path))
                return result
            except (OSError, ValueError):
                pass
        return {}
