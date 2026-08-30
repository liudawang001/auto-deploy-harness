"""Model runtime controller (Document B Phase B8).

Deterministic orchestration of the runtime phase after ``model_prepare``:

    Preparation Artifact Gate
      -> vLLM Adapter
      -> Runtime Policy
      -> managed container start
      -> startup readiness
      -> non-stream + SSE trace inference

Every dependency (gate, adapter, policy, runner, readiness, verifier, command
runner, urlopen) is injectable so the whole chain runs offline in tests. The
controller never lets the LLM alter a hard security field — the plan is
produced only by the adapter and authorized only by the policy.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auto_harness.model_runtime.evidence import ModelRuntimeEvidenceWriter
from auto_harness.model_runtime.local_adapter import LocalVllmRuntimeAdapter
from auto_harness.model_runtime.preparation_gate import (
    PreparationArtifactGate,
    PreparationBundle,
)
from auto_harness.model_runtime.policy import ModelRuntimePolicy
from auto_harness.model_runtime.readiness import ModelRuntimeReadiness
from auto_harness.model_runtime.schemas import (
    InferenceRuntimePlan,
    ModelInferenceEvidence,
    ModelRuntimeStartupEvidence,
)
from auto_harness.model_runtime.vllm_adapter import VllmRuntimeAdapter


def stable_operation_id(task_id: str, plan_hash: str) -> str:
    """Stable runner operation id (no container IDs, PIDs, or timestamps)."""
    digest = hashlib.sha256(("%s:%s" % (task_id, plan_hash)).encode("utf-8")).hexdigest()
    return "run-%s" % digest[:20]


def runtime_labels(task_id: str, operation_id: str, plan) -> Dict[str, str]:
    return {
        "auto-harness.task-id": str(task_id),
        "auto-harness.operation-id": str(operation_id),
        "auto-harness.plan-hash": plan.plan_hash,
        "auto-harness.model-hash": plan.resolved_model_hash,
    }


def runtime_resource_identity(plan: InferenceRuntimePlan) -> Dict[str, Any]:
    """Resource identity for the DockerReconciler (matches graph_adapter)."""
    gpu_indexes = list(plan.gpu_indexes or [])
    return {
        "container_name": plan.container_name,
        "plan_hash": plan.plan_hash,
        "runtime_plan_hash": plan.plan_hash,
        "image": plan.image,
        "image_digest": plan.image_digest,
        "model_hash": plan.resolved_model_hash,
        "model_identity": plan.model_identity,
        "model_host_path": plan.model_host_path,
        "gpu_indexes": gpu_indexes,
        "ports": [int(plan.expected_port)] if plan.expected_port else [],
        "network": "bridge",
        "gpus": ("device=%d" % gpu_indexes[0]) if gpu_indexes else "none",
        "shm_size": "8g",
        "memory": "32g",
        "cpus": 8.0,
        "pids_limit": 1024,
        "read_only_rootfs": True,
        "user": "",
    }


@dataclass
class RuntimePhaseResult:
    status: str
    bundle: Optional[PreparationBundle] = None
    plan: Optional[InferenceRuntimePlan] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    startup_evidence: Optional[ModelRuntimeStartupEvidence] = None
    container_id: str = ""
    errors: List[str] = field(default_factory=list)


class ModelRuntimeController:
    """Deterministic orchestration of the managed inference runtime phase."""

    def __init__(
        self,
        *,
        adapter=None,
        policy=None,
        runner=None,
        readiness=None,
        verifier=None,
        local_adapter=None,
    ) -> None:
        self.adapter = adapter or VllmRuntimeAdapter()
        self.local_adapter = local_adapter or LocalVllmRuntimeAdapter()
        self.policy = policy or ModelRuntimePolicy()
        self.runner = runner
        self.readiness = readiness
        self.verifier = verifier

    def run_runtime_phase(
        self,
        *,
        run_dir,
        task_id,
        config,
        cache_root=None,
        host_facts_provider=None,
        execute: bool = False,
        allow_start: bool = False,
        command_runner=None,
        urlopen=None,
        gpu_indexes=None,
        operation_id: str = "",
        reconciler=None,
        process_launcher=None,
    ) -> RuntimePhaseResult:
        gate = PreparationArtifactGate(
            run_dir, cache_root=cache_root, host_facts_provider=host_facts_provider,
        )
        bundle = gate.validate()
        if not bundle.ok:
            return RuntimePhaseResult(status="blocked", bundle=bundle, errors=bundle.errors)

        mode = str(getattr(config, "model_runtime_mode", "managed_vllm") or "managed_vllm")
        adapter = self.local_adapter if mode == "local_vllm" else self.adapter
        execution_backend = "local" if mode == "local_vllm" else "docker"
        try:
            plan = adapter.build(bundle, config, task_id=task_id, gpu_indexes=gpu_indexes)
        except ValueError as exc:
            return RuntimePhaseResult(status="blocked", bundle=bundle, errors=[str(exc)])

        decision = self.policy.authorize(
            plan, bundle, config,
            execute=execute, allow_start=allow_start, execution_backend=execution_backend,
            require_start_auth=bool(execute and allow_start),
        )
        if not decision["allowed"]:
            return RuntimePhaseResult(status="blocked", plan=plan, bundle=bundle, policy=decision, errors=decision["reasons"])

        writer = ModelRuntimeEvidenceWriter(run_dir)
        writer.write_runtime_plan(plan)

        if not (execute and allow_start):
            # Dry-run: plan is generated and authorized, no container started.
            return RuntimePhaseResult(status="passed", plan=plan, bundle=bundle, policy=decision)

        from auto_harness.modules.runner import RunnerModule

        operation_id = operation_id or stable_operation_id(task_id, plan.plan_hash)
        runner = self.runner or RunnerModule()

        local_mode = mode == "local_vllm"

        # Reconcile any pre-existing managed container before starting a new one.
        container_id = ""
        if reconciler is not None and not local_mode:
            op = {
                "operation_id": operation_id,
                "task_id": task_id,
                "stage": "runner",
                "action": "start_service",
                "resource_type": "docker_service",
                "resource_identity": runtime_resource_identity(plan),
            }
            decision = reconciler.reconcile(op)
            kind = decision.get("decision", "manual")
            if kind == "reuse":
                container_id = (decision.get("observed_state") or {}).get("id", "")
            elif kind == "conflict":
                return RuntimePhaseResult(
                    status="failed", plan=plan, bundle=bundle, policy=decision,
                    errors=["container conflict: %s" % decision.get("reason", "")],
                )
            elif kind in ("manual", "cleanup_then_retry"):
                return RuntimePhaseResult(
                    status="failed", plan=plan, bundle=bundle, policy=decision,
                    errors=["%s: %s" % (kind, decision.get("reason", ""))],
                )
            # retry / continue fall through to a fresh docker run

        if not container_id:
            result = runner.run_model_runtime(
                run_dir=run_dir,
                task_id=task_id,
                runtime_plan=plan,
                bundle=bundle,
                execute=True,
                command_runner=command_runner,
                operation_id=operation_id,
                process_launcher=process_launcher,
            )
            if result.status != "passed":
                return RuntimePhaseResult(
                    status="failed", plan=plan, bundle=bundle, policy=decision,
                    errors=[result.error or "container_start_failed"],
                )
            container_id = str(result.data.get("container_id", "") or result.data.get("pid", ""))

        readiness = self.readiness or ModelRuntimeReadiness(
            command_runner=command_runner, urlopen=urlopen,
        )
        startup = readiness.wait(
            runtime_plan=plan,
            task_id=task_id,
            operation_id=operation_id,
            container_id=container_id,
            labels=runtime_labels(task_id, operation_id, plan),
            local_log_path=str(result.data.get("log_path", "")) if local_mode else "",
        )
        writer.write_startup_evidence(startup)
        if startup.status != "ready":
            return RuntimePhaseResult(
                status="failed", plan=plan, bundle=bundle, policy=decision,
                startup_evidence=startup, container_id=container_id,
                errors=[startup.failure_reason],
            )
        return RuntimePhaseResult(
            status="passed", plan=plan, bundle=bundle, policy=decision,
            startup_evidence=startup, container_id=container_id,
        )

    def verify_phase(
        self,
        *,
        run_dir,
        task_id,
        runtime_plan,
        startup_evidence,
        urlopen=None,
        operation_id: str = "",
    ):
        from auto_harness.modules.verify import VerifyModule

        verifier = self.verifier or VerifyModule(urlopen=urlopen)
        return verifier.verify_model_runtime(
            run_dir,
            runtime_plan,
            startup_evidence,
            task_id=task_id,
            operation_id=operation_id or stable_operation_id(task_id, runtime_plan.plan_hash),
        )
