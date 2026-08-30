"""Deterministic local vLLM runtime adapter (docker-less GPU hosts).

Some GPU hosts (rented container instances, bare workstations) cannot run
Docker. ``local_vllm`` mode serves the same deterministic contract as the
managed container mode with one honest difference: there is no container
isolation. The plan therefore carries a distinct security profile
(``model_runtime_local_v1``) so audits can tell the two execution stories
apart, and the runtime policy only authorizes it when the operator has
explicitly selected ``model_runtime_mode=local_vllm``.

Like the managed adapter, this class is a pure plan generator:
- it never executes anything;
- the command is re-derivable via ``command_for`` and the plan hash binds
  every parameter source.
"""
import sys

from auto_harness.model_runtime.preparation_gate import PreparationBundle
from auto_harness.model_runtime.schemas import InferenceRuntimePlan
from auto_harness.utils.files import short_hash

LOCAL_SECURITY_PROFILE = "model_runtime_local_v1"

_FORBIDDEN_FLAGS = (
    "--trust-remote-code",
    "--cpu-offload-gb",
    "--enable-lora",
    "--lora-modules",
    "--pipeline-parallel-size",
    "--worker-use-ray",
    "--distributed-executor-backend",
)


class LocalVllmRuntimeAdapter:
    """Deterministic host-process vLLM command and runtime plan generator."""

    def __init__(self, python_executable: str = "") -> None:
        self._python_executable = str(python_executable or "").strip()

    def build(
        self,
        bundle: PreparationBundle,
        config,
        *,
        task_id: str = "",
        host_port=None,
        gpu_indexes=None,
    ) -> InferenceRuntimePlan:
        if not isinstance(bundle, PreparationBundle) or not bundle.ok:
            raise ValueError("local adapter requires a ready PreparationBundle (got %s)" % getattr(bundle, "status", "unknown"))

        spec = bundle.spec
        decision = bundle.decision

        host_path = str(bundle.model_host_path or "").strip()
        if not host_path:
            raise ValueError("empty model host path")
        if "://" in host_path:
            raise ValueError("model host path must be a local path, not a remote URL")

        from auto_harness.model_runtime.compatibility import runtime_blockers
        blockers = runtime_blockers(spec)
        if blockers:
            raise ValueError("model is blocked by runtime compatibility: %s" % "; ".join(blockers))

        host = int(host_port) if host_port is not None else int(getattr(config, "model_runtime_port", 8000) or 8000)
        command = self.command_for(bundle, config, host_port=host)

        plan = InferenceRuntimePlan(
            runtime="vllm",
            deployment_mode="local_vllm",
            image="",
            image_digest="",
            model_identity=spec.model_identity,
            resolved_model_hash=bundle.resolved_model_hash,
            file_plan_hash=bundle.file_plan_hash,
            cache_marker_hash=bundle.cache_marker_hash,
            resource_decision_hash=bundle.resource_decision_hash,
            model_host_path=host_path,
            model_container_path="",
            served_model_name=self._served_name(spec.repo_id),
            command=command,
            expected_host="127.0.0.1",
            expected_port=host,
            startup_timeout_seconds=int(getattr(config, "model_runtime_startup_timeout_seconds", 900) or 900),
            request_timeout_seconds=int(getattr(config, "model_runtime_request_timeout_seconds", 120) or 120),
            health_path="/v1/models",
            container_name="",
            gpu_indexes=list(gpu_indexes) if gpu_indexes is not None else list(decision.gpu_indexes or [0]),
            security_profile=LOCAL_SECURITY_PROFILE,
        )
        plan.plan_hash = plan.compute_plan_hash()
        return plan

    def command_for(self, bundle: PreparationBundle, config, host_port: int = None):
        """Build the fixed host-process vLLM command for a validated bundle."""
        spec = bundle.spec
        decision = bundle.decision
        host_path = str(bundle.model_host_path or "").strip()
        if not host_path:
            raise ValueError("empty model host path")
        port = int(host_port) if host_port is not None else int(getattr(config, "model_runtime_port", 8000) or 8000)
        dtype = decision.selected_dtype or "float16"
        command = [
            self._python_executable or sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", host_path,
            "--served-model-name", self._served_name(spec.repo_id),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--dtype", dtype,
            "--max-model-len", str(int(decision.max_model_len or 1)),
            "--gpu-memory-utilization", str(float(decision.gpu_memory_utilization or 0.9)),
            "--max-num-seqs", str(int(decision.max_num_seqs or 1)),
            "--tensor-parallel-size", str(int(decision.tensor_parallel_size or 1)),
        ]
        for flag in _FORBIDDEN_FLAGS:
            if flag in command:
                raise ValueError("forbidden vLLM flag in generated command: %s" % flag)
        return command

    @staticmethod
    def _served_name(repo_id: str) -> str:
        name = str(repo_id or "").strip()
        if not name or "/" not in name:
            raise ValueError("invalid repo_id for served-model-name: %r" % repo_id)
        return name

    @staticmethod
    def local_operation_tag(task_id: str) -> str:
        return "local-%s" % short_hash(str(task_id or "task"), 8)
