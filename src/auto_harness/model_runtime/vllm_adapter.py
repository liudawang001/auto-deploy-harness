"""Deterministic vLLM Runtime Adapter (Document B Phase B1).

Turns a validated ``PreparationBundle`` + model runtime config into a single
``InferenceRuntimePlan``. The adapter is a pure plan generator:

- It never executes Docker or touches the daemon.
- It only accepts a validated bundle + config + task id + allocated host port.
- The vLLM command argument order is fixed; every parameter has exactly one
  source (the Resource Decision or a fixed constant).
- The plan hash binds the model/file/cache/resource hashes, image digest,
  command, port, GPU index and security profile.

The adapter is the ONLY producer of the runtime plan; the LLM cannot assemble
a vLLM command.
"""
from auto_harness.model_runtime.compatibility import runtime_blockers
from auto_harness.model_runtime.preparation_gate import PreparationBundle
from auto_harness.model_runtime.schemas import InferenceRuntimePlan
from auto_harness.utils.files import short_hash

# Fixed runtime boundaries.
CONTAINER_MODEL_PATH = "/models/current"
CONTAINER_PORT = 8000
SECURITY_PROFILE = "model_runtime_v1"

# Parameters that must never appear in a generated plan. The first-phase
# boundary is single-GPU, no remote code, no LoRA/plugins, no Ray/distributed.
_FORBIDDEN_FLAGS = (
    "--trust-remote-code",
    "--download-dir",
    "--enable-lora",
    "--lora-modules",
    "--pipeline-parallel-size",
    "--worker-use-ray",
    "--distributed-executor-backend",
    "--cpu-offload-gb",
)

# vLLM command argv[0] (resolved inside the official vllm image).
_VLLM_ENTRYPOINT = ["python3", "-m", "vllm.entrypoints.openai.api_server"]


class VllmRuntimeAdapter:
    """Deterministic vLLM command and runtime plan generator."""

    def __init__(self, container_port: int = CONTAINER_PORT) -> None:
        self.container_port = int(container_port)

    def build(
        self,
        bundle: PreparationBundle,
        config,
        *,
        task_id: str = "",
        host_port=None,
        gpu_indexes=None,
        require_image_digest: bool = True,
    ) -> InferenceRuntimePlan:
        if not isinstance(bundle, PreparationBundle) or not bundle.ok:
            raise ValueError("vLLM adapter requires a ready PreparationBundle (got %s)" % getattr(bundle, "status", "unknown"))

        spec = bundle.spec
        decision = bundle.decision

        host_path = str(bundle.model_host_path or "").strip()
        if not host_path:
            raise ValueError("empty model host path")
        if "://" in host_path:
            raise ValueError("model host path must be a local path, not a remote URL")

        blockers = runtime_blockers(spec)
        if blockers:
            raise ValueError("model is blocked by runtime compatibility: %s" % "; ".join(blockers))

        image = self._image(config)
        image_digest = self._image_digest(image)
        if require_image_digest and not image_digest:
            raise ValueError("model_runtime_image must be pinned to an immutable digest (<tag>@sha256:<digest>)")

        served_model_name = self._served_name(spec.repo_id)
        host = int(host_port) if host_port is not None else int(getattr(config, "model_runtime_port", 8000) or 8000)
        command = self.command_for(bundle)

        plan = InferenceRuntimePlan(
            runtime="vllm",
            deployment_mode="managed_vllm",
            image=image,
            image_digest=image_digest,
            model_identity=spec.model_identity,
            resolved_model_hash=bundle.resolved_model_hash,
            file_plan_hash=bundle.file_plan_hash,
            cache_marker_hash=bundle.cache_marker_hash,
            resource_decision_hash=bundle.resource_decision_hash,
            model_host_path=host_path,
            model_container_path=CONTAINER_MODEL_PATH,
            served_model_name=served_model_name,
            command=command,
            expected_host="127.0.0.1",
            expected_port=host,
            startup_timeout_seconds=int(getattr(config, "model_runtime_startup_timeout_seconds", 900) or 900),
            request_timeout_seconds=int(getattr(config, "model_runtime_request_timeout_seconds", 120) or 120),
            health_path="/v1/models",
            container_name=self._container_name(task_id),
            gpu_indexes=list(gpu_indexes) if gpu_indexes is not None else list(decision.gpu_indexes or [0]),
            security_profile=SECURITY_PROFILE,
        )
        plan.plan_hash = plan.compute_plan_hash()
        return plan

    def command_for(self, bundle: PreparationBundle):
        """Build the fixed vLLM command for a validated bundle.

        Exposed so the runtime policy can re-derive the exact command and
        reject any plan whose command does not match.
        """
        spec = bundle.spec
        decision = bundle.decision
        served_model_name = self._served_name(spec.repo_id)
        dtype = decision.selected_dtype or "float16"
        command = [
            *_VLLM_ENTRYPOINT,
            "--model", CONTAINER_MODEL_PATH,
            "--served-model-name", served_model_name,
            "--host", "0.0.0.0",
            "--port", str(self.container_port),
            "--dtype", dtype,
            "--max-model-len", str(int(decision.max_model_len or 1)),
            "--gpu-memory-utilization", str(float(decision.gpu_memory_utilization or 0.9)),
            "--max-num-seqs", str(int(decision.max_num_seqs or 1)),
            "--tensor-parallel-size", str(int(decision.tensor_parallel_size or 1)),
        ]
        self._assert_no_forbidden_flags(command)
        return command

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _image(config) -> str:
        return str(getattr(config, "model_runtime_image", "") or "").strip()

    @staticmethod
    def _image_digest(image: str) -> str:
        if "@" not in image:
            return ""
        digest = image.rsplit("@", 1)[1]
        return digest if digest.startswith("sha256:") else ""

    @staticmethod
    def _served_name(repo_id: str) -> str:
        name = str(repo_id or "").strip()
        if not name or "/" not in name:
            raise ValueError("invalid repo_id for served-model-name: %r" % repo_id)
        return name

    @staticmethod
    def _container_name(task_id: str) -> str:
        return "auto-harness-%s-vllm" % short_hash(str(task_id or "task"), 8)

    @staticmethod
    def _assert_no_forbidden_flags(command):
        for flag in _FORBIDDEN_FLAGS:
            if flag in command:
                raise ValueError("forbidden vLLM flag in generated command: %s" % flag)
