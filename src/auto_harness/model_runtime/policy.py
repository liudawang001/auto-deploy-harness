"""Managed inference runtime policy (Document B Phase B2).

``ModelRuntimePolicy`` is the single deterministic gate for a runtime plan
produced by the built-in vLLM adapter. A plan is auto-allowed only when every
hard condition holds; otherwise it is denied with a precise reason code.

The policy is plan-first: it never accepts a free-form LLM command. The only
valid command is the one the adapter re-derives from the validated bundle.
"""
import json
from typing import List, Optional

from auto_harness.model_runtime.local_adapter import (
    LOCAL_SECURITY_PROFILE,
    LocalVllmRuntimeAdapter,
)
from auto_harness.model_runtime.preparation_gate import PreparationBundle
from auto_harness.model_runtime.schemas import InferenceRuntimePlan
from auto_harness.model_runtime.vllm_adapter import (
    CONTAINER_MODEL_PATH,
    SECURITY_PROFILE,
    VllmRuntimeAdapter,
)
from auto_harness.utils.redaction import check_redaction

# Allowlist of tested vLLM image repositories (registry/repository only; the
# digest is additionally required so no floating tag can pass).
DEFAULT_IMAGE_ALLOWLIST = ("vllm/vllm-openai",)

# Forbidden host-escape markers that must never appear in a runtime plan.
_FORBIDDEN_MARKERS = (
    "--trust-remote-code",
    "--privileged",
    "--network=host",
    "--network",
    "docker.sock",
    "/var/run/docker.sock",
)


class ModelRuntimePolicy:
    """Hard gate for managed vLLM runtime plans."""

    def __init__(
        self,
        adapter: Optional[VllmRuntimeAdapter] = None,
        image_allowlist=DEFAULT_IMAGE_ALLOWLIST,
        security_profile: str = SECURITY_PROFILE,
        local_adapter: Optional[LocalVllmRuntimeAdapter] = None,
    ) -> None:
        self.adapter = adapter or VllmRuntimeAdapter()
        self.local_adapter = local_adapter or LocalVllmRuntimeAdapter()
        self.image_allowlist = tuple(image_allowlist or ())
        self.security_profile = security_profile

    def authorize(
        self,
        plan: InferenceRuntimePlan,
        bundle: PreparationBundle,
        config,
        *,
        execute: bool = False,
        allow_start: bool = False,
        execution_backend: str = "docker",
        image_allowlist=None,
        require_start_auth: bool = True,
    ) -> dict:
        """Return an authorization dict for a runtime plan.

        ``allowed`` is True only when all conditions hold; otherwise the dict
        carries ``verdict: hard_denied`` and a ``reason_code``.
        """
        allowlist = tuple(image_allowlist) if image_allowlist is not None else self.image_allowlist

        # 0. Feature flag must be on (managed source does not exist otherwise).
        enabled = getattr(config, "model_inference_enabled", False)
        if not enabled:
            return self._deny("model_inference_disabled", "model inference feature is disabled")

        # 1. Preparation Gate passed this round (checked before plan fields so a
        #    missing/bogus bundle cannot be bypassed).
        if not isinstance(bundle, PreparationBundle) or not bundle.ok:
            return self._deny("preparation_not_ready", "preparation artifacts are not ready")

        # 2. Runtime Plan schema + hash valid
        if not isinstance(plan, InferenceRuntimePlan):
            return self._deny("plan_schema_unsupported", "runtime plan is not an InferenceRuntimePlan")
        if plan.schema_version != 1:
            return self._deny("plan_schema_unsupported", "unsupported runtime plan schema")
        if plan.compute_plan_hash() != plan.plan_hash:
            return self._deny("plan_hash_mismatch", "runtime plan hash does not match its content")

        # 3. runtime == vllm
        if plan.runtime != "vllm":
            return self._deny("unsupported_runtime", "runtime must be vllm")

        if plan.deployment_mode == "local_vllm":
            return self._authorize_local(
                plan, bundle, config,
                execute=execute, allow_start=allow_start,
                execution_backend=execution_backend,
                require_start_auth=require_start_auth,
            )

        # 4. Image registry/repository in allowlist and digest fixed
        image_name = plan.image.split("@", 1)[0]
        if not any(image_name == allowed or image_name.startswith(allowed + ":") for allowed in allowlist):
            return self._deny("image_not_in_allowlist", "image %r is not in the allowlist" % image_name)
        if not plan.image_digest:
            return self._deny("image_digest_missing", "image must be pinned to an immutable digest")

        # 5. Model host path inside Harness cache; container path fixed
        if not self._path_in_cache(plan.model_host_path, bundle.cache_root):
            return self._deny("model_path_outside_cache", "model host path escapes the cache root")
        if plan.model_container_path != CONTAINER_MODEL_PATH:
            return self._deny("model_container_path_not_fixed", "container model path must be fixed")

        # 6. Command matches the adapter's re-derived command exactly
        try:
            expected_command = self.adapter.command_for(bundle)
        except ValueError as exc:
            return self._deny("command_recompute_failed", str(exc))
        if plan.command != expected_command:
            return self._deny("command_mismatch", "runtime command does not match the adapter")

        # 7. Backend must be Docker
        if execution_backend != "docker":
            return self._deny("local_backend_denied", "managed inference runtime requires Docker")

        # 8. --execute and --allow-start authorized (skipped for plan-only
        #    dry-run, where the plan is generated but not executed).
        if require_start_auth and not (execute and allow_start):
            return self._deny("start_not_authorized", "execute and allow-start must both be authorized")

        # 9. GPU Preflight / Resource Decision allowed
        if not bundle.decision or bundle.decision.status != "allowed":
            return self._deny("resource_not_allowed", "resource decision is not allowed")

        # 10. Security profile is a supported version
        if plan.security_profile != self.security_profile:
            return self._deny("unsupported_security_profile", "security profile %r not supported" % plan.security_profile)

        # Defense-in-depth: no host-escape markers and no secrets anywhere.
        joined = " ".join(plan.command).lower()
        for marker in _FORBIDDEN_MARKERS:
            if marker.lower() in joined:
                return self._deny("host_escape_rejected", "forbidden marker in command: %s" % marker)
        text = json.dumps(plan.to_dict(), ensure_ascii=False)
        if check_redaction(text):
            return self._deny("secret_in_plan", "runtime plan contains unredacted sensitive content")

        return {
            "allowed": True,
            "verdict": "auto_allowed",
            "reason_code": "managed_inference_runtime",
            "reasons": ["managed inference runtime plan authorized"],
            "plan_hash": plan.plan_hash,
        }

    def _authorize_local(
        self,
        plan: InferenceRuntimePlan,
        bundle: PreparationBundle,
        config,
        *,
        execute: bool,
        allow_start: bool,
        execution_backend: str,
        require_start_auth: bool,
    ) -> dict:
        """Hard gate for docker-less host-process vLLM plans.

        Same fail-closed conditions as the managed mode, minus the container
        checks, plus an explicit operator opt-in: without
        ``model_runtime_mode=local_vllm`` the plan is denied even when every
        other condition holds.
        """
        if str(getattr(config, "model_runtime_mode", "managed_vllm")) != "local_vllm":
            return self._deny("local_mode_not_selected", "local_vllm plans require model_runtime_mode=local_vllm")
        if execution_backend != "local":
            return self._deny("backend_mode_mismatch", "local_vllm plans require the local execution backend")
        if plan.security_profile != LOCAL_SECURITY_PROFILE:
            return self._deny("unsupported_security_profile", "security profile %r not supported" % plan.security_profile)
        if plan.image or plan.image_digest or plan.model_container_path or plan.container_name:
            return self._deny("local_plan_polluted", "local plans must not carry container fields")
        if not self._path_in_cache(plan.model_host_path, bundle.cache_root):
            return self._deny("model_path_outside_cache", "model host path escapes the cache root")
        if plan.expected_host not in ("127.0.0.1", "localhost"):
            return self._deny("local_bind_denied", "local runtime must bind loopback")

        try:
            expected_command = self.local_adapter.command_for(bundle, config, host_port=plan.expected_port)
        except ValueError as exc:
            return self._deny("command_recompute_failed", str(exc))
        if plan.command != expected_command:
            return self._deny("command_mismatch", "runtime command does not match the adapter")

        if require_start_auth and not (execute and allow_start):
            return self._deny("start_not_authorized", "execute and allow-start must both be authorized")
        if not bundle.decision or bundle.decision.status != "allowed":
            return self._deny("resource_not_allowed", "resource decision is not allowed")

        joined = " ".join(plan.command).lower()
        for marker in _FORBIDDEN_MARKERS:
            if marker.lower() in joined:
                return self._deny("host_escape_rejected", "forbidden marker in command: %s" % marker)
        text = json.dumps(plan.to_dict(), ensure_ascii=False)
        if check_redaction(text):
            return self._deny("secret_in_plan", "runtime plan contains unredacted sensitive content")

        return {
            "allowed": True,
            "verdict": "auto_allowed",
            "reason_code": "local_inference_runtime",
            "reasons": ["local inference runtime plan authorized (no container isolation)"],
            "plan_hash": plan.plan_hash,
        }

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _path_in_cache(host_path: str, cache_root: str) -> bool:
        if not host_path or not cache_root:
            return False
        from pathlib import Path

        try:
            host = Path(host_path).resolve()
            root = Path(cache_root).resolve()
            host.relative_to(root)
            return True
        except (ValueError, OSError):
            return False

    @staticmethod
    def _deny(reason_code: str, reason: str) -> dict:
        return {
            "allowed": False,
            "verdict": "hard_denied",
            "reason_code": reason_code,
            "reasons": [reason],
            "plan_hash": "",
        }
