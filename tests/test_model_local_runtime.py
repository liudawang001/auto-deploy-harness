"""local_vllm runtime mode tests: docker-less host-process vLLM chain.

Covers the policy opt-in gating, command re-derivation binding, and the full
offline chain (gate -> adapter -> policy -> process launch -> readiness ->
non-stream/SSE inference evidence) without Docker.
"""
import json
import sys
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.model_runtime.controller import ModelRuntimeController
from auto_harness.model_runtime.local_adapter import LocalVllmRuntimeAdapter
from auto_harness.model_runtime.policy import ModelRuntimePolicy
from auto_harness.model_runtime.readiness import ModelRuntimeReadiness

from tests.test_model_inference_pipeline import _build_artifacts, FakeHTTP


def _config(tmp_path, **overrides):
    data = dict(
        model_inference_enabled=True,
        model_runtime_mode="local_vllm",
        model_runtime_local_python=sys.executable,
        model_cache_dir=str(tmp_path / "model_cache"),
    )
    data.update(overrides)
    return HarnessConfig(**data)


def _local_bundle_plan(tmp_path):
    return _build_artifacts(tmp_path)


def test_local_policy_requires_explicit_mode(tmp_path):
    run_dir, cache_root = _local_bundle_plan(tmp_path)
    from auto_harness.model_runtime.preparation_gate import PreparationArtifactGate

    bundle = PreparationArtifactGate(run_dir, cache_root=cache_root).validate()
    assert bundle.ok
    # default (managed) mode: a locally-built plan must be denied by the
    # policy even though every other condition holds
    config = HarnessConfig(model_inference_enabled=True)
    plan = LocalVllmRuntimeAdapter().build(bundle, config, task_id="task-1")
    verdict = ModelRuntimePolicy().authorize(
        plan, bundle, config,
        execute=True, allow_start=True, execution_backend="local",
    )
    assert verdict["allowed"] is False
    assert verdict["reason_code"] == "local_mode_not_selected"


def test_local_policy_rejects_docker_backend(tmp_path):
    run_dir, cache_root = _local_bundle_plan(tmp_path)
    config = _config(tmp_path)
    controller = ModelRuntimeController()
    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=config, cache_root=cache_root,
        execute=False, allow_start=False,
    )
    assert phase.status == "passed"
    # the controller derives execution_backend from the mode; feeding a
    # managed (docker) backend into the policy must deny a local plan
    verdict = ModelRuntimePolicy().authorize(
        phase.plan, phase.bundle, config,
        execute=True, allow_start=True, execution_backend="docker",
    )
    assert verdict["allowed"] is False
    assert verdict["reason_code"] == "backend_mode_mismatch"


def test_local_plan_shape_and_command_binding(tmp_path):
    run_dir, cache_root = _local_bundle_plan(tmp_path)
    config = _config(tmp_path)
    controller = ModelRuntimeController()
    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=config, cache_root=cache_root,
        execute=False, allow_start=False,
    )
    assert phase.status == "passed"
    plan = phase.plan
    assert plan.deployment_mode == "local_vllm"
    assert plan.image == "" and plan.image_digest == ""
    assert plan.model_container_path == "" and plan.container_name == ""
    assert plan.security_profile == "model_runtime_local_v1"
    assert plan.command[0] == sys.executable
    assert plan.command[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert "--trust-remote-code" not in plan.command
    # the command must be exactly what the policy re-derives from the bundle
    expected = LocalVllmRuntimeAdapter().command_for(
        phase.bundle, config, host_port=plan.expected_port,
    )
    assert plan.command == expected
    # tampering is rejected by the hash + re-derivation gates
    plan.command.append("--trust-remote-code")
    plan.plan_hash = plan.compute_plan_hash()  # mirror the managed tamper case
    policy = ModelRuntimePolicy()
    verdict = policy.authorize(
        plan, phase.bundle, config,
        execute=True, allow_start=True, execution_backend="local",
    )
    assert verdict["allowed"] is False
    assert verdict["reason_code"] == "command_mismatch"


def test_local_full_chain_passes(tmp_path):
    run_dir, cache_root = _local_bundle_plan(tmp_path)
    http = FakeHTTP()
    launched = []
    config = _config(tmp_path)

    def fake_launcher(cmd, log_path):
        launched.append((list(cmd), str(log_path)))
        return 424242

    readiness = ModelRuntimeReadiness(
        urlopen=http,
        local_liveness=lambda pid: True,
        sleeper=lambda seconds: None,
    )
    controller = ModelRuntimeController(readiness=readiness)
    phase = controller.run_runtime_phase(
        run_dir=run_dir, task_id="task-1", config=config, cache_root=cache_root,
        execute=True, allow_start=True,
        urlopen=http,
        process_launcher=fake_launcher,
    )
    assert phase.status == "passed", phase.errors
    assert phase.container_id == "424242"
    assert phase.startup_evidence.status == "ready"
    assert len(launched) == 1
    cmd, log_path = launched[0]
    assert cmd == phase.plan.command
    assert log_path == str(run_dir / "logs" / "model_runtime_local.log")
    model_dir = run_dir / "reports" / "model"
    assert (model_dir / "runtime_plan.json").exists()
    assert (model_dir / "startup_evidence.json").exists()
    # no docker command may appear anywhere in a local launch
    assert not any("docker" in part for part in cmd)

    verify = controller.verify_phase(
        run_dir=run_dir, task_id="task-1", runtime_plan=phase.plan,
        startup_evidence=phase.startup_evidence, urlopen=http,
    )
    assert verify.status == "passed"
    assert verify.data["non_stream"]["status"] == "passed"
    assert verify.data["stream"]["status"] == "passed"
    assert (model_dir / "inference_non_stream_evidence.json").exists()
    assert (model_dir / "inference_stream_evidence.json").exists()
