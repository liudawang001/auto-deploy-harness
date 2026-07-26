import json
from pathlib import Path

from auto_harness.models.result import StageResult
from auto_harness.models.task import RuntimePolicy
from auto_harness.repair.actions import RepairActionNormalizer, RepairActionRegistry
from auto_harness.repair.apply import RepairApplier
from auto_harness.repair.planner import RepairPlanner
from auto_harness.repair.policy import RepairPolicy


def test_registry_is_authoritative_for_supported_repair_actions():
    registry = RepairActionRegistry()
    assert registry.validate(
        {"type": "install_package", "payload": {"package": "numpy<2"}}
    )["allowed"] is True
    decision = registry.validate(
        {"type": "change_cache_dir", "payload": {"config": "model_cache_dir"}}
    )
    assert decision["allowed"] is False
    assert decision["reasons"] == ["unsupported repair action type"]


def test_legacy_env_request_alias_normalizes_to_supported_action():
    action = RepairActionNormalizer().normalize(
        {
            "type": "request_env_var_name_only",
            "payload": {"env_vars": ["HF_TOKEN"]},
        }
    )
    assert action["type"] == "set_env_var_name_only"
    assert RepairActionRegistry().validate(action)["allowed"] is True


def test_registry_rejects_unsafe_env_names_and_rerun_stage():
    registry = RepairActionRegistry()
    env_decision = registry.validate(
        {
            "type": "set_env_var_name_only",
            "payload": {"env_vars": ["HF_TOKEN", "bad=value"]},
        }
    )
    rerun_decision = registry.validate(
        {"type": "rerun_from_stage", "payload": {"stage": "analyze"}}
    )
    assert env_decision["allowed"] is False
    assert rerun_decision["allowed"] is False


def test_conda_channels_survive_action_normalization():
    action = RepairActionNormalizer().normalize(
        {
            "type": "install_conda_package",
            "payload": {
                "package": "pytorch-cuda=12.1",
                "channels": ["pytorch", "nvidia"],
            },
        }
    )[0]
    assert action["payload"]["channels"] == ["pytorch", "nvidia"]


def test_policy_rejects_unknown_action_even_with_operator_approval(tmp_path):
    runtime = RuntimePolicy(workspace_root=str(tmp_path), allow_dependency_install=True)
    result = RepairPolicy().check(
        {
            "actions": [
                {
                    "type": "change_cache_dir",
                    "requires": {"operator_approval": True},
                    "payload": {"config": "model_cache_dir"},
                }
            ]
        },
        runtime,
        operator_approval={
            "approved": True,
            "approved_action_types": ["change_cache_dir"],
        },
    )
    assert result["allowed"] is False
    assert result["decisions"][0]["contract_kind"] == "unsupported"


def test_applier_rejects_entire_plan_before_partial_execution(tmp_path):
    run_dir = Path(tmp_path)
    (run_dir / "workspace" / "repo").mkdir(parents=True)
    calls = []

    result = RepairApplier().apply(
        run_dir,
        {
            "actions": [
                {"type": "install_package", "payload": {"package": "numpy<2"}},
                {"type": "adjust_runtime", "payload": {"strategy": "cpu fallback"}},
            ]
        },
        {"allowed": True, "decisions": []},
        execute=True,
        command_runner=lambda cmd, cwd, timeout: calls.append(cmd),
    )

    assert result["status"] == "rejected"
    assert result["executed_action_count"] == 0
    assert calls == []
    persisted = json.loads(
        (run_dir / "repairs" / "repair_apply_result.json").read_text(encoding="utf-8")
    )
    assert persisted["policy"]["allowed"] is False
    assert persisted["policy"]["contract_decisions"][1]["action_type"] == "adjust_runtime"


def test_planner_marks_unsupported_deterministic_action_for_manual_review():
    result = StageResult(
        stage="runner",
        status="failed",
        summary="CUDA out of memory",
        data={"diagnosis": {"category": "cuda_oom", "confidence": 0.9}},
    )
    plan = RepairPlanner().propose("runner", result)
    assert plan["status"] == "needs_manual_review"
    assert plan["action_contract"]["valid"] is False
    assert plan["actions"][0]["type"] == "adjust_runtime"


def test_pin_dependency_uses_controlled_package_installer(tmp_path):
    run_dir = Path(tmp_path)
    (run_dir / "workspace" / "repo").mkdir(parents=True)
    calls = []

    result = RepairApplier().apply(
        run_dir,
        {"actions": [{"type": "pin_dependency", "payload": {"package": "numpy<2"}}]},
        {"allowed": True, "decisions": []},
        execute=True,
        command_runner=lambda cmd, cwd, timeout: (
            calls.append(cmd)
            or {"exit_code": 0, "stdout": "ok", "stderr": "", "timed_out": False}
        ),
        env_context={"backend": "venv", "python_executable": ".venv/bin/python"},
    )

    assert result["status"] == "applied"
    assert result["executed_action_count"] == 1
    assert calls == [[".venv/bin/python", "-m", "pip", "install", "numpy<2"]]
