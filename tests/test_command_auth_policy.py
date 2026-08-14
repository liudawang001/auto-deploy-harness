from datetime import datetime, timedelta, timezone

from auto_harness.command_auth import (
    CommandAuthorizationEngine,
    CommandCandidateSelector,
    CommandDiscoveryService,
)
from auto_harness.command_auth.approval import build_command_approval_request
from auto_harness.agent_runtime.plan_policy import PlanPolicyGate
from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
from auto_harness.modules.runner import RunnerModule


def _files(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _registry(tmp_path, readme="demo serve\n"):
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\ndemo = "demo.cli:main"\n', encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo")


def test_declared_cli_auto_allowed_and_forces_docker(tmp_path):
    registry = _registry(tmp_path)
    decision = CommandAuthorizationEngine().authorize(
        registry.candidates[0], registry, repo_dir=tmp_path, execution_backend="local"
    )
    assert decision.verdict == "auto_allowed"
    assert decision.effective_backend == "docker"
    assert decision.reason_code == "declared_cli_bound_to_owned_env"


def test_evidence_change_rejects_candidate(tmp_path):
    registry = _registry(tmp_path)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    decision = CommandAuthorizationEngine().authorize(
        registry.candidates[0], registry, repo_dir=tmp_path,
    )
    assert decision.verdict == "candidate_rejected"
    assert decision.reason_code == "evidence_hash_mismatch"


def test_make_requires_bound_one_shot_approval(tmp_path):
    (tmp_path / "Makefile").write_text("serve:\n\tpython app.py\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("make serve\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo")
    candidate = registry.candidates[0]
    engine = CommandAuthorizationEngine()
    pending = engine.authorize(candidate, registry, repo_dir=tmp_path, sandbox_policy_fingerprint="box")
    assert pending.verdict == "approval_required"
    evidence = [registry.evidence_by_id()[item] for item in candidate.evidence_ids]
    request = build_command_approval_request(
        candidate, "repo", evidence, "box",
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    approval = {**request, "decision": "approve", "request_hash": request["request_hash"]}
    allowed = engine.authorize(
        candidate, registry, repo_dir=tmp_path,
        sandbox_policy_fingerprint="box", approval=approval,
    )
    assert allowed.verdict == "auto_allowed"


def test_approval_cannot_override_hard_deny():
    result = CommandAuthorizationEngine().authorize_argv(
        ["bash", "-c", "echo ok"], allowed_commands=["bash"],
    )
    assert result["verdict"] == "hard_denied"
    assert result["reason_code"] == "shell_wrapper_hard_denied"


def test_selector_prefers_safe_fallback(tmp_path):
    registry = _registry(tmp_path)
    safe = registry.candidates[0]
    dangerous = type(safe).build(
        phase="run", argv=["bash", "-c", "bad"], source_kind="pep621_script",
        evidence_ids=safe.evidence_ids, score=1.0,
    )
    registry.candidates.insert(0, dangerous)
    engine = CommandAuthorizationEngine()
    decisions = [engine.authorize(item, registry, repo_dir=tmp_path) for item in registry.candidates]
    selected = CommandCandidateSelector().select(registry.candidates, decisions)
    assert selected["candidate_id"] == safe.candidate_id
    assert dangerous.candidate_id in selected["rejected_candidate_ids"]


def _plan(command):
    return {
        "status": "ok",
        "plan_id": "p1",
        "grounding": [{"file": "README.md", "claim": "run", "reason": "documented"}],
        "environment": {"install_commands": [["python3", "-m", "venv", ".venv"]]},
        "run": {
            "candidates": [{"id": "run", "cmd": command, "expected_port": 8000}],
            "selected_candidate_id": "run",
        },
        "verify": {
            "request": {"method": "GET", "path": "/?trace={{trace_id}}"},
            "success_evidence": "response contains trace_id",
        },
    }


def test_plan_gate_rejects_unknown_command_when_registry_exists(tmp_path):
    (tmp_path / "README.md").write_text("mystery serve\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder().build(tmp_path)
    result = PlanPolicyGate().validate(_plan(["mystery", "serve"]), snapshot)
    assert result["status"] == "rejected"
    assert result["rejected_items"][0]["reason"] == "repository_command_not_declared"


def test_unknown_python_entrypoint_requires_human_approval(tmp_path):
    (tmp_path / "app.py").write_text("print('app')\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder().build(tmp_path)
    result = PlanPolicyGate().validate(_plan([".venv/bin/python", "app.py"]), snapshot)
    assert result["status"] == "approval_required"
    assert result["approval_request"]["normalized_argv"] == [".venv/bin/python", "app.py"]
    assert result["approval_request"]["expires_at"]
    assert result["approval_preview_candidates"][0]["cmd"] == [
        ".venv/bin/python", "app.py",
    ]


def test_plan_gate_emits_approval_then_accepts_bound_make_target(tmp_path):
    (tmp_path / "Makefile").write_text("serve:\n\tpython app.py\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("make serve\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder().build(tmp_path)
    gate = PlanPolicyGate()
    pending = gate.validate(_plan(["make", "serve"]), snapshot)
    assert pending["status"] == "approval_required"
    request = pending["approval_request"]

    approved = gate.validate(
        _plan(["make", "serve"]),
        snapshot,
        approval={
            "request": request,
            "decision": {
                "decision": "approve",
                "request_hash": request["request_hash"],
                "approval_id": request["approval_id"],
                "operation_id": request["operation_id"],
            },
        },
    )
    assert approved["allowed"] is True
    candidate = approved["normalized_plan"]["run"]["candidates"][0]
    assert candidate["required_backend"] == "docker"


def test_runner_jit_revalidates_evidence_and_forces_docker(tmp_path):
    registry = _registry(tmp_path)
    declared = registry.candidates[0]
    analysis = {
        "command_registry": registry.to_dict(),
        "sandbox_policy_fingerprint": "",
        "run_candidates": [{
            "id": "cli",
            "cmd": declared.argv,
            "command_candidate_id": declared.candidate_id,
            "expected_port": 8000,
        }],
    }
    dry = RunnerModule().run(tmp_path, analysis, execute=False)
    assert dry.status == "passed"
    assert dry.data["execution_backend"] == "docker"

    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    rejected = RunnerModule().run(tmp_path, analysis, execute=True)
    assert rejected.status == "failed"
    assert rejected.error == "no_safe_command_candidate"
    assert rejected.data["authorization_attempts"][0]["reason_code"] == "evidence_hash_mismatch"


def test_jit_requires_owned_environment_marker(tmp_path):
    registry = _registry(tmp_path)
    declared = registry.candidates[0]
    executable = tmp_path / declared.argv[0]
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    engine = CommandAuthorizationEngine()

    missing = engine.authorize(
        declared, registry, repo_dir=tmp_path, require_executable=True,
    )
    assert missing.reason_code == "owned_environment_marker_missing"

    marker = tmp_path / "venv_owner.json"
    marker.write_text(
        '{"repository_fingerprint":"repo","environment_path":"%s"}'
        % str((tmp_path / ".venv").resolve()), encoding="utf-8",
    )
    allowed = engine.authorize(
        declared, registry, repo_dir=tmp_path, require_executable=True,
        environment_ownership_marker=marker,
    )
    assert allowed.verdict == "auto_allowed"


def test_runner_falls_back_after_process_exit(tmp_path):
    (tmp_path / "first.py").write_text("raise SystemExit(1)\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    result = RunnerModule().run(
        tmp_path,
        {"run_candidates": [
            {"id": "first", "cmd": ["python3", "first.py"], "expected_port": 65431},
            {"id": "second", "cmd": ["python3", "second.py"], "expected_port": 65432},
        ]},
        execute=True,
        allowed_commands=["python3"],
        wait_seconds=0.2,
        max_candidate_attempts=2,
    )
    assert result.status == "failed"
    assert [item["candidate_id"] for item in result.data["fallbacks"]] == ["first", "second"]


def test_command_approval_is_consumed_once(tmp_path):
    runner = RunnerModule()
    candidate = {
        "command_candidate_id": "cmd1",
        "_authorization_operation_id": "op_cmd_test",
    }
    assert runner._consume_command_approval(tmp_path, candidate) is True
    assert runner._consume_command_approval(tmp_path, candidate) is False
