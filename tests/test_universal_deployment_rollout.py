"""Phase B5/B6 regression tests: rollout modes, shadow diff, readiness gate."""

import json
import shutil

import pytest

from auto_harness.capabilities.shadow_diff import compute_shadow_diff, enforce_blockers
from auto_harness.config import HarnessConfig
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.reporter import ReportGenerator
from auto_harness.universal_readiness import UniversalDeploymentReadinessGate


def _procfile_repo(root):
    (root / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (root / "app.py").write_text("print('service')\n", encoding="utf-8")
    (root / "Procfile").write_text("web: python3 app.py\n", encoding="utf-8")
    return root


def test_shadow_mode_computes_auditable_diff(tmp_path):
    _procfile_repo(tmp_path)

    analysis = ProjectAnalyzer(deployment_capability_mode="shadow").analyze(tmp_path).data

    diff = analysis["rollout_shadow_diff"]
    assert diff["computed"] is True
    assert diff["mode"] == "shadow"
    assert diff["classification"] in (
        "equivalent", "new_more_complete", "new_less_complete",
        "new_less_safe", "incomparable", "new_safer",
    )
    gained = diff["run_candidate_set"]["gained"]
    assert ["python3", "app.py"] in gained


def test_legacy_mode_keeps_baseline_chain(tmp_path):
    _procfile_repo(tmp_path)

    analysis = ProjectAnalyzer(deployment_capability_mode="legacy").analyze(tmp_path).data

    assert not [
        item for item in analysis["run_candidates"]
        if item.get("source_kind") == "procfile_web"
    ]
    assert "command_registry" not in analysis
    assert analysis["rollout_shadow_diff"]["computed"] is False


def test_off_mode_keeps_baseline_chain(tmp_path):
    _procfile_repo(tmp_path)

    analysis = ProjectAnalyzer(deployment_capability_mode="off").analyze(tmp_path).data

    assert not [
        item for item in analysis["run_candidates"]
        if item.get("source_kind") == "procfile_web"
    ]


def test_enforce_mode_with_clean_diff_is_not_blocked(tmp_path):
    _procfile_repo(tmp_path)

    analysis = ProjectAnalyzer(deployment_capability_mode="enforce").analyze(tmp_path).data

    assert analysis["rollout_enforce_blockers"] == []
    assert analysis["deployability"]["status"] != "blocked"


def test_enforce_mode_fails_closed_on_less_safe_diff(tmp_path, monkeypatch):
    _procfile_repo(tmp_path)

    from auto_harness.capabilities import shadow_diff as shadow_diff_module

    def poisoned(baseline, candidate):
        return {
            "classification": "new_less_safe",
            "new_less_safe_commands": [["python3", "app.py"]],
            "run_candidate_set": {"gained": [], "lost": []},
        }

    monkeypatch.setattr(shadow_diff_module, "compute_shadow_diff", poisoned)
    import auto_harness.modules.analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "compute_shadow_diff", poisoned)

    analysis = ProjectAnalyzer(deployment_capability_mode="enforce").analyze(tmp_path).data

    assert analysis["deployability"]["status"] == "blocked"
    assert "shadow_diff_new_less_safe" in analysis["deployability"]["risk_reasons"]


def test_shadow_diff_classification_matrix():
    baseline = {
        "frameworks": ["unknown"],
        "install_plan": [["python3", "-m", "venv", ".venv"]],
        "run_candidates": [{
            "cmd": [".venv/bin/python", "app.py"], "expected_port": 8000,
        }],
        "verify_hint": {"service_type": "unknown"},
        "deployability": {
            "status": "partial",
            "missing_capabilities": ["verify.strong_evidence"],
        },
    }
    same = dict(baseline, run_candidates=[
        {"cmd": [".venv/bin/python", "app.py"], "expected_port": 8000},
    ])
    assert compute_shadow_diff(baseline, same)["classification"] == "equivalent"

    richer = dict(baseline, run_candidates=[
        {"cmd": [".venv/bin/python", "app.py"], "expected_port": 8000},
        {"cmd": [".venv/bin/gunicorn", "app:application"], "expected_port": 8000},
    ], deployability={
        "status": "partial",
        "missing_capabilities": ["verify.strong_evidence"],
    })
    assert compute_shadow_diff(baseline, richer)["classification"] == "new_more_complete"

    fewer = dict(baseline, run_candidates=[])
    assert compute_shadow_diff(baseline, fewer)["classification"] == "new_less_complete"

    rejected = dict(baseline, authorization_attempts=[{
        "normalized_argv": [".venv/bin/python", "app.py"],
        "verdict": "candidate_rejected",
    }])
    baseline_authorized = dict(baseline, authorization_attempts=[{
        "normalized_argv": [".venv/bin/python", "app.py"],
        "verdict": "auto_allowed",
    }])
    assert compute_shadow_diff(
        baseline_authorized, rejected,
    )["classification"] == "new_less_safe"

    incomparable = dict(baseline, verify_hint={"service_type": "openai_compatible"})
    assert compute_shadow_diff(baseline, incomparable)["classification"] == "incomparable"


def test_enforce_blockers_fail_closed():
    assert enforce_blockers({}) == ["shadow_diff_missing"]
    assert enforce_blockers({"classification": "new_less_safe"}) == [
        "shadow_diff_new_less_safe",
    ]
    assert enforce_blockers({"classification": "equivalent"}) == []


def test_rollout_decision_artifact_binds_shadow_diff_and_gate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _procfile_repo(repo)
    analysis = ProjectAnalyzer(deployment_capability_mode="shadow").analyze(repo).data

    ReportGenerator().generate(
        tmp_path,
        {"project": {"name": "rollout", "repo_url": "local"}},
        {
            "analyze": {"status": "passed", "summary": "analyzed", "data": analysis},
            "verify": {"status": "uncertain", "summary": "not run", "data": {}},
        },
    )

    payload = json.loads(
        (tmp_path / "reports" / "rollout_decision.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert payload["data"]["mode"] == "shadow"
    assert payload["data"]["shadow_diff"]["computed"] is True
    assert payload["data"]["enforce"]["allowed"] is False
    assert payload["data"]["enforce"]["blockers"] == []


def test_config_accepts_legacy_rollout_mode():
    config = HarnessConfig(deployment_capability_mode="legacy")
    assert config.deployment_capability_mode == "legacy"
    with pytest.raises(ValueError):
        HarnessConfig(deployment_capability_mode="bogus")


def _write_handoff(root, **overrides):
    evidence_dir = root / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    handoff = {
        "schema_version": 1,
        "commit": "0" * 40,
        "source_state": "committed",
        "config_hash": "b" * 64,
        "capability_schema_version": 2,
        "contract_schema_version": 1,
        "adapter_registry_version": 1,
        "verifier_registry_version": 1,
        "tests": {},
        "false_success_count": 0,
        "unsafe_command_execution_count": 0,
        "secret_leak_count": 0,
    }
    handoff.update(overrides)
    (evidence_dir / "universal-deployment-foundation-handoff.json").write_text(
        json.dumps(handoff), encoding="utf-8",
    )
    return handoff


def test_readiness_ready_with_valid_evidence(tmp_path):
    _write_handoff(tmp_path)
    gate = UniversalDeploymentReadinessGate(tmp_path)
    evidence = gate.build_expansion_evidence(
        command=["pytest", "-q", "tests/test_universal_deployment_expansion.py"],
        passed=47,
        failed=0,
        config_hash="c" * 64,
        execution_backend="local",
    )

    result = gate.evaluate(
        expansion_evidence=evidence,
        shadow_diff={"computed": True, "classification": "new_more_complete"},
        config_hash="c" * 64,
        execution_backend="local",
    )

    assert result["status"] == "ready", result["reason_codes"]
    assert result["evidence_bindings"]["commit_sha"] == evidence["commit_sha"]
    assert result["evidence_sha256"]


def test_readiness_blocked_when_evidence_missing(tmp_path):
    _write_handoff(tmp_path)
    gate = UniversalDeploymentReadinessGate(tmp_path)

    result = gate.evaluate(
        shadow_diff={"computed": True, "classification": "equivalent"},
    )

    assert result["status"] == "blocked"
    assert "expansion_evidence_missing" in result["reason_codes"]


def test_readiness_blocked_on_stale_fixture_hash(tmp_path):
    _write_handoff(tmp_path)
    gate = UniversalDeploymentReadinessGate(tmp_path)
    evidence = gate.build_expansion_evidence(
        command=["pytest", "-q"],
        passed=10,
        failed=0,
    )

    # The bound test fixture changed after the evidence was generated.
    fixture = tmp_path / "tests" / "test_universal_deployment_expansion.py"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text("# tampered\n", encoding="utf-8")

    result = gate.evaluate(
        expansion_evidence=evidence,
        shadow_diff={"computed": True, "classification": "equivalent"},
    )

    assert result["status"] == "blocked"
    assert any("fixture" in code for code in result["reason_codes"])


def test_readiness_blocked_on_less_safe_shadow(tmp_path):
    _write_handoff(tmp_path)
    gate = UniversalDeploymentReadinessGate(tmp_path)
    evidence = gate.build_expansion_evidence(
        command=["pytest", "-q"], passed=10, failed=0,
    )

    result = gate.evaluate(
        expansion_evidence=evidence,
        shadow_diff={"computed": True, "classification": "new_less_safe"},
    )

    assert result["status"] == "blocked"
    assert "shadow_diff_new_less_safe" in result["reason_codes"]


def test_readiness_blocked_on_nonzero_security_counters(tmp_path):
    _write_handoff(tmp_path, false_success_count=1)
    gate = UniversalDeploymentReadinessGate(tmp_path)
    evidence = gate.build_expansion_evidence(
        command=["pytest", "-q"], passed=10, failed=0,
    )

    result = gate.evaluate(
        expansion_evidence=evidence,
        shadow_diff={"computed": True, "classification": "equivalent"},
    )

    assert result["status"] == "blocked"
    assert "handoff_false_success_count_nonzero" in result["reason_codes"]


def test_readiness_blocked_when_handout_schema_unsupported(tmp_path):
    _write_handoff(tmp_path, capability_schema_version=99)
    gate = UniversalDeploymentReadinessGate(tmp_path)
    evidence = gate.build_expansion_evidence(
        command=["pytest", "-q"], passed=10, failed=0,
    )

    result = gate.evaluate(
        expansion_evidence=evidence,
        shadow_diff={"computed": True, "classification": "equivalent"},
    )

    assert result["status"] == "blocked"
    assert "handoff_capability_schema_version_unsupported" in result["reason_codes"]
