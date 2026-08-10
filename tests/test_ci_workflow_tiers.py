from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_push_ci_uses_offline_development_gate_only():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "development-readiness:" in workflow
    assert "python -m auto_harness.release_gates" not in workflow
    assert "python -m auto_harness.cli readiness" not in workflow
    assert "--agent-provider mock" in workflow
    assert "--agent-plan-first-provider mock" in workflow


def test_release_readiness_is_manual_and_keeps_strict_gates():
    workflow = (
        ROOT / ".github" / "workflows" / "release-readiness.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "\n  push:" not in workflow
    assert "\n  pull_request:" not in workflow
    assert "python -m auto_harness.release_gates" in workflow
    assert "python -m auto_harness.cli readiness" in workflow
