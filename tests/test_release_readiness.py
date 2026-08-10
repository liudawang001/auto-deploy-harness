import json
import subprocess
from pathlib import Path

from auto_harness import release_gates
from auto_harness.readiness import ReadinessAuditor
from auto_harness.release_evidence import build_evidence, evidence_hash
from auto_harness.resources.installer import initialize_workspace
from auto_harness.models.base import write_json


def _complete_fixture(root: Path) -> None:
    for relative in ReadinessAuditor.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("manifest.json"):
            write_json(path, {
                "cases": [{"id": value} for value in ReadinessAuditor.REQUIRED_BENCHMARK_CASES]
            })
        else:
            path.write_text("fixture\n", encoding="utf-8")
    for relative in ReadinessAuditor.REQUIRED_EVIDENCE.values():
        write_json(root / relative, build_evidence(root, ["fixture"], "passed", 1, 0))


def test_readiness_requires_commit_bound_evidence(tmp_path):
    _complete_fixture(tmp_path)
    report = ReadinessAuditor().audit(tmp_path)
    assert report["status"] == "ready_for_external_smoke"
    assert report["local_readiness_percent"] == 100


def test_readiness_rejects_tampered_evidence(tmp_path):
    _complete_fixture(tmp_path)
    path = tmp_path / ReadinessAuditor.REQUIRED_EVIDENCE["package_smoke"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["passed"] = 99
    write_json(path, payload)
    report = ReadinessAuditor().audit(tmp_path)
    assert report["status"] == "incomplete"
    gate = next(item for item in report["local_gates"] if item["id"] == "evidence:package_smoke")
    assert "evidence hash mismatch" in gate["errors"]


def test_workspace_initializer_preserves_then_forces(tmp_path):
    result = initialize_workspace(tmp_path)
    assert result["status"] == "initialized"
    config = tmp_path / "configs" / "default.json"
    assert config.exists()
    assert len(list((tmp_path / "skills").glob("*/SKILL.md"))) == 8
    config.write_text('{"custom": true}\n', encoding="utf-8")
    initialize_workspace(tmp_path)
    assert json.loads(config.read_text(encoding="utf-8"))["custom"] is True
    initialize_workspace(tmp_path, force=True)
    assert "default_controller" in json.loads(config.read_text(encoding="utf-8"))


def test_packaged_resources_match_repository_defaults():
    repository = Path(__file__).resolve().parents[1]
    packaged = repository / "src" / "auto_harness" / "resources"
    assert (packaged / "default.json").read_bytes() == (repository / "configs" / "default.json").read_bytes()
    for source in sorted((repository / "skills").glob("*/SKILL.md")):
        target = packaged / "skills" / source.parent.name / "SKILL.md"
        assert target.read_bytes() == source.read_bytes()


def test_default_cli_smoke_uses_offline_mock_without_llm_keys(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, cwd, env=None):
        captured["command"] = command
        captured["env"] = env
        state = Path(cwd) / "runs" / "task" / "state.json"
        state.parent.mkdir(parents=True)
        write_json(state, {"status": "completed_dry_run"})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "inherited-key-must-not-pass-through")
    monkeypatch.setenv("AUTO_HARNESS_LLM_API_KEY", "generic-key-must-not-pass-through")
    monkeypatch.setattr(release_gates, "_run", fake_run)

    evidence = release_gates._default_cli_smoke(tmp_path)

    assert evidence["status"] == "passed"
    assert "--dry-run" in captured["command"]
    agent_provider_index = captured["command"].index("--agent-provider")
    plan_provider_index = captured["command"].index("--agent-plan-first-provider")
    assert captured["command"][agent_provider_index + 1] == "mock"
    assert captured["command"][plan_provider_index + 1] == "mock"
    assert "DEEPSEEK_API_KEY" not in captured["env"]
    assert "AUTO_HARNESS_LLM_API_KEY" not in captured["env"]
