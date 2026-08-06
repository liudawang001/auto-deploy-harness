import json
from pathlib import Path

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
