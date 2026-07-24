import json
from pathlib import Path

from auto_harness.benchmarks.runner import BenchmarkRunner


MANIFEST = Path("tests/fixtures/benchmarks/manifest.json")


def test_every_manifest_case_has_test_level():
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    assert cases
    for case in cases:
        assert case["case_id"] == case["id"]
        assert case["test_level"] in BenchmarkRunner.TEST_LEVELS
        assert isinstance(case["requires"], list)


def test_simulation_not_reported_as_e2e():
    cases = {
        case["id"]: case
        for case in json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    }
    assert "agent_full_self_healing_pipeline" not in cases
    assert cases["agent_self_healing_control_flow_simulation"]["test_level"] == "unit_simulation"
    assert cases["agent_loop_dependency_self_repair_e2e"]["test_level"] == "module_integration"


def test_report_includes_level_and_environment_metadata():
    report = BenchmarkRunner().run(MANIFEST, case_ids=["tool_registry_policy_gate"])
    assert report["status"] == "passed"
    case = report["cases"][0]
    assert case["case_id"] == "tool_registry_policy_gate"
    assert case["test_level"] == "unit_simulation"
    assert case["environment_status"] == "available"
    assert isinstance(case["assertions"], list)
    assert isinstance(case["artifact_paths"], list)
    assert case["duration_ms"] >= 0


def test_environment_blocked_is_not_passed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "cases": [{
            "id": "blocked",
            "case_id": "blocked",
            "test_level": "controller_e2e",
            "requires": ["socket_bind"],
            "purpose": "",
            "expected_signal": "",
        }]
    }), encoding="utf-8")

    class BlockedRunner(BenchmarkRunner):
        def _run_case(self, case, fixture_dir):
            return self._environment_blocked(case, "socket_bind_not_permitted")

    report = BlockedRunner().run(manifest)
    assert report["status"] == "partial"
    assert report["cases"][0]["status"] == "not_run"
    assert report["cases"][0]["environment_status"] == "blocked"
