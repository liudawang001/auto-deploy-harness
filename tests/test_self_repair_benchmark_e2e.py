import inspect
import json
from pathlib import Path

from auto_harness.benchmarks.runner import BenchmarkRunner
from auto_harness.repair.evidence import compute_fresh_trace, compute_repair_verified


def test_controller_e2e_does_not_mock_controller():
    manifest = json.loads(
        Path("tests/fixtures/benchmarks/manifest.json").read_text(encoding="utf-8")
    )
    case = next(
        item for item in manifest["cases"]
        if item["id"] == "langgraph_self_repair_controller_e2e"
    )
    assert case["test_level"] == "controller_e2e"
    source = inspect.getsource(BenchmarkRunner._case_langgraph_self_repair_controller_e2e)
    assert "_build_controller =" not in source
    assert "LangGraphController.run =" not in source
    assert "runner.deploy(" in source
    assert 'controller="langgraph"' in source


def test_self_repair_e2e_requires_fresh_trace():
    assert compute_repair_verified(
        effective_action_count=1,
        resume_executed=True,
        verify_status_after="passed",
        evidence_contains_after_trace=True,
        fresh_trace=compute_fresh_trace("trace-before", "trace-after"),
    )
    assert not compute_repair_verified(
        effective_action_count=1,
        resume_executed=True,
        verify_status_after="passed",
        evidence_contains_after_trace=True,
        fresh_trace=compute_fresh_trace("trace-same", "trace-same"),
    )
