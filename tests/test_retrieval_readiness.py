import json

from auto_harness.readiness import CapabilityMatrix


def test_retrieval_readiness_keeps_fake_below_live(tmp_path):
    project = tmp_path / "project"
    (project / "src" / "auto_harness" / "retrieval").mkdir(parents=True)
    (project / "src" / "auto_harness" / "retrieval" / "service.py").write_text("# implemented\n")
    reports = project / "reports"
    reports.mkdir()
    (reports / "retrieval_eval_result.json").write_text(json.dumps({
        "status": "completed", "case_count": 30,
        "lexical_tests_passed": True, "hybrid_fake_tests_passed": True,
    }), encoding="utf-8")
    result = CapabilityMatrix(project)._retrieval_readiness(reports)
    assert result["status"] == "integrated"
    assert result["details"]["readiness_level"] == "hybrid_fake_verified"
    assert result["details"]["hybrid_live_verified"] is False
