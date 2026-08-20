import json

from auto_harness.evals.retrieval import RetrievalEvaluator, compare_modes
from auto_harness.retrieval.artifacts import RetrievalArtifacts
from auto_harness.modules.reporter import ReportGenerator


def test_labeled_fixture_has_thirty_cases_and_metrics_are_real():
    manifest = json.loads(
        (__import__("pathlib").Path(__file__).parent / "fixtures" / "retrieval" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    cases = manifest["cases"]
    assert len(cases) >= 30
    perfect = {item["case_id"]: item["relevant_chunk_labels"] for item in cases}
    result = RetrievalEvaluator().evaluate(cases, perfect, mode="lexical")
    assert result["case_count"] == 30
    assert result["metrics"]["recall_at_k"] == 1.0
    assert result["metrics"]["forbidden_hit_count"] == 0
    assert compare_modes(result, result)["recall_delta"] == 0.0


def test_artifacts_summarize_observable_behavior_without_false_attribution(tmp_path):
    artifacts = RetrievalArtifacts(tmp_path)
    artifacts.write_manifest({
        "completed": True, "manifest_hash": "sha256:test",
        "document_count": 2, "chunk_count": 3,
        "embedding": {"enabled": True, "provider": "fake"},
    })
    query_path = tmp_path / "retrieval" / "queries.jsonl"
    query_path.write_text(json.dumps({
        "query": {"sources": ["repository"]},
        "candidate_counts": {"returned": 2},
        "budgets": {"returned_tokens": 40},
        "latency_ms": {"total": 5},
        "degradation": {"occurred": False, "to": "hybrid", "reason": ""},
    }) + "\n", encoding="utf-8")
    summary = artifacts.finalize(requested_mode="hybrid")
    contribution = json.loads(
        (tmp_path / "reports" / "retrieval_contribution.json").read_text(encoding="utf-8")
    )
    assert summary["queries"] == 1
    assert summary["external_embedding_used"] is False
    assert contribution["rag_helped"] is False
    assert contribution["reason"] == "usage_without_baseline_delta_is_not_causal_evidence"
    ReportGenerator().generate(tmp_path, {"project": {}}, {})
    report = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    assert "## Evidence Retrieval" in report
    assert "RAG helped/required: `false` / `false`" in report
