import hashlib

from auto_harness.agent_runtime.plan_policy import PlanPolicyGate


def _plan(grounding):
    return {
        "status": "ok",
        "grounding": grounding,
        "environment": {"install_commands": [["python", "-m", "pip", "install", "."]]},
        "run": {"candidates": [{"id": "app", "cmd": ["python", "app.py"], "expected_port": 8000}], "selected_candidate_id": "app"},
        "verify": {"request": {"method": "GET", "path": "/?_trace={{trace_id}}"}, "success_evidence": "response contains {{trace_id}}"},
    }


def test_layered_grounding_requires_observed_fresh_digest(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    content = b"print('ok')\n"
    (repo / "app.py").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    snapshot = {
        "context_mode": "layered",
        "repo_dir": str(repo),
        "file_tree": ["app.py"],
        "selected_files": {"app.py": {"observation_id": "core_1", "sha256": digest, "line_start": 1, "line_end": 1}},
    }
    grounding = [{"claim": "entrypoint", "file": "app.py", "reason": "starts app", "observation_id": "core_1", "sha256": digest, "line_start": 1, "line_end": 1}]
    accepted = PlanPolicyGate().validate(_plan(grounding), snapshot)
    assert accepted["allowed"] is True
    grounding[0]["sha256"] = "stale"
    rejected = PlanPolicyGate().validate(_plan(grounding), snapshot)
    assert rejected["allowed"] is False
    assert any("grounding_digest_stale" in item["reason"] for item in rejected["rejected_items"])


def test_layered_grounding_rejects_unobserved_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('ok')\n", encoding="utf-8")
    snapshot = {"context_mode": "layered", "repo_dir": str(repo), "file_tree": ["app.py"], "selected_files": {}}
    result = PlanPolicyGate().validate(_plan([{"claim": "entry", "file": "app.py", "reason": "guess", "line_start": 1, "line_end": 1, "sha256": "x"}]), snapshot)
    assert result["allowed"] is False
    assert any("grounding_not_observed" in item["reason"] for item in result["rejected_items"])
