import hashlib
import json

from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
from auto_harness.agent_runtime.repository_inventory import (
    rebuild_repository_inventory,
)
from auto_harness.models.base import write_json
from auto_harness.modules.reporter import ReportGenerator


def test_layered_snapshot_has_inventory_core_evidence_and_no_secret(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (repo / "src" / "server.py").write_text("TOKEN=sk-abcdefghijklmnop\nuvicorn.run(app, port=8000)\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder(
        context_mode="layered", core_budget_tokens=1000, max_tree_entries=100,
    ).build(repo, task_id="t")
    assert snapshot["schema_version"] == 2
    assert snapshot["context_mode"] == "layered"
    assert snapshot["repository_fingerprint"]
    assert snapshot["repository_inventory"]["tree"]["total_file_count"] == 2
    assert "requirements.txt" in snapshot["selected_files"]
    for item in snapshot["selected_files"].values():
        assert item["observation_id"].startswith("core_")
        assert "sk-abcdefghijklmnop" not in item["content"]


def test_layered_core_budget_is_bounded(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(20):
        (repo / ("module_%02d.py" % index)).write_text("x = 1\n" * 1000, encoding="utf-8")
    snapshot = ProjectSnapshotBuilder(
        context_mode="layered", core_budget_tokens=100, max_file_chars=6000,
    ).build(repo)
    total = sum(len(item["content"]) for item in snapshot["selected_files"].values())
    assert total <= 500


def test_layered_snapshot_uses_full_file_digest_and_excludes_sensitive_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    raw = ("print('x')\n" * 1000).encode("utf-8")
    (repo / "app.py").write_bytes(raw)
    (repo / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder(
        context_mode="layered", max_file_chars=100,
    ).build(repo)
    assert ".env" not in snapshot["file_tree"]
    assert snapshot["repository_inventory"]["excluded"]["sensitive_files"] >= 1
    assert snapshot["selected_files"]["app.py"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert snapshot["selected_files"]["app.py"]["truncated"] is True


def test_repository_fingerprint_detects_added_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("print('ok')\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder(context_mode="layered").build(repo)
    (repo / "new.py").write_text("print('new')\n", encoding="utf-8")
    current = rebuild_repository_inventory(repo, snapshot)
    assert current["repository_fingerprint"] != snapshot["repository_fingerprint"]


def test_report_includes_layered_repository_metrics(tmp_path):
    reports = tmp_path / "reports"
    turns = reports / "planner_turns"
    turns.mkdir(parents=True)
    write_json(reports / "project_snapshot.json", {"context_mode": "layered"})
    write_json(turns / "turn_001.json", {
        "context": {"estimated_input_tokens": 321},
    })
    (reports / "observation_ledger.jsonl").write_text(
        json.dumps({
            "round": 1,
            "status": "passed",
            "content_tokens": 10,
            "cache_hit": False,
            "evidence": {"files": [{"path": "app.py"}]},
        }) + "\n",
        encoding="utf-8",
    )
    result = ReportGenerator().generate(
        tmp_path,
        {"project": {"name": "demo", "repo_url": "local"}},
        {},
    )
    text = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")
    assert result.status == "passed"
    assert "## Repository Context" in text
    assert "Initial estimated input tokens: `321`" in text
