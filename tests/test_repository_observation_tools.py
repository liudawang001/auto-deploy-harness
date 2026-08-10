from auto_harness.agent_runtime.schemas import ToolCall
from auto_harness.tools.repository_executor import RepositoryToolExecutor


def test_read_selected_files_is_bounded_and_redacted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("TOKEN=sk-abcdefghijklmnop\nprint('ok')\n", encoding="utf-8")
    executor = RepositoryToolExecutor()
    executor.validate_contract()
    result = executor.execute(ToolCall(
        name="read_selected_files",
        input={"files": [{"path": "app.py", "start_line": 1, "end_line": 20}]},
    ), {"repo_dir": str(repo)})
    assert result.status == "passed"
    item = result.evidence["files"][0]
    assert "sk-abcdefghijklmnop" not in item["content"]
    assert "[REDACTED_SECRET]" in item["content"]
    assert item["sha256"]


def test_sensitive_and_escape_paths_are_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("SECRET=value", encoding="utf-8")
    executor = RepositoryToolExecutor()
    for path in (".env", "../outside.txt", str(tmp_path / "outside.txt")):
        result = executor.execute(ToolCall(
            name="read_selected_files",
            input={"files": [{"path": path, "start_line": 1, "end_line": 2}]},
        ), {"repo_dir": str(repo)})
        assert result.status == "rejected"


def test_symlink_escape_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (repo / "link.txt").symlink_to(outside)
    result = RepositoryToolExecutor().execute(ToolCall(
        name="read_selected_files",
        input={"files": [{"path": "link.txt", "start_line": 1, "end_line": 2}]},
    ), {"repo_dir": str(repo)})
    assert result.status == "rejected"

    search = RepositoryToolExecutor().execute(ToolCall(
        name="search_repo",
        input={"query": "secret", "path_glob": "**/*.txt"},
    ), {"repo_dir": str(repo)})
    assert search.status == "passed"
    assert search.evidence["results"] == []


def test_search_and_tree_are_bounded(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "server.py").write_text("uvicorn.run(app, port=8000)\n", encoding="utf-8")
    executor = RepositoryToolExecutor()
    search = executor.execute(ToolCall(
        name="search_repo",
        input={"query": "uvicorn.run", "path_glob": "**/*.py", "max_results": 5},
    ), {"repo_dir": str(repo)})
    assert search.status == "passed"
    assert search.evidence["results"][0]["path"] == "src/server.py"
    tree = executor.execute(ToolCall(
        name="inspect_repo_tree",
        input={"path": "src", "max_depth": 2, "max_entries": 5},
    ), {"repo_dir": str(repo)})
    assert tree.status == "passed"
    assert tree.evidence["entries"]
