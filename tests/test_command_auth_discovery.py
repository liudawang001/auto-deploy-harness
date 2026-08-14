import json
from pathlib import Path

from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
from auto_harness.command_auth import (
    CommandAuthorizationEngine,
    CommandDiscoveryService,
    CommandRegistry,
)


def _files(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def test_discovers_pep621_and_poetry_cli(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\ndemo = "demo.cli:main"\n'
        '[tool.poetry.scripts]\npaw = "paw.cli:main"\n', encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "demo serve --port 8080\npaw start\n", encoding="utf-8",
    )
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo1")
    assert {tuple(item.argv) for item in registry.candidates} == {
        (".venv/bin/demo", "serve", "--port", "8080"),
        (".venv/bin/paw", "start"),
    }
    assert {item.source_kind for item in registry.candidates} == {"pep621_script", "poetry_script"}


def test_discovers_locked_node_managers(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"serve": "node server.js"}}), encoding="utf-8",
    )
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("pnpm run serve\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo2")
    assert registry.candidates[0].argv == ["pnpm", "--dir", ".", "run", "serve"]
    assert len(registry.candidates[0].evidence_ids) == 3
    install = next(item for item in registry.candidates if item.source_kind == "node_install")
    assert install.argv == ["pnpm", "--dir", ".", "install", "--frozen-lockfile"]
    assert install.network_profile == "registry_only"


def test_discovers_make_and_repository_scripts_as_high_risk(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "serve.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("serve:\n\tpython scripts/serve.py\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("make serve\npython scripts/serve.py\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo3")
    assert {item.source_kind for item in registry.candidates} == {"make_target", "repository_script"}
    assert all(item.risk_level == "high" for item in registry.candidates)


def test_readme_only_unknown_command_is_not_candidate(tmp_path):
    (tmp_path / "README.md").write_text("mystery serve\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo4")
    assert registry.candidates == []


def test_shell_chaining_not_discovered(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\ndemo = "demo.cli:main"\n', encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("demo serve && rm -rf /\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo5")
    assert registry.candidates == []


def test_discovers_hash_bound_shebang_script(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "serve").write_text(
        "#!/usr/bin/env python3\nprint('ok')\n", encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("./scripts/serve --port 8000\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo6")
    candidate = registry.candidates[0]
    assert candidate.argv == [".venv/bin/python", "scripts/serve", "--port", "8000"]
    assert candidate.source_kind == "repository_script"


def test_unknown_common_python_entrypoint_is_discovered_as_approval_only(tmp_path):
    (tmp_path / "app.py").write_text("print('app')\n", encoding="utf-8")
    registry = CommandDiscoveryService().discover(tmp_path, _files(tmp_path), "repo7")
    candidate = registry.candidates[0]
    assert candidate.argv == [".venv/bin/python", "app.py"]
    assert candidate.source_kind == "python_entrypoint"
    decision = CommandAuthorizationEngine().authorize(candidate, registry, repo_dir=tmp_path)
    assert decision.verdict == "approval_required"


def test_snapshot_embeds_round_trippable_command_registry(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project.scripts]\ndemo = "demo:main"\n', encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("demo serve\n", encoding="utf-8")
    snapshot = ProjectSnapshotBuilder().build(tmp_path)

    registry = CommandRegistry.from_dict(snapshot["command_registry"])
    assert registry.repository_fingerprint == snapshot["repository_fingerprint"]
    assert registry.candidates[0].argv == [".venv/bin/demo", "serve"]
