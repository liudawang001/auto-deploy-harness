"""Execute local release gates and emit commit-bound JSON evidence."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tarfile
import venv
import zipfile
from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json, write_json
from auto_harness.release_evidence import build_evidence


def _run(command: List[str], cwd: Path, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd), env=env, capture_output=True, text=True)


def _counts(output: str) -> tuple[int, int, int]:
    values = {name: 0 for name in ("passed", "failed", "skipped")}
    for count, name in re.findall(r"(\d+)\s+(passed|failed|skipped)", output):
        values[name] = int(count)
    return values["passed"], values["failed"], values["skipped"]


def run_local_gates(project_root: Path, skip_tests: bool = False) -> Dict[str, object]:
    root = Path(project_root).resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results: Dict[str, object] = {}

    if not skip_tests:
        command = [sys.executable, "-m", "pytest", "-q"]
        completed = _run(command, root)
        passed, failed, skipped = _counts(completed.stdout + "\n" + completed.stderr)
        evidence = build_evidence(
            root, command, "passed" if completed.returncode == 0 else "failed",
            passed, failed if failed else int(completed.returncode != 0), skipped,
            returncode=completed.returncode,
        )
        write_json(reports / "test-summary.json", evidence)
        results["tests"] = evidence
        if completed.returncode != 0:
            return {"status": "failed", "gates": results}

    benchmark_command = [
        sys.executable, "-m", "auto_harness.cli", "benchmark", "--manifest",
        "tests/fixtures/benchmarks/manifest.json", "--output", "reports/benchmark.json",
    ]
    benchmark = _run(benchmark_command, root)
    results["benchmark"] = (
        read_json(reports / "benchmark.json")
        if (reports / "benchmark.json").exists() else {}
    )
    if benchmark.returncode != 0:
        return {"status": "failed", "gates": results}

    package_evidence = _package_smoke(root)
    write_json(reports / "package-smoke.json", package_evidence)
    results["package_smoke"] = package_evidence
    if package_evidence["status"] != "passed":
        return {"status": "failed", "gates": results}

    cli_evidence = _default_cli_smoke(root)
    write_json(reports / "default-cli-smoke.json", cli_evidence)
    results["default_cli_smoke"] = cli_evidence
    return {
        "status": "passed" if cli_evidence["status"] == "passed" else "failed",
        "gates": results,
    }


def _package_smoke(root: Path) -> Dict[str, object]:
    command = [
        sys.executable, "-m", "build", "--wheel", "--sdist",
    ]
    with tempfile.TemporaryDirectory(prefix="auto-harness-wheel-") as tmp:
        temp = Path(tmp)
        build = _run(
            command + ["--outdir", str(temp / "dist"), str(root)],
            temp,
        )
        wheels = list((temp / "dist").glob("*.whl"))
        sdists = list((temp / "dist").glob("*.tar.gz"))
        errors: List[str] = []
        if build.returncode != 0 or len(wheels) != 1:
            errors.append("wheel build failed")
        if build.returncode != 0 or len(sdists) != 1:
            errors.append("sdist build failed")
        members: List[str] = []
        sdist_members: List[str] = []
        if wheels:
            with zipfile.ZipFile(wheels[0]) as archive:
                members = archive.namelist()
            if not any(name.endswith("resources/default.json") for name in members):
                errors.append("bundled default configuration is missing")
            if len([name for name in members if "/resources/skills/" in name and name.endswith("/SKILL.md")]) != 8:
                errors.append("bundled skill set is incomplete")
            if any(name.startswith("docs/") or "/docs/" in name for name in members):
                errors.append("private docs leaked into wheel")
        try:
            if len(sdists) != 1:
                raise OSError("expected exactly one sdist")
            with tarfile.open(sdists[0], "r:gz") as archive:
                sdist_members = archive.getnames()
            if any("/docs/" in name or name.startswith("docs/") for name in sdist_members):
                errors.append("private docs leaked into sdist")
        except (OSError, RuntimeError, tarfile.TarError) as exc:
            errors.append("sdist build failed: %s" % exc)
        init_returncode = 1
        if wheels and not errors:
            env_dir = temp / "venv"
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
            python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            install = _run([str(python), "-m", "pip", "install", "--no-deps", str(wheels[0])], temp)
            workspace = temp / "workspace"
            workspace.mkdir()
            clean_env = dict(os.environ)
            clean_env.pop("PYTHONPATH", None)
            init = _run([str(python), "-m", "auto_harness.cli", "init"], workspace, clean_env)
            init_returncode = init.returncode
            if install.returncode != 0 or init.returncode != 0:
                errors.append("installed wheel init smoke failed")
            if not (workspace / "configs" / "default.json").exists():
                errors.append("init did not install default configuration")
            if len(list((workspace / "skills").glob("*/SKILL.md"))) != 8:
                errors.append("init did not install all bundled skills")
        return build_evidence(
            root, command, "passed" if not errors else "failed",
            1 if not errors else 0, 0 if not errors else 1,
            wheel_member_count=len(members), sdist_member_count=len(sdist_members),
            init_returncode=init_returncode, errors=errors,
        )


def _default_cli_smoke(root: Path) -> Dict[str, object]:
    fixture = root / "tests" / "fixtures" / "e2e" / "http_trace_echo"
    with tempfile.TemporaryDirectory(prefix="auto-harness-cli-") as tmp:
        work = Path(tmp)
        command = [
            sys.executable, "-m", "auto_harness.cli", "deploy",
            "--repo", str(fixture), "--name", "release-default-smoke", "--dry-run",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "src")
        completed = _run(command, work, env)
        states = list((work / "runs").glob("*/state.json"))
        terminal = read_json(states[0]).get("status") if len(states) == 1 else "missing"
        ok = completed.returncode == 0 and terminal == "completed_dry_run"
        return build_evidence(
            root, command, "passed" if ok else "failed", int(ok), int(not ok),
            returncode=completed.returncode, terminal_status=terminal,
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m auto_harness.release_gates")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--skip-tests", action="store_true", default=False)
    args = parser.parse_args(argv)
    result = run_local_gates(Path(args.project_root), skip_tests=args.skip_tests)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
