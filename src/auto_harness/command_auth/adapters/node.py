"""npm, pnpm and yarn script discovery with lockfile binding."""

import json
import re
from pathlib import Path

from auto_harness.command_auth.adapters.common import read_text
from auto_harness.command_auth.evidence import build_evidence
from auto_harness.command_auth.schemas import CommandCandidate


LOCKFILES = {
    "npm": "package-lock.json",
    "pnpm": "pnpm-lock.yaml",
    "yarn": "yarn.lock",
}


def _documented_script(argv):
    if not argv or argv[0] not in LOCKFILES:
        return "", ""
    manager = argv[0]
    if manager in {"npm", "pnpm"} and len(argv) >= 3 and argv[1] == "run":
        return manager, argv[2]
    if manager == "yarn" and len(argv) >= 2:
        return manager, argv[2] if argv[1] == "run" and len(argv) >= 3 else argv[1]
    return "", ""


def discover_node(repo_dir, file_tree, readme, repository_fingerprint):
    file_set = set(file_tree)
    evidence = []
    candidates = []
    package_files = [
        item for item in file_tree
        if Path(item).name == "package.json" and "node_modules" not in Path(item).parts
    ]
    package_files.sort(key=lambda item: (
        0 if item in {"package.json", "src/frontend/package.json", "frontend/package.json"} else 1,
        1 if "docs" in Path(item).parts else 0,
        len(Path(item).parts),
        item,
    ))
    package_files = package_files[:50]
    for package_path in package_files:
        try:
            package = json.loads(read_text(repo_dir, package_path))
        except (TypeError, ValueError):
            continue
        scripts = package.get("scripts") if isinstance(package, dict) else {}
        if not isinstance(scripts, dict):
            continue
        directory = str(Path(package_path).parent).replace("\\", "/")
        directory = "." if directory == "." else directory
        for manager, lock_name in LOCKFILES.items():
            lock_path = str(Path(directory) / lock_name).replace("\\", "/")
            if lock_path.startswith("./"):
                lock_path = lock_path[2:]
            if lock_path not in file_set:
                continue
            manifest = build_evidence(
                repo_dir, "package_manifest", package_path,
                repository_fingerprint, declaration_key="packageManager",
                declared_value=str(package.get("packageManager", "")),
            )
            lock = build_evidence(
                repo_dir, "lockfile", lock_path, repository_fingerprint,
                declaration_key=manager,
            )
            evidence.extend([manifest, lock])
            if manager == "npm":
                install_argv = ["npm", "--prefix", directory, "ci"]
            elif manager == "pnpm":
                install_argv = ["pnpm", "--dir", directory, "install", "--frozen-lockfile"]
            else:
                package_manager = str(package.get("packageManager", ""))
                immutable = package_manager.startswith("yarn@") and not package_manager.startswith("yarn@1.")
                install_argv = ["yarn", "--cwd", directory, "install", "--immutable" if immutable else "--frozen-lockfile"]
            candidates.append(CommandCandidate.build(
                phase="install", argv=install_argv, cwd=directory,
                source_kind="node_install",
                evidence_ids=[manifest.evidence_id, lock.evidence_id],
                declared_executable=manager,
                environment_binding={"kind": "package_manager", "manager": manager, "lockfile": lock_path},
                required_backend="docker", network_profile="registry_only",
                filesystem_profile="install_workspace", risk_level="medium",
                score=0.88,
                score_reasons=["package manifest", "matching lockfile", "frozen dependency install"],
                fallback_group="install",
            ))
            makefile = read_text(repo_dir, "Makefile") if "Makefile" in file_set else ""
            if (
                manager == "npm"
                and directory in {"src/frontend", "frontend"}
                and "build" in scripts
                and re.search(r"(?m)^run_cli:\s*[^\n]*build_frontend", makefile)
            ):
                build_declaration = build_evidence(
                    repo_dir, "package_json_script", package_path,
                    repository_fingerprint, declaration_key="scripts.build",
                    declared_value=str(scripts["build"]),
                )
                make_reference = build_evidence(
                    repo_dir, "make_reference", "Makefile",
                    repository_fingerprint, declaration_key="run_cli",
                    declared_value="build_frontend",
                )
                evidence.extend([build_declaration, make_reference])
                candidates.append(CommandCandidate.build(
                    phase="install",
                    argv=["npm", "--prefix", directory, "run", "build"],
                    cwd=directory,
                    source_kind="source_build",
                    evidence_ids=[
                        build_declaration.evidence_id,
                        lock.evidence_id,
                        make_reference.evidence_id,
                    ],
                    declared_executable=manager,
                    environment_binding={
                        "kind": "package_manager",
                        "manager": manager,
                        "lockfile": lock_path,
                    },
                    required_backend="docker",
                    network_profile="none",
                    filesystem_profile="install_workspace",
                    risk_level="medium",
                    score=0.92,
                    score_reasons=[
                        "declared package build script",
                        "matching lockfile",
                        "source run target requires frontend build",
                    ],
                    fallback_group="install",
                ))
        for documented in readme:
            manager, script = _documented_script(documented["argv"])
            if not manager or script not in scripts:
                continue
            lock_path = str(Path(directory) / LOCKFILES[manager]).replace("\\", "/")
            if lock_path.startswith("./"):
                lock_path = lock_path[2:]
            if lock_path not in file_set:
                continue
            declaration = build_evidence(
                repo_dir, "package_json_script", package_path,
                repository_fingerprint, declaration_key="scripts.%s" % script,
                declared_value=str(scripts[script]),
            )
            lock = build_evidence(
                repo_dir, "lockfile", lock_path, repository_fingerprint,
                declaration_key=manager,
            )
            reference = build_evidence(
                repo_dir, "readme_reference", documented["path"],
                repository_fingerprint, line_start=documented["line"],
                line_end=documented["line"], declaration_key=script,
                declared_value=" ".join(documented["argv"]),
            )
            evidence.extend([declaration, lock, reference])
            if manager == "yarn":
                argv = ["yarn", "--cwd", directory, "run", script]
            else:
                argv = [manager, "--dir" if manager == "pnpm" else "--prefix", directory, "run", script]
            candidates.append(CommandCandidate.build(
                phase="run", argv=argv, cwd=directory,
                source_kind="package_json_script",
                evidence_ids=[declaration.evidence_id, lock.evidence_id, reference.evidence_id],
                declared_executable=manager,
                environment_binding={"kind": "package_manager", "manager": manager, "lockfile": lock_path},
                required_backend="docker", network_profile="none",
                filesystem_profile="runtime_read_only", risk_level="medium",
                score=0.9,
                score_reasons=["declared package script", "matching lockfile", "README command reference"],
                fallback_group="run",
            ))
    return evidence, candidates
