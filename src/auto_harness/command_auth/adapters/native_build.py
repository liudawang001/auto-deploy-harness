"""JVM, Go and Rust build/run evidence discovery (Phase B3).

An ecosystem counts as supported only with the full chain: dependency
detection, reproducible build, run candidate, command policy, sandbox,
readiness, protocol verify, repair boundary and offline E2E.  The discovery
here never executes anything; wrapper/lockfile evidence is mandatory so a
missing lock or wrapper can never look like reproducible automatic
deployment.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from auto_harness.command_auth.adapters.common import read_text
from auto_harness.command_auth.adapters.entrypoint import _selectable
from auto_harness.command_auth.evidence import build_evidence
from auto_harness.command_auth.schemas import CommandCandidate


# Harness-owned build output locations. Build artifacts are only runnable
# from these controlled paths, never from arbitrary repository paths.
GO_BIN_DIR = ".harness/bin"
CARGO_TARGET_DIR = "target/release"

_SPRING_BOOT_MAVEN = re.compile(r"<groupId>\s*org\.springframework\.boot\s*</groupId>", re.IGNORECASE)
_SPRING_BOOT_GRADLE = re.compile(
    r"(?:id\s*[('\"]+org\.springframework\.boot[)'\"]+|plugins\s*\{[^}]*springframework)",
    re.IGNORECASE | re.DOTALL,
)
_JAVA_EXECUTABLE_MAIN = re.compile(r"(?m)^\s*public\s+static\s+void\s+main\s*\(")
_GO_PACKAGE_MAIN = re.compile(r"(?m)^\s*package\s+main\s*$")
_CARGO_BIN_SECTION = re.compile(r"(?m)^\s*\[\[bin\]\]\s*$")


def discover_native_builds(repo_dir, file_tree, readme, repository_fingerprint):
    """Discover JVM (Maven/Gradle wrapper), Go module and Cargo build/run candidates."""
    repo_dir = Path(repo_dir)
    file_set = set(file_tree)
    evidence: List = []
    candidates: List = []
    rejections: List[Dict] = []

    rejections.extend(_maven(repo_dir, file_set, evidence, candidates, repository_fingerprint))
    rejections.extend(_gradle(repo_dir, file_set, evidence, candidates, repository_fingerprint))
    rejections.extend(_go(repo_dir, file_set, evidence, candidates, repository_fingerprint))
    rejections.extend(_cargo(repo_dir, file_set, evidence, candidates, repository_fingerprint))
    return evidence, candidates, rejections


def _maven(repo_dir, file_set, evidence, candidates, repository_fingerprint):
    rejections = []
    if "pom.xml" not in file_set:
        return rejections
    pom_text = read_text(repo_dir, "pom.xml")
    if not pom_text:
        return rejections
    pom = build_evidence(
        repo_dir, "jvm_maven_pom", "pom.xml", repository_fingerprint,
        declaration_key="project", declared_value="maven",
    )
    wrapper_properties = ".mvn/wrapper/maven-wrapper.properties"
    wrapper_files = [item for item in file_set if item.startswith(".mvn/wrapper/")]
    if "mvnw" not in file_set or wrapper_properties not in file_set:
        # A pom alone is not reproducible: no wrapper, no automatic build.
        evidence.append(pom)
        rejections.append({
            "source_type": "jvm_maven_build",
            "path": "pom.xml",
            "reason_code": "maven_wrapper_missing",
            "declared_value": "mvnw package",
        })
        return rejections
    wrapper = build_evidence(
        repo_dir, "maven_wrapper", wrapper_properties, repository_fingerprint,
        declaration_key="distribution", declared_value=wrapper_properties,
    )
    jar_hash = ""
    wrapper_jar = next(
        (item for item in wrapper_files if item.endswith(".jar")), "",
    )
    if wrapper_jar:
        jar_hash = build_evidence(
            repo_dir, "maven_wrapper", wrapper_jar, repository_fingerprint,
            declaration_key="wrapper_jar", declared_value=wrapper_jar,
        ).evidence_id
    spring_boot = bool(_SPRING_BOOT_MAVEN.search(pom_text))
    evidence.extend([pom, wrapper])
    build_evidence_ids = [
        pom.evidence_id, wrapper.evidence_id,
        *( [jar_hash] if jar_hash else [] ),
    ]
    reasons = ["Maven wrapper build", "pinned wrapper distribution"]
    if spring_boot:
        reasons.append("Spring Boot plugin declared")
    candidates.append(CommandCandidate.build(
        phase="install",
        argv=["./mvnw", "-DskipTests", "package"],
        source_kind="jvm_maven_build",
        evidence_ids=build_evidence_ids,
        declared_executable="mvnw",
        environment_binding={"kind": "jvm_wrapper_build", "ecosystem": "maven"},
        required_backend="docker",
        network_profile="registry_only",
        filesystem_profile="install_workspace",
        risk_level="medium",
        score=0.9,
        score_reasons=reasons,
        fallback_group="install",
    ))
    _artifact_run_candidate(
        repo_dir, file_set, evidence, candidates, repository_fingerprint,
        pattern="target/*.jar", source_kind="jvm_artifact_run",
        artifact_type="jvm_build_artifact", ecosystem="maven",
        interpreter="java", interpreter_args=["-jar"],
    )
    return rejections


def _gradle(repo_dir, file_set, evidence, candidates, repository_fingerprint):
    rejections = []
    build_file = next(
        (
            item for item in ("build.gradle", "build.gradle.kts")
            if item in file_set
        ),
        "",
    )
    if not build_file:
        return rejections
    build_text = read_text(repo_dir, build_file)
    if not build_text:
        return rejections
    if "gradlew" not in file_set or "gradle/wrapper/gradle-wrapper.properties" not in file_set:
        evidence.append(build_evidence(
            repo_dir, "jvm_gradle_build", build_file, repository_fingerprint,
            declaration_key="project", declared_value="gradle",
        ))
        rejections.append({
            "source_type": "jvm_gradle_build",
            "path": build_file,
            "reason_code": "gradle_wrapper_missing",
            "declared_value": "gradlew bootJar",
        })
        return rejections
    manifest = build_evidence(
        repo_dir, "jvm_gradle_build", build_file, repository_fingerprint,
        declaration_key="project", declared_value="gradle",
    )
    wrapper = build_evidence(
        repo_dir, "gradle_wrapper", "gradle/wrapper/gradle-wrapper.properties",
        repository_fingerprint, declaration_key="distribution",
        declared_value="gradle/wrapper/gradle-wrapper.properties",
    )
    spring_boot = bool(_SPRING_BOOT_GRADLE.search(build_text))
    evidence.extend([manifest, wrapper])
    reasons = ["Gradle wrapper build", "pinned wrapper distribution"]
    if spring_boot:
        reasons.append("Spring Boot plugin declared")
    candidates.append(CommandCandidate.build(
        phase="install",
        argv=["./gradlew", "bootJar", "--no-daemon"]
        if spring_boot else ["./gradlew", "build", "--no-daemon"],
        source_kind="jvm_gradle_build",
        evidence_ids=[manifest.evidence_id, wrapper.evidence_id],
        declared_executable="gradlew",
        environment_binding={"kind": "jvm_wrapper_build", "ecosystem": "gradle"},
        required_backend="docker",
        network_profile="registry_only",
        filesystem_profile="install_workspace",
        risk_level="medium",
        score=0.9,
        score_reasons=reasons,
        fallback_group="install",
    ))
    _artifact_run_candidate(
        repo_dir, file_set, evidence, candidates, repository_fingerprint,
        pattern="build/libs/*.jar", source_kind="jvm_artifact_run",
        artifact_type="jvm_build_artifact", ecosystem="gradle",
        interpreter="java", interpreter_args=["-jar"],
    )
    return rejections


def _go(repo_dir, file_set, evidence, candidates, repository_fingerprint):
    rejections = []
    if "go.mod" not in file_set:
        return rejections
    go_mod_text = read_text(repo_dir, "go.mod")
    if not go_mod_text:
        return rejections
    module = build_evidence(
        repo_dir, "go_module", "go.mod", repository_fingerprint,
        declaration_key="module",
        declared_value=_go_module_name(go_mod_text),
    )
    if "go.sum" not in file_set:
        evidence.append(module)
        rejections.append({
            "source_type": "go_build",
            "path": "go.mod",
            "reason_code": "go_lockfile_missing",
            "declared_value": "go build",
        })
        return rejections
    lock = build_evidence(
        repo_dir, "go_lockfile", "go.sum", repository_fingerprint,
        declaration_key="go_sum", declared_value="go.sum",
    )
    main_evidence_file, main_package = _go_main_package(repo_dir, file_set)
    if not main_package:
        evidence.extend([module, lock])
        rejections.append({
            "source_type": "go_build",
            "path": "go.mod",
            "reason_code": "go_main_package_missing",
            "declared_value": "package main",
        })
        return rejections
    main_evidence = build_evidence(
        repo_dir, "go_main_package", main_evidence_file, repository_fingerprint,
        declaration_key="package", declared_value="main",
    )
    binary_name = _go_module_name(go_mod_text).rsplit("/", 1)[-1] or "service"
    output_path = "%s/%s" % (GO_BIN_DIR, binary_name)
    evidence.extend([module, lock, main_evidence])
    candidates.append(CommandCandidate.build(
        phase="install",
        argv=["go", "build", "-mod=readonly", "-o", output_path, "./%s" % main_package]
        if main_package != "." else ["go", "build", "-mod=readonly", "-o", output_path, "."],
        source_kind="go_build",
        evidence_ids=[module.evidence_id, lock.evidence_id, main_evidence.evidence_id],
        declared_executable="go",
        environment_binding={"kind": "go_module_build", "output": output_path},
        required_backend="docker",
        network_profile="registry_only",
        filesystem_profile="install_workspace",
        risk_level="medium",
        score=0.9,
        score_reasons=["Go module build", "pinned go.sum", "deterministic main package"],
        fallback_group="install",
    ))
    _binary_run_candidate(
        repo_dir, evidence, candidates, repository_fingerprint,
        output_path=output_path, source_kind="go_binary_run",
        artifact_type="go_build_artifact", ecosystem="go",
    )
    return rejections


def _cargo(repo_dir, file_set, evidence, candidates, repository_fingerprint):
    rejections = []
    if "Cargo.toml" not in file_set:
        return rejections
    toml_text = read_text(repo_dir, "Cargo.toml")
    if not toml_text:
        return rejections
    manifest = build_evidence(
        repo_dir, "cargo_manifest", "Cargo.toml", repository_fingerprint,
        declaration_key="package", declared_value=_cargo_package_name(toml_text),
    )
    if "Cargo.lock" not in file_set:
        evidence.append(manifest)
        rejections.append({
            "source_type": "cargo_build",
            "path": "Cargo.toml",
            "reason_code": "cargo_lockfile_missing",
            "declared_value": "cargo build --locked",
        })
        return rejections
    lock = build_evidence(
        repo_dir, "cargo_lockfile", "Cargo.lock", repository_fingerprint,
        declaration_key="cargo_lock", declared_value="Cargo.lock",
    )
    bin_name = _cargo_bin_name(toml_text, file_set)
    if not bin_name:
        evidence.extend([manifest, lock])
        rejections.append({
            "source_type": "cargo_build",
            "path": "Cargo.toml",
            "reason_code": "cargo_bin_target_missing",
            "declared_value": "[[bin]] or src/main.rs",
        })
        return rejections
    bin_evidence = build_evidence(
        repo_dir, "cargo_bin_target", "Cargo.toml", repository_fingerprint,
        declaration_key="bin", declared_value=bin_name,
    )
    evidence.extend([manifest, lock, bin_evidence])
    candidates.append(CommandCandidate.build(
        phase="install",
        argv=["cargo", "build", "--locked", "--release", "--bin", bin_name],
        source_kind="cargo_build",
        evidence_ids=[manifest.evidence_id, lock.evidence_id, bin_evidence.evidence_id],
        declared_executable="cargo",
        environment_binding={"kind": "cargo_locked_build", "bin": bin_name},
        required_backend="docker",
        network_profile="registry_only",
        filesystem_profile="install_workspace",
        risk_level="medium",
        score=0.9,
        score_reasons=["Cargo locked build", "pinned Cargo.lock", "declared bin target"],
        fallback_group="install",
    ))
    _binary_run_candidate(
        repo_dir, evidence, candidates, repository_fingerprint,
        output_path="%s/%s" % (CARGO_TARGET_DIR, bin_name),
        source_kind="cargo_binary_run",
        artifact_type="cargo_build_artifact", ecosystem="rust",
    )
    return rejections


def _artifact_run_candidate(
    repo_dir, file_set, evidence, candidates, repository_fingerprint,
    *, pattern, source_kind, artifact_type, ecosystem, interpreter, interpreter_args,
):
    """Run candidate bound to a hash-pinned, already-built artifact."""
    prefix, suffix = pattern.split("*")
    artifacts = sorted(
        item for item in file_set
        if item.startswith(prefix) and item.endswith(suffix) and _artifact_selectable(item)
    )
    if not artifacts:
        # No build output yet: the run candidate only appears after an
        # authorized build produced it; report the gap explicitly.
        return
    artifact_rel = artifacts[0]
    artifact_evidence = build_evidence(
        repo_dir, artifact_type, artifact_rel, repository_fingerprint,
        declaration_key="artifact", declared_value=artifact_rel,
    )
    evidence.append(artifact_evidence)
    candidates.append(CommandCandidate.build(
        phase="run",
        argv=[interpreter, *interpreter_args, artifact_rel],
        source_kind=source_kind,
        evidence_ids=[artifact_evidence.evidence_id],
        declared_executable=interpreter,
        environment_binding={
            "kind": "built_artifact_run",
            "ecosystem": ecosystem,
            "artifact": artifact_rel,
        },
        required_backend="docker",
        network_profile="none",
        filesystem_profile="runtime_read_only",
        risk_level="high",
        score=0.82,
        score_reasons=[
            "hash-pinned build artifact",
            "build and run backend separated",
        ],
        fallback_group="run",
    ))


def _binary_run_candidate(
    repo_dir, evidence, candidates, repository_fingerprint,
    *, output_path, source_kind, artifact_type, ecosystem,
):
    artifact_rel = output_path
    root = Path(repo_dir)
    path = root / artifact_rel
    if not path.is_file() or path.is_symlink():
        # No build output yet; the run candidate appears only after an
        # authorized build into the Harness-owned path.
        return
    artifact_evidence = build_evidence(
        repo_dir, artifact_type, artifact_rel, repository_fingerprint,
        declaration_key="artifact", declared_value=artifact_rel,
    )
    evidence.append(artifact_evidence)
    candidates.append(CommandCandidate.build(
        phase="run",
        argv=["./%s" % artifact_rel],
        source_kind=source_kind,
        evidence_ids=[artifact_evidence.evidence_id],
        declared_executable=artifact_rel,
        environment_binding={
            "kind": "built_artifact_run",
            "ecosystem": ecosystem,
            "artifact": artifact_rel,
        },
        required_backend="docker",
        network_profile="none",
        filesystem_profile="runtime_read_only",
        risk_level="high",
        score=0.82,
        score_reasons=[
            "hash-pinned build artifact in Harness-owned path",
            "build and run backend separated",
        ],
        fallback_group="run",
    ))


def _artifact_selectable(relative: str) -> bool:
    """Artifact output paths legitimately live under build/dist directories."""
    parts = Path(relative).parts
    return not any(
        part in {"node_modules", ".venv", "tests", "test", "docs", "examples"}
        for part in parts[:-1]
    )


def _go_module_name(text: str) -> str:
    match = re.search(r"(?m)^\s*module\s+(\S+)", text)
    return match.group(1) if match else ""


def _go_main_package(repo_dir, file_set) -> Tuple[str, str]:
    """Return (evidence_file, package_dir) for the deterministic main package."""
    root = Path(repo_dir)
    for relative in sorted(file_set):
        if not relative.endswith(".go") or not _selectable(relative):
            continue
        text = read_text(repo_dir, relative)
        if text and _GO_PACKAGE_MAIN.search(text) and _go_has_main_func(text):
            package_dir = str(Path(relative).parent).replace("\\", "/")
            return relative, ("." if package_dir == "." else package_dir)
    return "", ""


def _go_has_main_func(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*func\s+main\s*\(", text))


def _cargo_package_name(text: str) -> str:
    match = re.search(
        r"(?m)^\s*\[package\]\s*(?:[^\[]*?)^\s*name\s*=\s*\"([^\"]+)\"",
        text,
        re.DOTALL,
    )
    return match.group(1) if match else ""


def _cargo_bin_name(text: str, file_set) -> str:
    if _CARGO_BIN_SECTION.search(text):
        section_match = re.search(
            r"\[\[bin\]\]([^\[]*)", text, re.DOTALL,
        )
        body = section_match.group(1) if section_match else ""
        name_match = re.search(r"name\s*=\s*\"([^\"]+)\"", body)
        if name_match:
            return name_match.group(1)
        return ""
    if "src/main.rs" in set(file_set):
        return _cargo_package_name(text) or "main"
    return ""
