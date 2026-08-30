"""Native ecosystem deployment adapters (Phase B3).

JVM (Maven/Gradle wrapper), Go module and Cargo adapters propose run and
verify candidates for hash-pinned build outputs.  They never execute or
authorize anything; proposals still go through the unified command
authorization engine.
"""

import hashlib
from pathlib import Path

from auto_harness.deployment_adapters.builtin import BuiltinAdapter
from auto_harness.deployment_adapters.schemas import (
    AdapterDetection,
    EnvironmentProposal,
    RunProposal,
    VerifyProposal,
)


def _file_evidence(context, relatives):
    evidence_ids = []
    evidence = []
    root = Path(context.repo_dir).resolve()
    for relative in relatives:
        path = root / relative
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
            payload = resolved.read_bytes()
        except (OSError, ValueError):
            continue
        sha256 = hashlib.sha256(payload).hexdigest()
        evidence_id = "nativeev_%s" % hashlib.sha256(
            ("%s\0%s" % (relative, sha256)).encode("utf-8")
        ).hexdigest()[:20]
        evidence_ids.append(evidence_id)
        evidence.append({
            "evidence_id": evidence_id,
            "source_type": "build_metadata",
            "path": str(relative),
            "sha256": sha256,
        })
    return evidence_ids, evidence


class _NativeBuildAdapter(BuiltinAdapter):
    """Base for build-metadata driven ecosystem adapters."""

    build_files = ()
    adapter_id = ""
    ecosystem = ""

    def detect(self, context):
        file_set = set(context.files)
        present = [item for item in self.build_files if item in file_set]
        matched = len(present) == len(self.build_files)
        evidence_ids, evidence = _file_evidence(context, present)
        return AdapterDetection(
            adapter_id=self.adapter_id,
            matched=matched and bool(evidence_ids),
            confidence=0.9 if matched and evidence_ids else 0.0,
            evidence_ids=evidence_ids,
            evidence=evidence,
            reasons=[
                "build metadata %s" % name for name in present
            ] if matched else [],
        )

    def _run_proposals(self, context, detection, run_source_kinds):
        if not detection.matched:
            return []
        from auto_harness.command_auth.adapters.common import readme_commands
        from auto_harness.command_auth.adapters.native_build import discover_native_builds

        _, found_candidates, _ = discover_native_builds(
            Path(context.repo_dir),
            list(context.files),
            readme_commands(Path(context.repo_dir), list(context.files)),
            "",
        )
        proposals = []
        for item in found_candidates:
            if item.source_kind not in run_source_kinds:
                continue
            proposals.append(RunProposal(
                adapter_id=self.adapter_id,
                argv=list(item.argv),
                expected_port=int(getattr(item, "expected_port", 0) or 0),
                confidence=0.75,
                evidence_ids=list(detection.evidence_ids),
                reasons=["%s build output candidate" % self.ecosystem],
            ))
        return proposals

    def propose_environment(self, context, detection):
        if not detection.matched:
            return []
        return [EnvironmentProposal(
            adapter_id=self.adapter_id,
            backend="docker",
            confidence=0.8,
            evidence_ids=list(detection.evidence_ids),
            reasons=["%s toolchain builds in Docker" % self.ecosystem],
        )]

    def propose_verify_candidates(self, context, detection):
        if not detection.matched:
            return []
        return [VerifyProposal(
            adapter_id=self.adapter_id,
            protocol="http",
            confidence=detection.confidence,
            evidence_ids=list(detection.evidence_ids),
            reasons=["%s HTTP service trace echo" % self.ecosystem],
            verify_hint={
                "service_type": "http",
                "expected_output": "trace_echo",
                "request": {
                    "method": "GET",
                    "path": "/?_auto_harness_trace={{trace_id}}",
                },
            },
        )]


class MavenWrapperAdapter(_NativeBuildAdapter):
    adapter_id = "builtin.maven_wrapper"
    priority = 79
    ecosystem = "maven"
    build_files = ("pom.xml", "mvnw", ".mvn/wrapper/maven-wrapper.properties")
    run_source_kinds = frozenset({"jvm_artifact_run"})


class GradleWrapperAdapter(_NativeBuildAdapter):
    adapter_id = "builtin.gradle_wrapper"
    priority = 79
    ecosystem = "gradle"
    build_files = ("build.gradle", "gradlew", "gradle/wrapper/gradle-wrapper.properties")
    run_source_kinds = frozenset({"jvm_artifact_run"})

    def detect(self, context):
        detection = super().detect(context)
        if detection.matched:
            return detection
        # build.gradle.kts projects are equally valid when a wrapper exists.
        file_set = set(context.files)
        if "build.gradle.kts" in file_set and "gradlew" in file_set:
            adjusted = GradleWrapperAdapter()
            adjusted.build_files = (
                "build.gradle.kts", "gradlew",
                "gradle/wrapper/gradle-wrapper.properties",
            )
            return adjusted.detect(context)
        return detection


class GoModuleAdapter(_NativeBuildAdapter):
    adapter_id = "builtin.go_module"
    priority = 78
    ecosystem = "go"
    build_files = ("go.mod", "go.sum")
    run_source_kinds = frozenset({"go_binary_run"})


class CargoAdapter(_NativeBuildAdapter):
    adapter_id = "builtin.cargo"
    priority = 77
    ecosystem = "cargo"
    build_files = ("Cargo.toml", "Cargo.lock")
    run_source_kinds = frozenset({"cargo_binary_run"})
