"""Foundation regression tests for framework-independent deployment.

Phase A0 intentionally records existing behaviour before the capability model,
deployment contract, adapter registry, and protocol verifier are introduced.
"""

from datetime import datetime, timedelta, timezone
import json

import pytest

from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
from auto_harness.command_auth import CommandAuthorizationEngine, CommandRegistry
from auto_harness.command_auth.approval import build_command_approval_request
from auto_harness.capabilities import CapabilityDetector
from auto_harness.config import HarnessConfig
from auto_harness.deployment_adapters import (
    CandidateComposer,
    DeploymentAdapterRegistry,
    DetectionContext,
    RunProposal,
)
from auto_harness.deployment_contract import DeploymentContractParser
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.reporter import ReportGenerator
from auto_harness.modules.verify import VerifyModule
from auto_harness.verify.protocols import ProbeEvidence, ProtocolVerifierRegistry


def _write_contract(root, *, port=8123, extra=""):
    (root / "auto-deploy.yaml").write_text(
        """schema_version: 1
project:
  workload_type: service
  runtime_family: python
environment:
  backend: venv
  python: \"3.11\"
  dependency_files: [requirements.txt]
  install_commands:
    - [python3, -m, venv, .venv]
    - [.venv/bin/python, -m, pip, install, -r, requirements.txt]
service:
  command: [.venv/bin/python, app.py]
  cwd: .
  host: 0.0.0.0
  port: %d
  startup_timeout_seconds: 60
  required_env_names: []
verify:
  protocol: http
  request:
    method: GET
    path: /health?trace={{trace_id}}
  success:
    response_contains: \"{{trace_id}}\"
  timeout_seconds: 20
security:
  required_backend: docker
  network_profile: none
  allow_source_edit: false
%s""" % (port, extra),
        encoding="utf-8",
    )


def _baseline_status(analysis):
    frameworks = set(analysis.get("frameworks") or [])
    run_candidates = analysis.get("run_candidates") or []
    verify_hint = analysis.get("verify_hint") or {}
    service_type = str(verify_hint.get("service_type") or "unknown")
    return {
        "recognized": frameworks != {"unknown"},
        "startable": bool(run_candidates),
        "verifiable": service_type != "unknown",
        "verified": False,
    }


def test_baseline_unknown_python_can_be_startable_without_being_recognized(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('service')\n", encoding="utf-8")

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["frameworks"] == ["unknown"]
    assert analysis["run_candidates"][0]["cmd"] == [".venv/bin/python", "app.py"]
    assert analysis["verify_hint"]["service_type"] == "unknown"
    assert _baseline_status(analysis) == {
        "recognized": False,
        "startable": True,
        "verifiable": False,
        "verified": False,
    }


def test_baseline_unknown_project_without_entrypoint_has_explicit_gap(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["frameworks"] == ["unknown"]
    assert analysis["run_candidates"] == []
    assert _baseline_status(analysis)["startable"] is False


def test_baseline_node_installs_but_has_no_deterministic_runner(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"scripts":{"start":"node server.js"}}', encoding="utf-8",
    )
    (tmp_path / "server.js").write_text("console.log('service')\n", encoding="utf-8")

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["frameworks"] == ["node"]
    assert analysis["install_plan"] == [["npm", "install"]]
    assert analysis["run_candidates"] == []


def test_baseline_framework_signal_does_not_prove_deployment(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\ntorch\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("FastAPI inference service\n", encoding="utf-8")

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["frameworks"] == ["fastapi", "torch"]
    assert analysis["run_candidates"] == []
    assert _baseline_status(analysis) == {
        "recognized": True,
        "startable": False,
        "verifiable": True,
        "verified": False,
    }


def test_baseline_readiness_and_http_200_are_not_strong_evidence():
    verifier = VerifyModule()
    service = {"process_alive": True, "port_ready": True}

    assert verifier._can_pass(service, []) is False
    assert verifier._can_pass(
        service,
        [{
            "name": "http_trace_response",
            "status": "uncertain",
            "evidence": "HTTP 200",
        }],
    ) is False


def test_baseline_current_trace_check_is_strong_evidence():
    verifier = VerifyModule()
    service = {"process_alive": True, "port_ready": True}

    assert verifier._can_pass(
        service,
        [{
            "name": "http_trace_response",
            "status": "pass",
            "evidence": "trace_id=verify_current",
        }],
    ) is True


def test_capabilities_separate_service_frameworks_from_ml_libraries(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "fastapi==0.116.0\ntorch>=2\ntransformers[torch]\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from fastapi import FastAPI\nimport torch\n",
        encoding="utf-8",
    )

    analysis = ProjectAnalyzer().analyze(tmp_path).data
    capabilities = analysis["capabilities"]

    assert analysis["schema_version"] == 2
    assert capabilities["service_frameworks"] == ["fastapi"]
    assert capabilities["ml_libraries"] == ["torch", "transformers"]
    assert capabilities["languages"] == ["python"]
    assert capabilities["package_ecosystems"] == ["pip"]
    assert capabilities["protocols"] == ["http", "openapi"]
    assert analysis["frameworks"] == ["fastapi", "torch", "transformers"]


def test_capability_detection_does_not_promote_readme_competitor_text(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "This project does not use Gradio; Gradio is only a comparison.\n",
        encoding="utf-8",
    )

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    # The legacy field remains unchanged in Phase A1, while the new structured
    # capability model avoids treating prose as a declared dependency/import.
    assert "gradio" in analysis["frameworks"]
    assert analysis["capabilities"]["ui_frameworks"] == []
    gradio_detection = next(
        item for item in analysis["adapter_detections"]
        if item["adapter_id"] == "builtin.gradio"
    )
    assert gradio_detection["evidence_ids"]
    assert gradio_detection["evidence"][0]["path"] == "README.md"


def test_capability_evidence_is_hash_bound_and_serializable(tmp_path):
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")

    analysis = ProjectAnalyzer().analyze(tmp_path).data
    evidence = analysis["capability_evidence"]

    flask = next(item for item in evidence if item["capability_value"] == "flask")
    assert flask["source_type"] == "dependency"
    assert flask["path"] == "requirements.txt"
    assert len(flask["sha256"]) == 64
    assert flask["evidence_id"].startswith("cap_")


def test_invalid_pyproject_records_parse_failure_without_breaking_analysis(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project\ndependencies = [\"fastapi\"]\n", encoding="utf-8",
    )

    analysis = ProjectAnalyzer().analyze(tmp_path).data
    manifest = analysis["dependency_manifests"][0]

    assert manifest["path"] == "pyproject.toml"
    assert manifest["status"] == "parse_failed"
    assert manifest["reason_code"].startswith("invalid_toml:")
    assert analysis["legacy_compatibility"]["compiled"] is True


def test_deployability_reports_entrypoint_and_verify_gaps(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["deployability"] == {
        "status": "partial",
        "selected_candidate_id": "",
        "candidate_ids": [],
        "missing_capabilities": ["run.entrypoint", "verify.strong_evidence"],
        "risk_reasons": [],
        "next_resolution": "contract_required",
    }


def test_unknown_project_contract_compiles_to_evidence_bound_candidate(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('custom service')\n", encoding="utf-8")
    _write_contract(tmp_path, port=9123)

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["frameworks"] == ["unknown"]
    assert analysis["deployment_contract"]["valid"] is True
    assert analysis["run_candidates"][0]["expected_port"] == 9123
    assert analysis["run_candidates"][0]["selected_by"] == "deployment_contract"
    assert analysis["deployability"]["status"] == "ready"
    assert analysis["deployment_candidates"][0]["source"] == "manifest"
    registry = CommandRegistry.from_dict(analysis["command_registry"])
    manifest_commands = [
        item for item in registry.candidates if item.source_kind == "manifest_command"
    ]
    assert {item.phase for item in manifest_commands} == {"install", "run"}
    assert all(item.required_backend == "docker" for item in manifest_commands)
    assert all(item.network_profile == "none" for item in manifest_commands)


def test_snapshot_binds_contract_and_commands_to_repository_fingerprint(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('service')\n", encoding="utf-8")
    _write_contract(tmp_path)

    snapshot = ProjectSnapshotBuilder().build(tmp_path)

    assert snapshot["deployment_contract"]["valid"] is True
    assert snapshot["deployment_candidates"][0]["source"] == "manifest"
    assert any(
        item["source_kind"] == "manifest_command"
        for item in snapshot["command_registry"]["candidates"]
    )
    assert snapshot["repository_inventory"]["manifest_sha256"]["auto-deploy.yaml"]


@pytest.mark.parametrize(
    ("contract", "reason_code"),
    [
        (
            """schema_version: 1
service: {command: python app.py, port: 8000}
verify: {protocol: http, request: {method: GET, path: '/?trace={{trace_id}}'}, success: {response_contains: '{{trace_id}}'}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "service_command_invalid",
        ),
        (
            """schema_version: 1
environment: {dependency_files: ['../requirements.txt']}
service: {command: [python3, app.py], port: 8000}
verify: {protocol: http, request: {method: GET, path: '/?trace={{trace_id}}'}, success: {response_contains: '{{trace_id}}'}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "dependency_path_invalid",
        ),
        (
            """schema_version: 1
service: {command: [python3, app.py], port: 8000}
verify: {protocol: http, request: {method: GET, path: 'https://example.com/?trace={{trace_id}}'}, success: {response_contains: '{{trace_id}}'}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "verify_path_invalid",
        ),
        (
            """schema_version: 1
service: {command: [python3, app.py], port: 8000}
verify: {protocol: http, request: {method: GET, path: '/health'}, success: {response_contains: ok}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "verify_trace_required",
        ),
        (
            """schema_version: 1
project: {api_key: plaintext-secret}
service: {command: [python3, app.py], port: 8000}
verify: {protocol: http, request: {method: GET, path: '/?trace={{trace_id}}'}, success: {response_contains: '{{trace_id}}'}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "secret_value_not_allowed:project.api_key",
        ),
        (
            """schema_version: 1
environment: {dependency_files: requirements.txt}
service: {command: [python3, app.py], port: 8000}
verify: {protocol: http, request: {method: GET, path: '/?trace={{trace_id}}'}, success: {response_contains: '{{trace_id}}'}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "dependency_files_schema_invalid",
        ),
        (
            """schema_version: 1
service: {command: [/usr/bin/python3, app.py], port: 8000}
verify: {protocol: http, request: {method: GET, path: '/?trace={{trace_id}}'}, success: {response_contains: '{{trace_id}}'}}
security: {required_backend: docker, network_profile: none, allow_source_edit: false}
""",
            "manifest_absolute_path_not_allowed",
        ),
    ],
)
def test_contract_validator_fails_closed(tmp_path, contract, reason_code):
    (tmp_path / "auto-deploy.yaml").write_text(contract, encoding="utf-8")

    result = DeploymentContractParser().parse_repo(tmp_path)

    assert result == {
        "found": True,
        "valid": False,
        "path": "auto-deploy.yaml",
        "reason_code": reason_code,
    }
    analysis = ProjectAnalyzer().analyze(tmp_path).data
    assert analysis["deployability"]["status"] == "blocked"
    assert analysis["deployability"]["next_resolution"] == "fix_contract"


def test_manifest_command_requires_bound_one_shot_approval(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('service')\n", encoding="utf-8")
    _write_contract(tmp_path)
    analysis = ProjectAnalyzer().analyze(tmp_path).data
    registry = CommandRegistry.from_dict(analysis["command_registry"])
    candidate = next(
        item for item in registry.candidates
        if item.source_kind == "manifest_command" and item.phase == "run"
    )
    engine = CommandAuthorizationEngine()

    pending = engine.authorize(
        candidate,
        registry,
        repo_dir=tmp_path,
        sandbox_policy_fingerprint="box",
    )
    assert pending.verdict == "approval_required"
    assert pending.reason_code == "manifest_command_requires_approval"

    evidence = [registry.evidence_by_id()[item] for item in candidate.evidence_ids]
    request = build_command_approval_request(
        candidate,
        registry.repository_fingerprint,
        evidence,
        "box",
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    approval = {
        **request,
        "decision": "approve",
        "request_hash": request["request_hash"],
        "approval_id": request["approval_id"],
        "operation_id": request["operation_id"],
    }
    allowed = engine.authorize(
        candidate,
        registry,
        repo_dir=tmp_path,
        sandbox_policy_fingerprint="box",
        approval=approval,
    )
    assert allowed.verdict == "auto_allowed"


def test_manifest_change_invalidates_old_candidate_and_approval(tmp_path):
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('service')\n", encoding="utf-8")
    _write_contract(tmp_path, port=8123)
    analysis = ProjectAnalyzer().analyze(tmp_path).data
    registry = CommandRegistry.from_dict(analysis["command_registry"])
    candidate = next(
        item for item in registry.candidates
        if item.source_kind == "manifest_command" and item.phase == "run"
    )
    evidence = [registry.evidence_by_id()[item] for item in candidate.evidence_ids]
    request = build_command_approval_request(
        candidate,
        registry.repository_fingerprint,
        evidence,
        "box",
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    approval = {
        **request,
        "decision": "approve",
        "request_hash": request["request_hash"],
        "approval_id": request["approval_id"],
        "operation_id": request["operation_id"],
    }

    _write_contract(tmp_path, port=8124)
    decision = CommandAuthorizationEngine().authorize(
        candidate,
        registry,
        repo_dir=tmp_path,
        sandbox_policy_fingerprint="box",
        approval=approval,
    )

    assert decision.verdict == "candidate_rejected"
    assert decision.reason_code == "evidence_hash_mismatch"


def test_builtin_adapter_registry_has_stable_priority_order():
    registry = DeploymentAdapterRegistry.builtins()

    assert [item.adapter_id for item in registry.all()] == [
        "builtin.generic_python",
        "builtin.gradio",
        "builtin.streamlit",
        "builtin.fastapi",
        "builtin.flask",
        "builtin.django",
        "builtin.stdlib_http",
        "builtin.gradle_wrapper",
        "builtin.maven_wrapper",
        "builtin.go_module",
        "builtin.cargo",
        "builtin.vllm",
        "builtin.openai_compatible",
        "builtin.node_package",
    ]


def test_fastapi_adapter_detection_is_evidence_backed(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    capabilities, _ = CapabilityDetector().detect(tmp_path, ["requirements.txt"])
    context = DetectionContext(
        repo_dir=tmp_path,
        files=("requirements.txt",),
        capabilities=capabilities,
        legacy_frameworks=("fastapi",),
    )

    detections = DeploymentAdapterRegistry.builtins().detect_all(context)
    fastapi = next(item for item in detections if item.adapter_id == "builtin.fastapi")

    assert fastapi.matched is True
    assert fastapi.evidence_ids
    assert fastapi.confidence == 0.9


def test_multi_adapter_analysis_keeps_ml_and_service_capabilities(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "fastapi\ntorch\ntransformers\n", encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "from fastapi import FastAPI\nimport torch\n", encoding="utf-8",
    )

    analysis = ProjectAnalyzer().analyze(tmp_path).data

    assert analysis["capabilities"]["service_frameworks"] == ["fastapi"]
    assert analysis["capabilities"]["ml_libraries"] == ["torch", "transformers"]
    assert {item["adapter_id"] for item in analysis["adapter_detections"]} >= {
        "builtin.generic_python", "builtin.fastapi",
    }
    assert analysis["deployment_candidates"][0]["adapter_ids"] == [
        "builtin.generic_python", "builtin.fastapi",
    ]


def test_candidate_composer_deduplicates_argv_and_exposes_port_conflict():
    proposals = [
        RunProposal(
            adapter_id="a",
            argv=["python3", "app.py"],
            expected_port=8000,
            confidence=0.7,
            evidence_ids=["ev_a"],
            reasons=["first"],
        ),
        RunProposal(
            adapter_id="b",
            argv=["python3", "app.py"],
            expected_port=9000,
            confidence=0.8,
            evidence_ids=["ev_b"],
            reasons=["second"],
        ),
    ]

    merged = CandidateComposer().merge_run_proposals(proposals)
    deployments = CandidateComposer().compose(merged, [], [])

    assert len(merged) == 1
    assert merged[0].expected_port == 0
    assert merged[0].evidence_ids == ["ev_a", "ev_b"]
    assert "conflicting expected ports" in merged[0].reasons
    assert deployments[0].missing_capabilities == [
        "run.expected_port", "verify.strong_evidence",
    ]


def test_adapter_proposal_cannot_change_command_policy_verdict():
    proposal = RunProposal(
        adapter_id="untrusted.test",
        argv=["bash", "-c", "echo unsafe"],
        expected_port=8000,
        confidence=1.0,
    )

    decision = CommandAuthorizationEngine().authorize_argv(proposal.argv)

    assert decision["verdict"] == "hard_denied"
    assert decision["reason_code"] == "shell_wrapper_hard_denied"


def test_adapter_registry_rejects_duplicate_ids():
    class DuplicateAdapter:
        adapter_id = "duplicate"
        priority = 1

    registry = DeploymentAdapterRegistry([DuplicateAdapter()])

    with pytest.raises(ValueError, match="duplicate_adapter_id:duplicate"):
        registry.register(DuplicateAdapter())


def test_protocol_verifier_registry_has_stable_order():
    registry = ProtocolVerifierRegistry.builtins()

    assert [item.verifier_id for item in registry.all()] == [
        "builtin.http_trace",
        "builtin.openapi_trace",
        "builtin.openai_compatible",
        "builtin.gradio",
        "builtin.streamlit_browser",
        "builtin.browser_dom_trace",
    ]


def test_manifest_protocol_has_priority_over_framework_default():
    verifier, selection = ProtocolVerifierRegistry.builtins().select({
        "frameworks": ["gradio"],
        "verify_hint": {"service_type": "webui"},
        "deployment_contract": {
            "valid": True,
            "verify": {"protocol": "openapi"},
        },
    })

    assert verifier.verifier_id == "builtin.openapi_trace"
    assert selection.source == "deployment_contract"
    assert selection.protocol == "openapi"


def _probe_evidence(**overrides):
    values = {
        "verifier_id": "builtin.http_trace",
        "protocol": "http",
        "trace_id": "trace_current",
        "endpoint": "http://127.0.0.1:8123",
        "expected_port": 8123,
        "process_alive": True,
        "port_ready": True,
        "status": "pass",
        "trace_observed": True,
    }
    values.update(overrides)
    return ProbeEvidence(**values)


def test_protocol_verifier_accepts_only_current_bound_trace():
    verifier, _ = ProtocolVerifierRegistry.builtins().select({})

    passed = verifier.evaluate(
        _probe_evidence(), {"trace_id": "trace_current"},
    )
    stale = verifier.evaluate(
        _probe_evidence(trace_id="trace_old"), {"trace_id": "trace_current"},
    )
    bare_200 = verifier.evaluate(
        _probe_evidence(trace_observed=False), {"trace_id": "trace_current"},
    )

    assert passed.status == "passed"
    assert passed.strong_evidence is True
    assert stale.reason_code == "stale_trace_rejected"
    assert bare_200.reason_code == "current_trace_not_observed"


@pytest.mark.parametrize(
    ("overrides", "reason_code"),
    [
        ({"endpoint": "https://example.com:8123"}, "external_endpoint_rejected"),
        ({"endpoint": "ftp://127.0.0.1:8123"}, "endpoint_scheme_rejected"),
        ({"endpoint": "http://127.0.0.1:9000"}, "service_port_mismatch"),
        ({"process_alive": False}, "service_process_not_alive"),
        ({"port_ready": False}, "service_port_not_ready"),
    ],
)
def test_protocol_verifier_rejects_unbound_service_evidence(overrides, reason_code):
    verifier, _ = ProtocolVerifierRegistry.builtins().select({})

    decision = verifier.evaluate(
        _probe_evidence(**overrides), {"trace_id": "trace_current"},
    )

    assert decision.strong_evidence is False
    assert decision.reason_code == reason_code


def test_verify_writes_protocol_selection_artifact(tmp_path):
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
        body = (request.data or b"").decode("utf-8", errors="ignore")
        return Response("%s %s" % (request.full_url, body))

    repo = tmp_path / "workspace" / "repo"
    repo.mkdir(parents=True)
    result = VerifyModule(urlopen=echo_urlopen).verify(
        tmp_path,
        {
            "frameworks": ["unknown"],
            "verify_hint": {
                "service_type": "http",
                "request": {
                    "method": "GET",
                    "path": "/health?trace={{trace_id}}",
                },
            },
        },
        {"pid": 123, "expected_port": 8123, "service_ready": True},
    )

    selection = result.data["protocol_verify_selection"]
    assert result.status == "passed"
    assert selection["verifier_id"] == "builtin.http_trace"
    assert selection["shadow_decision"]["strong_evidence"] is True
    assert (tmp_path / "evidence" / "protocol_verify_selection.json").is_file()


def test_deployment_foundation_feature_defaults_are_shadow_safe():
    config = HarnessConfig()

    assert config.deployment_capability_mode == "shadow"
    assert config.deployment_contract_enabled is True
    assert config.deployment_adapter_registry_enabled is False
    assert config.protocol_verify_registry_enabled is False


def test_report_writes_all_foundation_audit_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (repo / "app.py").write_text("print('service')\n", encoding="utf-8")
    analysis = ProjectAnalyzer().analyze(repo).data
    selection = {
        "verifier_id": "builtin.openapi_trace",
        "protocol": "openapi",
        "source": "service_metadata",
        "reason": "FastAPI OpenAPI metadata",
    }

    ReportGenerator().generate(
        tmp_path,
        {"project": {"name": "audit", "repo_url": "local"}},
        {
            "analyze": {
                "status": "passed",
                "summary": "analyzed",
                "data": analysis,
            },
            "verify": {
                "status": "uncertain",
                "summary": "not executed",
                "data": {"protocol_verify_selection": selection},
            },
        },
    )

    names = {
        "project_capabilities.json",
        "capability_evidence.json",
        "deployment_contract.json",
        "adapter_detections.json",
        "deployment_candidates.json",
        "deployability_assessment.json",
        "protocol_verify_selection.json",
    }
    for name in names:
        payload = json.loads(
            (tmp_path / "reports" / name).read_text(encoding="utf-8")
        )
        assert payload["schema_version"] == 1
        assert payload["repository_fingerprint"] == analysis[
            "repository_fingerprint"
        ]
        assert payload["config_hash"] == analysis[
            "deployment_foundation_config_hash"
        ]


def test_unknown_manifest_service_reaches_current_trace_verification(tmp_path):
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

    repo = tmp_path / "workspace" / "repo"
    repo.mkdir(parents=True)
    (repo / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (repo / "app.py").write_text("print('custom server')\n", encoding="utf-8")
    _write_contract(repo, port=8123)
    analysis = ProjectAnalyzer().analyze(repo).data

    result = VerifyModule(urlopen=echo_urlopen).verify(
        tmp_path,
        analysis,
        {"pid": 123, "expected_port": 8123, "service_ready": True},
    )

    assert analysis["frameworks"] == ["unknown"]
    assert analysis["deployability"]["status"] == "ready"
    assert result.status == "passed"
    assert result.data["protocol_verify_selection"]["source"] == (
        "deployment_contract"
    )
    assert result.data["protocol_verify_selection"]["shadow_decision"][
        "strong_evidence"
    ] is True
