"""Compile validated contracts into evidence-bound pipeline candidates."""

from pathlib import Path
from typing import Dict, Tuple

from auto_harness.capabilities.schemas import DeploymentCandidate
from auto_harness.command_auth.evidence import build_evidence
from auto_harness.command_auth.schemas import (
    CommandCandidate,
    CommandRegistry,
    canonical_hash,
)


class DeploymentContractCompiler:
    def compile_registry(
        self,
        repo_dir: Path,
        contract,
        registry: CommandRegistry,
    ) -> Tuple[CommandRegistry, DeploymentCandidate]:
        evidence = list(registry.evidence)
        candidates = list(registry.candidates)
        install_ids = []
        for index, argv in enumerate(contract.environment.install_commands):
            command = self._command_candidate(
                repo_dir,
                contract,
                registry.repository_fingerprint,
                phase="install",
                argv=argv,
                declaration_key="environment.install_commands[%d]" % index,
                network_profile=contract.security.network_profile,
            )
            evidence.append(command[0])
            candidates.append(command[1])
            install_ids.append(command[1].candidate_id)
        run_evidence, run_candidate = self._command_candidate(
            repo_dir,
            contract,
            registry.repository_fingerprint,
            phase="run",
            argv=contract.service.command,
            declaration_key="service.command",
            network_profile=contract.security.network_profile,
        )
        evidence.append(run_evidence)
        candidates.append(run_candidate)
        registry = CommandRegistry(
            repository_fingerprint=registry.repository_fingerprint,
            evidence=sorted({item.evidence_id: item for item in evidence}.values(), key=lambda item: item.evidence_id),
            candidates=sorted({item.candidate_id: item for item in candidates}.values(), key=lambda item: (-item.score, item.candidate_id)),
        )
        verify_id = "verify_%s" % canonical_hash(contract.verify.__dict__)[:20]
        identity = {
            "source": "manifest",
            "contract_sha256": contract.sha256,
            "run_candidate_id": run_candidate.candidate_id,
            "verify_candidate_id": verify_id,
        }
        deployment = DeploymentCandidate(
            candidate_id="deploy_%s" % canonical_hash(identity)[:20],
            source="manifest",
            adapter_ids=[],
            environment_candidate_id="env_%s" % canonical_hash(contract.environment.__dict__)[:20],
            install_candidate_ids=install_ids,
            run_candidate_id=run_candidate.candidate_id,
            expected_port=contract.service.port,
            protocol_hints=[contract.verify.protocol],
            verify_candidate_ids=[verify_id],
            required_backend=contract.security.required_backend,
            confidence=0.95,
            evidence_ids=[run_evidence.evidence_id],
            score_reasons=["validated auto-deploy.yaml contract"],
        )
        return registry, deployment

    def compile_analysis(
        self,
        analysis: Dict,
        contract,
        registry: CommandRegistry,
        deployment_candidate: DeploymentCandidate,
    ) -> Dict:
        result = dict(analysis)
        manifest_run = {
            "id": deployment_candidate.run_candidate_id,
            "cmd": list(contract.service.command),
            "expected_port": int(contract.service.port),
            "confidence": 0.95,
            "score": 0.95,
            "selected_by": "deployment_contract",
            "score_reasons": ["validated auto-deploy.yaml contract"],
            "command_candidate_id": deployment_candidate.run_candidate_id,
            "required_backend": contract.security.required_backend,
        }
        existing_runs = [
            item for item in result.get("run_candidates", [])
            if item.get("cmd") != manifest_run["cmd"]
        ]
        result["run_candidates"] = [manifest_run] + existing_runs
        result["install_plan"] = [list(item) for item in contract.environment.install_commands]
        result["verify_hint"] = {
            "service_type": contract.verify.protocol,
            "expected_output": contract.verify.success.get("response_contains", "trace_evidence"),
            "request": {
                **contract.verify.request,
                "timeout": contract.verify.timeout_seconds,
            },
        }
        result["environment_strategy"] = {
            "backend": contract.environment.backend,
            "preferred_tool": contract.environment.backend,
            "python": contract.environment.python or "python3",
            "channels": [],
            "source": "deployment_contract",
            "confidence": 0.95,
            "reasons": ["validated auto-deploy.yaml contract"],
        }
        result["deployment_contract"] = {
            "found": True,
            "valid": True,
            **contract.to_dict(),
        }
        existing_deployments = list(result.get("deployment_candidates") or [])
        result["deployment_candidates"] = [
            deployment_candidate.to_dict(),
            *[
                item for item in existing_deployments
                if item.get("candidate_id") != deployment_candidate.candidate_id
            ],
        ]
        result["selected_candidate"] = manifest_run
        result["selection_source"] = "deployment_contract"
        result["command_registry"] = registry.to_dict()
        return result

    @staticmethod
    def _command_candidate(
        repo_dir,
        contract,
        repository_fingerprint,
        *,
        phase,
        argv,
        declaration_key,
        network_profile,
    ):
        evidence = build_evidence(
            repo_dir,
            "manifest_command",
            contract.path,
            repository_fingerprint,
            declaration_key=declaration_key,
            declared_value=str(list(argv)),
        )
        root = str(argv[0])
        binding = {
            "kind": "owned_python_env" if root.startswith(".venv/bin/") else "system_tool",
            "contract_sha256": contract.sha256,
        }
        candidate = CommandCandidate.build(
            phase=phase,
            argv=list(argv),
            source_kind="manifest_command",
            evidence_ids=[evidence.evidence_id],
            declared_executable=Path(root).name,
            environment_binding=binding,
            required_backend=contract.security.required_backend,
            network_profile=network_profile,
            filesystem_profile="install_workspace" if phase == "install" else "runtime_read_only",
            risk_level="high",
            score=0.98 if phase == "run" else 0.9,
            score_reasons=["validated auto-deploy.yaml %s" % declaration_key],
            fallback_group=phase,
        )
        return evidence, candidate
