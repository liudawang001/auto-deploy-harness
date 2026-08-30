"""Deterministic deployability assessment from known pipeline inputs."""

from typing import Dict, List

from auto_harness.capabilities.schemas import DeployabilityAssessment


class DeployabilityAssessor:
    def assess(
        self,
        *,
        run_candidates: List[Dict],
        verify_hint: Dict,
        deployment_candidates=None,
    ) -> DeployabilityAssessment:
        deployment_candidates = list(deployment_candidates or [])
        missing = []
        if not run_candidates and not any(item.run_candidate_id for item in deployment_candidates):
            missing.append("run.entrypoint")
        service_type = str((verify_hint or {}).get("service_type") or "unknown")
        if service_type == "unknown" and not any(item.verify_candidate_ids for item in deployment_candidates):
            missing.append("verify.strong_evidence")
        selected = deployment_candidates[0].candidate_id if deployment_candidates else ""
        status = "ready" if not missing else "partial"
        next_resolution = "compile" if status == "ready" else "contract_required"
        return DeployabilityAssessment(
            status=status,
            selected_candidate_id=selected,
            candidate_ids=[item.candidate_id for item in deployment_candidates],
            missing_capabilities=missing,
            next_resolution=next_resolution,
        )
