"""Compose adapter proposals without granting execution authority."""

from typing import Iterable, List

from auto_harness.capabilities.schemas import DeploymentCandidate
from auto_harness.command_auth.schemas import canonical_hash


class CandidateComposer:
    def merge_run_proposals(self, proposals: Iterable) -> List:
        merged = {}
        conflicts = set()
        for proposal in proposals:
            key = tuple(proposal.argv)
            current = merged.get(key)
            if current is None:
                merged[key] = proposal
                continue
            current.evidence_ids = sorted(set(current.evidence_ids + proposal.evidence_ids))
            current.reasons = list(dict.fromkeys(current.reasons + proposal.reasons))
            current.confidence = max(current.confidence, proposal.confidence)
            if current.expected_port != proposal.expected_port:
                current.expected_port = 0
                conflicts.add(key)
        for key in conflicts:
            merged[key].reasons.append("conflicting expected ports")
        return list(merged.values())

    def compose(self, run_proposals, environment_proposals, verify_proposals):
        runs = self.merge_run_proposals(run_proposals)
        environment = environment_proposals[0] if environment_proposals else None
        verify = verify_proposals[0] if verify_proposals else None
        result = []
        for run in runs:
            run_id = "adapter_run_%s" % canonical_hash(run.to_dict())[:20]
            missing = []
            if run.expected_port <= 0:
                missing.append("run.expected_port")
            if verify is None:
                missing.append("verify.strong_evidence")
            adapter_ids = [run.adapter_id]
            evidence_ids = list(run.evidence_ids)
            reasons = list(run.reasons)
            if environment is not None:
                adapter_ids.append(environment.adapter_id)
                evidence_ids.extend(environment.evidence_ids)
                reasons.extend(environment.reasons)
            if verify is not None:
                adapter_ids.append(verify.adapter_id)
                evidence_ids.extend(verify.evidence_ids)
                reasons.extend(verify.reasons)
            identity = {
                "run": run_id,
                "environment": environment.to_dict() if environment else {},
                "verify": verify.to_dict() if verify else {},
            }
            result.append(DeploymentCandidate(
                candidate_id="deploy_%s" % canonical_hash(identity)[:20],
                source="adapter",
                adapter_ids=list(dict.fromkeys(adapter_ids)),
                environment_candidate_id=(
                    "adapter_env_%s" % canonical_hash(environment.to_dict())[:20]
                    if environment else ""
                ),
                run_candidate_id=run_id,
                # The run argv lets grounded LLM selection map a deployment
                # candidate back to its registry command candidate (Phase B2).
                run_cmd=list(run.argv),
                expected_port=run.expected_port,
                protocol_hints=[verify.protocol] if verify else [],
                verify_candidate_ids=(
                    ["adapter_verify_%s" % canonical_hash(verify.to_dict())[:20]]
                    if verify else []
                ),
                required_backend="docker",
                confidence=min(
                    run.confidence,
                    verify.confidence if verify else run.confidence,
                ),
                evidence_ids=sorted(set(evidence_ids)),
                missing_capabilities=missing,
                score_reasons=list(dict.fromkeys(reasons)),
            ))
        return result
