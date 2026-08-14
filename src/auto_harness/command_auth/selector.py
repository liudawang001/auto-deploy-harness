"""Deterministic safety-first command candidate selection."""

from typing import Dict, Iterable, List

from auto_harness.command_auth.schemas import CommandCandidate, CommandDecision


VERDICT_ORDER = {
    "auto_allowed": 0,
    "approval_required": 1,
    "candidate_rejected": 2,
    "hard_denied": 3,
}


class CommandCandidateSelector:
    def select(
        self,
        candidates: Iterable[CommandCandidate],
        decisions: Iterable[CommandDecision],
        excluded_ids: Iterable[str] = (),
    ) -> Dict:
        excluded = set(excluded_ids or ())
        by_id = {item.candidate_id: item for item in candidates}
        ranked: List[CommandDecision] = sorted(
            [item for item in decisions if item.candidate_id in by_id and item.candidate_id not in excluded],
            key=lambda item: (
                VERDICT_ORDER.get(item.verdict, 99),
                -float(by_id[item.candidate_id].score),
                item.candidate_id,
            ),
        )
        selected = next(
            (item for item in ranked if item.verdict in {"auto_allowed", "approval_required"}),
            None,
        )
        return {
            "status": "selected" if selected else "no_safe_command_candidate",
            "candidate_id": selected.candidate_id if selected else "",
            "verdict": selected.verdict if selected else "",
            "reason_code": selected.reason_code if selected else "no_safe_command_candidate",
            "ordered_candidate_ids": [item.candidate_id for item in ranked],
            "rejected_candidate_ids": [
                item.candidate_id for item in ranked
                if item.verdict in {"candidate_rejected", "hard_denied"}
            ],
        }
