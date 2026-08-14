"""Build deterministic command registry from repository evidence."""

from pathlib import Path
from typing import List

from auto_harness.command_auth.adapters import (
    discover_make,
    discover_node,
    discover_python_cli,
    discover_repository_scripts,
)
from auto_harness.command_auth.adapters.common import readme_commands
from auto_harness.command_auth.schemas import CommandRegistry


class CommandDiscoveryService:
    def discover(
        self,
        repo_dir: Path,
        file_tree: List[str],
        repository_fingerprint: str,
    ) -> CommandRegistry:
        repo_dir = Path(repo_dir)
        documented = readme_commands(repo_dir, file_tree)
        evidence = []
        candidates = []
        for adapter in (
            discover_python_cli,
            discover_node,
            discover_make,
            discover_repository_scripts,
        ):
            found_evidence, found_candidates = adapter(
                repo_dir, file_tree, documented, repository_fingerprint
            )
            evidence.extend(found_evidence)
            candidates.extend(found_candidates)

        unique_evidence = {item.evidence_id: item for item in evidence}
        unique_candidates = {item.candidate_id: item for item in candidates}
        return CommandRegistry(
            repository_fingerprint=repository_fingerprint,
            evidence=sorted(unique_evidence.values(), key=lambda item: item.evidence_id),
            candidates=sorted(
                unique_candidates.values(),
                key=lambda item: (-item.score, item.candidate_id),
            ),
        )
