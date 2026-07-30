"""Ownership marker for Harness-managed Conda environments."""
from pathlib import Path
from typing import Dict

from auto_harness.models.base import read_json
from auto_harness.utils.atomic import atomic_write_text
from auto_harness.utils.time import utc_now_iso


class EnvironmentOwnership:
    FILE_NAME = ".auto-harness-owner.json"

    def marker_path(self, prefix: Path) -> Path:
        return Path(prefix) / self.FILE_NAME

    def read(self, prefix: Path) -> Dict:
        path = self.marker_path(prefix)
        if not path.exists():
            return {}
        try:
            value = read_json(path)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def write(
        self,
        prefix: Path,
        project_id: str,
        repo_fingerprint: str,
        operation_id: str,
        spec_hash: str,
        python_version: str = "",
    ) -> Dict:
        import json
        payload = {
            "schema_version": 1,
            "created_by": "auto-deploy-harness",
            "project_id": project_id,
            "repo_fingerprint": repo_fingerprint,
            "operation_id": operation_id,
            "spec_hash": spec_hash,
            "python_version": python_version,
            "created_at": utc_now_iso(),
        }
        path = self.marker_path(prefix)
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        )
        return payload

    def is_valid(self, marker: Dict) -> bool:
        return (
            isinstance(marker, dict)
            and marker.get("schema_version") == 1
            and marker.get("created_by") == "auto-deploy-harness"
            and bool(marker.get("project_id"))
            and bool(marker.get("spec_hash"))
        )

    def matches(self, prefix: Path, project_id: str, spec_hash: str) -> bool:
        marker = self.read(prefix)
        return (
            self.is_valid(marker)
            and marker.get("project_id") == project_id
            and marker.get("spec_hash") == spec_hash
        )
