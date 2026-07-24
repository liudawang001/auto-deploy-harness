"""Tests for Evidence manifest and exporter.

Validates:
- Export computes hash, not trusts input
- Export rejects missing required artifact
- Export rejects path traversal
- Export rejects external symlink
- Manifest contains commit SHA
- Dirty worktree is recorded
- Secret values not in archive
"""
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from auto_harness.evidence.exporter import (
    EvidenceExporter,
    REQUIRED_ARTIFACTS,
    sha256_file,
    _validate_artifact_path,
)
from auto_harness.models.base import write_json


class TestSha256File:
    """Test sha256_file hash computation."""

    def test_export_computes_hash_not_trust_input(self):
        """Hash must be computed from file content, not from input metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "test.json"
            test_file.write_text('{"key": "value"}', encoding="utf-8")
            # Compute expected hash
            expected = hashlib.sha256(test_file.read_bytes()).hexdigest()
            # Verify sha256_file computes the same
            result = sha256_file(test_file)
            assert result == expected
            assert len(result) == 64  # SHA-256 hex digest length


class TestEvidenceExporter:
    """Test EvidenceExporter."""

    def _create_valid_run_dir(self, tmp: str) -> Path:
        """Create a minimal valid run directory with all required artifacts."""
        run_dir = Path(tmp) / "runs" / "test_task"
        run_dir.mkdir(parents=True)

        # Create required files
        write_json(run_dir / "task.json", {"task_id": "test_task"})
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True)
        write_json(reports_dir / "controller_result.json", {
            "task_id": "test_task",
            "controller": "langgraph",
            "final_status": "completed",
            "verify": {"status": "passed"},
        })
        write_json(reports_dir / "project_snapshot.json", {"files": ["app.py"]})
        write_json(reports_dir / "llm_contribution_evidence.json", {"llm_helped": False})

        return run_dir

    def test_export_rejects_missing_required_artifact(self):
        """Missing required artifacts must cause export failure."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "test_task"
            run_dir.mkdir(parents=True)
            # Only create some required files, not all
            write_json(run_dir / "task.json", {"task_id": "test"})

            exporter = EvidenceExporter(project_root=Path(tmp))
            manifest = exporter.export(run_dir, "test_task")

            assert manifest["status"] == "failed"
            assert len(manifest.get("missing_artifacts", [])) > 0

    def test_export_rejects_path_traversal(self):
        """Artifact paths with traversal should be rejected."""
        # Path traversal in artifact path
        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp) / "evidence"
            evidence_root.mkdir()
            test_file = evidence_root / "../../../etc/passwd"

            # _validate_artifact_path should reject this
            result = _validate_artifact_path(
                Path(evidence_root / "../../../etc/passwd"),
                evidence_root,
            )
            assert result is False

    def test_export_rejects_external_symlink(self):
        """Symlinks pointing outside evidence root should be rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            evidence_root = Path(tmp) / "evidence"
            evidence_root.mkdir()

            # Create a real file outside
            outside_file = Path(tmp) / "outside.txt"
            outside_file.write_text("secret", encoding="utf-8")

            # Create symlink inside pointing outside
            symlink = evidence_root / "link.txt"
            try:
                symlink.symlink_to(outside_file)
            except OSError:
                pytest.skip("symlinks not supported on this platform")

            result = _validate_artifact_path(symlink, evidence_root)
            # The resolved path should be within evidence_root
            # Since it points outside, it should be rejected
            assert result is False

    def test_manifest_contains_commit_sha(self):
        """Manifest must contain commit SHA from git."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            exporter = EvidenceExporter(project_root=Path.cwd())
            manifest = exporter.export(run_dir, "test_task")

            assert "project_commit_sha" in manifest
            # In a git repo, it should be a hex string or "unknown"
            assert manifest["project_commit_sha"] is not None

    def test_dirty_worktree_is_recorded(self):
        """Dirty worktree status must be recorded."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            exporter = EvidenceExporter(project_root=Path.cwd())
            manifest = exporter.export(run_dir, "test_task")

            assert "dirty_worktree" in manifest
            assert isinstance(manifest["dirty_worktree"], bool)

    def test_secret_value_not_in_archive(self):
        """Source secrets must be replaced in the actual archive payload."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            secret = "sk_test_secret_123456789"
            write_json(run_dir / "evidence" / "request.json", {
                "api_key": secret,
                "authorization": "Bearer abcdefghijklmnop",
            })
            exporter = EvidenceExporter(project_root=Path.cwd())
            output = Path(tmp) / "evidence.tar.gz"
            manifest = exporter.export(run_dir, "test_task", output_path=output)

            with tarfile.open(output, "r:gz") as archive:
                payload = archive.extractfile("evidence/request.json").read().decode("utf-8")
            assert secret not in payload
            assert "abcdefghijklmnop" not in payload
            assert "[REDACTED]" in payload
            artifact = next(item for item in manifest["artifacts"] if item["path"] == "evidence/request.json")
            assert artifact["redacted"] is True

    def test_current_controller_schema_is_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            write_json(run_dir / "reports" / "controller_result.json", {
                "task_id": "test_task",
                "controller": "langgraph",
                "status": "completed",
                "verify_status": "passed",
            })
            manifest = EvidenceExporter(project_root=Path.cwd()).export(run_dir, "test_task")
            assert manifest["final_status"] == "completed"
            assert manifest["verify_status"] == "passed"
            assert Path(manifest["archive_path"]).is_file()

    def test_artifacts_have_computed_hashes(self):
        """All artifacts must have sha256 hashes computed by the exporter."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            exporter = EvidenceExporter(project_root=Path.cwd())
            manifest = exporter.export(run_dir, "test_task")

            for artifact in manifest.get("artifacts", []):
                if artifact.get("size_bytes", 0) > 0:
                    assert len(artifact.get("sha256", "")) == 64

    def test_manifest_schema_version(self):
        """Manifest must have schema_version=1."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            exporter = EvidenceExporter(project_root=Path.cwd())
            manifest = exporter.export(run_dir, "test_task")

            assert manifest["schema_version"] == 1

    def test_manifest_contains_environment_info(self):
        """Manifest must contain environment information."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._create_valid_run_dir(tmp)
            exporter = EvidenceExporter(project_root=Path.cwd())
            manifest = exporter.export(run_dir, "test_task")

            assert "environment" in manifest
            assert manifest["environment"]["os"] is not None
            assert manifest["environment"]["python"] is not None
