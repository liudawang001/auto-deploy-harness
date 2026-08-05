"""Tests for Capability Matrix.

Validates:
- Matrix does not mark missing evidence as validated
- External not_run is not validated
- Matrix records commit SHA
- Readiness returns nonzero for failed required capability
- No production_ready status allowed
"""
import json
import tempfile
from pathlib import Path

import pytest

from auto_harness.models.base import write_json
from auto_harness.readiness import CapabilityMatrix, ALLOWED_STATUSES


class TestCapabilityMatrix:
    """Test CapabilityMatrix generation."""

    def test_matrix_does_not_mark_missing_evidence_validated(self):
        """Capabilities without evidence artifacts must not be validated."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            # Without any report artifacts, capabilities should not be validated
            for cap_name, cap in matrix["capabilities"].items():
                if cap["status"] == "validated":
                    pytest.fail(
                        "capability %s marked validated without evidence" % cap_name
                    )

    def test_external_not_run_is_not_validated(self):
        """docker_gpu must always be not_run (external)."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            assert matrix["capabilities"]["docker_gpu"]["status"] == "not_run"

    def test_matrix_records_commit_sha(self):
        """Matrix must contain a commit SHA."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path.cwd())
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            assert "commit_sha" in matrix
            assert matrix["commit_sha"] is not None

    def test_readiness_returns_nonzero_for_failed_required_capability(self):
        """If a required capability is not validated, readiness must return nonzero."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            # Without evidence, required capabilities won't be validated
            exit_code = matrix_obj.check_readiness(matrix)
            assert exit_code == 1

    def test_readiness_returns_zero_when_validated(self):
        """If all required capabilities are validated, readiness returns zero."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()

            # Create artifacts that signal success
            write_json(reports_dir / "controller_result.json", {"status": "completed"})
            write_json(reports_dir / "fault_injection_result.json", {"status": "completed"})
            write_json(reports_dir / "approval_e2e_result.json", {"status": "completed"})
            write_json(reports_dir / "skill_memory_result.json", {"status": "completed"})

            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            exit_code = matrix_obj.check_readiness(matrix)
            assert exit_code == 0

    def test_no_production_ready_status(self):
        """production_ready must never appear in allowed statuses."""
        assert "production_ready" not in ALLOWED_STATUSES

    def test_matrix_schema_version(self):
        """Matrix must have schema_version=1."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            assert matrix["schema_version"] == 1

    def test_matrix_contains_all_expected_capabilities(self):
        """Matrix must contain all expected capability keys."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            expected = [
                "default_langgraph",
                "crash_safe_reconcile",
                "approval_resume",
                "memory_skill_mainline",
                "llm_necessity",
                "docker_gpu",
                "tool_registry_contract",
                "provider_protocol",
                "self_repair_closure",
                "docker_sandbox_policy",
                "evidence_provenance",
                "deepseek_provider",
            ]
            for cap in expected:
                assert cap in matrix["capabilities"], "missing capability: %s" % cap

    def test_deepseek_registration_is_not_reported_as_configuration(
        self, monkeypatch
    ):
        for name in (
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_API_BASE",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_API_URL",
            "DEEPSEEK_MODEL",
            "DEEPSEEK_CONTEXT_WINDOW_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            write_json(root / "configs" / "default.json", {"provider_configs": {}})
            reports = root / "reports"
            reports.mkdir()
            details = CapabilityMatrix(root).generate(reports)["capabilities"][
                "deepseek_provider"
            ]["details"]
            assert details["registered"] is True
            assert details["configured"] is False
            assert details["live_smoke_status"] == "not_run"

    def test_all_statuses_are_allowed(self):
        """All capability statuses must be in ALLOWED_STATUSES."""
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            matrix_obj = CapabilityMatrix(project_root=Path(tmp))
            matrix = matrix_obj.generate(reports_dir=reports_dir)

            for cap_name, cap in matrix["capabilities"].items():
                assert cap["status"] in ALLOWED_STATUSES, (
                    "capability %s has invalid status %s" % (cap_name, cap["status"])
                )
