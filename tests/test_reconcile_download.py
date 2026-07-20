"""Tests for DownloadReconciler and partial download metadata helpers.

Phase 2 tests: model download reconciliation, partial metadata,
SHA-256 verification, and offset continuation.
"""
import json
import pytest
from pathlib import Path

from auto_harness.recovery.download import (
    DownloadReconciler,
    reconcile_result,
    sha256_file,
    partial_metadata_path,
    write_partial_metadata,
    read_partial_metadata,
    meta_matches,
    PARTIAL_IDENTITY_KEYS,
)
from auto_harness.recovery.schemas import compute_operation_id, canonical_json


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_download_operation(
    tmp_path,
    target_path=None,
    source="huggingface",
    repo_id="org/model",
    revision="main",
    relative_path="model.bin",
    expected_size=1024,
    sha256="",
    etag="",
):
    """Build a download operation dict for testing."""
    if target_path is None:
        target_path = str(tmp_path / "cache" / "model.bin")
    identity = {
        "source": source,
        "repo_id": repo_id,
        "revision": revision,
        "relative_path": relative_path,
        "target_path": str(target_path),
        "expected_size": str(expected_size),
        "sha256": sha256,
        "etag": etag,
    }
    normalized_input = {
        "source": source,
        "repo_id": repo_id,
        "revision": revision,
        "relative_path": relative_path,
    }
    operation_id = compute_operation_id(
        "test_task", "model_prepare", "download",
        normalized_input, identity,
    )
    return {
        "operation_id": operation_id,
        "task_id": "test_task",
        "stage": "model_prepare",
        "action": "download",
        "resource_type": "model_download",
        "resource_identity": identity,
        "observed_resource": {},
        "normalized_input_hash": canonical_json(normalized_input),
        "status": "planned",
    }


def write_test_file(path, size=1024, content=None):
    """Write a test file of given size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        path.write_bytes(content)
    else:
        path.write_bytes(b"\x00" * size)


# -------------------------------------------------------------------
# Partial Metadata Tests
# -------------------------------------------------------------------

class TestPartialMetadata:
    def test_write_and_read(self, tmp_path):
        part = tmp_path / "model.bin.part"
        part.write_bytes(b"partial")
        identity = {
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "main",
            "relative_path": "model.bin",
            "expected_size": "1024",
            "etag": "abc",
            "sha256": "def",
        }
        write_partial_metadata(part, identity)
        meta = read_partial_metadata(part)
        assert meta is not None
        assert meta["source"] == "huggingface"
        assert meta["repo_id"] == "org/model"
        assert meta["revision"] == "main"

    def test_read_nonexistent(self, tmp_path):
        part = tmp_path / "nonexistent.part"
        assert read_partial_metadata(part) is None

    def test_read_invalid_json(self, tmp_path):
        part = tmp_path / "model.bin.part"
        part.write_bytes(b"partial")
        meta_path = partial_metadata_path(part)
        meta_path.write_text("not json", encoding="utf-8")
        assert read_partial_metadata(part) is None

    def test_meta_path(self, tmp_path):
        part = tmp_path / "model.bin.part"
        expected = tmp_path / "model.bin.part.auto_harness_meta.json"
        assert partial_metadata_path(part) == expected


class TestMetaMatches:
    def test_matching_identity(self):
        meta = {
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "main",
            "relative_path": "model.bin",
            "expected_size": "1024",
            "etag": "abc",
            "sha256": "def",
        }
        identity = dict(meta)
        assert meta_matches(meta, identity) is True

    def test_mismatched_revision(self):
        meta = {
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "v1",
            "relative_path": "model.bin",
            "expected_size": "1024",
            "etag": "abc",
            "sha256": "def",
        }
        identity = dict(meta)
        identity["revision"] = "v2"
        assert meta_matches(meta, identity) is False

    def test_none_meta(self):
        assert meta_matches(None, {}) is False

    def test_empty_meta(self):
        assert meta_matches({}, {}) is False


# -------------------------------------------------------------------
# SHA-256 File Tests
# -------------------------------------------------------------------

class TestSha256File:
    def test_known_content(self, tmp_path):
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello world\n")
        digest = sha256_file(path)
        assert len(digest) == 64
        assert digest.isalnum()

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        digest = sha256_file(path)
        # SHA-256 of empty string
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# -------------------------------------------------------------------
# DownloadReconciler Tests
# -------------------------------------------------------------------

class TestDownloadReconciler:
    def test_reuse_when_target_valid(self, tmp_path):
        """Complete file with valid size → reuse."""
        target = tmp_path / "cache" / "model.bin"
        write_test_file(target, size=1024)
        op = make_download_operation(tmp_path, target_path=str(target), expected_size=1024)
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"
        assert result["observed_state"]["size"] == 1024

    def test_reuse_with_hash_verification(self, tmp_path):
        """Complete file with matching SHA-256 → reuse."""
        target = tmp_path / "cache" / "model.bin"
        content = b"hello world\n"
        write_test_file(target, content=content)
        import hashlib
        expected_hash = hashlib.sha256(content).hexdigest()
        op = make_download_operation(
            tmp_path, target_path=str(target),
            expected_size=len(content), sha256=expected_hash,
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"

    def test_conflict_when_hash_mismatch(self, tmp_path):
        """Complete file with wrong SHA-256 → conflict."""
        target = tmp_path / "cache" / "model.bin"
        write_test_file(target, size=1024)
        op = make_download_operation(
            tmp_path, target_path=str(target),
            expected_size=1024, sha256="wrong_hash",
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"
        assert "integrity" in result["reason"]

    def test_conflict_when_size_mismatch(self, tmp_path):
        """Complete file with wrong size → conflict."""
        target = tmp_path / "cache" / "model.bin"
        write_test_file(target, size=512)
        op = make_download_operation(
            tmp_path, target_path=str(target), expected_size=1024,
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "conflict"

    def test_continue_when_partial_matches(self, tmp_path):
        """Partial file with matching metadata → continue with offset."""
        target = tmp_path / "cache" / "model.bin"
        part = Path(str(target) + ".part")
        write_test_file(part, size=4096)
        identity = {
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "main",
            "relative_path": "model.bin",
            "target_path": str(target),
            "expected_size": "8192",
            "sha256": "",
            "etag": "",
        }
        write_partial_metadata(part, identity)
        op = make_download_operation(
            tmp_path, target_path=str(target),
            expected_size=8192,
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "continue"
        assert result["observed_state"]["offset"] == 4096

    def test_manual_when_partial_no_metadata(self, tmp_path):
        """Partial file without sidecar metadata → manual."""
        target = tmp_path / "cache" / "model.bin"
        part = Path(str(target) + ".part")
        write_test_file(part, size=4096)
        # No metadata sidecar
        op = make_download_operation(tmp_path, target_path=str(target))
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"
        assert "no identity metadata" in result["reason"]

    def test_manual_when_partial_mismatched_metadata(self, tmp_path):
        """Partial file with mismatched metadata → manual."""
        target = tmp_path / "cache" / "model.bin"
        part = Path(str(target) + ".part")
        write_test_file(part, size=4096)
        # Write metadata for a different revision
        wrong_identity = {
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "v1",  # Different from operation's "main"
            "relative_path": "model.bin",
            "target_path": str(target),
            "expected_size": "8192",
            "sha256": "",
            "etag": "",
        }
        write_partial_metadata(part, wrong_identity)
        op = make_download_operation(
            tmp_path, target_path=str(target),
            revision="main",  # Operation wants "main"
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "manual"
        assert "does not match" in result["reason"]

    def test_retry_when_no_file(self, tmp_path):
        """No target or partial file → retry."""
        target = tmp_path / "cache" / "model.bin"
        op = make_download_operation(tmp_path, target_path=str(target))
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "retry"
        assert "no cached" in result["reason"]

    def test_reuse_without_size_check(self, tmp_path):
        """Target exists, no expected_size specified → reuse (size not checked)."""
        target = tmp_path / "cache" / "model.bin"
        write_test_file(target, size=512)
        op = make_download_operation(
            tmp_path, target_path=str(target), expected_size=0,
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "reuse"

    def test_resource_type(self):
        assert DownloadReconciler.resource_type == "model_download"


# -------------------------------------------------------------------
# Integration: continue offset passed to executor
# -------------------------------------------------------------------

class TestContinueOffsetIntegration:
    def test_continue_offset_is_available(self, tmp_path):
        """When reconciler returns 'continue', the offset is in observed_state."""
        target = tmp_path / "cache" / "model.bin"
        part = Path(str(target) + ".part")
        write_test_file(part, size=8192)
        identity = {
            "source": "huggingface",
            "repo_id": "org/model",
            "revision": "main",
            "relative_path": "model.bin",
            "target_path": str(target),
            "expected_size": "16384",
            "sha256": "",
            "etag": "",
        }
        write_partial_metadata(part, identity)
        op = make_download_operation(
            tmp_path, target_path=str(target),
            expected_size=16384,
        )
        reconciler = DownloadReconciler()
        result = reconciler.reconcile(op)
        assert result["decision"] == "continue"
        assert result["observed_state"]["offset"] == 8192
        assert result["observed_state"]["offset"] == 8192
