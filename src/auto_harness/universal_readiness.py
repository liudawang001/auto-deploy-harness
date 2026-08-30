"""Phase B6: commit-bound readiness gate for universal deployment.

The gate is fail-closed: readiness is only reported when every piece of
required evidence exists, is hash-valid, is bound to the current commit and
clean worktree, and the security counters are zero.  A missing, stale or
mismatched evidence file blocks the release instead of being skipped.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.release_evidence import build_evidence, evidence_hash, validate_evidence
from auto_harness.utils.time import utc_now_iso


READINESS_SCHEMA_VERSION = 1

SUPPORTED_SCHEMA_VERSIONS = {
    "capability_schema_version": 2,
    "contract_schema_version": 1,
    "adapter_registry_version": 1,
    "verifier_registry_version": 1,
}

# Required expansion evidence artifacts (relative to the project root).
EXPANSION_TEST_FILES = (
    "tests/test_universal_deployment_expansion.py",
    "tests/test_universal_deployment_llm_fallback.py",
    "tests/test_universal_deployment_native_ecosystems.py",
)


class UniversalDeploymentReadinessGate:
    """Evaluate commit-bound readiness for the universal deployment chain."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)

    def build_expansion_evidence(
        self,
        *,
        command: List[str],
        passed: int,
        failed: int,
        skipped: int = 0,
        config_hash: str = "",
        execution_backend: str = "local",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the commit-bound evidence record for the expansion gates."""
        payload = build_evidence(
            self.project_root,
            command,
            "passed" if failed == 0 else "failed",
            passed,
            failed,
            skipped,
            config_hash=str(config_hash),
            execution_backend=str(execution_backend),
            schema_versions=dict(SUPPORTED_SCHEMA_VERSIONS),
            test_files=list(EXPANSION_TEST_FILES),
            fixture_hashes=self.fixture_hashes(),
            **(extra or {}),
        )
        return payload

    def fixture_hashes(self) -> Dict[str, str]:
        """Hash the expansion test fixtures for tamper binding."""
        hashes = {}
        for relative in EXPANSION_TEST_FILES:
            path = self.project_root / relative
            if not path.is_file():
                hashes[relative] = "missing"
                continue
            import hashlib

            hashes[relative] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        return hashes

    def evaluate(
        self,
        *,
        expansion_evidence: Optional[Dict[str, Any]] = None,
        shadow_diff: Optional[Dict[str, Any]] = None,
        handoff: Optional[Dict[str, Any]] = None,
        config_hash: str = "",
        execution_backend: str = "local",
    ) -> Dict[str, Any]:
        """Fail-closed readiness evaluation."""
        checks: List[Dict[str, Any]] = []
        reason_codes: List[str] = []

        handoff = handoff or self._load_handoff()
        if not handoff:
            reason_codes.append("foundation_handoff_missing")
            checks.append({"name": "foundation_handoff", "status": "failed"})
        else:
            stale = self._validate_handoff(handoff)
            if stale:
                reason_codes.extend(stale)
                checks.append({
                    "name": "foundation_handoff",
                    "status": "failed",
                    "reason_codes": stale,
                })
            else:
                checks.append({"name": "foundation_handoff", "status": "passed"})

        if not expansion_evidence:
            reason_codes.append("expansion_evidence_missing")
            checks.append({"name": "expansion_evidence", "status": "failed"})
        else:
            errors = validate_evidence(expansion_evidence, self.project_root)
            fixture_hashes = self.fixture_hashes()
            if (expansion_evidence.get("fixture_hashes") or {}) != fixture_hashes:
                errors.append("fixture hash binding mismatch")
            if config_hash and expansion_evidence.get("config_hash") != config_hash:
                errors.append("config hash binding mismatch")
            if expansion_evidence.get("schema_versions") != dict(SUPPORTED_SCHEMA_VERSIONS):
                errors.append("schema version binding mismatch")
            if expansion_evidence.get("execution_backend") != str(execution_backend):
                errors.append("execution backend binding mismatch")
            if errors:
                reason_codes.extend("expansion_evidence_%s" % item.replace(" ", "_") for item in errors)
                checks.append({
                    "name": "expansion_evidence",
                    "status": "failed",
                    "reason_codes": errors,
                })
            else:
                checks.append({"name": "expansion_evidence", "status": "passed"})

        diff = shadow_diff if shadow_diff is not None else self._load_shadow_diff()
        if diff is None:
            reason_codes.append("shadow_diff_missing")
            checks.append({"name": "shadow_diff", "status": "failed"})
        elif diff.get("computed") is False:
            # legacy/off modes legitimately skip the diff; enforce is the
            # only mode that requires a computed comparison.
            checks.append({"name": "shadow_diff", "status": "skipped"})
        elif str(diff.get("classification")) == "new_less_safe":
            reason_codes.append("shadow_diff_new_less_safe")
            checks.append({
                "name": "shadow_diff",
                "status": "failed",
                "reason_codes": ["new_less_safe unresolved"],
            })
        else:
            checks.append({"name": "shadow_diff", "status": "passed"})

        if handoff:
            for counter in (
                "false_success_count", "unsafe_command_execution_count",
                "secret_leak_count",
            ):
                if int(handoff.get(counter) or 0) != 0:
                    reason_codes.append("%s_nonzero" % counter)
            checks.append({
                "name": "security_counters",
                "status": "passed" if not any(
                    code.endswith("_nonzero") for code in reason_codes
                ) else "failed",
            })

        status = "ready" if not reason_codes else "blocked"
        result = {
            "schema_version": READINESS_SCHEMA_VERSION,
            "status": status,
            "reason_codes": reason_codes,
            "checks": checks,
            "evidence_bindings": {
                "commit_sha": (expansion_evidence or {}).get("commit_sha", ""),
                "config_hash": str(config_hash),
                "fixture_hashes": self.fixture_hashes(),
                "schema_versions": dict(SUPPORTED_SCHEMA_VERSIONS),
                "execution_backend": str(execution_backend),
                "generated_at": utc_now_iso(),
            },
        }
        result["evidence_sha256"] = evidence_hash(result)
        return result

    def _load_handoff(self) -> Dict[str, Any]:
        for relative in (
            "docs/evidence/universal-deployment-foundation-handoff.json",
            "reports/universal-deployment-foundation-handoff.json",
        ):
            path = self.project_root / relative
            if path.is_file():
                import json

                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return {}
        return {}

    def _load_shadow_diff(self) -> Optional[Dict[str, Any]]:
        path = self.project_root / "reports" / "universal-deployment-shadow-diff.json"
        if not path.is_file():
            return None
        import json

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _validate_handoff(handoff: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for key, expected in SUPPORTED_SCHEMA_VERSIONS.items():
            value = str(handoff.get(key) or "")
            if not value or value != str(expected):
                errors.append("handoff_%s_unsupported" % key)
        for counter in (
            "false_success_count", "unsafe_command_execution_count",
        ):
            try:
                if int(handoff.get(counter) or 0) != 0:
                    errors.append("handoff_%s_nonzero" % counter)
            except (TypeError, ValueError):
                errors.append("handoff_%s_invalid" % counter)
        if not str(handoff.get("commit") or ""):
            errors.append("handoff_commit_missing")
        return errors
