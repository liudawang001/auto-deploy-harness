"""Memory Evolution E2E smoke test (Phase 3).

Verifies the full memory-to-skill evolution pipeline end-to-end:
  verified memory fixture → propose → regression → shadow → promote → rollback

All using mock provider, no real LLM, no GPU, no real model.
"""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.memory.evolution import MemoryEvolutionManager
from auto_harness.memory.outcomes import SkillOutcomeRecorder
from auto_harness.providers.memory_evolution_mock import MemoryEvolutionMockProvider
from auto_harness.skills.rollback import SkillRollbackManager
from auto_harness.skills.shadow import ShadowSkillEvaluator


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_verified_memory_fixture(memory_dir: Path, count: int = 3) -> Path:
    """Write verified memory entries to deployment_issues.jsonl."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    issue_path = memory_dir / "deployment_issues.jsonl"
    entries = []
    for i in range(count):
        entry = {
            "id": "mem_e2e_%03d" % i,
            "memory_type": "verified_success",
            "created_at": "2026-07-09T00:00:00+00:00",
            "task_id": "task_e2e_%03d" % i,
            "stage": "verify",
            "category": "verification_gap",
            "frameworks": ["gradio"],
            "signature": "sig_e2e_%03d" % i,
            "symptom": "HTTP 200 but no trace_id observed",
            "root_cause": "non-default Gradio API shape",
            "repair_action_hash": "hash_e2e_%03d" % i,
            "repair_actions": ["discover /config with discover_gradio_api"],
            "repair_action_status": "executed",
            "verification_trace_id": "trace_e2e_%03d" % i,
            "verify_status": "passed",
            "regression_case_ids": ["gradio_config_discovery"],
            "regression_status": "passed",
            "verified_success": True,
            "policy_rejected_high_risk": False,
            "suggested_next_action": "Promote after regression.",
        }
        entries.append(json.dumps(entry, ensure_ascii=False))
    issue_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return issue_path


def _write_skill_file(skills_dir: Path, skill_name: str, content: str) -> Path:
    """Write a skill file and return its path."""
    skill_dir = skills_dir / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")
    return skill_path


# Minimal verify-evidence skill content for E2E
_VERIFY_SKILL_CONTENT = """\
---
name: verify-evidence
description: Evidence-based verification skill.
---

# Evidence Verify

Verify deployment by sending trace requests and checking responses.
"""


class _PassingBenchmarkRunner:
    def run(self, manifest_path, output_path=None, case_ids=None):
        return {
            "status": "passed",
            "cases": [
                {"id": case_id, "status": "passed"}
                for case_id in (case_ids or [])
            ],
        }


class TestMemoryEvolutionE2E(unittest.TestCase):
    """End-to-end smoke test for memory-to-skill evolution pipeline."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.memory_dir = Path(self.tmp) / "memory"
        self.skills_dir = Path(self.tmp) / "skills"
        self.candidate_dir = self.memory_dir / "skill_candidates"
        self.provider = MemoryEvolutionMockProvider()

        # Write verified memory fixture
        _write_verified_memory_fixture(self.memory_dir, count=3)

        # Write target skill file
        self.skill_path = _write_skill_file(
            self.skills_dir, "verify-evidence", _VERIFY_SKILL_CONTENT
        )
        self.base_sha = _sha256(_VERIFY_SKILL_CONTENT)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_pipeline_propose_to_rollback(self):
        """Full E2E: propose → regression → shadow → promote → rollback."""
        manager = MemoryEvolutionManager(
            memory_dir=self.memory_dir,
            skills_dir=self.skills_dir,
            provider=self.provider,
        )

        # Step 1: Propose
        propose_result = manager.propose(min_verified_count=1)
        self.assertEqual(propose_result["status"], "proposed")
        self.assertGreater(len(propose_result["candidates"]), 0)

        candidate = propose_result["candidates"][0]
        candidate_id = candidate["candidate_id"]
        candidate_path = self.candidate_dir / ("candidate_%s.json" % candidate_id)

        # Verify candidate json exists
        self.assertTrue(candidate_path.exists(), "candidate json must exist")

        # Verify candidate md exists
        candidate_md_path = self.candidate_dir / ("candidate_%s.md" % candidate_id)
        self.assertTrue(candidate_md_path.exists(), "candidate md must exist")

        # Verify candidate content
        loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["candidate_id"], candidate_id)
        self.assertEqual(loaded["status"], "proposed")
        self.assertTrue(loaded["quality_gate"]["passed"])
        self.assertIn("patch", loaded)
        self.assertIn("markdown", loaded["patch"])

        # Step 2: explicit approval, then execute the regression gate.
        self.assertEqual(manager.approve(candidate_path, reviewer="e2e")["status"], "approved")
        regression = manager.run_regression(
            candidate_path,
            benchmark_runner=_PassingBenchmarkRunner(),
        )
        self.assertEqual(regression["status"], "passed")

        # Step 3: Shadow evaluation
        shadow_eval = ShadowSkillEvaluator()
        # Record shadow results to pass the gate
        for i in range(2):
            shadow_result = {
                "candidate_id": candidate_id,
                "run_id": "e2e_shadow_run_%d" % i,
                "matched": True,
                "would_help": True,
                "would_harm": False,
                "reason": "candidate tool matches trace-verified successful memory",
                "evaluated_at": "2026-07-09T00:00:00+00:00",
            }
            shadow_eval.record(candidate_path, shadow_result)

        # Verify shadow artifact exists
        shadow_artifact = candidate_path.with_suffix(".shadow.json")
        self.assertTrue(shadow_artifact.exists(), "shadow artifact must exist")

        # Verify candidate shadow status updated
        loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(loaded["shadow"]["helped_count"], 2)
        self.assertEqual(loaded["shadow"]["harmful_count"], 0)

        # Step 4: Promote (with shadow required)
        promote_result = manager.promote(candidate_path, require_shadow=True)
        self.assertEqual(promote_result["status"], "promoted")
        self.assertEqual(promote_result["candidate_id"], candidate_id)

        # Verify skill file now has marker
        skill_content = self.skill_path.read_text(encoding="utf-8")
        marker = "auto-harness-skill-evolution:%s" % candidate_id
        self.assertIn(marker, skill_content, "skill file must contain evolution marker")

        # Verify candidate status is active
        loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["status"], "active")
        self.assertEqual(loaded["promotion"]["status"], "promoted")

        # Step 5: Rollback
        rollback_mgr = SkillRollbackManager()
        rollback_result = rollback_mgr.rollback_candidate(candidate_path)
        self.assertEqual(rollback_result["status"], "rolled_back")

        # Verify marker is gone after rollback
        skill_content = self.skill_path.read_text(encoding="utf-8")
        self.assertNotIn(marker, skill_content, "marker must be gone after rollback")

        # Verify skill content restored
        self.assertEqual(
            _sha256(skill_content),
            self.base_sha,
            "skill content must be restored to original",
        )

    def test_propose_writes_candidate_artifacts(self):
        """Propose step writes both .json and .md candidate files."""
        manager = MemoryEvolutionManager(
            memory_dir=self.memory_dir,
            skills_dir=self.skills_dir,
            provider=self.provider,
        )
        result = manager.propose(min_verified_count=1)
        self.assertEqual(result["status"], "proposed")

        candidate = result["candidates"][0]
        cid = candidate["candidate_id"]

        json_path = self.candidate_dir / ("candidate_%s.json" % cid)
        md_path = self.candidate_dir / ("candidate_%s.md" % cid)

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        # MD file should contain the patch markdown
        md_content = md_path.read_text(encoding="utf-8")
        self.assertIn(cid, md_content)

    def test_skill_outcomes_summarizable_after_e2e(self):
        """After E2E pipeline, skill_outcomes.jsonl can be summarized."""
        recorder = SkillOutcomeRecorder(self.memory_dir)

        # Record some outcomes
        recorder.record_run(
            run_id="e2e_run_001",
            stage="verify",
            selected_skills=[{"name": "verify-evidence", "sha256": "abc"}],
            result={"status": "passed"},
            agent_metadata={"llm_helped": True, "trace_verified": True},
        )
        recorder.record_run(
            run_id="e2e_run_002",
            stage="verify",
            selected_skills=[{"name": "verify-evidence", "sha256": "abc"}],
            result={"status": "uncertain"},
            agent_metadata={"llm_helped": False, "trace_verified": False},
        )

        summary = recorder.summarize()
        self.assertEqual(summary["total"], 2)
        self.assertIn("by_skill_sha", summary)

    def test_promote_without_shadow_fails_when_required(self):
        """Promotion fails when shadow is required but not passed."""
        manager = MemoryEvolutionManager(
            memory_dir=self.memory_dir,
            skills_dir=self.skills_dir,
            provider=self.provider,
        )
        result = manager.propose(min_verified_count=1)
        candidate = result["candidates"][0]
        candidate_path = self.candidate_dir / ("candidate_%s.json" % candidate["candidate_id"])

        manager.approve(candidate_path, reviewer="e2e")
        manager.run_regression(candidate_path, benchmark_runner=_PassingBenchmarkRunner())

        # Promote with shadow required should fail
        promote_result = manager.promote(candidate_path, require_shadow=True)
        self.assertEqual(promote_result["status"], "failed")
        self.assertIn("shadow", promote_result.get("error", "").lower())

    def test_promote_without_shadow_passes_when_not_required(self):
        """Promotion succeeds when shadow is not required and other gates pass."""
        manager = MemoryEvolutionManager(
            memory_dir=self.memory_dir,
            skills_dir=self.skills_dir,
            provider=self.provider,
        )
        result = manager.propose(min_verified_count=1)
        candidate = result["candidates"][0]
        candidate_path = self.candidate_dir / ("candidate_%s.json" % candidate["candidate_id"])

        manager.approve(candidate_path, reviewer="e2e")
        manager.run_regression(candidate_path, benchmark_runner=_PassingBenchmarkRunner())

        # Promote without shadow requirement
        promote_result = manager.promote(candidate_path, require_shadow=False)
        self.assertEqual(promote_result["status"], "promoted")


if __name__ == "__main__":
    unittest.main()
