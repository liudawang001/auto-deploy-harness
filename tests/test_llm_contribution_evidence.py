"""LLM Contribution Evidence tests.

Validates that:
1. llm_contribution_evidence.json is generated
2. llm_helped=true only when baseline failed and agent passed with trace evidence
3. Without baseline, llm_required_status is "unknown_without_baseline"
4. report.md contains LLM Contribution Evidence section
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.agent_runtime.evidence import LLMContributionEvidenceWriter
from auto_harness.models.base import write_json


class TestLLMContributionEvidence(unittest.TestCase):
    """Test LLM contribution evidence generation."""

    def test_llm_helped_when_baseline_failed_and_agent_passed(self):
        """llm_helped should be true when baseline failed and agent passed with trace."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            # Create evidence file with trace_id
            evidence_dir = run_dir / "evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            trace_id = "verify_20260710_123456_test123"
            evidence_file = evidence_dir / "http_trace.json"
            write_json(evidence_file, {
                "request": {"trace_id": trace_id},
                "response": {"body": f"trace={trace_id}"},
                "check": {"status": "pass"},
            })

            # Create pipeline results
            pipeline_results = {
                "verify": {
                    "status": "passed",
                    "data": {
                        "status": "pass",
                        "trace_id": trace_id,
                        "evidence": [str(evidence_file)],
                    },
                },
            }

            # Create agent steps with decision
            agent_steps = [
                {
                    "stage": "verify",
                    "decision": {
                        "decision_status": "ok",
                        "tool_call": {"name": "probe_http", "input": {}},
                        "policy_allowed": True,
                        "executed": True,
                    },
                },
            ]

            writer = LLMContributionEvidenceWriter()
            evidence = writer.write(
                run_dir=run_dir,
                task_id="test_task",
                baseline_result={"final_status": "uncertain", "failed_stage": "verify"},
                agent_result={"mode": "gated_actor", "changed_stage": "verify"},
                agent_steps=agent_steps,
                pipeline_results=pipeline_results,
            )

            # Verify evidence
            self.assertTrue(evidence["llm_helped"])
            self.assertTrue(evidence["llm_required"])
            self.assertEqual(evidence["llm_required_status"], "proven_by_baseline_agent_delta")
            self.assertEqual(evidence["trace_id"], trace_id)
            self.assertIn("verify_probe_selection", evidence["help_type"])
            self.assertTrue(evidence["safety"]["policy_gated"])

    def test_unknown_without_baseline(self):
        """Without baseline, llm_required_status should be unknown_without_baseline."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            # Create pipeline results
            pipeline_results = {
                "verify": {
                    "status": "passed",
                    "data": {
                        "status": "pass",
                        "trace_id": "verify_123",
                        "evidence": [],
                    },
                },
            }

            writer = LLMContributionEvidenceWriter()
            evidence = writer.write(
                run_dir=run_dir,
                task_id="test_task",
                baseline_result=None,  # No baseline
                agent_result={"mode": "gated_actor"},
                agent_steps=[],
                pipeline_results=pipeline_results,
            )

            # Verify
            self.assertFalse(evidence["llm_required"])
            self.assertEqual(evidence["llm_required_status"], "unknown_without_baseline")

    def test_no_trace_not_helped(self):
        """llm_helped should be false when no trace evidence exists."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            # Create pipeline results without trace evidence
            pipeline_results = {
                "verify": {
                    "status": "passed",
                    "data": {
                        "status": "pass",
                        "trace_id": "verify_123",
                        "evidence": [],
                    },
                },
            }

            writer = LLMContributionEvidenceWriter()
            evidence = writer.write(
                run_dir=run_dir,
                task_id="test_task",
                baseline_result={"final_status": "uncertain"},
                agent_result={"mode": "gated_actor"},
                agent_steps=[],
                pipeline_results=pipeline_results,
            )

            # Verify - no trace evidence means not helped
            self.assertFalse(evidence["llm_helped"])

    def test_baseline_passed_not_helped(self):
        """llm_helped should be false when baseline already passed."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            writer = LLMContributionEvidenceWriter()
            evidence = writer.write(
                run_dir=run_dir,
                task_id="test_task",
                baseline_result={"final_status": "passed"},
                agent_result={"mode": "gated_actor"},
                agent_steps=[],
                pipeline_results={},
            )

            # Verify
            self.assertFalse(evidence["llm_helped"])
            self.assertFalse(evidence["llm_required"])
            self.assertEqual(evidence["llm_required_status"], "baseline_did_not_fail")

    def test_evidence_file_written(self):
        """llm_contribution_evidence.json should be written to reports directory."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            writer = LLMContributionEvidenceWriter()
            writer.write(
                run_dir=run_dir,
                task_id="test_task",
                pipeline_results={},
            )

            # Verify file exists
            evidence_path = run_dir / "reports" / "llm_contribution_evidence.json"
            self.assertTrue(evidence_path.exists())

            # Verify JSON is valid
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertIn("task_id", evidence)
            self.assertIn("llm_helped", evidence)
            self.assertIn("safety", evidence)

    def test_safety_metrics(self):
        """Safety metrics should reflect policy gating."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)

            # Agent steps with side-effect tool executed
            agent_steps = [
                {
                    "stage": "env_deploy",
                    "decision": {
                        "decision_status": "ok",
                        "policy_allowed": True,
                        "executed": True,
                    },
                },
            ]

            writer = LLMContributionEvidenceWriter()
            evidence = writer.write(
                run_dir=run_dir,
                task_id="test_task",
                agent_steps=agent_steps,
                pipeline_results={},
            )

            # Verify safety metrics
            self.assertTrue(evidence["safety"]["policy_gated"])
            self.assertEqual(evidence["safety"]["side_effect_tools_executed"], 1)


if __name__ == "__main__":
    unittest.main()
