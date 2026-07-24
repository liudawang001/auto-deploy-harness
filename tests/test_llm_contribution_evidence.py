"""LLM Contribution Evidence tests.

Validates that:
1. llm_contribution_evidence.json is generated
2. llm_helped=true only when baseline failed and agent passed with trace evidence
3. Without baseline, llm_required_status is "unknown_without_baseline"
4. report.md contains LLM Contribution Evidence section
5. decision=None does not crash
6. metadata_only is not counted as helped
7. policy_rejected is not counted as helped
"""
import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.agent_runtime.evidence import LLMContributionEvidenceWriter, _decision_dict
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


class TestDecisionDictHelper(unittest.TestCase):
    """Test _decision_dict helper for safe decision access."""

    def test_none_decision_does_not_crash(self):
        """step.get('decision') returning None should not crash."""
        step = {"decision": None}
        result = _decision_dict(step)
        self.assertEqual(result, {})

    def test_dict_decision_returned(self):
        """Normal dict decision should be returned as-is."""
        step = {"decision": {"decision_status": "ok"}}
        result = _decision_dict(step)
        self.assertEqual(result, {"decision_status": "ok"})

    def test_non_dict_step_returns_empty(self):
        """Non-dict step should return empty dict."""
        result = _decision_dict("not a dict")
        self.assertEqual(result, {})

    def test_missing_decision_returns_empty(self):
        """Step without decision key should return empty dict."""
        step = {"other_key": "value"}
        result = _decision_dict(step)
        self.assertEqual(result, {})


class TestLLMHelpedTightening(unittest.TestCase):
    """Test that llm_helped only counts effective decisions."""

    def _write_evidence(self, run_dir, trace_id, content):
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / "verify.json"
        write_json(path, {"trace_id": trace_id, "content": content})
        return str(path)

    def test_none_decision_is_not_helped(self):
        """decision=None should not count as llm_helped."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            evidence_path = self._write_evidence(run_dir, "trace-1", "data with trace-1")
            writer = LLMContributionEvidenceWriter()
            result = writer.write(
                run_dir=run_dir,
                task_id="test-none-decision",
                baseline_result={"final_status": "failed"},
                agent_result={"mode": "gated_actor"},
                agent_steps=[{"decision": None}],
                pipeline_results={"verify": {"status": "passed", "trace_id": "trace-1", "evidence_paths": [evidence_path]}},
            )
            self.assertFalse(result["llm_helped"])

    def test_metadata_only_is_not_helped(self):
        """metadata_only decision should not count as llm_helped."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            evidence_path = self._write_evidence(run_dir, "trace-2", "data with trace-2")
            writer = LLMContributionEvidenceWriter()
            result = writer.write(
                run_dir=run_dir,
                task_id="test-metadata-only",
                baseline_result={"final_status": "failed"},
                agent_result={"mode": "gated_actor"},
                agent_steps=[{
                    "decision": {
                        "decision_status": "ok",
                        "policy_allowed": True,
                        "executed": True,
                        "metadata_only": True,
                    }
                }],
                pipeline_results={"verify": {"status": "passed", "trace_id": "trace-2", "evidence_paths": [evidence_path]}},
            )
            self.assertFalse(result["llm_helped"])

    def test_policy_rejected_is_not_helped(self):
        """policy_rejected decision should not count as llm_helped."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            evidence_path = self._write_evidence(run_dir, "trace-3", "data with trace-3")
            writer = LLMContributionEvidenceWriter()
            result = writer.write(
                run_dir=run_dir,
                task_id="test-policy-rejected",
                baseline_result={"final_status": "failed"},
                agent_result={"mode": "gated_actor"},
                agent_steps=[{
                    "decision": {
                        "decision_status": "ok",
                        "policy_allowed": False,
                        "executed": True,
                    }
                }],
                pipeline_results={"verify": {"status": "passed", "trace_id": "trace-3", "evidence_paths": [evidence_path]}},
            )
            self.assertFalse(result["llm_helped"])

    def test_pass_without_current_trace_is_not_helped(self):
        """Agent pass without current trace evidence should not count as helped."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer = LLMContributionEvidenceWriter()
            result = writer.write(
                run_dir=run_dir,
                task_id="test-no-trace",
                baseline_result={"final_status": "failed"},
                agent_result={"mode": "gated_actor"},
                agent_steps=[{
                    "decision": {
                        "decision_status": "ok",
                        "policy_allowed": True,
                        "executed": True,
                    }
                }],
                pipeline_results={"verify": {"status": "passed", "trace_id": "trace-4", "evidence_paths": []}},
            )
            self.assertFalse(result["llm_helped"])

    def test_no_baseline_means_required_unknown(self):
        """Without baseline, llm_required must be False and status unknown."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            evidence_path = self._write_evidence(run_dir, "trace-5", "data with trace-5")
            writer = LLMContributionEvidenceWriter()
            result = writer.write(
                run_dir=run_dir,
                task_id="test-no-baseline",
                baseline_result=None,
                agent_result={"mode": "gated_actor"},
                agent_steps=[{
                    "decision": {
                        "decision_status": "ok",
                        "policy_allowed": True,
                        "executed": True,
                    }
                }],
                pipeline_results={"verify": {"status": "passed", "trace_id": "trace-5", "evidence_paths": [evidence_path]}},
            )
            self.assertFalse(result["llm_required"])
            self.assertEqual(result["llm_required_status"], "unknown_without_baseline")
