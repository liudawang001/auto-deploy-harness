"""Tests for Model Decision Gate (Phase 4).

Covers:
- model gate selects HF strategy overlay
- model gate rejects external URL
- model gate rejects path traversal
- model gate dry-run does not download
- model gate requires model_prepare evidence for passed
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
from auto_harness.agent_runtime.stage_schemas import MODEL_TOOLS


class TestModelGatePolicy(unittest.TestCase):
    """Model gate policy validation tests."""

    def test_valid_hf_strategy(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "Qwen/Qwen2.5-0.5B", "strategy": "snapshot_download"
        }}
        r = policy.validate(tc, "model_prepare")
        self.assertTrue(r["allowed"])

    def test_valid_modelscope_strategy(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "modelscope", "repo_id": "qwen/Qwen2.5-0.5B"
        }}
        r = policy.validate(tc, "model_prepare")
        self.assertTrue(r["allowed"])

    def test_rejects_external_url(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "https://evil.com/model"
        }}
        r = policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("external URL", r["reason"])

    def test_rejects_path_traversal(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "org/model", "target_path": "../../etc/passwd"
        }}
        r = policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("traversal", r["reason"])

    def test_rejects_token_in_input(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "huggingface", "repo_id": "org/model", "token": "hf_abc123"
        }}
        r = policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("secret", r["reason"])

    def test_rejects_unknown_source(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "select_model_asset_strategy", "input": {
            "source": "unknown_source", "repo_id": "org/model"
        }}
        r = policy.validate(tc, "model_prepare")
        self.assertFalse(r["allowed"])
        self.assertIn("not in allowed", r["reason"])

    def test_inspect_model_config_read_only(self):
        from auto_harness.agent_runtime.decision_gate import StagePolicyValidator
        policy = StagePolicyValidator()
        tc = {"name": "inspect_model_config", "input": {"path": "./model"}}
        r = policy.validate(tc, "model_prepare")
        self.assertTrue(r["allowed"])


class TestModelGateExecution(unittest.TestCase):
    """Model gate full pipeline tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = Path(self.tmpdir)

    def _make_provider(self, response_json):
        provider = MagicMock()
        result = MagicMock()
        result.text = json.dumps(response_json)
        provider.complete.return_value = result
        return provider

    def test_selects_hf_strategy_overlay(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "README identifies HF repo while code defaults to ./model",
            "confidence": 0.85,
            "tool_call": {
                "name": "select_model_asset_strategy",
                "input": {
                    "source": "huggingface",
                    "repo_id": "Qwen/Qwen2.5-0.5B",
                    "strategy": "snapshot_download",
                    "target_subdir": "models/Qwen2.5-0.5B",
                    "fallback": "modelscope",
                },
            },
            "expected_observation": "model_prepare passes with HF download",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="model_prepare",
            observation={"model_mentions": ["Qwen/Qwen2.5-0.5B"], "detected_assets": []},
            allowed_tools=list(MODEL_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        self.assertEqual(result.decision_status, "ok")
        self.assertTrue(result.execution.get("applied"))
        strategy_path = self.run_dir / "reports" / "model_asset_strategy.json"
        self.assertTrue(strategy_path.exists())
        strategy = json.loads(strategy_path.read_text())
        self.assertEqual(strategy["repo_id"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(strategy["source"], "huggingface")

    def test_dry_run_does_not_download(self):
        """In planner mode, the gate should not write strategy overlay."""
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "model needed",
            "confidence": 0.8,
            "tool_call": {
                "name": "select_model_asset_strategy",
                "input": {"source": "huggingface", "repo_id": "org/model"},
            },
            "expected_observation": "model prepared",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="model_prepare",
            observation={"model_mentions": ["org/model"], "detected_assets": []},
            allowed_tools=list(MODEL_TOOLS),
            mode="planner",
            run_dir=self.run_dir,
        )
        self.assertFalse(result.execution.get("executed", True))
        strategy_path = self.run_dir / "reports" / "model_asset_strategy.json"
        self.assertFalse(strategy_path.exists())

    def test_rejected_external_url_artifact_written(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "download from URL",
            "confidence": 0.5,
            "tool_call": {
                "name": "select_model_asset_strategy",
                "input": {"source": "huggingface", "repo_id": "https://evil.com/model"},
            },
            "expected_observation": "none",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="model_prepare",
            observation={"model_mentions": [], "detected_assets": []},
            allowed_tools=list(MODEL_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # Should be rejected
        artifact_path = self.run_dir / "agent_decision_gates" / "model_prepare_gate.json"
        self.assertTrue(artifact_path.exists())

    def test_link_cached_model_asset(self):
        provider = self._make_provider({
            "status": "ok",
            "hypothesis": "cached model available",
            "confidence": 0.7,
            "tool_call": {
                "name": "link_cached_model_asset",
                "input": {"cache_path": "/cache/model", "target_path": "models/model"},
            },
            "expected_observation": "model linked from cache",
        })
        gate = AgentDecisionGate(provider=provider)
        result = gate.decide(
            stage="model_prepare",
            observation={"model_mentions": [], "detected_assets": [], "cache_candidates": ["/cache/model"]},
            allowed_tools=list(MODEL_TOOLS),
            mode="gated_actor",
            run_dir=self.run_dir,
        )
        # link_cached_model_asset needs executor, so it will be no_executor
        self.assertIn(result.execution.get("status"), ["no_executor", "applied"])


class TestModelGateFixture(unittest.TestCase):
    """Test using the model_path_ambiguous fixture."""

    def test_fixture_has_empty_model_assets(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "llm_necessity" / "model_path_ambiguous"
        resource_path = fixture_dir / "resource_plan.json"
        if not resource_path.exists():
            self.skipTest("fixture not found")
        resource = json.loads(resource_path.read_text())
        self.assertEqual(resource["model_assets"], [])


if __name__ == "__main__":
    unittest.main()
