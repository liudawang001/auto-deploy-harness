"""Tests for PlanCompiler.

Phase 3 of LLM Plan-first Deployment Agent.

Covers:
- install_commands compile to analysis.install_plan
- run candidates compile to analysis.run_candidates
- selected candidate ranked first
- verify request compile to analysis.verify_hint
- deterministic facts preserved
- rejected items not compiled
- LLM candidates prepend deterministic
"""
import json
import unittest

from auto_harness.agent_runtime.plan_compiler import PlanCompiler


# A normalized plan (after policy gate)
NORMALIZED_PLAN = {
    "status": "ok",
    "plan_id": "plan_http_trace",
    "summary": "Run HTTP trace echo in venv",
    "grounding": [
        {"claim": "app.py is entrypoint", "file": "app.py", "reason": "contains HTTPServer"}
    ],
    "environment": {
        "backend": "venv",
        "python": "3.10",
        "install_commands": [
            ["python3", "-m", "venv", ".venv"],
            [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
        ],
    },
    "model_assets": {"required": False, "strategy": "none", "env_vars": []},
    "run": {
        "candidates": [
            {"id": "llm_app_py", "cmd": [".venv/bin/python", "app.py"], "expected_port": 8917, "reason": "app.py starts HTTPServer on 8917"},
            {"id": "llm_main_py", "cmd": [".venv/bin/python", "main.py"], "expected_port": 7860, "reason": "fallback to main.py"},
        ],
        "selected_candidate_id": "llm_app_py",
    },
    "verify": {
        "service_type": "http",
        "request": {"method": "GET", "path": "/?_auto_harness_trace={{trace_id}}"},
        "success_evidence": "response contains trace_id",
    },
    "risks": [],
    "fallbacks": [],
}

DETERMINISTIC_ANALYSIS = {
    "files": ["app.py", "main.py", "requirements.txt", "README.md"],
    "frameworks": ["http.server"],
    "install_plan": [["python3", "-m", "venv", ".venv"], [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"]],
    "run_candidates": [
        {"cmd": [".venv/bin/python", "app.py"], "expected_port": 8917, "confidence": 0.75, "selected_by": "deterministic"},
        {"cmd": [".venv/bin/python", "main.py"], "expected_port": 7860, "confidence": 0.7, "selected_by": "deterministic"},
    ],
    "verify_hint": {"service_type": "http", "expected_output": "trace_echo", "request": {"method": "GET", "path": "/?_auto_harness_trace={{trace_id}}"}},
    "environment_strategy": {"backend": "venv", "preferred_tool": "venv", "python": "python3", "source": "deterministic"},
    "deterministic_facts": {
        "file_count": 4,
        "frameworks": ["http.server"],
        "has_requirements": True,
        "has_environment_yml": False,
    },
}


class TestPlanCompiler(unittest.TestCase):
    """Test PlanCompiler compilation logic."""

    def setUp(self):
        self.compiler = PlanCompiler()

    def test_install_commands_compile(self):
        """LLM install_commands should compile to analysis.install_plan."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        self.assertEqual(analysis["install_plan"], NORMALIZED_PLAN["environment"]["install_commands"])

    def test_run_candidates_compile(self):
        """LLM run.candidates should compile to analysis.run_candidates."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        self.assertTrue(len(analysis["run_candidates"]) >= 2)
        # First candidate should be the selected one
        self.assertEqual(analysis["run_candidates"][0]["id"], "llm_app_py")
        self.assertEqual(analysis["run_candidates"][0]["selected_by"], "llm_plan_first")

    def test_selected_candidate_first(self):
        """Selected candidate should be ranked first in run_candidates."""
        # Plan with selected_candidate_id pointing to the second candidate
        plan = json.loads(json.dumps(NORMALIZED_PLAN))
        plan["run"]["selected_candidate_id"] = "llm_main_py"
        result = self.compiler.compile(plan, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        self.assertEqual(analysis["run_candidates"][0]["id"], "llm_main_py")
        self.assertEqual(analysis["selected_candidate"]["id"], "llm_main_py")

    def test_verify_hint_compile(self):
        """LLM verify.request should compile to analysis.verify_hint."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        self.assertEqual(analysis["verify_hint"]["service_type"], "http")
        self.assertEqual(analysis["verify_hint"]["request"]["method"], "GET")
        self.assertIn("{{trace_id}}", analysis["verify_hint"]["request"]["path"])

    def test_deterministic_facts_preserved(self):
        """Deterministic facts should be preserved in compiled output."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        self.assertEqual(analysis["frameworks"], ["http.server"])
        self.assertEqual(analysis["files"], ["app.py", "main.py", "requirements.txt", "README.md"])
        self.assertTrue(analysis["deterministic_facts"]["has_requirements"])

    def test_rejected_items_not_compiled(self):
        """Items removed by policy gate should not appear in compiled output."""
        # Simulate a plan where install_commands was cleared by policy
        plan = json.loads(json.dumps(NORMALIZED_PLAN))
        plan["environment"]["install_commands"] = []  # policy removed all commands
        result = self.compiler.compile(plan, DETERMINISTIC_ANALYSIS)
        # Should fall back to deterministic install_plan
        analysis = result["analysis"]
        self.assertEqual(analysis["install_plan"], DETERMINISTIC_ANALYSIS["install_plan"])

    def test_llm_candidates_prepend_deterministic(self):
        """LLM candidates should come before deterministic candidates."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        candidates = analysis["run_candidates"]
        # LLM candidates (2) should come before deterministic ones
        llm_count = sum(1 for c in candidates if c.get("selected_by") == "llm_plan_first")
        self.assertEqual(llm_count, 2)
        # First candidate should be LLM-selected
        self.assertEqual(candidates[0]["selected_by"], "llm_plan_first")

    def test_selection_source_is_llm_plan_first(self):
        """selection_source should be llm_plan_first when LLM plan is used."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        self.assertEqual(analysis["selection_source"], "llm_plan_first")

    def test_effective_plan_in_output(self):
        """Output should include the effective_plan for artifact writing."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        self.assertIn("effective_plan", result)
        self.assertEqual(result["effective_plan"]["plan_id"], "plan_http_trace")

    def test_environment_strategy_from_llm(self):
        """Environment strategy should come from LLM plan when specified."""
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        strategy = analysis["environment_strategy"]
        self.assertEqual(strategy["source"], "llm_plan_first")
        self.assertEqual(strategy["backend"], "venv")

    def test_no_deterministic_analysis(self):
        """Compiler should work without deterministic analysis."""
        result = self.compiler.compile(NORMALIZED_PLAN)
        analysis = result["analysis"]
        self.assertEqual(analysis["install_plan"], NORMALIZED_PLAN["environment"]["install_commands"])
        self.assertEqual(len(analysis["run_candidates"]), 2)
        self.assertEqual(analysis["selection_source"], "llm_plan_first")

    def test_empty_model_assets_override_repository_wide_optional_detections(self):
        plan = json.loads(json.dumps(NORMALIZED_PLAN))
        plan["model_assets"] = {}
        result = self.compiler.compile(plan, DETERMINISTIC_ANALYSIS)
        self.assertIn("model_assets", result["analysis"])
        self.assertEqual(result["analysis"]["model_assets"], {})

    def test_duplicate_candidates_deduped(self):
        """Deterministic candidates duplicating LLM candidates should be removed."""
        # Deterministic has same cmd as LLM candidate
        result = self.compiler.compile(NORMALIZED_PLAN, DETERMINISTIC_ANALYSIS)
        analysis = result["analysis"]
        candidates = analysis["run_candidates"]
        # Should not have duplicate .venv/bin/python app.py entries
        cmds = [tuple(c.get("cmd", [])) for c in candidates]
        self.assertEqual(len(cmds), len(set(cmds)), "Duplicate commands found in run_candidates")


if __name__ == "__main__":
    unittest.main()
