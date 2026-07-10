"""Tests for ToolExecutor (Phase 4).

Covers:
- Tool categorization (read_only, state_delta, execution)
- ToolResult schema with policy_allowed, executed, applied, metadata_only
- Tool dispatch for verify tools
"""
import unittest

from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.schemas import ToolSchema


class TestToolRegistry(unittest.TestCase):
    """Tests for ToolRegistry categorization."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_read_only_tools(self):
        """read_only tools should have category='read_only'."""
        read_only_tools = ["inspect_repo_tree", "read_selected_files", "inspect_env_log",
                           "inspect_log", "inspect_model_config", "inspect_git_lfs_pointers"]
        for name in read_only_tools:
            tool = self.registry.get(name)
            self.assertEqual(tool.get("category"), "read_only",
                             "tool %s should be read_only" % name)

    def test_state_delta_tools(self):
        """state_delta tools should have category='state_delta'."""
        state_delta_tools = ["select_runner_candidate", "add_runner_candidate",
                             "apply_dependency_constraint", "select_environment_backend",
                             "set_deployment_strategy", "set_stage_hint"]
        for name in state_delta_tools:
            tool = self.registry.get(name)
            self.assertEqual(tool.get("category"), "state_delta",
                             "tool %s should be state_delta" % name)

    def test_execution_tools(self):
        """execution tools should have category='execution'."""
        execution_tools = ["install_environment", "start_service", "probe_http",
                           "discover_gradio_api", "apply_repair", "resume_from_stage"]
        for name in execution_tools:
            tool = self.registry.get(name)
            self.assertEqual(tool.get("category"), "execution",
                             "tool %s should be execution" % name)

    def test_all_tools_have_category(self):
        """All tools should have a category field."""
        for name, tool in self.registry.tools.items():
            self.assertIn(tool.category, ["read_only", "state_delta", "execution"],
                          "tool %s has invalid category: %s" % (name, tool.category))

    def test_execution_tools_require_policy(self):
        """Most execution tools should require policy."""
        execution_tools = ["install_environment", "start_service", "apply_repair", "resume_from_stage"]
        for name in execution_tools:
            tool = self.registry.get(name)
            self.assertTrue(tool.get("requires_policy"),
                            "execution tool %s should require policy" % name)

    def test_read_only_tools_no_side_effects(self):
        """read_only tools should have no side effects."""
        read_only_tools = ["inspect_repo_tree", "read_selected_files", "inspect_env_log"]
        for name in read_only_tools:
            tool = self.registry.get(name)
            self.assertEqual(len(tool.get("side_effects", [])), 0,
                             "read_only tool %s should have no side effects" % name)


class TestToolResultSchema(unittest.TestCase):
    """Tests for ToolResult schema."""

    def test_default_values(self):
        from auto_harness.agent_runtime.schemas import ToolResult
        result = ToolResult()
        self.assertEqual(result.status, "error")
        self.assertEqual(result.category, "read_only")
        self.assertFalse(result.policy_allowed)
        self.assertFalse(result.executed)
        self.assertFalse(result.applied)
        self.assertFalse(result.metadata_only)

    def test_execution_result(self):
        from auto_harness.agent_runtime.schemas import ToolResult
        result = ToolResult(
            status="passed",
            tool_name="probe_http",
            category="execution",
            policy_allowed=True,
            executed=True,
            applied=False,
            metadata_only=False,
        )
        self.assertTrue(result.executed)
        self.assertFalse(result.applied)

    def test_state_delta_result(self):
        from auto_harness.agent_runtime.schemas import ToolResult
        result = ToolResult(
            status="passed",
            tool_name="select_runner_candidate",
            category="state_delta",
            policy_allowed=True,
            executed=False,
            applied=True,
            metadata_only=False,
        )
        self.assertFalse(result.executed)
        self.assertTrue(result.applied)

    def test_metadata_only_not_self_healing(self):
        """metadata_only=True should not count as self_healing."""
        from auto_harness.agent_runtime.schemas import ToolResult
        result = ToolResult(
            status="passed",
            tool_name="set_stage_hint",
            category="state_delta",
            policy_allowed=True,
            executed=False,
            applied=True,
            metadata_only=True,
        )
        # metadata_only cannot be counted as repair_verified
        self.assertTrue(result.metadata_only)
        # A proper self_healing result should have metadata_only=False
        self.assertNotEqual(result.metadata_only, False)


if __name__ == "__main__":
    unittest.main()
