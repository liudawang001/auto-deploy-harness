"""Tool policy boundary tests.

Validates that:
1. planner mode requests side-effect tool are recorded as would_execute, not executed
2. gated_actor mode requests unknown tool are rejected
3. Requests with external URL are rejected
4. LLM tool input with shell command fields are rejected
5. Verify tool returning HTTP 200 without trace cannot pass
"""
import unittest

from auto_harness.agent_runtime.policy import ToolPolicy
from auto_harness.agent_runtime.schemas import ToolCall
from auto_harness.tools.registry import ToolRegistry


class TestToolPolicyBoundaries(unittest.TestCase):
    """Test tool policy enforcement boundaries."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.policy = ToolPolicy(registry=self.registry, allowed_hosts=["127.0.0.1", "localhost", "::1"])

    def test_planner_mode_side_effect_tool_rejected(self):
        """planner mode requesting side-effect tool should be rejected."""
        tool_call = ToolCall(
            name="install_environment",
            input={"package": "numpy"},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="planner",
        )
        # planner mode does not execute tools
        self.assertFalse(decision.allowed)
        self.assertIn("planner", decision.reason.lower())

    def test_gated_actor_unknown_tool_rejected(self):
        """gated_actor mode requesting unknown tool should be rejected."""
        tool_call = ToolCall(
            name="unknown_tool_that_does_not_exist",
            input={"param": "value"},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("not found", decision.reason.lower())

    def test_external_url_rejected(self):
        """Request with external URL should be rejected."""
        tool_call = ToolCall(
            name="probe_http",
            input={
                "url": "https://evil.example.com/steal",
                "trace_template": "test_{{trace_id}}",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("external", decision.reason.lower())

    def test_shell_command_field_rejected(self):
        """LLM tool input with shell command field containing metacharacters should be rejected."""
        tool_call = ToolCall(
            name="probe_http",
            input={
                "command": "rm -rf / ; cat /etc/passwd",
                "trace_template": "test_{{trace_id}}",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("metacharacters", decision.reason.lower())

    def test_shell_in_cmd_field_rejected(self):
        """LLM tool input with cmd field containing shell metacharacters should be rejected."""
        tool_call = ToolCall(
            name="install_environment",
            input={
                "package": "numpy; rm -rf /",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)

    def test_verify_probe_without_trace_rejected(self):
        """Verify probe tool without trace_template should be rejected."""
        tool_call = ToolCall(
            name="probe_http",
            input={
                "url": "http://127.0.0.1:8080/",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("trace_template", decision.reason.lower())

    def test_local_url_allowed(self):
        """Request with local URL should be allowed."""
        tool_call = ToolCall(
            name="probe_http",
            input={
                "url": "http://127.0.0.1:8080/",
                "trace_template": "test_{{trace_id}}",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        self.assertTrue(decision.allowed)

    def test_read_only_tool_any_mode(self):
        """read_only tools should be allowed in any mode."""
        tool_call = ToolCall(
            name="inspect_repo_tree",
            input={"path": "."},
        )
        # In planner mode, read_only tools are allowed (for observation)
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="analyze",
            agent_mode="planner",
        )
        self.assertTrue(decision.allowed)

        # In gated_actor mode
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="analyze",
            agent_mode="gated_actor",
        )
        self.assertTrue(decision.allowed)

    def test_secret_field_rejected(self):
        """Tool input with secret-like field should be rejected."""
        tool_call = ToolCall(
            name="probe_http",
            input={
                "url": "http://127.0.0.1:8080/",
                "api_key": "sk-1234567890",
                "trace_template": "test_{{trace_id}}",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("secret", decision.reason.lower())

    def test_path_traversal_rejected(self):
        """Tool input with path traversal should be rejected."""
        tool_call = ToolCall(
            name="read_selected_files",
            input={"path": "../../../etc/passwd"},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="analyze",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("traversal", decision.reason.lower())

    def test_tool_registry_categories(self):
        """Tool registry should have correct categories."""
        registry = ToolRegistry()

        # read_only tools
        read_only_tools = ["inspect_repo_tree", "read_selected_files", "parse_dependency_files",
                          "inspect_log", "classify_failure"]
        for tool_name in read_only_tools:
            tool = registry.get(tool_name)
            self.assertEqual(tool.get("category"), "read_only", f"{tool_name} should be read_only")

        # state_delta tools
        state_delta_tools = ["select_runner_candidate", "add_runner_candidate",
                            "set_deployment_strategy", "set_stage_hint", "propose_repair"]
        for tool_name in state_delta_tools:
            tool = registry.get(tool_name)
            self.assertEqual(tool.get("category"), "state_delta", f"{tool_name} should be state_delta")

        # side_effect tools
        side_effect_tools = ["install_environment", "start_service", "apply_repair"]
        for tool_name in side_effect_tools:
            tool = registry.get(tool_name)
            self.assertEqual(tool.get("category"), "side_effect", f"{tool_name} should be side_effect")
            self.assertTrue(tool.get("requires_policy"), f"{tool_name} should require policy")

        # evidence tools
        evidence_tools = ["probe_http", "discover_gradio_api", "verify_evidence"]
        for tool_name in evidence_tools:
            tool = registry.get(tool_name)
            self.assertEqual(tool.get("category"), "evidence", f"{tool_name} should be evidence")

    def test_side_effect_tool_requires_gated_actor(self):
        """Side-effect tools should only work in gated_actor mode."""
        tool_call = ToolCall(
            name="install_environment",
            input={"package": "numpy"},
        )

        # planner mode - rejected
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="planner",
        )
        self.assertFalse(decision.allowed)

        # gated_actor mode - should pass mode check (may fail other checks)
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="gated_actor",
        )
        # Should pass mode check (may have other rejections)
        # The key is that it doesn't reject on mode
        if not decision.allowed:
            self.assertNotIn("planner", decision.reason.lower())

    def test_verify_tool_cannot_declare_success(self):
        """Verify tools should not be able to declare final success directly.

        The verify tool returns evidence; final success is determined by Python evidence gate.
        """
        # This is a design invariant test - verify_evidence returns evidence, not pass/fail
        registry = ToolRegistry()
        verify_tool = registry.get("verify_evidence")
        self.assertIn("trace", verify_tool.get("success_signal", "").lower())


if __name__ == "__main__":
    unittest.main()
