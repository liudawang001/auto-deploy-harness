"""Tool policy boundary tests.

Validates that:
1. planner mode requests side-effect tool are rejected with planner reason
2. gated_actor mode requests unknown tool are rejected
3. Requests with external URL are rejected
4. LLM tool input with shell command fields are rejected
5. Verify tool returning HTTP 200 without trace cannot pass
6. read_only tools allowed in planner mode
7. Unimplemented tools rejected before risk check
8. Stage mismatch has stage reason
9. Path traversal has path reason
10. Side-effect tools require runtime permission
"""
import unittest

from auto_harness.agent_runtime.policy import ToolPolicy
from auto_harness.agent_runtime.schemas import ToolCall
from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.schemas import ToolSchema


class TestToolPolicyBoundaries(unittest.TestCase):
    """Test tool policy enforcement boundaries."""

    def setUp(self):
        self.registry = ToolRegistry()
        self.policy = ToolPolicy(registry=self.registry, allowed_hosts=["127.0.0.1", "localhost", "::1"])

    def _make_implemented(self, name):
        """Helper: mark a tool as implemented for testing."""
        tool = self.registry.tools.get(name)
        if tool:
            tool.implemented = True
            tool.executor = "test"
            if not tool.stages:
                tool.stages = ["verify"]

    def test_planner_mode_side_effect_tool_rejected(self):
        """planner mode requesting side-effect tool should be rejected."""
        # Make install_environment implemented so we get past the implemented check
        self._make_implemented("install_environment")
        self.registry.tools["install_environment"].stages = ["env_deploy"]
        tool_call = ToolCall(
            name="install_environment",
            input={"package": "numpy"},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="planner",
        )
        # planner mode does not execute non-read-only tools
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
            name="probe_http",
            input={
                "cmd": "rm -rf / ; cat /etc/passwd",
                "trace_template": "test_{{trace_id}}",
            },
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="verify",
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

    def test_read_only_tool_allowed_in_planner(self):
        """read_only tools should be allowed in planner mode."""
        # Make inspect_repo_tree implemented for this test
        self._make_implemented("inspect_repo_tree")
        self.registry.tools["inspect_repo_tree"].stages = ["analyze"]
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

    def test_path_traversal_has_path_reason(self):
        """Tool input with path traversal should be rejected with path reason."""
        # Make read_selected_files implemented for this test
        self._make_implemented("read_selected_files")
        self.registry.tools["read_selected_files"].stages = ["analyze"]
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

    def test_unimplemented_tool_rejected_before_risk(self):
        """Unimplemented tools should be rejected before risk/category checks."""
        tool_call = ToolCall(
            name="inspect_repo_tree",  # not implemented
            input={"path": "."},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="analyze",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("not implemented", decision.reason.lower())

    def test_stage_mismatch_has_stage_reason(self):
        """Tool not allowed for a stage should have stage-specific reason."""
        tool_call = ToolCall(
            name="probe_http",
            input={"url": "http://127.0.0.1/", "trace_template": "t_{{trace_id}}"},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="analyze",  # probe_http is verify-stage only
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("stage", decision.reason.lower())

    def test_windows_path_traversal_rejected(self):
        """Windows-style path traversal should be rejected."""
        self._make_implemented("read_selected_files")
        self.registry.tools["read_selected_files"].stages = ["analyze"]
        tool_call = ToolCall(
            name="read_selected_files",
            input={"path": "..\\..\\windows\\system32"},
        )
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="analyze",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("traversal", decision.reason.lower())

    def test_side_effect_requires_runtime_permission(self):
        """Side-effect tools require runtime permission even if policy allows."""
        self._make_implemented("install_environment")
        self.registry.tools["install_environment"].stages = ["env_deploy"]
        tool_call = ToolCall(
            name="install_environment",
            input={"package": "numpy"},
        )
        # Without runtime permission
        decision = self.policy.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="gated_actor",
        )
        self.assertFalse(decision.allowed)
        self.assertIn("runtime permission", decision.reason.lower())

        # With runtime permission
        policy_with_perm = ToolPolicy(
            registry=self.registry,
            runtime_policy={"allow_dependency_install": True},
        )
        decision = policy_with_perm.validate(
            tool_call=tool_call,
            stage="env_deploy",
            agent_mode="gated_actor",
        )
        # Should pass the runtime permission check (may fail on other checks)
        if not decision.allowed:
            self.assertNotIn("runtime permission", decision.reason.lower())

    def test_external_host_rejected(self):
        """External host in URL should be rejected."""
        tool_call = ToolCall(
            name="probe_http",
            input={
                "url": "http://evil.example.com:8080/",
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
        self._make_implemented("install_environment")
        self.registry.tools["install_environment"].stages = ["env_deploy"]
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
        # This is a design invariant test - verify_evidence test - verify_evidence returns evidence, not pass/fail
        registry = ToolRegistry()
        verify_tool = registry.get("verify_evidence")
        self.assertIn("trace", verify_tool.get("success_signal", "").lower())


if __name__ == "__main__":
    unittest.main()
