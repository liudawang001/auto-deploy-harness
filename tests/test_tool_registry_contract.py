"""Contract tests for ToolRegistry category unification and ToolExecutor dispatch.

Validates:
- Only four valid categories exist
- ToolResult never writes 'execution' category
- Unimplemented tools are not executable
- Verify-stage executable tools have an executor
- Executor contract validation catches mismatches
- Policy rejected tools don't reach handlers
- Result category comes from registry schema
"""
import pytest

from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.tools.executor import ToolExecutor
from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.schemas import TOOL_CATEGORIES, ToolSchema


class TestToolCategories:
    """Test that tool categories are unified and consistent."""

    def test_categories_are_from_one_enum(self):
        """Only the four canonical categories are allowed."""
        assert TOOL_CATEGORIES == (
            "read_only",
            "state_delta",
            "side_effect",
            "evidence",
        )

    def test_registry_uses_only_valid_categories(self):
        """Every tool in the registry must use a valid category."""
        registry = ToolRegistry()
        for name, tool in registry.tools.items():
            assert tool.category in TOOL_CATEGORIES, (
                "tool %s has invalid category %s" % (name, tool.category)
            )

    def test_new_tool_result_never_writes_execution_category(self):
        """New ToolResult instances must never have category='execution'."""
        result = ToolResult(status="passed", tool_name="probe_http")
        assert result.category != "execution"
        assert result.category in TOOL_CATEGORIES

    def test_tool_result_default_category_is_read_only(self):
        """Default ToolResult category must be read_only."""
        result = ToolResult()
        assert result.category == "read_only"

    def test_tool_schema_default_category_is_read_only(self):
        """Default ToolSchema category must be read_only."""
        schema = ToolSchema(name="test_tool")
        assert schema.category == "read_only"


class TestImplementedTools:
    """Test that implemented flag and executor are consistent."""

    def test_unimplemented_tool_is_not_executable(self):
        """Tools with implemented=False must not appear in executable_for_stage."""
        registry = ToolRegistry()
        # inspect_repo_tree is not implemented
        executable = registry.executable_for_stage("analyze", agent_mode="gated_actor")
        names = [t["name"] for t in executable]
        assert "inspect_repo_tree" not in names

    def test_verify_executable_tools_have_executor(self):
        """Every implemented tool for verify stage must have an executor."""
        registry = ToolRegistry()
        executable = registry.executable_for_stage("verify", agent_mode="gated_actor")
        for tool in executable:
            assert tool.get("executor"), (
                "implemented tool %s has no executor" % tool["name"]
            )

    def test_only_four_tools_are_implemented(self):
        """First version: only probe_http, discover_gradio_api,
        discover_openapi_schema, probe_browser_dom are implemented."""
        registry = ToolRegistry()
        implemented = [
            name for name, tool in registry.tools.items() if tool.implemented
        ]
        assert sorted(implemented) == sorted([
            "probe_http",
            "discover_gradio_api",
            "discover_openapi_schema",
            "probe_browser_dom",
        ])

    def test_unimplemented_tools_have_no_executor(self):
        """Unimplemented tools must not have an executor string."""
        registry = ToolRegistry()
        for name, tool in registry.tools.items():
            if not tool.implemented:
                assert tool.executor == "", (
                    "unimplemented tool %s has executor %s" % (name, tool.executor)
                )

    def test_executable_for_stage_respects_mode(self):
        """Tools with restricted allowed_modes must not appear for wrong mode."""
        registry = ToolRegistry()
        # probe_browser_dom requires planner or gated_actor
        executable = registry.executable_for_stage("verify", agent_mode="off")
        names = [t["name"] for t in executable]
        assert "probe_browser_dom" not in names

    def test_executable_for_stage_respects_stage(self):
        """Tools must only appear for stages they declare."""
        registry = ToolRegistry()
        # All implemented tools are verify-stage only
        executable = registry.executable_for_stage("analyze", agent_mode="gated_actor")
        names = [t["name"] for t in executable]
        assert "probe_http" not in names


class TestCategoryMigration:
    """Test that legacy 'execution' category is migrated on read."""

    def test_execution_category_migrated_to_side_effect(self):
        """When reading old data with category='execution', it should
        be treated as 'side_effect'."""
        from auto_harness.agent_runtime.evidence import _migrate_execution_category
        assert _migrate_execution_category("execution") == "side_effect"
        assert _migrate_execution_category("read_only") == "read_only"
        assert _migrate_execution_category("side_effect") == "side_effect"


class TestExecutorContract:
    """Test that ToolExecutor contract validation works correctly."""

    def test_executor_contract_passes_for_default_registry(self):
        """Default executor with default registry should pass contract validation."""
        executor = ToolExecutor()
        executor.validate_contract()  # Should not raise

    def test_missing_handler_fails_fast(self):
        """If a tool is marked implemented but has no handler, contract fails."""
        registry = ToolRegistry()
        # Force a tool to be implemented without a handler
        registry.tools["force_implemented"] = ToolSchema(
            name="force_implemented",
            implemented=True,
            executor="missing",
            stages=["verify"],
        )
        executor = ToolExecutor(registry=registry)
        with pytest.raises(RuntimeError, match="missing=\\['force_implemented'\\]"):
            executor.validate_contract()

    def test_unknown_handler_fails_fast(self):
        """If a handler is registered for an unimplemented tool, contract fails."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)
        executor.register("ghost_tool", lambda inp, ctx: ToolResult())
        with pytest.raises(RuntimeError, match="unknown=\\['ghost_tool'\\]"):
            executor.validate_contract()

    def test_unimplemented_tool_is_rejected(self):
        """Executing an unimplemented tool returns rejected status."""
        executor = ToolExecutor()
        tool_call = ToolCall(name="inspect_repo_tree", input={})
        result = executor.execute(tool_call, {})
        assert result.status == "rejected"
        assert "not implemented" in result.error

    def test_policy_rejected_tool_does_not_reach_handler(self):
        """If policy rejects, execute() is never called.
        This is enforced by the caller (AgentRuntime), but we verify
        that the executor correctly rejects unimplemented tools."""
        executor = ToolExecutor()
        tool_call = ToolCall(name="nonexistent_tool", input={})
        result = executor.execute(tool_call, {})
        assert result.status == "rejected"

    def test_result_category_comes_from_registry(self):
        """Execute result category must match the registry schema, not the handler."""
        executor = ToolExecutor()
        # probe_http is in the 'evidence' category per registry
        tool_call = ToolCall(
            name="probe_http",
            input={"url": "http://127.0.0.1:8000/", "trace_template": "_trace={{trace_id}}"},
        )
        context = {
            "trace_id": "test-trace-123",
            "evidence_dir": "/tmp/test_evidence",
        }
        result = executor.execute(tool_call, context)
        assert result.category == "evidence"

    def test_register_non_callable_raises_type_error(self):
        """Registering a non-callable handler must raise TypeError."""
        executor = ToolExecutor()
        with pytest.raises(TypeError, match="must be callable"):
            executor.register("bad_handler", "not_a_function")

    def test_custom_handler_overrides_default(self):
        """Custom handlers passed in constructor override defaults."""
        custom_result = ToolResult(status="passed", tool_name="probe_http")
        executor = ToolExecutor(
            handlers={"probe_http": lambda inp, ctx: custom_result}
        )
        tool_call = ToolCall(name="probe_http", input={})
        result = executor.execute(tool_call, {})
        assert result is custom_result
