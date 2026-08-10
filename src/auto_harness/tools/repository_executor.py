"""Executor for policy-normalized repository observation tools."""
from pathlib import Path
from typing import Any, Callable, Dict

from auto_harness.agent_runtime.schemas import ToolCall, ToolResult
from auto_harness.tools import repository_tools
from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.repository_policy import RepositoryReadPolicy


class RepositoryToolExecutor:
    def __init__(self, config: Any = None, registry=None, handlers=None):
        self.config = config
        self.registry = registry or ToolRegistry()
        self.policy = RepositoryReadPolicy(config)
        self.handlers: Dict[str, Callable] = {
            "inspect_repo_tree": repository_tools.inspect_repo_tree,
            "search_repo": repository_tools.search_repo,
            "read_selected_files": repository_tools.read_selected_files,
            "parse_dependency_files": repository_tools.parse_dependency_files,
        }
        self.handlers.update(handlers or {})

    def validate_contract(self) -> None:
        implemented = {
            name for name, schema in self.registry.tools.items()
            if schema.implemented and schema.executor == "repository"
        }
        registered = set(self.handlers)
        missing = sorted(implemented - registered)
        unknown = sorted(registered - implemented)
        if missing or unknown:
            raise RuntimeError("repository tool registry/executor mismatch: missing=%s unknown=%s" % (missing, unknown))

    def execute(self, tool_call: ToolCall, context: Dict) -> ToolResult:
        schema = self.registry.tools.get(tool_call.name)
        if schema is None or not schema.implemented or schema.executor != "repository":
            return ToolResult(status="rejected", tool_name=tool_call.name, category="read_only", error="tool is not an implemented repository tool")
        repo_dir = Path(context.get("repo_dir", ""))
        if not repo_dir.is_dir():
            return ToolResult(status="rejected", tool_name=tool_call.name, category="read_only", error="repository root is unavailable")
        decision = self.policy.validate_and_normalize(tool_call.name, tool_call.input or {}, repo_dir)
        if not decision["allowed"]:
            return ToolResult(status="rejected", tool_name=tool_call.name, category="read_only", policy_allowed=False, error=decision["reason"])
        handler = self.handlers[tool_call.name]
        try:
            result = handler(decision["normalized_input"], {**context, "config": self.config})
        except (OSError, TypeError, ValueError) as exc:
            return ToolResult(
                status="failed", tool_name=tool_call.name, category="read_only",
                policy_allowed=True, executed=True, metadata_only=True,
                error=("repository observation failed: %s" % str(exc))[:300],
            )
        result.category = "read_only"
        result.policy_allowed = True
        return result
