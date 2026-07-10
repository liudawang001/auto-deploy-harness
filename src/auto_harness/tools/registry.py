from typing import Dict, List

from auto_harness.models.base import to_plain
from auto_harness.tools.schemas import ToolSchema


class ToolRegistry:
    """Declarative registry for tools the agent runtime may request."""

    def __init__(self) -> None:
        self.tools: Dict[str, ToolSchema] = {}
        for tool in self._defaults():
            self.tools[tool.name] = tool

    def list(self) -> List[Dict]:
        return [to_plain(self.tools[name]) for name in sorted(self.tools)]

    def get(self, name: str) -> Dict:
        tool = self.tools.get(name)
        return to_plain(tool) if tool else {}

    def _defaults(self) -> List[ToolSchema]:
        return [
            # read_only tools
            ToolSchema("inspect_repo_tree", "low", [], False, success_signal="repo files are listed", category="read_only"),
            ToolSchema("read_selected_files", "low", [], False, success_signal="selected file snippets are redacted", category="read_only"),
            ToolSchema("parse_dependency_files", "low", [], False, success_signal="requirements and environment files parsed", category="read_only"),
            ToolSchema("inspect_env_log", "low", [], False, success_signal="env log extracted", category="read_only"),
            ToolSchema("inspect_log", "low", [], False, success_signal="log tail extracted", category="read_only"),
            ToolSchema("inspect_model_config", "low", [], False, success_signal="model config extracted", category="read_only"),
            ToolSchema("inspect_git_lfs_pointers", "low", [], False, success_signal="git lfs pointers extracted", category="read_only"),
            ToolSchema("classify_failure", "low", [], False, success_signal="structured diagnosis emitted", category="read_only"),
            # state_delta tools
            ToolSchema("select_runner_candidate", "low", [], False, ["planner", "gated_actor"], success_signal="candidate selected", category="state_delta"),
            ToolSchema("add_runner_candidate", "low", [], False, ["planner", "gated_actor"], success_signal="candidate added", category="state_delta"),
            ToolSchema("reject_runner_candidate", "low", [], False, ["planner", "gated_actor"], success_signal="candidate rejected", category="state_delta"),
            ToolSchema("apply_dependency_constraint", "low", [], False, ["planner", "gated_actor"], success_signal="constraint applied", category="state_delta"),
            ToolSchema("propose_dependency_constraint", "low", [], False, success_signal="constraint proposed", category="state_delta"),
            ToolSchema("select_environment_backend", "low", [], False, ["planner", "gated_actor"], success_signal="backend selected", category="state_delta"),
            ToolSchema("select_torch_variant", "low", [], False, ["planner", "gated_actor"], success_signal="torch variant selected", category="state_delta"),
            ToolSchema("select_model_source", "low", [], False, ["planner", "gated_actor"], success_signal="model source selected", category="state_delta"),
            ToolSchema("select_model_asset_strategy", "low", [], False, ["planner", "gated_actor"], success_signal="model strategy selected", category="state_delta"),
            ToolSchema("link_cached_model_asset", "low", [], False, ["planner", "gated_actor"], success_signal="cached model linked", category="state_delta"),
            ToolSchema("set_deployment_strategy", "low", [], False, ["planner", "gated_actor"], success_signal="strategy set", category="state_delta"),
            ToolSchema("set_stage_hint", "low", [], False, ["planner", "gated_actor"], success_signal="stage hint set", category="state_delta"),
            ToolSchema("propose_repair", "low", [], False, success_signal="hypothesis-driven repair plan generated", category="state_delta"),
            # execution tools
            ToolSchema("solve_environment", "low", [], False, success_signal="env_solution generated", category="execution"),
            ToolSchema("install_environment", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="install command exits 0", category="execution"),
            ToolSchema("prepare_model_assets", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="asset manifest ready", category="execution"),
            ToolSchema("download_model_asset", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="asset checksum/etag recorded", category="execution"),
            ToolSchema("start_service", "medium", ["process", "network"], True, ["gated_actor"], success_signal="process alive and port ready", category="execution"),
            ToolSchema("probe_http", "low", ["network"], False, success_signal="response contains current trace id", category="execution"),
            ToolSchema("probe_browser_dom", "medium", ["browser", "network"], True, ["planner", "gated_actor"], success_signal="DOM contains current trace id", category="execution"),
            ToolSchema("discover_gradio_api", "low", ["network"], False, success_signal="trace-capable Gradio endpoint found", category="execution"),
            ToolSchema("discover_openapi_schema", "low", ["network"], False, success_signal="trace-capable OpenAPI endpoint found", category="execution"),
            ToolSchema("discover_openai_compatible_model", "low", ["network"], False, success_signal="model id discovered", category="execution"),
            ToolSchema("apply_repair", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="policy-approved repair applied", category="execution"),
            ToolSchema("resume_from_stage", "medium", ["process"], True, ["gated_actor"], success_signal="pipeline resumes from safe stage", category="execution"),
            ToolSchema("verify_evidence", "low", ["network"], False, success_signal="trace evidence passes", category="execution"),
            ToolSchema("verify_after_repair", "low", ["network"], False, success_signal="verify after repair passed", category="execution"),
        ]
