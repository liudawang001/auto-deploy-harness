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
            # Existing tools
            ToolSchema("inspect_repo_tree", "low", [], False, success_signal="repo files are listed"),
            ToolSchema("read_selected_files", "low", [], False, success_signal="selected file snippets are redacted"),
            ToolSchema("parse_dependency_files", "low", [], False, success_signal="requirements and environment files parsed"),
            ToolSchema("solve_environment", "low", [], False, success_signal="env_solution generated"),
            ToolSchema("install_environment", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="install command exits 0"),
            ToolSchema("prepare_model_assets", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="asset manifest ready"),
            ToolSchema("start_service", "medium", ["process", "network"], True, ["gated_actor"], success_signal="process alive and port ready"),
            ToolSchema("probe_http", "low", ["network"], False, success_signal="response contains current trace id"),
            ToolSchema("probe_browser_dom", "medium", ["browser", "network"], True, ["planner", "gated_actor"], success_signal="DOM contains current trace id"),
            ToolSchema("discover_gradio_api", "low", ["network"], False, success_signal="trace-capable Gradio endpoint found"),
            ToolSchema("discover_openapi_schema", "low", ["network"], False, success_signal="trace-capable OpenAPI endpoint found"),
            ToolSchema("discover_openai_compatible_model", "low", ["network"], False, success_signal="model id discovered"),
            ToolSchema("download_model_asset", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="asset checksum/etag recorded"),
            ToolSchema("inspect_log", "low", [], False, success_signal="log tail extracted"),
            ToolSchema("classify_failure", "low", [], False, success_signal="structured diagnosis emitted"),
            ToolSchema("propose_repair", "low", [], False, success_signal="hypothesis-driven repair plan generated"),
            ToolSchema("apply_repair", "medium", ["filesystem", "network"], True, ["gated_actor"], success_signal="policy-approved repair applied"),
            ToolSchema("resume_from_stage", "medium", ["process"], True, ["gated_actor"], success_signal="pipeline resumes from safe stage"),
            ToolSchema("verify_evidence", "low", ["network"], False, success_signal="trace evidence passes"),
            # Runner gate tools
            ToolSchema("select_runner_candidate", "low", [], False, ["planner", "gated_actor"], success_signal="candidate selected"),
            ToolSchema("add_runner_candidate", "low", [], False, ["planner", "gated_actor"], success_signal="candidate added"),
            ToolSchema("reject_runner_candidate", "low", [], False, ["planner", "gated_actor"], success_signal="candidate rejected"),
            # Env gate tools
            ToolSchema("inspect_env_log", "low", [], False, success_signal="env log extracted"),
            ToolSchema("propose_dependency_constraint", "low", [], False, success_signal="constraint proposed"),
            ToolSchema("apply_dependency_constraint", "low", [], False, ["planner", "gated_actor"], success_signal="constraint applied"),
            ToolSchema("select_environment_backend", "low", [], False, ["planner", "gated_actor"], success_signal="backend selected"),
            ToolSchema("select_torch_variant", "low", [], False, ["planner", "gated_actor"], success_signal="torch variant selected"),
            # Model gate tools
            ToolSchema("select_model_source", "low", [], False, ["planner", "gated_actor"], success_signal="model source selected"),
            ToolSchema("select_model_asset_strategy", "low", [], False, ["planner", "gated_actor"], success_signal="model strategy selected"),
            ToolSchema("link_cached_model_asset", "low", [], False, ["planner", "gated_actor"], success_signal="cached model linked"),
            ToolSchema("inspect_model_config", "low", [], False, success_signal="model config extracted"),
            ToolSchema("inspect_git_lfs_pointers", "low", [], False, success_signal="git lfs pointers extracted"),
            # Plan gate tools
            ToolSchema("set_deployment_strategy", "low", [], False, ["planner", "gated_actor"], success_signal="strategy set"),
            ToolSchema("set_stage_hint", "low", [], False, ["planner", "gated_actor"], success_signal="stage hint set"),
            # Repair actuator gate tools
            ToolSchema("verify_after_repair", "low", ["network"], False, success_signal="verify after repair passed"),
        ]
