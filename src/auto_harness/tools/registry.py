from typing import Dict, List

from auto_harness.models.base import to_plain
from auto_harness.tools.schemas import TOOL_CATEGORIES, ToolSchema


class ToolRegistry:
    """Declarative registry for tools the agent runtime may request.

    Tool categories:
    - read_only: no side effects, safe to run in any mode
    - state_delta: changes internal state but no external side effects
    - side_effect: has external side effects (filesystem, network, process)
    - evidence: produces evidence for verification
    """

    def __init__(self) -> None:
        self.tools: Dict[str, ToolSchema] = {}
        for tool in self._defaults():
            self.tools[tool.name] = tool

    def list(self) -> List[Dict]:
        return [to_plain(self.tools[name]) for name in sorted(self.tools)]

    def get(self, name: str) -> Dict:
        tool = self.tools.get(name)
        return to_plain(tool) if tool else {}

    def executable_for_stage(
        self,
        stage: str,
        *,
        agent_mode: str,
    ) -> List[Dict]:
        """Return tools that are implemented, allowed for the given stage and mode.

        LLM prompts should use this instead of list() to avoid showing
        unimplemented tools.
        """
        result = []
        for tool in self.tools.values():
            if not tool.implemented:
                continue
            if stage not in tool.stages:
                continue
            if tool.allowed_modes and agent_mode not in tool.allowed_modes:
                continue
            result.append(to_plain(tool))
        return result

    def _defaults(self) -> List[ToolSchema]:
        return [
            # read_only tools - no side effects, safe in any mode
            ToolSchema(
                name="inspect_repo_tree",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="repo files are listed",
                category="read_only",
            ),
            ToolSchema(
                name="read_selected_files",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="selected file snippets are redacted",
                category="read_only",
            ),
            ToolSchema(
                name="parse_dependency_files",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="requirements and environment files parsed",
                category="read_only",
            ),
            ToolSchema(
                name="inspect_env_log",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="env log extracted",
                category="read_only",
            ),
            ToolSchema(
                name="inspect_log",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="log tail extracted",
                category="read_only",
            ),
            ToolSchema(
                name="inspect_model_config",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="model config extracted",
                category="read_only",
            ),
            ToolSchema(
                name="inspect_git_lfs_pointers",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="git lfs pointers extracted",
                category="read_only",
            ),
            ToolSchema(
                name="classify_failure",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="structured diagnosis emitted",
                category="read_only",
            ),

            # state_delta tools - change internal state, no external side effects
            ToolSchema(
                name="select_runner_candidate",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="candidate selected",
                category="state_delta",
            ),
            ToolSchema(
                name="add_runner_candidate",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="candidate added",
                category="state_delta",
            ),
            ToolSchema(
                name="reject_runner_candidate",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="candidate rejected",
                category="state_delta",
            ),
            ToolSchema(
                name="apply_dependency_constraint",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="constraint applied",
                category="state_delta",
            ),
            ToolSchema(
                name="propose_dependency_constraint",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="constraint proposed",
                category="state_delta",
            ),
            ToolSchema(
                name="select_environment_backend",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="backend selected",
                category="state_delta",
            ),
            ToolSchema(
                name="select_torch_variant",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="torch variant selected",
                category="state_delta",
            ),
            ToolSchema(
                name="select_model_source",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="model source selected",
                category="state_delta",
            ),
            ToolSchema(
                name="select_model_asset_strategy",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="model strategy selected",
                category="state_delta",
            ),
            ToolSchema(
                name="link_cached_model_asset",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="cached model linked",
                category="state_delta",
            ),
            ToolSchema(
                name="set_deployment_strategy",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="strategy set",
                category="state_delta",
            ),
            ToolSchema(
                name="set_stage_hint",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="stage hint set",
                category="state_delta",
            ),
            ToolSchema(
                name="propose_repair",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="hypothesis-driven repair plan generated",
                category="state_delta",
            ),

            # side_effect tools - have external side effects, require policy gate
            ToolSchema(
                name="solve_environment",
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="env_solution generated",
                category="side_effect",
            ),
            ToolSchema(
                name="install_environment",
                risk_level="medium",
                side_effects=["filesystem", "network"],
                requires_policy=True,
                allowed_modes=["gated_actor"],
                success_signal="install command exits 0",
                category="side_effect",
            ),
            ToolSchema(
                name="prepare_model_assets",
                risk_level="medium",
                side_effects=["filesystem", "network"],
                requires_policy=True,
                allowed_modes=["gated_actor"],
                success_signal="asset manifest ready",
                category="side_effect",
            ),
            ToolSchema(
                name="download_model_asset",
                risk_level="medium",
                side_effects=["filesystem", "network"],
                requires_policy=True,
                allowed_modes=["gated_actor"],
                success_signal="asset checksum/etag recorded",
                category="side_effect",
            ),
            ToolSchema(
                name="start_service",
                risk_level="medium",
                side_effects=["process", "network"],
                requires_policy=True,
                allowed_modes=["gated_actor"],
                success_signal="process alive and port ready",
                category="side_effect",
            ),
            ToolSchema(
                name="apply_repair",
                risk_level="medium",
                side_effects=["filesystem", "network"],
                requires_policy=True,
                allowed_modes=["gated_actor"],
                success_signal="policy-approved repair applied",
                category="side_effect",
            ),
            ToolSchema(
                name="resume_from_stage",
                risk_level="medium",
                side_effects=["process"],
                requires_policy=True,
                allowed_modes=["gated_actor"],
                success_signal="pipeline resumes from safe stage",
                category="side_effect",
            ),

            # evidence tools - produce verification evidence
            # Only these 4 are implemented with real executor handlers
            ToolSchema(
                name="probe_http",
                risk_level="low",
                side_effects=["network"],
                requires_policy=False,
                success_signal="response contains current trace id",
                category="evidence",
                implemented=True,
                executor="verify",
                stages=["verify"],
            ),
            ToolSchema(
                name="probe_browser_dom",
                risk_level="medium",
                side_effects=["browser", "network"],
                requires_policy=True,
                allowed_modes=["planner", "gated_actor"],
                success_signal="DOM contains current trace id",
                category="evidence",
                implemented=True,
                executor="verify",
                stages=["verify"],
            ),
            ToolSchema(
                name="discover_gradio_api",
                risk_level="low",
                side_effects=["network"],
                requires_policy=False,
                success_signal="trace-capable Gradio endpoint found",
                category="evidence",
                implemented=True,
                executor="verify",
                stages=["verify"],
            ),
            ToolSchema(
                name="discover_openapi_schema",
                risk_level="low",
                side_effects=["network"],
                requires_policy=False,
                success_signal="trace-capable OpenAPI endpoint found",
                category="evidence",
                implemented=True,
                executor="verify",
                stages=["verify"],
            ),
            # Unimplemented evidence tools
            ToolSchema(
                name="discover_openai_compatible_model",
                risk_level="low",
                side_effects=["network"],
                requires_policy=False,
                success_signal="model id discovered",
                category="evidence",
            ),
            ToolSchema(
                name="verify_evidence",
                risk_level="low",
                side_effects=["network"],
                requires_policy=False,
                success_signal="trace evidence passes",
                category="evidence",
            ),
            ToolSchema(
                name="verify_after_repair",
                risk_level="low",
                side_effects=["network"],
                requires_policy=False,
                success_signal="verify after repair passed",
                category="evidence",
            ),
        ]
