from typing import Any, Dict, List

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

    def __init__(self, config: Any = None) -> None:
        self.config = config
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
            if tool.name == "retrieve_deployment_context" and not self._retrieval_enabled():
                continue
            if stage not in tool.stages:
                continue
            if tool.allowed_modes and agent_mode not in tool.allowed_modes:
                continue
            result.append(to_plain(tool))
        return result

    def _retrieval_enabled(self) -> bool:
        if self.config is None:
            return False
        if isinstance(self.config, dict):
            settings = self.config.get("retrieval", self.config)
        else:
            settings = getattr(self.config, "retrieval", {})
        return isinstance(settings, dict) and bool(settings.get("enabled", False))

    def _defaults(self) -> List[ToolSchema]:
        return [
            # read_only tools - no side effects, safe in any mode
            ToolSchema(
                name="inspect_repo_tree",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "max_depth": {"type": "integer", "minimum": 1, "maximum": 8},
                        "max_entries": {"type": "integer", "minimum": 1},
                        "path_glob": {"type": "string", "default": "**/*"},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="repo files are listed",
                category="read_only",
                implemented=True,
                executor="repository",
                stages=["plan", "replan"],
            ),
            ToolSchema(
                name="read_selected_files",
                input_schema={
                    "type": "object",
                    "required": ["files"],
                    "properties": {
                        "files": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["path"],
                                "properties": {
                                    "path": {"type": "string"},
                                    "start_line": {"type": "integer", "minimum": 1},
                                    "end_line": {"type": "integer", "minimum": 1},
                                },
                            },
                        },
                        "retrieved_from_query_id": {"type": "string", "maxLength": 80},
                        "retrieval_chunk_ids": {
                            "type": "array", "maxItems": 12,
                            "items": {"type": "string", "maxLength": 80},
                        },
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="selected file snippets are redacted",
                category="read_only",
                implemented=True,
                executor="repository",
                stages=["plan", "replan"],
            ),
            ToolSchema(
                name="search_repo",
                input_schema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "path_glob": {"type": "string", "default": "**/*"},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "minimum": 1},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="bounded repository matches are returned",
                category="read_only",
                implemented=True,
                executor="repository",
                stages=["plan", "replan"],
            ),
            ToolSchema(
                name="parse_dependency_files",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 16,
                        },
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                success_signal="requirements and environment files parsed",
                category="read_only",
                implemented=True,
                executor="repository",
                stages=["plan", "replan"],
            ),
            ToolSchema(
                name="retrieve_deployment_context",
                input_schema={
                    "type": "object",
                    "required": ["query", "purpose"],
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "purpose": {
                            "type": "string",
                            "enum": [
                                "plan_repository", "diagnose_failure",
                                "select_repair", "select_verify_strategy", "replan",
                            ],
                        },
                        "sources": {
                            "type": "array", "maxItems": 5,
                            "items": {
                                "type": "string",
                                "enum": [
                                    "repository", "runtime_evidence", "runtime_log",
                                    "issue_memory", "verified_memory", "active_skill",
                                ],
                            },
                        },
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 12},
                    },
                    "additionalProperties": False,
                },
                risk_level="low", side_effects=[], requires_policy=False,
                success_signal="bounded candidate evidence returned",
                category="read_only", implemented=True, executor="retrieval",
                stages=["plan", "replan", "repair", "verify"],
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
                input_schema={
                    "type": "object",
                    "required": ["candidate_id"],
                    "properties": {
                        "candidate_id": {"type": "string", "minLength": 1, "maxLength": 200},
                        "reason": {"type": "string", "maxLength": 1000},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="candidate selected",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["runner"],
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
                input_schema={
                    "type": "object",
                    "required": ["package"],
                    "properties": {
                        "package": {"type": "string", "minLength": 1, "maxLength": 200},
                        "version_spec": {"type": "string", "maxLength": 100},
                        "scope": {"type": "string", "enum": ["temporary_overlay"]},
                        "reason": {"type": "string", "maxLength": 1000},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="constraint applied",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["env_solve", "repair"],
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
                input_schema={
                    "type": "object",
                    "required": ["backend"],
                    "properties": {
                        "backend": {"type": "string", "enum": ["venv", "conda", "mamba", "pip"]},
                        "reason": {"type": "string", "maxLength": 1000},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="backend selected",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["env_solve"],
            ),
            ToolSchema(
                name="select_torch_variant",
                input_schema={
                    "type": "object",
                    "properties": {
                        "variant": {"type": "string", "maxLength": 100},
                        "index": {"type": "string", "maxLength": 200},
                        "reason": {"type": "string", "maxLength": 1000},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="torch variant selected",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["env_solve"],
            ),
            ToolSchema(
                name="select_model_source",
                input_schema={
                    "type": "object",
                    "required": ["source"],
                    "properties": {
                        "source": {"type": "string", "enum": ["huggingface", "modelscope", "local_cache"]},
                        "repo_id": {"type": "string", "maxLength": 300},
                        "target_path": {"type": "string", "maxLength": 500},
                        "fallback": {"type": "string", "maxLength": 300},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="model source selected",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["model_prepare"],
            ),
            ToolSchema(
                name="select_model_asset_strategy",
                input_schema={
                    "type": "object",
                    "required": ["strategy"],
                    "properties": {
                        "strategy": {"type": "string", "maxLength": 200},
                        "fallback": {"type": "string", "maxLength": 300},
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="model strategy selected",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["model_prepare"],
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
                input_schema={
                    "type": "object",
                    "required": ["stage", "hints"],
                    "properties": {
                        "stage": {
                            "type": "string",
                            "enum": ["analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy", "model_prepare", "runner", "verify", "report"],
                        },
                        "hints": {
                            "type": "object",
                            "properties": {
                                "strategy": {"type": "string", "maxLength": 200},
                                "prefer_entrypoint_patterns": {
                                    "type": "array",
                                    "maxItems": 20,
                                    "items": {"type": "string", "maxLength": 200},
                                },
                                "method": {"type": "string", "enum": ["GET", "POST"]},
                                "path": {"type": "string", "maxLength": 500},
                                "service_type": {"type": "string", "maxLength": 100},
                                "backend": {"type": "string", "maxLength": 100},
                                "source": {"type": "string", "maxLength": 100},
                                "reason": {"type": "string", "maxLength": 1000},
                            },
                        },
                    },
                },
                risk_level="low",
                side_effects=[],
                requires_policy=False,
                allowed_modes=["planner", "gated_actor"],
                success_signal="stage hint set",
                category="state_delta",
                implemented=True,
                executor="state_delta",
                stages=["plan", "replan"],
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
