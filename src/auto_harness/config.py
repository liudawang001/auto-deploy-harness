import json
import os
from importlib import resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_deepseek_provider_configs() -> Dict[str, Dict[str, Any]]:
    """Return a fresh, secret-free DeepSeek configuration for direct use."""
    return {
        "deepseek": {
            "api_base": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "require_api_key": True,
            "model": "deepseek-v4-pro",
            "context_window_tokens": 262144,
            "max_tokens": 16384,
            "thinking": {
                "agent": "disabled",
                "plan_first": "enabled",
                "memory_evolution": "disabled",
                "llm_test": "disabled",
                "live_smoke": "disabled",
            },
            "reasoning_effort": {"plan_first": "high"},
            "json_mode": {
                "agent": True,
                "plan_first": True,
                "memory_evolution": True,
                "llm_test": False,
                "live_smoke": True,
            },
            "timeout_seconds": 60,
            "max_retries": 2,
            "retry_base_seconds": 1.0,
            "retry_max_seconds": 8.0,
        }
    }


@dataclass
class HarnessConfig:
    runs_dir: str = "runs"
    default_timeout_seconds: int = 900
    max_stage_attempts: int = 2
    max_repair_attempts: int = 2
    allow_source_edit: bool = False
    allow_dependency_install: bool = False
    allow_service_start: bool = False
    execution_backend: str = "local"
    docker_image: str = "python:3.10-slim"
    docker_network: str = "bridge"
    docker_gpus: str = "none"
    docker_model_cache_dir: str = ""
    docker_read_only_rootfs: bool = False
    docker_user: str = ""
    docker_memory: str = "8g"
    docker_cpus: float = 4.0
    docker_pids_limit: int = 512
    docker_tmpfs_size: str = "1g"
    docker_cap_drop_all: bool = True
    docker_no_new_privileges: bool = True
    docker_repo_mount_mode: str = "rw"
    verify_workspace_name: str = "verify_workspace"
    allowed_commands: List[str] = None
    repository_command_policy: Dict[str, Any] = None
    use_agent_analyzer: bool = False
    agent_timeout_seconds: int = 900
    agent_mode: str = "off"
    agent_provider: str = "deepseek"
    provider_configs: Dict[str, Dict[str, Any]] = None
    agent_max_input_chars: int = 20000
    agent_max_file_chars: int = 6000
    agent_context_mode: str = "enforce"
    agent_context_window_tokens: Optional[int] = 262144
    agent_context_reserved_output_tokens: int = 16384
    agent_context_safety_margin_tokens: int = 2048
    agent_context_warn_ratio: float = 0.70
    agent_context_compact_ratio: float = 0.85
    agent_context_tokenizer: str = "auto"
    agent_context_unknown_model_fallback_tokens: int = 8192
    agent_context_max_overflow_retries: int = 1
    agent_context_trace_section_details: bool = True
    agent_context_skill_budget_tokens: int = 2000
    agent_context_memory_budget_tokens: int = 2000
    agent_decision_timeout_seconds: int = 60
    agent_enable_analyze_planner: bool = False
    agent_enable_log_diagnosis: bool = False
    agent_enable_verify_planner: bool = False
    agent_enable_verify: bool = False
    agent_enable_repair_actions: bool = False
    agent_auto_resume_after_repair: bool = False
    agent_max_loop_iterations: int = 2
    agent_auto_resume_stages: List[str] = None
    agent_stop_on_verify_pass: bool = True
    agent_verify_max_steps: int = 3
    agent_allowed_hosts: List[str] = None
    env_backend: str = "auto"
    conda_envs_dir: str = ".conda/envs"
    conda_prefer_mamba: bool = True
    conda_allowed_channels: List[str] = None
    conda_python_default: str = "3.10"
    torch_cuda_preference: str = "auto"
    preflight_enabled: bool = True
    preflight_fail_closed: bool = True
    preflight_require_gpu: bool = False
    gpu_probe_timeout_seconds: int = 5
    conda_probe_timeout_seconds: int = 10
    conda_inventory_timeout_seconds: int = 30
    conda_inventory_max_envs: int = 50
    conda_allow_venv_fallback: bool = False
    conda_allow_cpu_fallback: bool = False
    conda_reuse_owned_env: bool = True
    conda_reuse_external_env: bool = False
    min_gpu_memory_mb: int = 0
    skills_dir: str = "skills"
    memory_dir: str = "memory"
    model_cache_dir: str = "model_cache"
    task_queue_dir: str = "queue"
    queue_max_concurrent_tasks: int = 1
    queue_gpu_slots: Optional[int] = None
    queue_claim_ttl_seconds: int = 3600
    model_download_max_workers: int = 1
    model_download_retry_count: int = 2
    model_download_retry_backoff_seconds: float = 1.0
    model_cache_cleanup_max_total_bytes: Optional[int] = None
    model_cache_cleanup_older_than_days: Optional[float] = None
    model_cache_cleanup_source: Optional[str] = None
    model_cache_cleanup_repo_id: Optional[str] = None
    model_cache_cleanup_keep_cache_keys: List[str] = None
    model_cache_cleanup_keep_repo_ids: List[str] = None
    max_skill_chars: int = 6000
    max_memory_items: int = 5
    memory_evolution_enabled: bool = False
    memory_evolution_min_verified_count: int = 3
    memory_evolution_require_regression: bool = True
    memory_evolution_require_shadow: bool = True
    memory_evolution_shadow_helped_threshold: int = 2
    memory_evolution_shadow_harmful_threshold: int = 0
    memory_evolution_provider: str = "deepseek"
    skill_candidate_dir: str = "memory/skill_candidates"
    # Decision gate configuration
    agent_enable_plan_gate: bool = False
    agent_enable_runner_gate: bool = False
    agent_enable_env_gate: bool = False
    agent_enable_model_gate: bool = False
    agent_enable_repair_gate: bool = False
    agent_decision_gate_max_steps: int = 2
    agent_llm_required_eval: bool = False
    # Agent runtime loop configuration
    agent_enable_runtime_loop: bool = False
    agent_runtime_loop_max_iterations: int = 5
    agent_runtime_loop_stop_on_verify_pass: bool = True
    agent_runtime_loop_position: str = "primary"  # primary | post_pipeline
    # LLM Plan-first configuration
    agent_plan_first: bool = False
    agent_plan_first_provider: str = "deepseek"
    agent_plan_first_mode: str = "planner"  # planner | gated_actor
    agent_plan_first_max_replans: int = 2
    agent_plan_first_max_files: int = 80
    agent_plan_first_max_file_chars: int = 6000
    agent_plan_first_require_grounding: bool = True
    agent_plan_first_allow_external_network: bool = False
    # Layered repository context and bounded on-demand observation.
    agent_repo_context_mode: str = "layered"  # layered | eager_compat
    agent_repo_inventory_budget_tokens: int = 4000
    agent_repo_core_budget_tokens: int = 12000
    agent_repo_observation_budget_tokens: int = 24000
    agent_repo_max_observation_rounds: int = 4
    agent_repo_max_requests_per_round: int = 4
    agent_repo_max_observed_files: int = 20
    agent_repo_max_chars_per_read: int = 12000
    agent_repo_max_lines_per_read: int = 400
    agent_repo_search_max_results: int = 30
    agent_repo_search_max_files: int = 5000
    agent_repo_search_max_bytes: int = 50000000
    agent_repo_tree_max_entries: int = 5000
    default_controller: str = "langgraph"
    langgraph_require_llm: bool = False
    langgraph_allow_mock_in_dry_run: bool = True
    langgraph_allow_mock_in_execute: bool = False
    langgraph_enable_diagnose: bool = True
    langgraph_enable_repair: bool = True
    langgraph_enable_agent_verify: bool = True
    langgraph_enable_recovery: bool = True
    langgraph_fault_injection_points: List[str] = None
    langgraph_max_diagnoses: int = 2
    langgraph_max_repairs: int = 2
    langgraph_max_same_failure: int = 2
    langgraph_planner_mode: str = "auto"  # auto | llm | deterministic

    # Non-persistent runtime overrides (never loaded from or saved to JSON)
    llm_runtime_overrides: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.allowed_commands is None:
            self.allowed_commands = ["python", "python3", "pip", "curl", "git", "streamlit"]
        repository_command_defaults = {
            "enabled": True,
            "unknown_repository_backend": "docker",
            "approval_mode": "risk_based",
            "approval_ttl_seconds": 1800,
            "max_runner_candidate_attempts": 3,
            "max_strategy_attempts": 2,
            "allow_make_targets": True,
            "allow_repository_scripts": True,
            "auto_allow_declared_project_cli_in_docker": True,
            "auto_allow_locked_package_scripts_in_docker": True,
            "require_readme_reference_for_project_cli": True,
            "require_lockfile_for_node_install": True,
            "default_network_profile": "none",
            "install_network_profile": "registry_only",
        }
        configured_repository_policy = self.repository_command_policy or {}
        if not isinstance(configured_repository_policy, dict):
            raise ValueError("repository_command_policy must be an object")
        self.repository_command_policy = {
            **repository_command_defaults,
            **configured_repository_policy,
        }
        if self.repository_command_policy["unknown_repository_backend"] != "docker":
            raise ValueError("unknown_repository_backend must be docker")
        for name in ("approval_ttl_seconds", "max_runner_candidate_attempts", "max_strategy_attempts"):
            value = self.repository_command_policy.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("repository_command_policy.%s must be a positive integer" % name)
        if self.model_cache_cleanup_keep_cache_keys is None:
            self.model_cache_cleanup_keep_cache_keys = []
        if self.model_cache_cleanup_keep_repo_ids is None:
            self.model_cache_cleanup_keep_repo_ids = []
        if self.agent_auto_resume_stages is None:
            self.agent_auto_resume_stages = [
                "host_preflight", "env_solve", "env_deploy",
                "model_prepare", "runner", "verify",
            ]
        if self.agent_allowed_hosts is None:
            self.agent_allowed_hosts = ["127.0.0.1", "localhost", "::1"]
        if self.conda_allowed_channels is None:
            self.conda_allowed_channels = ["defaults", "conda-forge", "pytorch", "nvidia", "fastai"]
        if self.langgraph_fault_injection_points is None:
            self.langgraph_fault_injection_points = []
        if self.provider_configs is None:
            self.provider_configs = _default_deepseek_provider_configs()
        if self.gpu_probe_timeout_seconds <= 0:
            raise ValueError("gpu_probe_timeout_seconds must be positive")
        if self.conda_probe_timeout_seconds <= 0 or self.conda_inventory_timeout_seconds <= 0:
            raise ValueError("Conda probe timeouts must be positive")
        if self.conda_inventory_max_envs <= 0:
            raise ValueError("conda_inventory_max_envs must be positive")
        if self.default_controller not in {"legacy", "langgraph"}:
            raise ValueError("default_controller must be 'legacy' or 'langgraph', got: %s" % self.default_controller)
        if self.langgraph_max_diagnoses < 0:
            raise ValueError("langgraph_max_diagnoses must be non-negative, got: %s" % self.langgraph_max_diagnoses)
        if self.langgraph_max_repairs < 0:
            raise ValueError("langgraph_max_repairs must be non-negative, got: %s" % self.langgraph_max_repairs)
        if self.langgraph_max_same_failure < 0:
            raise ValueError("langgraph_max_same_failure must be non-negative, got: %s" % self.langgraph_max_same_failure)
        if self.langgraph_planner_mode not in ("auto", "llm", "deterministic"):
            raise ValueError(
                "langgraph_planner_mode must be 'auto', 'llm' or "
                "'deterministic', got: %s" % self.langgraph_planner_mode
            )
        if self.agent_repo_context_mode not in {"layered", "eager_compat"}:
            raise ValueError("agent_repo_context_mode must be layered or eager_compat")
        repo_budget_fields = (
            "agent_repo_inventory_budget_tokens",
            "agent_repo_core_budget_tokens",
            "agent_repo_observation_budget_tokens",
            "agent_repo_max_observation_rounds",
            "agent_repo_max_requests_per_round",
            "agent_repo_max_observed_files",
            "agent_repo_max_chars_per_read",
            "agent_repo_max_lines_per_read",
            "agent_repo_search_max_results",
            "agent_repo_search_max_files",
            "agent_repo_search_max_bytes",
            "agent_repo_tree_max_entries",
        )
        for name in repo_budget_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.agent_repo_max_observation_rounds > 8:
            raise ValueError("agent_repo_max_observation_rounds must be <= 8")
        if self.agent_repo_max_requests_per_round > 8:
            raise ValueError("agent_repo_max_requests_per_round must be <= 8")
        if self.agent_context_mode not in {"observe", "shadow", "enforce"}:
            raise ValueError(
                "agent_context_mode must be observe, shadow or enforce, got: %s"
                % self.agent_context_mode
            )
        if self.agent_context_window_tokens is not None and self.agent_context_window_tokens <= 0:
            raise ValueError("agent_context_window_tokens must be positive")
        if self.agent_context_reserved_output_tokens <= 0:
            raise ValueError("agent_context_reserved_output_tokens must be positive")
        if self.agent_decision_timeout_seconds <= 0:
            raise ValueError("agent_decision_timeout_seconds must be positive")
        if self.agent_context_safety_margin_tokens < 0:
            raise ValueError("agent_context_safety_margin_tokens must be non-negative")
        if not (
            0 < self.agent_context_warn_ratio < self.agent_context_compact_ratio <= 1
        ):
            raise ValueError(
                "context ratios must satisfy 0 < warn < compact <= 1"
            )
        if self.agent_context_unknown_model_fallback_tokens <= 0:
            raise ValueError(
                "agent_context_unknown_model_fallback_tokens must be positive"
            )
        if self.agent_context_max_overflow_retries not in {0, 1}:
            raise ValueError("agent_context_max_overflow_retries must be 0 or 1")
        if self.agent_context_skill_budget_tokens <= 0:
            raise ValueError("agent_context_skill_budget_tokens must be positive")
        if self.agent_context_memory_budget_tokens <= 0:
            raise ValueError("agent_context_memory_budget_tokens must be positive")
        if not isinstance(self.provider_configs, dict):
            raise ValueError("provider_configs must be an object")
        forbidden_provider_keys = {
            "api_key",
            "token",
            "secret",
            "password",
            "authorization",
        }
        for provider_name, settings in self.provider_configs.items():
            if not str(provider_name).strip():
                raise ValueError("provider_configs contains an empty provider name")
            if not isinstance(settings, dict):
                raise ValueError(
                    "provider_configs.%s must be an object" % provider_name
                )
            forbidden = forbidden_provider_keys.intersection(
                str(key).lower() for key in settings
            )
            if forbidden:
                raise ValueError(
                    "provider_configs.%s must not contain secret values: %s"
                    % (provider_name, ", ".join(sorted(forbidden)))
                )
            for key in (
                "timeout_seconds",
                "max_tokens",
                "context_window_tokens",
            ):
                if key in settings and (
                    isinstance(settings[key], bool)
                    or not isinstance(settings[key], (int, float))
                    or settings[key] <= 0
                ):
                    raise ValueError(
                        "provider_configs.%s.%s must be positive"
                        % (provider_name, key)
                    )
            # DeepSeek-specific validation
            _validate_deepseek_config(provider_name, settings)
            if (
                str(provider_name).strip().lower().replace("-", "_")
                == "deepseek"
                and settings.get("timeout_seconds") is not None
                and float(settings["timeout_seconds"])
                > float(self.agent_decision_timeout_seconds)
            ):
                raise ValueError(
                    "provider_configs.deepseek.timeout_seconds must not exceed "
                    "agent_decision_timeout_seconds"
                )
        valid_fault_windows = {
            "before_side_effect",
            "after_side_effect_before_commit",
            "after_commit_before_checkpoint",
        }
        for point in self.langgraph_fault_injection_points:
            parts = str(point).split(":", 1)
            if len(parts) != 2 or not parts[0] or parts[1] not in valid_fault_windows:
                raise ValueError("invalid langgraph fault injection point: %s" % point)
        if self.docker_network == "host":
            raise ValueError("host network is not allowed")
        if not self.docker_memory:
            raise ValueError("docker_memory must be non-empty")
        if self.docker_cpus <= 0:
            raise ValueError("docker_cpus must be positive, got: %s" % self.docker_cpus)
        if self.docker_pids_limit <= 0:
            raise ValueError("docker_pids_limit must be positive, got: %s" % self.docker_pids_limit)
        if self.docker_repo_mount_mode not in ("ro", "rw"):
            raise ValueError(
                "docker_repo_mount_mode must be 'ro' or 'rw', got: %s" % self.docker_repo_mount_mode
            )

    @classmethod
    def load(cls, path: str = None) -> "HarnessConfig":
        config_path = Path(path or os.environ.get("AUTO_HARNESS_CONFIG", "configs/default.json"))
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        elif path or os.environ.get("AUTO_HARNESS_CONFIG"):
            return cls(
                use_agent_analyzer=os.environ.get("AUTO_HARNESS_USE_AGENT_ANALYZER") == "1"
            )
        else:
            data = json.loads(
                resources.files("auto_harness.resources")
                .joinpath("default.json")
                .read_text(encoding="utf-8")
            )
        known: Dict[str, Any] = {}
        for key, definition in cls.__dataclass_fields__.items():
            if definition.init and key in data:
                known[key] = data[key]
        if os.environ.get("AUTO_HARNESS_USE_AGENT_ANALYZER"):
            known["use_agent_analyzer"] = os.environ.get("AUTO_HARNESS_USE_AGENT_ANALYZER") == "1"
        if os.environ.get("AUTO_HARNESS_AGENT_MODE"):
            known["agent_mode"] = os.environ["AUTO_HARNESS_AGENT_MODE"]
        if os.environ.get("AUTO_HARNESS_AGENT_PROVIDER"):
            known["agent_provider"] = os.environ["AUTO_HARNESS_AGENT_PROVIDER"]
        if os.environ.get("AUTO_HARNESS_AGENT_CONTEXT_MODE"):
            known["agent_context_mode"] = os.environ["AUTO_HARNESS_AGENT_CONTEXT_MODE"]
        if os.environ.get("AUTO_HARNESS_AGENT_CONTEXT_WINDOW_TOKENS"):
            known["agent_context_window_tokens"] = int(
                os.environ["AUTO_HARNESS_AGENT_CONTEXT_WINDOW_TOKENS"]
            )
        if os.environ.get("AUTO_HARNESS_ENABLE_ANALYZE_PLANNER"):
            known["agent_enable_analyze_planner"] = os.environ.get("AUTO_HARNESS_ENABLE_ANALYZE_PLANNER") == "1"
        if os.environ.get("AUTO_HARNESS_ENABLE_LOG_DIAGNOSIS"):
            known["agent_enable_log_diagnosis"] = os.environ.get("AUTO_HARNESS_ENABLE_LOG_DIAGNOSIS") == "1"
        if os.environ.get("AUTO_HARNESS_ENABLE_VERIFY_PLANNER"):
            known["agent_enable_verify_planner"] = os.environ.get("AUTO_HARNESS_ENABLE_VERIFY_PLANNER") == "1"
        if os.environ.get("AUTO_HARNESS_ENABLE_VERIFY"):
            known["agent_enable_verify"] = os.environ.get("AUTO_HARNESS_ENABLE_VERIFY") == "1"
        if os.environ.get("AUTO_HARNESS_ENABLE_REPAIR_ACTIONS"):
            known["agent_enable_repair_actions"] = os.environ.get("AUTO_HARNESS_ENABLE_REPAIR_ACTIONS") == "1"
        if os.environ.get("AUTO_HARNESS_AGENT_AUTO_RESUME_AFTER_REPAIR"):
            known["agent_auto_resume_after_repair"] = os.environ.get("AUTO_HARNESS_AGENT_AUTO_RESUME_AFTER_REPAIR") == "1"
        if os.environ.get("AUTO_HARNESS_AGENT_MAX_LOOP_ITERATIONS"):
            known["agent_max_loop_iterations"] = int(os.environ["AUTO_HARNESS_AGENT_MAX_LOOP_ITERATIONS"])
        if os.environ.get("AUTO_HARNESS_ENV_BACKEND"):
            known["env_backend"] = os.environ["AUTO_HARNESS_ENV_BACKEND"]
        if os.environ.get("AUTO_HARNESS_CONDA_PREFER_MAMBA"):
            known["conda_prefer_mamba"] = os.environ.get("AUTO_HARNESS_CONDA_PREFER_MAMBA") == "1"
        if os.environ.get("AUTO_HARNESS_TORCH_CUDA_PREFERENCE"):
            known["torch_cuda_preference"] = os.environ["AUTO_HARNESS_TORCH_CUDA_PREFERENCE"]
        if os.environ.get("AUTO_HARNESS_MEMORY_EVOLUTION_ENABLED"):
            known["memory_evolution_enabled"] = os.environ.get("AUTO_HARNESS_MEMORY_EVOLUTION_ENABLED") == "1"
        if os.environ.get("AUTO_HARNESS_MEMORY_EVOLUTION_PROVIDER"):
            known["memory_evolution_provider"] = os.environ["AUTO_HARNESS_MEMORY_EVOLUTION_PROVIDER"]
        # Plan-first env overrides
        if os.environ.get("AUTO_HARNESS_AGENT_PLAN_FIRST"):
            known["agent_plan_first"] = os.environ.get("AUTO_HARNESS_AGENT_PLAN_FIRST") == "1"
        if os.environ.get("AUTO_HARNESS_AGENT_PLAN_FIRST_PROVIDER"):
            known["agent_plan_first_provider"] = os.environ["AUTO_HARNESS_AGENT_PLAN_FIRST_PROVIDER"]
        if os.environ.get("AUTO_HARNESS_AGENT_PLAN_FIRST_MODE"):
            known["agent_plan_first_mode"] = os.environ["AUTO_HARNESS_AGENT_PLAN_FIRST_MODE"]
        if os.environ.get("AUTO_HARNESS_AGENT_PLAN_FIRST_MAX_REPLANS"):
            known["agent_plan_first_max_replans"] = int(os.environ["AUTO_HARNESS_AGENT_PLAN_FIRST_MAX_REPLANS"])
        if os.environ.get("AUTO_HARNESS_AGENT_REPO_CONTEXT_MODE"):
            known["agent_repo_context_mode"] = os.environ["AUTO_HARNESS_AGENT_REPO_CONTEXT_MODE"]
        if os.environ.get("AUTO_HARNESS_AGENT_REPO_MAX_OBSERVATION_ROUNDS"):
            known["agent_repo_max_observation_rounds"] = int(
                os.environ["AUTO_HARNESS_AGENT_REPO_MAX_OBSERVATION_ROUNDS"]
            )
        repo_integer_overrides = {
            "agent_repo_inventory_budget_tokens": "AUTO_HARNESS_AGENT_REPO_INVENTORY_BUDGET_TOKENS",
            "agent_repo_core_budget_tokens": "AUTO_HARNESS_AGENT_REPO_CORE_BUDGET_TOKENS",
            "agent_repo_observation_budget_tokens": "AUTO_HARNESS_AGENT_REPO_OBSERVATION_BUDGET_TOKENS",
            "agent_repo_max_requests_per_round": "AUTO_HARNESS_AGENT_REPO_MAX_REQUESTS_PER_ROUND",
            "agent_repo_max_observed_files": "AUTO_HARNESS_AGENT_REPO_MAX_OBSERVED_FILES",
            "agent_repo_max_chars_per_read": "AUTO_HARNESS_AGENT_REPO_MAX_CHARS_PER_READ",
            "agent_repo_max_lines_per_read": "AUTO_HARNESS_AGENT_REPO_MAX_LINES_PER_READ",
            "agent_repo_search_max_results": "AUTO_HARNESS_AGENT_REPO_SEARCH_MAX_RESULTS",
            "agent_repo_search_max_files": "AUTO_HARNESS_AGENT_REPO_SEARCH_MAX_FILES",
            "agent_repo_search_max_bytes": "AUTO_HARNESS_AGENT_REPO_SEARCH_MAX_BYTES",
            "agent_repo_tree_max_entries": "AUTO_HARNESS_AGENT_REPO_TREE_MAX_ENTRIES",
        }
        for field, environment_name in repo_integer_overrides.items():
            if os.environ.get(environment_name):
                known[field] = int(os.environ[environment_name])
        if os.environ.get("AUTO_HARNESS_LANGGRAPH_FAULT_INJECTION"):
            known["langgraph_fault_injection_points"] = [
                point.strip()
                for point in os.environ["AUTO_HARNESS_LANGGRAPH_FAULT_INJECTION"].split(",")
                if point.strip()
            ]
        return cls(**known)

    @property
    def runs_path(self) -> Path:
        return Path(self.runs_dir)

    @property
    def skills_path(self) -> Path:
        return Path(self.skills_dir)

    @property
    def memory_path(self) -> Path:
        return Path(self.memory_dir)

    @property
    def model_cache_path(self) -> Path:
        return Path(self.model_cache_dir)

    @property
    def task_queue_path(self) -> Path:
        return Path(self.task_queue_dir)


# ------------------------------------------------------------------
# DeepSeek configuration validation
# ------------------------------------------------------------------

_RUNTIME_OVERRIDE_WHITELIST = frozenset({
    "model",
    "context_window_tokens",
    "max_output_tokens",
})

_RUNTIME_OVERRIDE_FORBIDDEN = frozenset({
    "api_key",
    "authorization",
    "token",
    "password",
    "secret",
})


def validate_runtime_overrides(overrides: dict) -> None:
    """Reject secret-bearing or unknown keys in runtime overrides."""
    if not isinstance(overrides, dict):
        raise ValueError("llm_runtime_overrides must be an object")
    forbidden = _RUNTIME_OVERRIDE_FORBIDDEN.intersection(
        str(k).lower() for k in overrides
    )
    if forbidden:
        raise ValueError(
            "llm_runtime_overrides must not contain secret values: %s"
            % ", ".join(sorted(forbidden))
        )
    unknown = set(overrides) - _RUNTIME_OVERRIDE_WHITELIST
    if unknown:
        raise ValueError(
            "unknown llm_runtime_overrides keys: %s; allowed: %s"
            % (", ".join(sorted(unknown)), ", ".join(sorted(_RUNTIME_OVERRIDE_WHITELIST)))
        )

_RETIRED_DEEPSEEK_MODELS = frozenset({
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-chat-v1",
    "deepseek-coder",
    "deepseek-coder-v1",
})

_VALID_THINKING_VALUES = frozenset({"enabled", "disabled"})
_VALID_REASONING_EFFORT_VALUES = frozenset({"high", "max"})
_VALID_DEEPSEEK_PURPOSES = frozenset({
    "agent", "plan_first", "memory_evolution", "llm_test", "live_smoke",
})


def _validate_deepseek_config(provider_name: str, settings: dict) -> None:
    """Validate DeepSeek-specific provider configuration.

    Only runs when provider_name is 'deepseek' (normalized).
    """
    normalized = str(provider_name).strip().lower().replace("-", "_")
    if normalized != "deepseek":
        return

    # Both endpoint forms must be absolute HTTPS URLs. The Provider repeats
    # this validation after environment overrides have been resolved.
    from urllib.parse import urlparse

    for endpoint_key in ("api_base", "api_url"):
        endpoint = settings.get(endpoint_key, "")
        if not endpoint:
            continue
        parsed = urlparse(str(endpoint))
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ValueError(
                "provider_configs.deepseek.%s must be an absolute https URL, got: %s"
                % (endpoint_key, endpoint)
            )

    # /beta requires allow_beta=true
    effective_endpoint = settings.get("api_url") or settings.get("api_base", "")
    if effective_endpoint:
        endpoint_host = (urlparse(str(effective_endpoint)).hostname or "").lower()
        if (
            endpoint_host != "api.deepseek.com"
            and not settings.get("allow_custom_endpoint", False)
        ):
            raise ValueError(
                "provider_configs.deepseek custom endpoint requires "
                "allow_custom_endpoint=true"
            )
    if "/beta" in str(effective_endpoint):
        if not settings.get("allow_beta", False):
            raise ValueError(
                "provider_configs.deepseek.api_base contains /beta "
                "but allow_beta is not true"
            )

    # models validation
    models = settings.get("models")
    if "models" in settings and not isinstance(models, dict):
        raise ValueError("provider_configs.deepseek.models must be an object")
    if isinstance(models, dict):
        for purpose, model_name in models.items():
            if purpose not in _VALID_DEEPSEEK_PURPOSES:
                raise ValueError(
                    "provider_configs.deepseek.models has unsupported purpose '%s'"
                    % purpose
                )
            if not model_name or not str(model_name).strip():
                raise ValueError(
                    "provider_configs.deepseek.models.%s must be a "
                    "non-empty string" % purpose
                )
            if str(model_name).strip().lower() in _RETIRED_DEEPSEEK_MODELS:
                raise ValueError(
                    "provider_configs.deepseek.models.%s uses retired "
                    "model '%s'; use deepseek-v4-flash or deepseek-v4-pro"
                    % (purpose, model_name)
                )

    # Single model field validation
    single_model = settings.get("model")
    if single_model and str(single_model).strip().lower() in _RETIRED_DEEPSEEK_MODELS:
        raise ValueError(
            "provider_configs.deepseek.model uses retired model '%s'; "
            "use deepseek-v4-flash or deepseek-v4-pro" % single_model
        )

    configured_models = list(models.values()) if isinstance(models, dict) else []
    if single_model:
        configured_models.append(single_model)
    has_unknown_model = any(
        str(model).strip().lower()
        not in {"deepseek-v4-flash", "deepseek-v4-pro"}
        and str(model).strip().lower() not in _RETIRED_DEEPSEEK_MODELS
        for model in configured_models
    )
    if has_unknown_model:
        if settings.get("allow_unknown_model") is not True:
            raise ValueError(
                "provider_configs.deepseek unknown models require "
                "allow_unknown_model=true"
            )
        if not settings.get("context_window_tokens") or not settings.get(
            "max_tokens"
        ):
            raise ValueError(
                "provider_configs.deepseek unknown models require explicit "
                "context_window_tokens and max_tokens"
            )

    # thinking validation
    thinking = settings.get("thinking")
    if "thinking" in settings and not isinstance(thinking, dict):
        raise ValueError("provider_configs.deepseek.thinking must be an object")
    if isinstance(thinking, dict):
        for purpose, value in thinking.items():
            if purpose not in _VALID_DEEPSEEK_PURPOSES:
                raise ValueError(
                    "provider_configs.deepseek.thinking has unsupported purpose '%s'"
                    % purpose
                )
            if str(value) not in _VALID_THINKING_VALUES:
                raise ValueError(
                    "provider_configs.deepseek.thinking.%s must be "
                    "'enabled' or 'disabled', got: %s" % (purpose, value)
                )

    # reasoning_effort validation
    reasoning_effort = settings.get("reasoning_effort")
    if "reasoning_effort" in settings and not isinstance(
        reasoning_effort, dict
    ):
        raise ValueError(
            "provider_configs.deepseek.reasoning_effort must be an object"
        )
    if isinstance(reasoning_effort, dict):
        for purpose, value in reasoning_effort.items():
            if purpose not in _VALID_DEEPSEEK_PURPOSES:
                raise ValueError(
                    "provider_configs.deepseek.reasoning_effort has unsupported "
                    "purpose '%s'" % purpose
                )
            if str(value) not in _VALID_REASONING_EFFORT_VALUES:
                raise ValueError(
                    "provider_configs.deepseek.reasoning_effort.%s "
                    "must be 'high' or 'max', got: %s" % (purpose, value)
                )

    # json_mode must be boolean
    json_mode = settings.get("json_mode")
    if "json_mode" in settings and not isinstance(json_mode, dict):
        raise ValueError("provider_configs.deepseek.json_mode must be an object")
    if isinstance(json_mode, dict):
        for purpose, value in json_mode.items():
            if purpose not in _VALID_DEEPSEEK_PURPOSES:
                raise ValueError(
                    "provider_configs.deepseek.json_mode has unsupported purpose '%s'"
                    % purpose
                )
            if not isinstance(value, bool):
                raise ValueError(
                    "provider_configs.deepseek.json_mode.%s must be "
                    "boolean, got: %s" % (purpose, type(value).__name__)
                )

    # native_tool_calling must be boolean
    if "native_tool_calling" in settings and not isinstance(
        settings["native_tool_calling"], bool
    ):
        raise ValueError(
            "provider_configs.deepseek.native_tool_calling must be boolean"
        )
    if settings.get("native_tool_calling") is True:
        raise ValueError(
            "provider_configs.deepseek.native_tool_calling is not implemented; "
            "keep it false and use json_action"
        )

    # allow_beta must be boolean
    if "allow_beta" in settings and not isinstance(
        settings["allow_beta"], bool
    ):
        raise ValueError(
            "provider_configs.deepseek.allow_beta must be boolean"
        )

    if "allow_custom_endpoint" in settings and not isinstance(
        settings["allow_custom_endpoint"], bool
    ):
        raise ValueError(
            "provider_configs.deepseek.allow_custom_endpoint must be boolean"
        )

    for key in ("allow_unknown_model", "require_api_key"):
        if key in settings and not isinstance(settings[key], bool):
            raise ValueError(
                "provider_configs.deepseek.%s must be boolean" % key
            )

    # Retry config validation
    for key in ("max_retries",):
        if key in settings and (
            isinstance(settings[key], bool)
            or not isinstance(settings[key], int)
            or settings[key] < 0
        ):
            raise ValueError(
                "provider_configs.deepseek.%s must be non-negative" % key
            )

    for key in ("retry_base_seconds", "retry_max_seconds"):
        if key in settings and (
            isinstance(settings[key], bool)
            or not isinstance(settings[key], (int, float))
            or settings[key] <= 0
        ):
            raise ValueError(
                "provider_configs.deepseek.%s must be positive" % key
            )
    if (
        settings.get("retry_base_seconds") is not None
        and settings.get("retry_max_seconds") is not None
        and float(settings["retry_max_seconds"])
        < float(settings["retry_base_seconds"])
    ):
        raise ValueError(
            "provider_configs.deepseek.retry_max_seconds must be greater than "
            "or equal to retry_base_seconds"
        )
