import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    use_agent_analyzer: bool = False
    agent_timeout_seconds: int = 900
    agent_mode: str = "off"
    agent_provider: str = "mock"
    agent_max_input_chars: int = 20000
    agent_max_file_chars: int = 6000
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
    memory_evolution_provider: str = "mock"
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
    agent_plan_first_provider: str = "mock"
    agent_plan_first_mode: str = "planner"  # planner | gated_actor
    agent_plan_first_max_replans: int = 2
    agent_plan_first_max_files: int = 80
    agent_plan_first_max_file_chars: int = 6000
    agent_plan_first_require_grounding: bool = True
    agent_plan_first_allow_external_network: bool = False
    default_controller: str = "langgraph"
    langgraph_require_llm: bool = True
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
    langgraph_planner_mode: str = "llm"  # llm | deterministic

    def __post_init__(self) -> None:
        if self.allowed_commands is None:
            self.allowed_commands = ["python", "python3", "pip", "curl", "git", "streamlit"]
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
        if self.langgraph_planner_mode not in ("llm", "deterministic"):
            raise ValueError(
                "langgraph_planner_mode must be 'llm' or 'deterministic', got: %s" % self.langgraph_planner_mode
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
        if not config_path.exists():
            return cls(
                use_agent_analyzer=os.environ.get("AUTO_HARNESS_USE_AGENT_ANALYZER") == "1"
            )
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known: Dict[str, Any] = {}
        for key in cls.__dataclass_fields__:
            if key in data:
                known[key] = data[key]
        if os.environ.get("AUTO_HARNESS_USE_AGENT_ANALYZER"):
            known["use_agent_analyzer"] = os.environ.get("AUTO_HARNESS_USE_AGENT_ANALYZER") == "1"
        if os.environ.get("AUTO_HARNESS_AGENT_MODE"):
            known["agent_mode"] = os.environ["AUTO_HARNESS_AGENT_MODE"]
        if os.environ.get("AUTO_HARNESS_AGENT_PROVIDER"):
            known["agent_provider"] = os.environ["AUTO_HARNESS_AGENT_PROVIDER"]
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
