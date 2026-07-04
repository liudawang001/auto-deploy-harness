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
    verify_workspace_name: str = "verify_workspace"
    allowed_commands: List[str] = None
    use_agent_analyzer: bool = False
    agent_timeout_seconds: int = 900
    skills_dir: str = "skills"
    memory_dir: str = "memory"
    model_cache_dir: str = "model_cache"
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

    def __post_init__(self) -> None:
        if self.allowed_commands is None:
            self.allowed_commands = ["python", "python3", "pip", "curl", "git", "streamlit"]
        if self.model_cache_cleanup_keep_cache_keys is None:
            self.model_cache_cleanup_keep_cache_keys = []
        if self.model_cache_cleanup_keep_repo_ids is None:
            self.model_cache_cleanup_keep_repo_ids = []

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
