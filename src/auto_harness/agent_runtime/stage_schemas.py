"""Stage-specific schemas for LLM Decision Gates.

Extends the base agent_runtime/schemas.py with schemas for each
decision gate stage: runner, env_solve, model_prepare, repair, plan.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ------------------------------------------------------------------
# Stage tool sets (tools each gate can use)
# ------------------------------------------------------------------

RUNNER_TOOLS = ("select_runner_candidate", "add_runner_candidate", "reject_runner_candidate")

ENV_TOOLS = (
    "inspect_env_log",
    "propose_dependency_constraint",
    "apply_dependency_constraint",
    "select_environment_backend",
    "select_torch_variant",
)

MODEL_TOOLS = (
    "select_model_source",
    "select_model_asset_strategy",
    "download_model_asset",
    "link_cached_model_asset",
    "inspect_model_config",
    "inspect_git_lfs_pointers",
)

REPAIR_TOOLS = (
    "inspect_log",
    "classify_failure",
    "apply_dependency_constraint",
    "apply_repair",
    "resume_from_stage",
    "verify_after_repair",
)

PLAN_TOOLS = ("set_deployment_strategy", "set_stage_hint")  # Plan gate only generates strategy hints

# All stage tools combined (for registry)
ALL_STAGE_TOOLS = RUNNER_TOOLS + ENV_TOOLS + MODEL_TOOLS + REPAIR_TOOLS + PLAN_TOOLS

# Pipeline stages that the plan gate can reference
PIPELINE_STAGES = (
    "analyze", "resource_plan", "env_solve", "env_deploy",
    "model_prepare", "runner", "verify", "report",
)

# Stages where repair can rerun from
SAFE_RERUN_STAGES = ("env_deploy", "model_prepare", "runner", "verify")


# ------------------------------------------------------------------
# Decision Gate Result
# ------------------------------------------------------------------

@dataclass
class GateDecision:
    """Parsed LLM decision from a decision gate."""
    stage: str = ""
    status: str = "invalid"  # ok | no_action | invalid
    hypothesis: str = ""
    confidence: float = 0.0
    tool_call: Optional[Dict] = None  # {"name": ..., "input": {...}}
    expected_observation: str = ""
    stop_reason: Optional[str] = None
    raw_response: str = ""


@dataclass
class GateResult:
    """Complete result from a decision gate invocation.

    Captures the full observe -> plan -> schema -> critic -> policy -> execute pipeline.
    """
    stage: str = ""
    mode: str = ""  # planner | gated_actor
    decision_status: str = "skipped"  # ok | no_action | invalid | skipped | error
    hypothesis: str = ""
    tool_call: Optional[Dict] = None
    critic: Dict = field(default_factory=dict)
    policy: Dict = field(default_factory=dict)
    execution: Dict = field(default_factory=dict)
    state_delta: Dict = field(default_factory=dict)
    llm_helped: bool = False
    error: Optional[str] = None


# ------------------------------------------------------------------
# Plan Gate schemas
# ------------------------------------------------------------------

@dataclass
class StagePlanHint:
    """A single stage hint from the plan gate."""
    stage: str = ""
    strategy: str = ""
    hints: Dict = field(default_factory=dict)


@dataclass
class DeploymentStrategy:
    """LLM-generated deployment strategy from the plan gate."""
    status: str = "invalid"  # ok | invalid
    deployment_strategy: str = ""
    confidence: float = 0.0
    stage_plan: List[Dict] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    fallback: str = "deterministic_pipeline"
    raw_response: str = ""


# ------------------------------------------------------------------
# Repair Actuator schemas
# ------------------------------------------------------------------

REPAIR_STATUSES = ("planned", "applied", "executed", "verified")


@dataclass
class RepairActuatorResult:
    """Result from the repair actuator gate.

    Distinguishes planned/applied/executed/verified:
    - planned: LLM proposed repair, schema/critic/policy passed
    - applied: repair action was applied (state delta written)
    - executed: repair tool was executed (command ran)
    - verified: final verify passed after repair
    """
    repair_status: str = "planned"
    hypothesis: str = ""
    tool_call: Optional[Dict] = None
    policy_allowed: bool = False
    executed: bool = False
    metadata_only: bool = False
    repair_verified: bool = False
    resume_from_stage: str = ""
    final_verify_status: str = ""
    artifacts: List[str] = field(default_factory=list)
