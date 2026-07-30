"""Graph state for LangGraph deployment controller.

DeploymentGraphState is a TypedDict used by the StateGraph.
It only stores serializable types (paths, strings, dicts, lists).
No provider, module, Popen, callback, or file handle references.

schema_version: 2 — includes all fields from Phases 1–6.
Annotated list fields (errors, node_history, recovery_events,
approval_history, agent_trace_paths) use operator.add reducers so
that each node can append entries without overwriting previous ones.
stage_results is updated per-stage (not a list reducer).
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class DeploymentGraphState(TypedDict, total=False):
    """State for the LangGraph deployment StateGraph.

    All fields are optional (total=False) to support incremental
    node updates that return only changed fields.
    """

    # ── Core fields (Phase 1) ──────────────────────────────────────────

    schema_version: int
    task_id: str
    controller: str
    run_dir: str
    repo_dir: str
    dry_run: bool
    runtime_policy: Dict[str, Any]
    current_stage: str
    stage_results: Dict[str, Any]
    verify_status: str
    verify_evidence_paths: List[str]
    failed_stage: str
    resume_from_stage: str
    plan_revision_paths: Dict[str, str]
    replan_count: int
    max_replans: int
    stop_reason: str
    errors: Annotated[List[Dict[str, Any]], operator.add]
    node_history: Annotated[List[Dict[str, Any]], operator.add]

    # ── Recovery fields (Phase 2) ──────────────────────────────────────

    snapshot_path: str
    raw_plan_path: str
    parsed_plan_path: str
    policy_result_path: str
    effective_plan_path: str
    compiled_analysis: Dict[str, Any]
    operation_ids: Dict[str, str]
    recovery_capabilities: Dict[str, bool]
    recovery_events: Annotated[List[Dict[str, Any]], operator.add]
    pending_approval: Optional[Dict[str, Any]]
    approval_history: Annotated[List[Dict[str, Any]], operator.add]
    approved_operation_id: str
    approved_action: str

    # ── Agent failure reasoning (Phase 3) ──────────────────────────────

    failure_context: Dict[str, Any]
    failure_signature: str
    same_failure_count: int
    diagnosis: Dict[str, Any]
    diagnosis_path: str
    diagnose_count: int
    max_diagnoses: int

    # ── Controlled repair (Phase 4) ────────────────────────────────────

    repair_plan: Dict[str, Any]
    repair_plan_path: str
    repair_policy_result: Dict[str, Any]
    repair_policy_path: str
    repair_apply_result: Dict[str, Any]
    repair_apply_path: str
    repair_overlay: Dict[str, Any]
    repair_count: int
    max_repairs: int
    repair_resume_stage: str
    repair_resume_executed: bool

    # ── Recovery gate (Phase 6) ────────────────────────────────────────

    pending_operation: Optional[Dict[str, Any]]
    pending_operation_id: str
    recovery_stage: str
    recovery_decision: str
    recovery_result: Dict[str, Any]
    recovery_skip_stage: bool

    # ── Approval routing ───────────────────────────────────────────────

    approval_kind: str
    approval_resume_target: str

    # ── Agent audit ────────────────────────────────────────────────────

    agent_call_count: int
    agent_trace_paths: Annotated[List[str], operator.add]
    llm_required: bool
    llm_provider: str
    llm_error: str
    llm_context: Dict[str, Any]
    previous_plan_path: str

    # ── Memory/Skill fields (Task 8) ───────────────────────────────────

    memory_hits: List[Dict[str, Any]]
    selected_skills: Dict[str, List[Dict[str, Any]]]
    skill_contexts: Dict[str, Dict[str, Any]]
    skill_route_paths: Dict[str, str]
    verified_memory_path: str
    skill_outcome_paths: List[str]
