"""Graph state for LangGraph deployment controller.

DeploymentGraphState is a TypedDict used by the StateGraph.
It only stores serializable types (paths, strings, dicts, lists).
No provider, module, Popen, callback, or file handle references.

schema_version: starts at 1, incremented on breaking state changes.
errors and node_history use Annotated list reducers (operator.add).
stage_results is updated per-stage (not a list reducer).
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class DeploymentGraphState(TypedDict, total=False):
    """State for the LangGraph deployment StateGraph.

    All fields are optional (total=False) to support incremental
    node updates that return only changed fields.
    """
    schema_version: int
    task_id: str
    controller: str
    run_dir: str
    repo_dir: str
    dry_run: bool
    runtime_policy: Dict[str, Any]
    snapshot_path: str
    raw_plan_path: str
    parsed_plan_path: str
    policy_result_path: str
    effective_plan_path: str
    previous_plan_path: str
    compiled_analysis: Dict[str, Any]
    resume_from_stage: str
    plan_revision_paths: Dict[str, str]
    current_stage: str
    stage_results: Dict[str, Dict[str, Any]]
    verify_status: str
    verify_evidence_paths: List[str]
    failed_stage: str
    replan_count: int
    max_replans: int
    stop_reason: str
    errors: Annotated[List[Dict[str, Any]], operator.add]
    node_history: Annotated[List[Dict[str, Any]], operator.add]
    # Recovery fields (Phase 2)
    operation_ids: Dict[str, str]
    recovery_capabilities: Dict[str, bool]
    recovery_events: Annotated[List[Dict[str, Any]], operator.add]
    pending_approval: Optional[Dict[str, Any]]
    approval_history: Annotated[List[Dict[str, Any]], operator.add]
    approved_operation_id: str
    approved_action: str
