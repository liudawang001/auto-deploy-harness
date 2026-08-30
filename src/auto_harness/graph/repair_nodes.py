"""Repair nodes for the LangGraph deployment StateGraph.

Implements the controlled repair pipeline:
  repair_plan -> repair_policy -> approval(optional) -> repair_apply -> select_repair_resume

Key invariants:
- Repair actions are NOT executed until policy approves
- Repair apply != repair effective; only verify pass marks effective
- Source edit / cleanup / unapproved actions must interrupt
- repair attempt limit is enforced
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from auto_harness.models.base import read_json, write_json
from auto_harness.models.result import StageResult
from auto_harness.models.task import RuntimePolicy
from auto_harness.utils.time import utc_now_iso


def repair_plan_node(state, deps):
    """repair_plan node: propose repair actions from diagnosis.

    Uses RepairPlanner.propose() with the failed stage result and diagnosis.
    Writes repairs/repair_plan_<count>.json.
    """
    failed_stage = state.get("failed_stage", "")
    stage_results = state.get("stage_results", {})
    failed = stage_results.get(failed_stage, {})

    # Reconstruct StageResult from state dict
    stage_result = StageResult(
        stage=failed_stage,
        status=failed.get("status", "failed"),
        summary=failed.get("summary", ""),
        data=dict(failed.get("data") or {}),
        evidence=list(failed.get("evidence") or []),
        error=failed.get("error"),
    )

    # Inject diagnosis into stage result data
    stage_result.data["agent_diagnosis"] = state.get("diagnosis", {})

    # Call RepairPlanner with skill_context (Task 10)
    repair_planner = deps.repair_planner
    skill_context = state.get("skill_contexts", {}).get("repair", {})
    plan = repair_planner.propose(
        failed_stage,
        stage_result,
        analysis=state.get("compiled_analysis", {}),
        skill_context=skill_context,
    )

    # Write repair plan artifact
    repair_count = int(state.get("repair_count", 0))
    run_dir = Path(state["run_dir"])
    repairs_dir = run_dir / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = repairs_dir / ("repair_plan_%s.json" % (repair_count + 1))
    write_json(plan_path, plan)

    return {
        "repair_plan": plan,
        "repair_plan_path": str(plan_path),
        "current_stage": "repair_plan",
    }


def repair_policy_node(state, deps):
    """repair_policy node: check repair plan against policy and loop limits.

    Uses RepairPolicy.check() and RepairLoopController.gate().
    Writes repairs/repair_policy_<count>.json.
    """
    repair_plan = state.get("repair_plan", {})
    if not repair_plan:
        return {
            "stop_reason": "repair_plan_missing",
            "current_stage": "repair_policy",
        }

    # Check repair limit
    repair_count = int(state.get("repair_count", 0))
    max_repairs = int(state.get("max_repairs", 2))
    if repair_count >= max_repairs:
        return {
            "stop_reason": "repair_limit_reached",
            "current_stage": "repair_policy",
        }

    # Build RuntimePolicy from state
    policy_dict = state.get("runtime_policy", {})
    runtime = RuntimePolicy(
        workspace_root=str(Path(state["run_dir"]) / "workspace"),
        allow_network=bool(policy_dict.get("allow_network", True)),
        allow_gpu=bool(policy_dict.get("allow_gpu", False)),
        allow_source_edit=bool(policy_dict.get("allow_source_edit", False)),
        allow_dependency_install=bool(policy_dict.get("allow_dependency_install", False)),
        allow_service_start=bool(policy_dict.get("allow_service_start", False)),
    )

    # Check if any action requires operator approval
    current_approval = None
    actions = repair_plan.get("actions", [])
    needs_approval = False
    approval_reasons = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        requires = action.get("requires", {})
        if requires.get("operator_approval"):
            needs_approval = True
            approval_reasons.append(action.get("type", "unknown"))
        if action.get("type") == "source_edit" and not runtime.allow_source_edit:
            needs_approval = True
            approval_reasons.append("source_edit")
        if action.get("type") in ("cleanup_external", "cleanup_then_retry"):
            needs_approval = True
            approval_reasons.append(action.get("type"))

    # Call RepairPolicy
    repair_policy = deps.repair_policy
    raw_policy_result = repair_policy.check(
        repair_plan,
        runtime,
        operator_approval=current_approval,
    )

    # Call RepairLoopController
    repair_loop = deps.repair_loop
    effective_policy = repair_loop.gate(
        Path(state["run_dir"]),
        state.get("failed_stage", ""),
        {"signature": state.get("failure_signature", "")},
        repair_plan,
        raw_policy_result,
    )

    # Write policy result artifact
    run_dir = Path(state["run_dir"])
    repairs_dir = run_dir / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    policy_path = repairs_dir / ("repair_policy_%s.json" % (repair_count + 1))
    write_json(policy_path, effective_policy)

    update = {
        "repair_policy_result": effective_policy,
        "repair_policy_path": str(policy_path),
        "current_stage": "repair_policy",
    }

    # Determine if approval is needed
    if needs_approval and effective_policy.get("allowed"):
        update["approval_kind"] = "repair"
        update["approval_resume_target"] = "repair_apply"
        operation_id = state.get("pending_operation_id", "")
        from auto_harness.graph.approval import build_approval_request
        update["pending_approval"] = build_approval_request(
            approval_id="repair_%s_%s" % (repair_count + 1, (operation_id or "unknown")[:8]),
            operation_id=operation_id or "repair_%d" % (repair_count + 1),
            approval_kind="repair",
            requested_action="apply_repair",
            risk="high",
            reason="repair requires operator approval: %s" % ", ".join(approval_reasons),
        )

    # Check if policy rejected all actions
    if not effective_policy.get("allowed"):
        update["stop_reason"] = "repair_policy_rejected"

    return update


def repair_apply_node(state, deps, config=None):
    """repair_apply node: apply approved repair actions.

    Uses RepairApplier.apply() and loads RepairOverlay.
    Writes repairs/repair_apply_<count>.json.
    """
    repair_plan = state.get("repair_plan", {})
    repair_policy_result = state.get("repair_policy_result", {})

    if not repair_plan or not repair_policy_result.get("allowed"):
        return {
            "stop_reason": "repair_not_allowed",
            "current_stage": "repair_apply",
        }

    # Build env_context from state
    env_context = _extract_env_context(state, config=config)

    # Determine execute flag
    enable_repair = True
    if config and hasattr(config, "langgraph_enable_repair"):
        enable_repair = config.langgraph_enable_repair
    execute = not state.get("dry_run", True) and enable_repair

    # Call RepairApplier
    repair_applier = deps.repair_applier
    apply_result = repair_applier.apply(
        Path(state["run_dir"]),
        repair_plan,
        repair_policy_result,
        execute=execute,
        timeout_seconds=getattr(config, "default_timeout_seconds", 900) if config else 900,
        allowed_commands=getattr(config, "allowed_commands", None) if config else None,
        env_context=env_context,
    )
    from auto_harness.repair.evidence import (
        build_repair_attempt,
        is_effective_repair_action,
    )
    previous_verify = state.get("stage_results", {}).get("verify", {})
    previous_verify_data = (
        previous_verify.get("data", {})
        if isinstance(previous_verify, dict)
        else {}
    )
    action_results = list(apply_result.get("action_results") or [])
    effective_action_count = sum(
        1 for item in action_results if is_effective_repair_action(item)
    )
    metadata_only_count = sum(
        1
        for item in action_results
        if item.get("metadata_only") or item.get("status") == "metadata_only"
    )
    apply_result["verification_trace_before"] = str(
        previous_verify_data.get("trace_id") or ""
    )
    apply_result["effective_action_count"] = effective_action_count
    apply_result["metadata_only_count"] = metadata_only_count
    apply_result["applied_at"] = utc_now_iso()
    apply_result["resume_executed"] = False

    # Load repair overlay
    repair_overlay_mod = deps.repair_overlay
    overlay = repair_overlay_mod.load(Path(state["run_dir"]))

    # Write apply result artifact
    repair_count = int(state.get("repair_count", 0))
    run_dir = Path(state["run_dir"])
    repairs_dir = run_dir / "repairs"
    repairs_dir.mkdir(parents=True, exist_ok=True)
    apply_path = repairs_dir / ("repair_apply_%s.json" % (repair_count + 1))
    write_json(apply_path, apply_result)
    write_json(repairs_dir / "repair_apply_result.json", apply_result)
    attempt_path = repairs_dir / ("attempt_%s.json" % (repair_count + 1))
    write_json(attempt_path, build_repair_attempt(
        attempt=repair_count + 1,
        failure_signature_before=state.get("failure_signature", ""),
        diagnosis_path=state.get("diagnosis_path", ""),
        plan_path=state.get("repair_plan_path", ""),
        policy_path=state.get("repair_policy_path", ""),
        apply_path=str(apply_path),
        resume_from_stage=repair_plan.get("rerun_from_effective", "")
        or repair_plan.get("rerun_from", ""),
        effective_action_count=effective_action_count,
        metadata_only_count=metadata_only_count,
    ))

    # Determine rerun stage
    rerun_from = repair_plan.get("rerun_from_effective", "") or repair_plan.get("rerun_from", "")
    safe_stages = {"env_deploy", "model_prepare", "runner", "verify"}
    repair_resume_stage = rerun_from if rerun_from in safe_stages else "env_deploy"

    return {
        "repair_apply_result": apply_result,
        "repair_apply_path": str(apply_path),
        "repair_overlay": overlay,
        "repair_count": repair_count + 1,
        "repair_resume_stage": repair_resume_stage,
        "repair_resume_executed": False,
        "current_stage": "repair_apply",
    }


def select_repair_resume_node(state, deps):
    """select_repair_resume: determine which stage to rerun after repair.

    Routes to the repair_resume_stage from the repair plan,
    or to verify if only verify hints were updated.
    """
    resume_stage = state.get("repair_resume_stage", "verify")
    safe_stages = {"env_deploy", "model_prepare", "runner", "verify"}
    selected = resume_stage if resume_stage in safe_stages else "env_deploy"

    return {
        "resume_from_stage": selected,
        "repair_resume_executed": True,
        "current_stage": "select_repair_resume",
    }


def _extract_env_context(state, config=None):
    """Extract environment context for repair applier."""
    env_context = {}
    runtime_policy = state.get("runtime_policy", {})
    if runtime_policy.get("allow_dependency_install"):
        env_context["can_install"] = True
    if runtime_policy.get("allow_service_start"):
        env_context["can_start_service"] = True
    env_context["dry_run"] = state.get("dry_run", True)
    env_context["run_dir"] = state.get("run_dir", "")
    compiled = state.get("compiled_analysis") or {}
    candidates = compiled.get("run_candidates") or []
    if any(
        isinstance(candidate, dict)
        and candidate.get("required_backend") == "docker"
        for candidate in candidates
    ):
        env_context["execution_backend"] = "docker"
    elif config and getattr(config, "execution_backend", "local") == "docker":
        env_context["execution_backend"] = "docker"
    if config:
        env_context["docker_image"] = getattr(config, "docker_image", "python:3.13-slim")
        env_context["docker_network"] = getattr(config, "docker_network", "bridge")
        env_context["docker_gpus"] = getattr(config, "docker_gpus", "none")
    return env_context
