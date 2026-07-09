"""Repair tools for the LLM-driven repair agent.

This module provides tool implementations for the repair sub-agent.
Per design doc §11, repair is second priority after verify — this
file provides skeleton interfaces only.

Repair tools will be implemented when the repair agent loop is built.
The expected tools are:
- apply_repair: execute a policy-approved repair action
- resume_from_stage: resume the pipeline from a safe stage

Both tools must follow the same safety model as verify tools:
- Only accept policy-normalized input
- Only operate within allowed boundaries
- Write evidence files
- Return ToolResult with strong verification
"""
from typing import Dict

from auto_harness.agent_runtime.schemas import ToolResult
from auto_harness.utils.time import utc_now_iso


def apply_repair(tool_input: Dict, context: Dict) -> ToolResult:
    """Execute a policy-approved repair action.

    NOT YET IMPLEMENTED — skeleton for Phase 2 (repair agent).

    Input: action (dict with type, package, command, etc.)
    """
    started = utc_now_iso()
    return ToolResult(
        status="rejected",
        tool_name="apply_repair",
        evidence={},
        error="apply_repair not yet implemented",
        started_at=started,
        ended_at=utc_now_iso(),
    )


def resume_from_stage(tool_input: Dict, context: Dict) -> ToolResult:
    """Resume the pipeline from a safe stage after repair.

    NOT YET IMPLEMENTED — skeleton for Phase 2 (repair agent).

    Input: stage (string), run_dir (string)
    """
    started = utc_now_iso()
    return ToolResult(
        status="rejected",
        tool_name="resume_from_stage",
        evidence={},
        error="resume_from_stage not yet implemented",
        started_at=started,
        ended_at=utc_now_iso(),
    )
