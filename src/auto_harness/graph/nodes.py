"""Graph nodes for the LangGraph deployment StateGraph.

Each node does ONE responsibility and returns a state delta.
Nodes never modify the input state directly — they return a dict
that LangGraph merges according to the state's reducer annotations.

GraphNodeDependencies holds all injectable callables so that nodes
can be tested with mocks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_harness.agent_runtime.deployment_plan import DeploymentPlan
from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


@dataclass
class GraphNodeDependencies:
    """Dependencies injected into graph nodes.

    Each field is a callable or object that the node uses.
    In production, these point to real modules.
    In tests, they are replaced with mocks.
    """
    build_snapshot: Any
    build_replan_input: Any
    determine_resume_stage: Any
    merge_analysis: Any
    planner: Any
    parser: Any
    policy_gate: Any
    compiler: Any
    stage_executor: Any
    artifact_writer_factory: Any
    runtime_config: Any


def stage_data(state, stage):
    """Extract stage result data from the state."""
    result = state.get("stage_results", {}).get(stage, {})
    data = result.get("data", {}) if isinstance(result, dict) else {}
    return data if isinstance(data, dict) else {}


def make_stage_node(stage, deps):
    """Create a node function for a specific pipeline stage.

    The returned function takes state and returns a state delta
    with updated stage_results, current_stage, and node_history.
    """
    def execute_stage(state):
        deterministic_analysis = stage_data(state, "analyze")
        analysis = deps.merge_analysis(
            deterministic_analysis,
            state.get("compiled_analysis", {}),
        )
        resource_data = stage_data(state, "resource_plan")
        deploy_analysis = stage_data(state, "env_solve").get("analysis", analysis)
        runner_data = stage_data(state, "runner")

        executed = deps.stage_executor.execute_stage(
            task_id=state["task_id"],
            run_dir=Path(state["run_dir"]),
            repo_dir=Path(state["repo_dir"]),
            stage=stage,
            state=state,
            analysis=analysis,
            resource_data=resource_data,
            deploy_analysis=deploy_analysis,
            runner_data=runner_data,
            dry_run=state["dry_run"],
            stage_hints={},
            repair_overlay={},
        )

        results = dict(state.get("stage_results", {}))
        results[stage] = executed.result or {
            "status": executed.after_status,
            "error": executed.error,
            "data": {},
        }
        update = {
            "current_stage": stage,
            "stage_results": results,
            "node_history": [{
                "node": stage,
                "status": executed.after_status,
                "at": utc_now_iso(),
            }],
        }
        if stage == "verify":
            update["verify_status"] = executed.after_status
            update["verify_evidence_paths"] = list(executed.evidence_paths)
        if executed.after_status in ("failed", "uncertain"):
            update["failed_stage"] = stage
        return update

    return execute_stage


def merge_plan_analysis(deterministic, compiled):
    """Merge deterministic analysis with compiled plan analysis.

    Preserves PlanFirstDeploymentLoop._execute_analyze() override semantics:
    compiled plan keys take priority over deterministic ones.
    """
    merged = dict(deterministic or {})
    plan_owned_keys = (
        "install_plan", "run_candidates", "verify_hint", "environment_strategy",
        "selected_candidate", "selection_source", "llm_plan", "llm_candidates",
        "merged_candidates", "llm_required_reason",
    )
    for key in plan_owned_keys:
        if key in compiled:
            merged[key] = compiled[key]
    return merged


class DeploymentGraphNodes:
    """Collection of node functions for the deployment StateGraph."""

    def __init__(self, deps):
        self.deps = deps

    def _artifacts(self, state):
        return self.deps.artifact_writer_factory(Path(state["run_dir"]))

    def build_snapshot(self, state):
        """Build project snapshot and write to artifact."""
        snapshot = self.deps.build_snapshot(state)
        path = self._artifacts(state).write_project_snapshot(snapshot)
        return {"snapshot_path": str(path), "current_stage": "snapshot"}

    def plan(self, state):
        """Ask LLM to generate a deployment plan."""
        snapshot = read_json(Path(state["snapshot_path"]))
        raw = self.deps.planner.plan(
            snapshot,
            skill_context=snapshot.get("skill_context", {}),
        )
        path = self._artifacts(state).write_raw_plan({"raw_text": raw.text[:10000]})
        return {"raw_plan_path": str(path), "current_stage": "plan"}

    def parse(self, state):
        """Parse the raw plan text into a structured plan."""
        raw = read_json(Path(state["raw_plan_path"]))
        try:
            parsed = self.deps.parser.parse(raw.get("raw_text", ""))
            stop_reason = "" if parsed.status == "ok" else "plan_not_ok"
        except ValueError as exc:
            parsed = DeploymentPlan(status="invalid", summary=str(exc))
            stop_reason = "plan_parse_failed"
        path = self._artifacts(state).write_parsed_plan(parsed.to_dict())
        return {"parsed_plan_path": str(path), "stop_reason": stop_reason}

    def policy(self, state):
        """Validate the parsed plan through the policy gate."""
        parsed = read_json(Path(state["parsed_plan_path"]))
        snapshot = read_json(Path(state["snapshot_path"]))
        result = self.deps.policy_gate.validate(
            parsed,
            snapshot,
            runtime_policy=state.get("runtime_policy", {}),
            config=self.deps.runtime_config,
        )
        path = self._artifacts(state).write_policy_result(result)
        return {
            "policy_result_path": str(path),
            "stop_reason": "" if result.get("allowed") else "policy_rejected",
        }

    def compile(self, state):
        """Compile the policy-validated plan into effective plan and analysis."""
        parsed = read_json(Path(state["parsed_plan_path"]))
        policy = read_json(Path(state["policy_result_path"]))
        compiled = self.deps.compiler.compile(policy.get("normalized_plan", parsed))
        path = self._artifacts(state).write_effective_plan(
            compiled.get("effective_plan", {})
        )
        return {
            "effective_plan_path": str(path),
            "compiled_analysis": compiled.get("analysis", {}),
        }

    def select_resume(self, state):
        """Determine which stage to resume from after (re)plan."""
        if int(state.get("replan_count", 0)) == 0:
            return {"resume_from_stage": "analyze"}
        previous_path = state.get("previous_plan_path", "")
        if not previous_path:
            return {
                "resume_from_stage": "analyze",
                "errors": [{"node": "select_resume", "error": "previous_plan_missing"}],
            }
        previous = read_json(Path(previous_path))
        current = read_json(Path(state["parsed_plan_path"]))
        requested = self.deps.determine_resume_stage(previous, current)
        allowed = {
            "analyze", "resource_plan", "env_solve", "env_deploy",
            "model_prepare", "runner", "verify",
        }
        selected = requested if requested in allowed else "analyze"
        revision = int(state.get("replan_count", 0))
        revision_path = Path(state["run_dir"]) / "reports" / "replans" / (
            "replan_%s.revision.json" % revision
        )
        revision_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(revision_path, {
            "revision": revision,
            "trigger_stage": state.get("failed_stage", ""),
            "previous_plan_id": previous.get("plan_id", ""),
            "new_plan_id": current.get("plan_id", ""),
            "resume_from": selected,
            "policy_allowed": True,
        })
        revision_paths = dict(state.get("plan_revision_paths", {}))
        revision_paths[str(revision)] = str(revision_path)
        return {
            "resume_from_stage": selected,
            "plan_revision_paths": revision_paths,
        }

    def replan(self, state):
        """Generate a revised plan based on failure context."""
        snapshot, previous_plan, failure = self.deps.build_replan_input(state)
        raw = self.deps.planner.replan(snapshot, previous_plan, failure)
        revision = int(state.get("replan_count", 0)) + 1
        replans_dir = Path(state["run_dir"]) / "reports" / "replans"
        replans_dir.mkdir(parents=True, exist_ok=True)
        path = replans_dir / ("replan_%s.raw.json" % revision)
        previous_path = replans_dir / ("replan_%s.previous.json" % revision)
        # Save previous plan snapshot before it gets overwritten
        write_json(previous_path, previous_plan)
        write_json(path, {"raw_text": raw.text[:10000], "revision": revision})
        return {
            "raw_plan_path": str(path),
            "previous_plan_path": str(previous_path),
            "replan_count": revision,
            "stop_reason": "",
        }

    def report(self, state):
        """Write pipeline results and mark report stage."""
        self._artifacts(state).write_pipeline_results(state.get("stage_results", {}))
        return {"current_stage": "report"}

    def stop(self, state):
        """Terminal node: marks the graph as stopped."""
        return {"current_stage": "stop"}
