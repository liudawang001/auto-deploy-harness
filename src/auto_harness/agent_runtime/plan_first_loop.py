"""LLM Deployment Planner and Plan-first Deployment Loop.

LLMDeploymentPlanner: calls LLM to generate/replan deployment plans.
PlanFirstDeploymentLoop: orchestrates the plan-first flow:
  snapshot -> LLM plan -> parse -> policy gate -> compile -> execute stages -> verify
"""
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.agent_runtime.deployment_plan import DeploymentPlan, DeploymentPlanParser
from auto_harness.agent_runtime.plan_artifacts import PlanArtifactWriter
from auto_harness.agent_runtime.plan_compiler import PlanCompiler
from auto_harness.agent_runtime.plan_policy import PlanPolicyGate
from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder
from auto_harness.models.base import write_json


# System prompt for the deployment planner
PLANNER_SYSTEM_PROMPT = """You are an LLM deployment planner inside auto-deploy-harness.

Your job is to inspect a project snapshot and produce a deployment plan.
You do not execute commands. You only propose structured candidates.
The Python framework validates, compiles, and executes the plan.
Final success is decided only by trace-based verification.

Rules:
- Return JSON only.
- Do not include prose outside JSON.
- Do not mark deployment success yourself.
- Do not propose shell strings. Commands must be arrays of arguments.
- Do not use shell metacharacters such as ;, &&, |, >, <, `$()`, backticks.
- Do not read or exfiltrate secrets.
- Do not require source edits unless explicitly asked.
- Prefer local files and documented entrypoints.
- Every selected command must be grounded in a project file.
- Verify request must include {{trace_id}}.
- If no safe plan exists, return status=no_safe_plan."""


# User prompt template
PLANNER_USER_TEMPLATE = """Project snapshot:
{snapshot_json}

Generate a deployment plan using this JSON schema:
{schema_json}

Important:
- install_commands are untrusted proposals.
- run.candidates are untrusted proposals.
- verify.request must include {{trace_id}}.
- Explain grounding using file paths from selected_files."""


# Replan prompt template
REPLAN_TEMPLATE = """Previous deployment plan:
{previous_plan_json}

Failure context:
{failure_context_json}

Project snapshot:
{snapshot_json}

Revise the deployment plan.
Keep safe commands only.
Do not repeat failed command unless you explain why it should now work.
Return full JSON plan, not a patch."""


# Deployment plan JSON schema (for LLM prompt)
DEPLOYMENT_PLAN_SCHEMA = {
    "type": "object",
    "required": ["status", "plan_id", "summary", "grounding", "environment", "run", "verify"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "needs_human_input", "no_safe_plan"]},
        "plan_id": {"type": "string"},
        "summary": {"type": "string"},
        "grounding": {"type": "array", "items": {"type": "object", "required": ["claim", "file", "reason"]}},
        "environment": {
            "type": "object",
            "required": ["install_commands"],
            "properties": {
                "backend": {"type": "string"},
                "python": {"type": "string"},
                "install_commands": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
            },
        },
        "model_assets": {"type": "object"},
        "run": {
            "type": "object",
            "required": ["candidates", "selected_candidate_id"],
            "properties": {
                "candidates": {"type": "array"},
                "selected_candidate_id": {"type": "string"},
            },
        },
        "verify": {
            "type": "object",
            "required": ["request"],
            "properties": {
                "service_type": {"type": "string"},
                "request": {"type": "object"},
                "success_evidence": {"type": "string"},
            },
        },
        "risks": {"type": "array"},
        "fallbacks": {"type": "array"},
    },
}


class LLMDeploymentPlanner:
    """Calls LLM to generate or revise deployment plans."""

    def __init__(self, provider: Any, max_tokens: int = 4000) -> None:
        self.provider = provider
        self.max_tokens = max_tokens

    def plan(self, snapshot: Dict, mode: str = "planner") -> Any:
        """Ask LLM to generate a deployment plan from project snapshot."""
        from auto_harness.providers.base import Message

        system_msg = Message(role="system", content=PLANNER_SYSTEM_PROMPT)
        user_msg = Message(
            role="user",
            content=PLANNER_USER_TEMPLATE.format(
                snapshot_json=json.dumps(snapshot, ensure_ascii=False, indent=2)[:8000],
                schema_json=json.dumps(DEPLOYMENT_PLAN_SCHEMA, ensure_ascii=False, indent=2),
            ),
        )
        return self.provider.complete(messages=[system_msg, user_msg])

    def replan(
        self,
        snapshot: Dict,
        previous_plan: Dict,
        failure_context: Dict,
    ) -> Any:
        """Ask LLM to revise the deployment plan based on failure context."""
        from auto_harness.providers.base import Message

        system_msg = Message(role="system", content=PLANNER_SYSTEM_PROMPT)
        replan_msg = Message(
            role="user",
            content=REPLAN_TEMPLATE.format(
                previous_plan_json=json.dumps(previous_plan, ensure_ascii=False, indent=2)[:4000],
                failure_context_json=json.dumps(failure_context, ensure_ascii=False, indent=2)[:4000],
                snapshot_json=json.dumps(snapshot, ensure_ascii=False, indent=2)[:8000],
            ),
        )
        return self.provider.complete(messages=[system_msg, replan_msg])


class PlanFirstDeploymentLoop:
    """Orchestrates the plan-first deployment flow.

    Flow:
    1. Build project snapshot
    2. LLM generates deployment plan
    3. Parse and validate plan
    4. Policy gate validates commands/paths/verify
    5. Compile plan to pipeline-consumable format
    6. Execute stages using existing modules
    7. Verify with trace evidence
    8. On failure, LLM replan with failure context
    """

    # Pipeline stages to execute (in order)
    EXECUTION_STAGES = (
        "analyze", "resource_plan", "env_solve", "env_deploy",
        "model_prepare", "runner", "verify",
    )

    # Safe stages to resume from after replan
    SAFE_RESUME_STAGES = frozenset({
        "env_deploy", "model_prepare", "runner", "verify",
    })

    def __init__(
        self,
        provider: Any,
        config: Any,
        stage_executor: Any = None,
        runtime_policy: Optional[Dict] = None,
        max_replans: int = 2,
    ) -> None:
        self.provider = provider
        self.config = config
        self.stage_executor = stage_executor
        self.runtime_policy = runtime_policy or {}
        self.max_replans = max_replans
        self.planner = LLMDeploymentPlanner(provider)
        self.parser = DeploymentPlanParser()
        self.policy_gate = PlanPolicyGate()
        self.compiler = PlanCompiler()

    def run(
        self,
        task_id: str,
        run_dir: Path,
        repo_dir: Path,
        dry_run: bool = True,
    ) -> Dict:
        """Run the plan-first deployment loop."""
        run_dir = Path(run_dir)
        repo_dir = Path(repo_dir)
        artifacts = PlanArtifactWriter(run_dir)

        # 1. Build project snapshot
        snapshot_builder = ProjectSnapshotBuilder(
            max_files=getattr(self.config, "agent_plan_first_max_files", 80),
            max_file_chars=getattr(self.config, "agent_plan_first_max_file_chars", 6000),
        )
        snapshot = snapshot_builder.build(repo_dir, task_id=task_id)
        artifacts.write_project_snapshot(snapshot)

        # 2. LLM generates deployment plan
        raw_result = self.planner.plan(snapshot)
        artifacts.write_raw_plan({"raw_text": raw_result.text[:10000]})

        # 3. Parse the plan
        try:
            parsed_plan = self.parser.parse(raw_result.text)
        except ValueError as exc:
            parsed_plan = DeploymentPlan(status="invalid", summary=str(exc))
        artifacts.write_parsed_plan(parsed_plan.to_dict())

        # If plan is not ok, we're done
        if parsed_plan.status != "ok":
            policy_result = {"allowed": False, "status": "rejected", "rejected_items": [{"section": "status", "item_index": -1, "reason": "plan status: %s" % parsed_plan.status}]}
            artifacts.write_policy_result(policy_result)
            return self._build_result(
                task_id=task_id,
                plan=parsed_plan,
                policy_result=policy_result,
                stop_reason="plan_not_ok",
                artifacts=artifacts,
            )

        # 4. Policy gate
        policy_result = self.policy_gate.validate(
            parsed_plan.to_dict(),
            snapshot,
            runtime_policy=self.runtime_policy,
            config=self.config,
        )
        artifacts.write_policy_result(policy_result)

        if not policy_result["allowed"]:
            return self._build_result(
                task_id=task_id,
                plan=parsed_plan,
                policy_result=policy_result,
                stop_reason="policy_rejected",
                artifacts=artifacts,
            )

        # 5. Compile the plan
        compiled = self.compiler.compile(
            policy_result.get("normalized_plan", parsed_plan.to_dict()),
        )
        effective_plan = compiled.get("effective_plan", {})
        analysis = compiled.get("analysis", {})
        artifacts.write_effective_plan(effective_plan)

        # 6. Execute stages
        pipeline_results = self._execute_stages(
            task_id=task_id,
            run_dir=run_dir,
            repo_dir=repo_dir,
            analysis=analysis,
            dry_run=dry_run,
        )

        # 7. Check verify result
        verify_result = pipeline_results.get("verify", {})
        verify_status = verify_result.get("status", "")
        replan_count = 0

        # 8. Replan loop on failure
        while verify_status not in ("passed", "pass") and replan_count < self.max_replans:
            # Find the failed stage
            failed_stage = self._find_failed_stage(pipeline_results)
            if not failed_stage:
                break

            # Build failure context
            failure_context = self._build_failure_context(
                failed_stage=failed_stage,
                pipeline_results=pipeline_results,
                plan=parsed_plan,
            )

            # LLM replan
            replan_result = self.planner.replan(
                snapshot, parsed_plan.to_dict(), failure_context,
            )

            # Parse new plan
            try:
                new_plan = self.parser.parse(replan_result.text)
            except ValueError:
                break

            if new_plan.status != "ok":
                break

            # Policy gate new plan
            new_policy = self.policy_gate.validate(
                new_plan.to_dict(), snapshot,
                runtime_policy=self.runtime_policy,
                config=self.config,
            )
            if not new_policy["allowed"]:
                break

            # Determine resume stage
            resume_from = self._determine_resume_stage(parsed_plan.to_dict(), new_plan.to_dict())

            # Compile new plan
            new_compiled = self.compiler.compile(
                new_policy.get("normalized_plan", new_plan.to_dict()),
            )
            new_analysis = new_compiled.get("analysis", {})

            # Write revision
            revision = {
                "revision": replan_count + 1,
                "trigger_stage": failed_stage,
                "failure_summary": failure_context.get("summary", ""),
                "previous_plan_id": parsed_plan.plan_id,
                "new_plan_id": new_plan.plan_id,
                "policy_allowed": True,
                "resume_from": resume_from,
            }
            artifacts.write_plan_revision(revision)

            # Update current plan
            parsed_plan = new_plan
            analysis = new_analysis

            # Re-execute from resume stage
            pipeline_results = self._execute_stages(
                task_id=task_id,
                run_dir=run_dir,
                repo_dir=repo_dir,
                analysis=analysis,
                dry_run=dry_run,
                start_stage=resume_from,
            )
            verify_result = pipeline_results.get("verify", {})
            verify_status = verify_result.get("status", "")
            replan_count += 1

        # Write pipeline results
        artifacts.write_pipeline_results(pipeline_results)

        # Build contribution evidence
        contribution = self._build_contribution_evidence(
            task_id=task_id,
            plan=parsed_plan,
            policy_result=policy_result,
            pipeline_results=pipeline_results,
            replan_count=replan_count,
        )
        artifacts.write_contribution_evidence(contribution)

        stop_reason = "verify_passed" if verify_status in ("passed", "pass") else "verify_failed"
        return self._build_result(
            task_id=task_id,
            plan=parsed_plan,
            policy_result=policy_result,
            pipeline_results=pipeline_results,
            stop_reason=stop_reason,
            artifacts=artifacts,
            replan_count=replan_count,
        )

    def _execute_stages(
        self,
        task_id: str,
        run_dir: Path,
        repo_dir: Path,
        analysis: Dict,
        dry_run: bool = True,
        start_stage: str = "analyze",
    ) -> Dict:
        """Execute pipeline stages using existing modules."""
        results: Dict[str, Dict] = {}
        started = False

        for stage in self.EXECUTION_STAGES:
            if stage == start_stage:
                started = True
            if not started:
                continue

            try:
                if stage == "analyze":
                    result = self._execute_analyze(repo_dir, analysis)
                elif stage == "resource_plan":
                    result = self._execute_resource_plan(repo_dir, analysis)
                elif stage == "env_solve":
                    result = self._execute_env_solve(repo_dir, analysis)
                elif stage == "env_deploy":
                    result = self._execute_env_deploy(repo_dir, analysis, dry_run)
                elif stage == "model_prepare":
                    result = self._execute_model_prepare(run_dir, analysis, dry_run)
                elif stage == "runner":
                    result = self._execute_runner(repo_dir, analysis, results.get("env_deploy", {}), dry_run)
                elif stage == "verify":
                    result = self._execute_verify(run_dir, analysis, results.get("runner", {}))
                else:
                    result = {"status": "skipped", "summary": "unknown stage"}
            except Exception as exc:
                result = {"status": "failed", "summary": str(exc)[:2000]}

            results[stage] = result

            # Stop on failure (unless it's verify which we handle in replan)
            if result.get("status") == "failed" and stage != "verify":
                break

        return results

    def _execute_analyze(self, repo_dir: Path, compiled_analysis: Dict) -> Dict:
        """Run deterministic analyze and merge with compiled plan."""
        from auto_harness.modules.analyzer import ProjectAnalyzer
        analyzer = ProjectAnalyzer()
        stage_result = analyzer.analyze(repo_dir)
        # Merge compiled plan into deterministic analysis
        deterministic = stage_result.data or {}
        merged = dict(deterministic)
        # Compiled plan values take priority for key fields
        for key in ("install_plan", "run_candidates", "verify_hint", "environment_strategy",
                     "selected_candidate", "selection_source", "llm_plan", "llm_candidates",
                     "merged_candidates", "llm_required_reason"):
            if key in compiled_analysis:
                merged[key] = compiled_analysis[key]
        return {"status": "passed", "summary": "analysis completed", "data": merged}

    def _execute_resource_plan(self, repo_dir: Path, analysis: Dict) -> Dict:
        from auto_harness.modules.resource_plan import ResourcePlanner
        planner = ResourcePlanner()
        result = planner.plan(repo_dir, analysis)
        return {"status": result.status, "summary": result.summary, "data": result.data or {}}

    def _execute_env_solve(self, repo_dir: Path, analysis: Dict) -> Dict:
        from auto_harness.modules.env_solve import EnvSolveModule
        solver = EnvSolveModule()
        result = solver.solve(repo_dir, analysis, {})
        return {"status": result.status, "summary": result.summary, "data": result.data or {}}

    def _execute_env_deploy(self, repo_dir: Path, analysis: Dict, dry_run: bool) -> Dict:
        from auto_harness.modules.env_deploy import EnvDeployModule
        deployer = EnvDeployModule()
        execute = not dry_run and self.runtime_policy.get("allow_dependency_install", False)
        result = deployer.deploy(
            repo_dir, analysis,
            execute=execute,
            allowed_commands=getattr(self.config, "allowed_commands", ["python", "python3", "pip"]),
        )
        return {"status": result.status, "summary": result.summary, "data": result.data or {}}

    def _execute_model_prepare(self, run_dir: Path, analysis: Dict, dry_run: bool) -> Dict:
        # Model prepare is often a no-op for simple projects
        return {"status": "passed", "summary": "no model assets to prepare", "data": {}}

    def _execute_runner(self, repo_dir: Path, analysis: Dict, env_result: Dict, dry_run: bool) -> Dict:
        from auto_harness.modules.runner import RunnerModule
        runner = RunnerModule()
        execute = not dry_run and self.runtime_policy.get("allow_service_start", False)
        result = runner.run(
            repo_dir, analysis,
            execute=execute,
            allowed_commands=getattr(self.config, "allowed_commands", ["python", "python3", "pip"]),
            wait_seconds=10,
        )
        return {"status": result.status, "summary": result.summary, "data": result.data or {}}

    def _execute_verify(self, run_dir: Path, analysis: Dict, runner_result: Dict) -> Dict:
        from auto_harness.modules.verify import VerifyModule
        verifier = VerifyModule()
        result = verifier.verify(run_dir, analysis, runner_result.get("data", {}))
        return {"status": result.status, "summary": result.summary, "data": result.data or {}, "evidence": result.evidence or []}

    def _find_failed_stage(self, pipeline_results: Dict) -> Optional[str]:
        """Find the first failed stage."""
        for stage in self.EXECUTION_STAGES:
            result = pipeline_results.get(stage, {})
            if result.get("status") in ("failed", "uncertain"):
                return stage
        return None

    def _build_failure_context(self, failed_stage: str, pipeline_results: Dict, plan: DeploymentPlan) -> Dict:
        """Build failure context for replan."""
        result = pipeline_results.get(failed_stage, {})
        return {
            "failed_stage": failed_stage,
            "stage_status": result.get("status", ""),
            "summary": result.get("summary", ""),
            "error": str(result.get("data", {}).get("error", ""))[:2000],
            "log_tail": str(result.get("data", {}).get("log_tail", ""))[:4000],
            "previous_plan_id": plan.plan_id,
            "previous_command": plan.run.get("candidates", [{}])[0].get("cmd", []) if plan.run else [],
            "evidence_paths": result.get("evidence", []),
        }

    def _determine_resume_stage(self, old_plan: Dict, new_plan: Dict) -> str:
        """Determine which stage to resume from after replan."""
        old_env = old_plan.get("environment", {})
        new_env = new_plan.get("environment", {})
        if old_env.get("install_commands") != new_env.get("install_commands"):
            return "env_deploy"

        old_assets = old_plan.get("model_assets", {})
        new_assets = new_plan.get("model_assets", {})
        if old_assets != new_assets:
            return "model_prepare"

        old_run = old_plan.get("run", {})
        new_run = new_plan.get("run", {})
        if old_run.get("candidates") != new_run.get("candidates"):
            return "runner"

        old_verify = old_plan.get("verify", {})
        new_verify = new_plan.get("verify", {})
        if old_verify != new_verify:
            return "verify"

        # Default: safe resume from runner
        return "runner"

    def _build_contribution_evidence(
        self,
        task_id: str,
        plan: DeploymentPlan,
        policy_result: Dict,
        pipeline_results: Dict,
        replan_count: int = 0,
    ) -> Dict:
        """Build LLM contribution evidence."""
        verify_status = pipeline_results.get("verify", {}).get("status", "")
        return {
            "task_id": task_id,
            "mode": "plan_first",
            "llm_planned": True,
            "plan_id": plan.plan_id,
            "policy_status": policy_result.get("status", ""),
            "compiled_sections": policy_result.get("accepted_sections", []),
            "rejected_sections": [r.get("section", "") for r in policy_result.get("rejected_items", [])],
            "final_verify_status": verify_status,
            "replan_count": replan_count,
            "llm_changed_decision": True,
            "llm_helped": verify_status in ("passed", "pass"),
            "llm_required_status": "unknown_without_baseline",
            "help_type": ["initial_deployment_planning", "runner_candidate_selection", "verify_hint_generation"],
            "safety": {
                "raw_plan_executed_directly": False,
                "policy_gated": True,
                "command_allowlist_enforced": True,
                "verify_trace_required": True,
            },
        }

    def _build_result(
        self,
        task_id: str,
        plan: DeploymentPlan,
        policy_result: Dict,
        stop_reason: str,
        artifacts: PlanArtifactWriter = None,
        pipeline_results: Dict = None,
        replan_count: int = 0,
    ) -> Dict:
        """Build the final result dict."""
        return {
            "task_id": task_id,
            "plan_id": plan.plan_id,
            "plan_status": plan.status,
            "policy_status": policy_result.get("status", ""),
            "stop_reason": stop_reason,
            "replan_count": replan_count,
            "pipeline_results": pipeline_results or {},
        }
