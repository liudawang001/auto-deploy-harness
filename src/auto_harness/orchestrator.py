import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from auto_harness.agent import AgentDecisionEngine, AgentDiagnoser, AgentLoopController, AgentMetricsCollector, AgentObservation, AgentTraceWriter, AgentVerifyPlanner
from auto_harness.agent_runtime import AgentContributionAnalyzer, AgentGoal, AgentRuntime
from auto_harness.agent_runtime.loop import DeploymentAgentLoop
from auto_harness.config import HarnessConfig
from auto_harness.agents.claude_code import ClaudeCodeExecutor
from auto_harness.assets import HuggingFaceDownloader, ModelCache, ModelScopeDownloader
from auto_harness.models.base import read_json, to_plain, write_json
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.memory import MemoryStore
from auto_harness.memory.outcomes import SkillOutcomeRecorder
from auto_harness.memory.success import VerifiedMemoryRecorder
from auto_harness.modules import (
    EnvDeployModule,
    EnvSolveModule,
    ModelPrepareModule,
    ProjectAnalyzer,
    ReportGenerator,
    ResourcePlanner,
    RunnerModule,
    VerifyModule,
)
from auto_harness.providers import MockLLMProvider, XunfeiSparkProvider
from auto_harness.skills import SkillRegistry
from auto_harness.repair import RepairApplier, RepairLoopController, RepairOverlay, RepairPlanner, RepairPolicy
from auto_harness.state import StateStore
from auto_harness.utils.files import safe_name, short_hash
from auto_harness.utils.time import compact_timestamp, utc_now_iso


class TaskRunner:
    PIPELINE_STAGES = ("analyze", "resource_plan", "env_solve", "env_deploy", "model_prepare", "runner", "verify", "report")
    RERUN_STAGES = ("analyze", "resource_plan", "env_solve", "env_deploy", "model_prepare", "runner", "verify")

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.store = StateStore(config.runs_path)
        self.skills = SkillRegistry(config.skills_path, max_chars=config.max_skill_chars)
        self.memory = MemoryStore(config.memory_path)
        self.model_cache = ModelCache(config.model_cache_path)
        self.model_prepare = ModelPrepareModule(
            self.model_cache,
            huggingface_downloader=HuggingFaceDownloader(
                max_workers=config.model_download_max_workers,
                retry_count=config.model_download_retry_count,
                retry_backoff_seconds=config.model_download_retry_backoff_seconds,
            ),
            modelscope_downloader=ModelScopeDownloader(
                max_workers=config.model_download_max_workers,
                retry_count=config.model_download_retry_count,
                retry_backoff_seconds=config.model_download_retry_backoff_seconds,
            ),
        )
        self.repair_planner = RepairPlanner()
        self.repair_policy = RepairPolicy()
        self.repair_applier = RepairApplier()
        self.repair_overlay = RepairOverlay()
        self.repair_loop = RepairLoopController(config.agent_max_loop_iterations or config.max_repair_attempts)

    def create_spec(
        self,
        repo_url: str,
        name: str,
        dry_run: bool = True,
        allow_install: bool = False,
        allow_start: bool = False,
    ) -> TaskSpec:
        base_name = safe_name(name or repo_url.rsplit("/", 1)[-1].replace(".git", ""))
        task_id = "%s_%s_%s" % (base_name, compact_timestamp(), short_hash(repo_url, 6))
        workspace_root = str(self.config.runs_path / task_id / "workspace")
        return TaskSpec(
            task_id=task_id,
            project=ProjectSpec(name=base_name, repo_url=repo_url),
            runtime=RuntimePolicy(
                workspace_root=workspace_root,
                allow_dependency_install=not dry_run and allow_install,
                allow_service_start=not dry_run and allow_start,
                allow_source_edit=self.config.allow_source_edit,
            ),
            created_at=utc_now_iso(),
        )

    def deploy(
        self,
        repo_url: str,
        name: str,
        dry_run: bool = True,
        skip_clone: bool = False,
        allow_install: bool = False,
        allow_start: bool = False,
    ) -> str:
        spec = self.create_spec(
            repo_url,
            name,
            dry_run=dry_run,
            allow_install=allow_install,
            allow_start=allow_start,
        )
        run_dir = self.store.create_task(spec)
        repo_dir = run_dir / "workspace" / "repo"
        if not skip_clone and repo_url.startswith("http"):
            clone_result = subprocess.run(["git", "clone", repo_url, str(repo_dir)], text=True, capture_output=True)
            self.store.events(spec.task_id).append("clone", "git_clone", {"exit_code": clone_result.returncode, "stderr": clone_result.stderr[-2000:]})
            if clone_result.returncode != 0:
                self.store.update_stage(spec.task_id, "analyze", "failed", error=clone_result.stderr[-2000:])
                return spec.task_id
        elif not skip_clone and Path(repo_url).exists():
            source = Path(repo_url).resolve()
            for child in source.iterdir():
                target = repo_dir / child.name
                if child.is_dir():
                    shutil.copytree(str(child), str(target), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(child), str(target))
            self.store.events(spec.task_id).append("clone", "copy_local_repo", {"source": str(source)})
        return self.run_existing(spec.task_id, dry_run=dry_run)

    def run_existing(self, task_id: str, dry_run: bool = True, start_stage: str = "analyze") -> str:
        # Check if AgentLoop should be the primary controller
        if (self.config.agent_enable_runtime_loop
                and self.config.agent_mode == "gated_actor"
                and self.config.agent_runtime_loop_position == "primary"):
            # AgentLoop is the primary deployment controller
            self._run_agent_runtime_loop(task_id, dry_run=dry_run)
            return task_id

        # Legacy path: deterministic pipeline first, then optional AgentLoop
        current_start_stage = start_stage
        max_iterations = int(self.config.agent_max_loop_iterations or 0)
        for iteration in range(max_iterations + 1):
            self._run_existing_once(task_id, dry_run=dry_run, start_stage=current_start_stage)
            decision = self._next_agent_resume_decision(task_id, iteration)
            if not decision.get("should_resume"):
                break
            self.store.events(task_id).append("task", "agent_auto_resume", decision)
            current_start_stage = decision["start_stage"]

        # If agent runtime loop is enabled in post_pipeline mode
        if (self.config.agent_enable_runtime_loop
                and self.config.agent_mode == "gated_actor"
                and self.config.agent_runtime_loop_position == "post_pipeline"):
            self._run_agent_runtime_loop(task_id, dry_run=dry_run)

        return task_id

    def _run_agent_runtime_loop(self, task_id: str, dry_run: bool = True) -> None:
        """Run the unified DeploymentAgentLoop as primary controller."""
        run_dir = self.store.run_dir(task_id)
        repo_dir = run_dir / "workspace" / "repo"

        # Load task for runtime policy
        task = self.store.load_task(task_id)
        runtime_policy = {
            "allow_dependency_install": task.runtime.allow_dependency_install,
            "allow_service_start": task.runtime.allow_service_start,
            "allow_source_edit": task.runtime.allow_source_edit,
        }

        # Load pipeline results
        pipeline_path = run_dir / "reports" / "pipeline_results.json"
        initial_results = {}
        if pipeline_path.exists():
            try:
                initial_results = read_json(pipeline_path)
            except (OSError, ValueError):
                pass

        # Create provider
        provider = self._create_agent_provider()

        # Create AgentStageExecutor with real module dependencies
        from auto_harness.agent_runtime.stage_executor import AgentStageExecutor
        stage_executor = AgentStageExecutor(
            config=self.config,
            store=self.store,
            model_prepare=self.model_prepare,
            repair_components={
                "planner": self.repair_planner,
                "policy": self.repair_policy,
                "applier": self.repair_applier,
                "loop": self.repair_loop,
                "overlay": self.repair_overlay,
            },
            provider_factory=lambda: self._create_agent_provider(),
            runtime_policy=runtime_policy,
        )

        # Create and run the agent loop
        loop = DeploymentAgentLoop(
            provider=provider,
            config=self.config,
            stage_executor=stage_executor,
            max_iterations=self.config.agent_runtime_loop_max_iterations,
            stop_on_verify_pass=self.config.agent_runtime_loop_stop_on_verify_pass,
            runtime_policy=runtime_policy,
        )

        result = loop.run(
            task_id=task_id,
            run_dir=run_dir,
            repo_dir=repo_dir,
            initial_results=initial_results,
            dry_run=dry_run,
        )

        # Write agent loop result
        write_json(run_dir / "reports" / "agent_loop_result.json", result)
        self.store.events(task_id).append("task", "agent_runtime_loop_completed", {
            "stop_reason": result.get("stop_reason", ""),
            "iteration_count": result.get("iteration_count", 0),
        })

    def _run_existing_once(self, task_id: str, dry_run: bool = True, start_stage: str = "analyze") -> str:
        task = self.store.load_task(task_id)
        run_dir = self.store.run_dir(task_id)
        repo_dir = run_dir / "workspace" / "repo"
        requested_start_stage = start_stage
        start_stage = self._normalize_start_stage(task_id, start_stage)
        start_index = self.PIPELINE_STAGES.index(start_stage)
        results: Dict[str, Dict] = {} if start_stage == "analyze" else self._load_previous_results(run_dir)
        execution_audit = self._execution_audit(requested_start_stage, start_stage, dry_run)
        if execution_audit:
            write_json(run_dir / "reports" / "execution_audit.json", execution_audit)
            self.store.events(task_id).append("task", "resume_execution_plan", execution_audit)

        def should_run(stage: str) -> bool:
            return self.PIPELINE_STAGES.index(stage) >= start_index

        if should_run("analyze"):
            analyze_context = self._stage_context("analyze", {})
            trace_writer = self._agent_trace_writer(run_dir)
            analyzer = ProjectAnalyzer(
                agent_executor=ClaudeCodeExecutor() if self.config.use_agent_analyzer else None,
                use_agent=self.config.use_agent_analyzer,
                agent_timeout_seconds=self.config.agent_timeout_seconds,
                stage_context=analyze_context,
                agent_engine=self._agent_decision_engine(trace_writer) if self._agent_planner_enabled("analyze") else None,
                agent_mode=self.config.agent_mode,
                runtime_policy=task.runtime,
                task_id=task_id,
                agent_max_file_chars=self.config.agent_max_file_chars,
            )
            analyze_result = analyzer.analyze(repo_dir)
            results["analyze"] = to_plain(analyze_result)
            self._save_stage(task_id, "analyze", analyze_result)
            self._remember(task_id, "analyze", analyze_result, analyze_result.data, results)
            analyze_data = analyze_result.data
        else:
            analyze_data = results["analyze"]["data"]
        repair_overlay = self.repair_overlay.load(run_dir)
        effective_analysis = self.repair_overlay.merge_analysis(analyze_data, repair_overlay)

        # Plan Decision Gate: generate initial deployment strategy after analyze
        if self._plan_gate_enabled() and should_run("analyze"):
            self._apply_plan_gate(task_id, effective_analysis, results, run_dir, revision=False)

        # Load plan hints for subsequent stages
        plan_hints = self._load_plan_hints(run_dir)

        if should_run("resource_plan"):
            resource_context = self._stage_context("resource_plan", effective_analysis)
            resource_result = ResourcePlanner().plan(repo_dir, effective_analysis)
            self._attach_context(resource_result, resource_context)
            results["resource_plan"] = to_plain(resource_result)
            self._save_stage(task_id, "resource_plan", resource_result)
            self._remember(task_id, "resource_plan", resource_result, effective_analysis, results)
            resource_data = resource_result.data
        else:
            resource_data = results["resource_plan"]["data"]

        if should_run("env_solve"):
            env_solve_context = self._stage_context("env_solve", effective_analysis)
            env_solve_hints = plan_hints.get("stage_hints", {}).get("env_solve", {})
            env_solve_result = EnvSolveModule(
                env_backend=self.config.env_backend,
                conda_envs_dir=self.config.conda_envs_dir,
                conda_prefer_mamba=self.config.conda_prefer_mamba,
                conda_allowed_channels=self.config.conda_allowed_channels,
                conda_python_default=self.config.conda_python_default,
                torch_cuda_preference=self.config.torch_cuda_preference,
            ).solve(repo_dir, effective_analysis, resource_data, stage_hints=env_solve_hints)
            self._attach_context(env_solve_result, env_solve_context)
            results["env_solve"] = to_plain(env_solve_result)
            self._save_stage(task_id, "env_solve", env_solve_result)
            self._remember(task_id, "env_solve", env_solve_result, effective_analysis, results)
            deploy_analysis = env_solve_result.data.get("analysis", effective_analysis)
        else:
            deploy_analysis = results["env_solve"]["data"].get("analysis", effective_analysis)

        if should_run("env_deploy"):
            env_context = self._stage_context("env_deploy", deploy_analysis)
            env_result = EnvDeployModule().deploy(
                repo_dir,
                deploy_analysis,
                execute=not dry_run and task.runtime.allow_dependency_install,
                allowed_commands=self.config.allowed_commands,
                execution_backend=self.config.execution_backend,
                docker_image=self.config.docker_image,
                docker_network=self.config.docker_network,
                docker_gpus=self.config.docker_gpus,
                docker_model_cache_dir=self._docker_model_cache_dir(),
            )
            self._attach_context(env_result, env_context)
            self._attach_repair_overlay(env_result, repair_overlay)
            # Env Decision Gate: LLM diagnoses dependency conflicts on failure
            if env_result.status in ("failed", "uncertain") and self._env_gate_enabled():
                env_gate_analysis = self._apply_env_gate(task_id, env_result, deploy_analysis, run_dir)
                if env_gate_analysis is not None:
                    deploy_analysis = env_gate_analysis
            results["env_deploy"] = to_plain(env_result)
            self._save_stage(task_id, "env_deploy", env_result)
            self._remember(task_id, "env_deploy", env_result, deploy_analysis, results)

        if should_run("model_prepare"):
            # Model Decision Gate: LLM resolves model asset ambiguity
            if self._model_gate_enabled():
                resource_data = self._apply_model_gate(task_id, resource_data, effective_analysis, repo_dir, run_dir)
            model_context = self._stage_context("model_prepare", resource_data)
            model_result = self.model_prepare.prepare(
                run_dir,
                resource_data,
                execute=not dry_run,
                progress_callback=lambda progress: self.store.update_stage(task_id, "model_prepare", "waiting_download", progress=progress),
                repo_dir=repo_dir,
                allowed_commands=self.config.allowed_commands,
                timeout_seconds=self.config.default_timeout_seconds,
            )
            self._attach_context(model_result, model_context)
            results["model_prepare"] = to_plain(model_result)
            self._save_stage(task_id, "model_prepare", model_result)
            self._remember(task_id, "model_prepare", model_result, effective_analysis, results)

        if should_run("runner"):
            runner_context = self._stage_context("runner", deploy_analysis)
            # Runner Decision Gate: LLM selects best candidate before deterministic execution
            if self._runner_gate_enabled():
                deploy_analysis = self._apply_runner_gate(task_id, deploy_analysis, repo_dir, run_dir)
            runner_hints = plan_hints.get("stage_hints", {}).get("runner", {})
            runner_result = RunnerModule().run(
                repo_dir,
                deploy_analysis,
                execute=not dry_run and task.runtime.allow_service_start,
                allowed_commands=self.config.allowed_commands,
                execution_backend=self.config.execution_backend,
                docker_image=self.config.docker_image,
                docker_network=self.config.docker_network,
                docker_gpus=self.config.docker_gpus,
                docker_model_cache_dir=self._docker_model_cache_dir(),
                stage_hints=runner_hints,
            )
            self._attach_context(runner_result, runner_context)
            results["runner"] = to_plain(runner_result)
            self._save_stage(task_id, "runner", runner_result)
            self._remember(task_id, "runner", runner_result, deploy_analysis, results)
            runner_data = runner_result.data
        else:
            runner_data = results["runner"]["data"]

        verify_context = self._stage_context("verify", deploy_analysis)
        verify_trace_writer = self._agent_trace_writer(run_dir)
        agent_verify_config = self._agent_verify_config()
        verify_hints = plan_hints.get("stage_hints", {}).get("verify", {})
        verify_result = VerifyModule(
            stage_context=verify_context,
            progress_callback=lambda progress: self.store.update_stage(task_id, "verify", "running_verify", progress=progress),
            verify_planner=self._agent_verify_planner(verify_trace_writer) if self._agent_verify_planner_enabled() else None,
            agent_verify_config=agent_verify_config,
        ).verify(run_dir, deploy_analysis, runner_data, stage_hints=verify_hints)
        self._attach_repair_overlay(verify_result, repair_overlay)
        results["verify"] = to_plain(verify_result)
        self._save_stage(task_id, "verify", verify_result)
        self._remember(task_id, "verify", verify_result, deploy_analysis, results)

        metrics_payload = AgentMetricsCollector().collect(run_dir, results, output_path=run_dir / "reports" / "agent_metrics.json")
        verified_memory = VerifiedMemoryRecorder(self.config.memory_path).record_if_verified(
            run_dir,
            results,
            metrics_payload.get("agent_metrics", {}),
        )
        if verified_memory:
            self.store.events(task_id).append("memory", "verified_success_recorded", {"memory_id": verified_memory.get("id")})
        contribution = AgentContributionAnalyzer().analyze(run_dir, results, output_path=run_dir / "reports" / "agent_contribution.json")
        AgentRuntime().run(
            AgentGoal(
                task_id=task_id,
                objective="Deploy and verify the target AI demo with evidence-based success criteria.",
            ),
            run_dir,
            results,
            contribution=contribution,
        )
        task_data = read_json(run_dir / "task.json")
        report_result = ReportGenerator().generate(run_dir, task_data, results, execution_audit=execution_audit)
        results["report"] = to_plain(report_result)
        self._save_stage(task_id, "report", report_result)
        state = self.store.load_state(task_id)
        state.report_path = report_result.data.get("report_path")
        self.store.save_state(state)

        write_json(run_dir / "reports" / "pipeline_results.json", results)
        return task_id

    def _next_agent_resume_decision(self, task_id: str, iteration: int) -> Dict:
        if not self.config.agent_auto_resume_after_repair:
            return {"should_resume": False, "reason": "config_disabled", "loop_iteration": iteration}
        if iteration >= int(self.config.agent_max_loop_iterations or 0):
            return {"should_resume": False, "reason": "max_iterations", "loop_iteration": iteration}
        run_dir = self.store.run_dir(task_id)
        pipeline = self._load_previous_results(run_dir)
        verify = pipeline.get("verify") if isinstance(pipeline.get("verify"), dict) else {}
        if self.config.agent_stop_on_verify_pass and verify.get("status") in ("pass", "passed"):
            return {"should_resume": False, "reason": "verify_passed", "loop_iteration": iteration}
        apply_result = self._read_optional(run_dir / "repairs" / "repair_apply_result.json")
        if not isinstance(apply_result, dict) or apply_result.get("status") != "applied":
            return {"should_resume": False, "reason": "repair_not_applied", "loop_iteration": iteration}
        policy = apply_result.get("policy") if isinstance(apply_result.get("policy"), dict) else {}
        if not policy.get("allowed"):
            return {"should_resume": False, "reason": "policy_rejected", "loop_iteration": iteration}
        if not self._repair_apply_effective(apply_result):
            return {"should_resume": False, "reason": "no_effective_repair_action", "loop_iteration": iteration}
        requested = self._resume_request_from_pipeline(pipeline)
        if not requested.get("should_auto_resume"):
            return {"should_resume": False, "reason": "agent_loop_not_requesting_resume", "loop_iteration": iteration}
        start_stage = requested.get("next_rerun_from") or self._repair_resume_stage(run_dir)
        allowed_stages = set(self.config.agent_auto_resume_stages or [])
        if start_stage not in self.RERUN_STAGES or start_stage not in allowed_stages:
            return {
                "should_resume": False,
                "reason": "start_stage_not_allowed",
                "requested_start_stage": start_stage,
                "loop_iteration": iteration,
            }
        return {
            "should_resume": True,
            "start_stage": start_stage,
            "reason": "agent_loop_requested_resume",
            "source_stage": requested.get("source_stage"),
            "loop_iteration": iteration + 1,
        }

    def _repair_apply_effective(self, apply_result: Dict) -> bool:
        """Check if repair had a truly effective action.

        metadata_only actions (update_verify_hint, rerun_from_stage) NEVER count.
        Only executed actions with exit_code=0 or strong_verify_pass qualify.
        """
        for item in apply_result.get("action_results", []):
            if item.get("executed") is True and int(item.get("exit_code") or 0) == 0:
                return True
            if item.get("tool_result", {}).get("strong_verify_pass") is True:
                return True
        return False

    def _resume_request_from_pipeline(self, pipeline: Dict) -> Dict:
        for stage, result in pipeline.items():
            if not isinstance(result, dict):
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            loop = data.get("agent_loop") if isinstance(data.get("agent_loop"), dict) else {}
            if loop.get("should_auto_resume"):
                return {
                    "should_auto_resume": True,
                    "next_rerun_from": loop.get("next_rerun_from"),
                    "source_stage": stage,
                }
        return {"should_auto_resume": False}

    def resume(self, task_id: str, dry_run: bool = True) -> str:
        run_dir = self.store.run_dir(task_id)
        start_stage = self._repair_resume_stage(run_dir)
        self.store.events(task_id).append("task", "resume_requested", {"dry_run": dry_run, "start_stage": start_stage})
        return self.run_existing(task_id, dry_run=dry_run, start_stage=start_stage)

    def approve_repair(self, task_id: str, note: str = "") -> Dict:
        run_dir = self.store.run_dir(task_id)
        approval = self.repair_loop.approve_latest(run_dir, note=note)
        self.store.events(task_id).append("repair", "operator_approved", approval)
        return approval

    def _save_stage(self, task_id: str, stage: str, result) -> None:
        path = self.store.save_result(task_id, stage, result)
        stage_status = "passed" if result.status in ("passed", "pass") else result.status
        progress = result.data.get("progress", {}) if isinstance(result.data, dict) else {}
        self.store.update_stage(task_id, stage, stage_status, result_path=str(path), error=result.error, progress=progress)
        self._record_skill_outcome(task_id, stage, result)
        # Plan revision at failed/uncertain stage boundaries
        if result.status in ("failed", "uncertain") and self._plan_gate_enabled():
            self._maybe_plan_revision(task_id, stage, result)

    def _record_skill_outcome(self, task_id: str, stage: str, result) -> None:
        """Record skill outcome after each stage. Failure must not affect the pipeline."""
        try:
            data = result.data if isinstance(result.data, dict) else {}
            control_context = data.get("control_context") if isinstance(data.get("control_context"), dict) else {}
            selected_skills = control_context.get("selected_skills", [])

            # Build agent metadata from verify result if available
            agent_metadata = {}
            agent_verify = data.get("agent_verify") if isinstance(data.get("agent_verify"), dict) else None
            if agent_verify:
                agent_metadata["llm_helped"] = agent_verify.get("llm_helped", False)
                agent_metadata["policy_rejected"] = agent_verify.get("rejected_tool_count", 0) > 0
            trace_id = str(data.get("trace_id") or "")
            if trace_id and result.status in ("pass", "passed"):
                agent_metadata["trace_verified"] = True

            recorder = SkillOutcomeRecorder(self.config.memory_path)
            recorder.record_run(
                run_id=task_id,
                stage=stage,
                selected_skills=selected_skills,
                result={"status": result.status},
                agent_metadata=agent_metadata if agent_metadata else None,
            )
        except Exception:
            # Outcome recording must never affect the deployment pipeline.
            # Swallow all errors silently.
            pass

    def _stage_context(self, stage: str, analysis: Dict) -> Dict:
        skills = [skill.to_context() for skill in self.skills.select_for_stage(stage, analysis, limit=3)]
        memories = self.memory.query(stage, analysis, limit=self.config.max_memory_items)
        return {
            "selected_skills": skills,
            "memory_hits": memories,
        }

    def _attach_context(self, result, context: Dict) -> None:
        if context:
            result.data.setdefault("control_context", context)

    def _attach_repair_overlay(self, result, overlay: Dict) -> None:
        if overlay.get("active"):
            result.data.setdefault("repair_overlay", {
                "active": True,
                "install_command_count": len(overlay.get("install_commands") or []),
                "verify_hint_count": len(overlay.get("verify_hints") or []),
                "source_dir": overlay.get("source_dir"),
            })

    def _docker_model_cache_dir(self) -> str:
        return self.config.docker_model_cache_dir or str(self.config.model_cache_path)

    def _normalize_start_stage(self, task_id: str, start_stage: str) -> str:
        requested = start_stage if start_stage in self.RERUN_STAGES else "analyze"
        if requested == "analyze":
            return "analyze"
        run_dir = self.store.run_dir(task_id)
        previous = self._load_previous_results(run_dir)
        missing = [
            stage for stage in self.PIPELINE_STAGES[:self.PIPELINE_STAGES.index(requested)]
            if not isinstance(previous.get(stage), dict) or "data" not in previous[stage]
        ]
        if missing:
            self.store.events(task_id).append(
                "task",
                "resume_stage_fallback",
                {
                    "requested_start_stage": start_stage,
                    "effective_start_stage": "analyze",
                    "missing_previous_results": missing,
                },
            )
            return "analyze"
        return requested

    def _repair_resume_stage(self, run_dir: Path) -> str:
        apply_result = self._read_optional(run_dir / "repairs" / "repair_apply_result.json")
        if isinstance(apply_result, dict):
            if apply_result.get("status") != "applied":
                return "analyze"
            policy = apply_result.get("policy") if isinstance(apply_result.get("policy"), dict) else {}
            loop = policy.get("loop") if isinstance(policy.get("loop"), dict) else {}
            stage = loop.get("rerun_from_effective")
            if stage in self.RERUN_STAGES:
                return stage
            return "analyze"
        plan = self._read_optional(run_dir / "repairs" / "repair_plan.json")
        if isinstance(plan, dict) and plan.get("rerun_from_effective") in self.RERUN_STAGES:
            return plan["rerun_from_effective"]
        return "analyze"

    def _load_previous_results(self, run_dir: Path) -> Dict[str, Dict]:
        pipeline = self._read_optional(run_dir / "reports" / "pipeline_results.json")
        if isinstance(pipeline, dict):
            return {stage: result for stage, result in pipeline.items() if isinstance(result, dict)}
        results = {}
        for stage in self.PIPELINE_STAGES:
            result = self._read_optional(run_dir / "reports" / ("%s_result.json" % stage))
            if isinstance(result, dict):
                results[stage] = result
        return results

    def _execution_audit(self, requested_start_stage: str, effective_start_stage: str, dry_run: bool) -> Dict:
        if requested_start_stage == "analyze" and effective_start_stage == "analyze":
            return {}
        start_index = self.PIPELINE_STAGES.index(effective_start_stage)
        return {
            "requested_start_stage": requested_start_stage,
            "effective_start_stage": effective_start_stage,
            "dry_run": dry_run,
            "reused_stages": list(self.PIPELINE_STAGES[:start_index]),
            "rerun_stages": list(self.PIPELINE_STAGES[start_index:]),
            "fallback_applied": requested_start_stage != effective_start_stage,
            "generated_at": utc_now_iso(),
        }

    def _agent_provider(self):
        if self.config.agent_provider == "xunfei":
            return XunfeiSparkProvider()
        return MockLLMProvider()

    def _create_agent_provider(self):
        """Create LLM provider for the agent runtime loop."""
        return self._agent_provider()

    def _agent_trace_writer(self, run_dir: Path) -> AgentTraceWriter:
        return AgentTraceWriter(run_dir / "logs" / "agent_calls")

    def _agent_decision_engine(self, trace_writer: AgentTraceWriter) -> AgentDecisionEngine:
        return AgentDecisionEngine(self._agent_provider(), config=self.config, trace_writer=trace_writer)

    def _agent_loop_controller(self) -> AgentLoopController:
        return AgentLoopController(
            self.config,
            self.store,
            self.memory,
            self.repair_planner,
            self.repair_policy,
            self.repair_applier,
            self.repair_loop,
            self._agent_provider,
        )

    def _agent_planner_enabled(self, stage: str) -> bool:
        return (
            self.config.agent_mode in ("planner", "gated_actor")
            and stage == "analyze"
            and self.config.agent_enable_analyze_planner
        )

    def _agent_verify_planner_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_verify_planner

    def _agent_verify_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_verify

    def _agent_verify_config(self) -> Dict:
        """Build config dict for the agent verify integration in VerifyModule."""
        if not self._agent_verify_enabled():
            return {}
        return {
            "agent_mode": self.config.agent_mode,
            "agent_enable_verify": self.config.agent_enable_verify,
            "agent_verify_max_steps": self.config.agent_verify_max_steps,
            "agent_allowed_hosts": self.config.agent_allowed_hosts,
            "provider": self._agent_provider(),
        }

    def _agent_repair_execute_enabled(self, runtime: RuntimePolicy) -> bool:
        return (
            self.config.agent_mode == "gated_actor"
            and self.config.agent_enable_repair_actions
            and runtime.allow_dependency_install
        )

    # ------------------------------------------------------------------
    # Decision Gate helpers
    # ------------------------------------------------------------------

    def _runner_gate_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_runner_gate

    def _env_gate_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_env_gate

    def _model_gate_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_model_gate

    def _repair_gate_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_repair_gate

    def _plan_gate_enabled(self) -> bool:
        return self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_plan_gate

    def _apply_runner_gate(self, task_id: str, analysis: Dict, repo_dir: Path, run_dir: Path) -> Dict:
        """Apply runner decision gate: LLM selects/reorders run candidates."""
        from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
        from auto_harness.agent_runtime.stage_schemas import RUNNER_TOOLS

        candidates = analysis.get("run_candidates", [])
        if not candidates:
            return analysis

        # Build observation for runner gate
        observation = self._build_runner_observation(analysis, repo_dir)

        gate = AgentDecisionGate(provider=self._agent_provider())
        gate_result = gate.decide(
            stage="runner",
            observation=observation,
            allowed_tools=list(RUNNER_TOOLS),
            mode=self.config.agent_mode,
            run_dir=run_dir,
            max_steps=self.config.agent_decision_gate_max_steps,
        )

        self.store.events(task_id).append("runner", "decision_gate", {
            "decision_status": gate_result.decision_status,
            "policy_allowed": gate_result.policy.get("allowed", False),
            "executed": gate_result.execution.get("executed", False),
            "state_changed": gate_result.state_delta.get("changed", False),
        })

        # Apply state delta if candidates were reordered
        if gate_result.state_delta.get("changed") and gate_result.state_delta.get("reordered_candidates"):
            reordered = gate_result.state_delta["reordered_candidates"]
            updated_analysis = dict(analysis)
            updated_analysis["run_candidates"] = reordered
            # Mark selection metadata
            if reordered:
                reordered[0]["selected_by"] = "llm_runner_gate"
                reordered[0]["selection_reason"] = gate_result.hypothesis
            return updated_analysis

        return analysis

    def _build_runner_observation(self, analysis: Dict, repo_dir: Path) -> Dict:
        """Build observation for runner decision gate."""
        from auto_harness.agent.safety import AgentInputSanitizer
        sanitizer = AgentInputSanitizer()

        candidates = analysis.get("run_candidates", [])
        # Normalize candidates with IDs
        normalized = []
        for i, c in enumerate(candidates):
            cand = dict(c)
            if "id" not in cand:
                cand["id"] = "cand_%d" % i
            normalized.append(cand)

        # Read selected files for context
        selected_files = {}
        for name in ("README.md", "readme.md", "app.py", "main.py", "gradio_app.py", "server.py"):
            path = repo_dir / name
            if path.is_file():
                try:
                    selected_files[name] = path.read_text(encoding="utf-8", errors="ignore")[:3000]
                except OSError:
                    pass
        selected_files = sanitizer.sanitize_selected_files(selected_files)

        frameworks = analysis.get("frameworks", [])
        allowed_roots = list(self.config.allowed_commands or ["python", "python3", "streamlit", "gradio", "uvicorn"])

        return {
            "stage": "runner",
            "frameworks": frameworks,
            "run_candidates": normalized,
            "selected_files": selected_files,
            "constraints": [
                "Only select from existing candidates or add safe candidates.",
                "Command roots must be in allowed list: %s" % ", ".join(allowed_roots),
                "No shell metacharacters in commands.",
                "Do not start services yourself; the runner module handles execution.",
            ],
        }

    def _apply_model_gate(self, task_id: str, resource_data: Dict, analysis: Dict, repo_dir: Path, run_dir: Path) -> Dict:
        """Apply model decision gate: LLM resolves model asset ambiguity."""
        from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
        from auto_harness.agent_runtime.stage_schemas import MODEL_TOOLS

        model_assets = resource_data.get("model_assets", [])
        # Check if model gate is needed
        needs_gate = (
            not model_assets  # no assets detected but README/code may imply models
            or any(a.get("status") in ("uncertain", "failed") for a in model_assets)
        )
        if not needs_gate:
            return resource_data

        # Build observation
        readme_excerpt = ""
        for name in ("README.md", "readme.md"):
            path = repo_dir / name
            if path.is_file():
                try:
                    readme_excerpt = path.read_text(encoding="utf-8", errors="ignore")[:3000]
                except OSError:
                    pass

        observation = {
            "stage": "model_prepare",
            "model_mentions": self._extract_model_mentions(analysis, readme_excerpt),
            "detected_assets": model_assets,
            "cache_candidates": [],
            "git_lfs": resource_data.get("git_lfs", {}),
            "constraints": [
                "Model source must be huggingface, model_scope, or local_cache.",
                "repo_id format must be namespace/name (no whitespace, no URL).",
                "target_path must be relative, no path traversal.",
                "No token values in tool input.",
                "Do not download directly; strategy overlay only.",
            ],
        }

        gate = AgentDecisionGate(provider=self._agent_provider())
        gate_result = gate.decide(
            stage="model_prepare",
            observation=observation,
            allowed_tools=list(MODEL_TOOLS),
            mode=self.config.agent_mode,
            run_dir=run_dir,
            max_steps=self.config.agent_decision_gate_max_steps,
        )

        self.store.events(task_id).append("model_prepare", "decision_gate", {
            "decision_status": gate_result.decision_status,
            "policy_allowed": gate_result.policy.get("allowed", False),
            "state_changed": gate_result.state_delta.get("changed", False),
        })

        # Apply model strategy overlay if produced
        if gate_result.state_delta.get("changed") and gate_result.state_delta.get("strategy_path"):
            strategy_path = Path(gate_result.state_delta["strategy_path"])
            if strategy_path.exists():
                import json
                strategy = json.loads(strategy_path.read_text())
                updated_resource = dict(resource_data)
                if strategy.get("repo_id"):
                    # Add model asset from strategy
                    new_asset = {
                        "source": strategy.get("source", "huggingface"),
                        "repo_id": strategy["repo_id"],
                        "target_path": strategy.get("target_path", ""),
                        "strategy": strategy.get("strategy", "snapshot_download"),
                        "selected_by": "llm_model_gate",
                    }
                    existing = list(updated_resource.get("model_assets", []))
                    existing.append(new_asset)
                    updated_resource["model_assets"] = existing
                return updated_resource

        return resource_data

    def _extract_model_mentions(self, analysis: Dict, readme: str) -> list:
        """Extract model mentions from analysis and README."""
        mentions = []
        # From analysis
        for hint in analysis.get("model_hints", []):
            if isinstance(hint, str):
                mentions.append(hint)
        # From README: look for HF-style model references
        import re
        hf_pattern = re.compile(r'[\w.-]+/[\w.-]+')
        for match in hf_pattern.finditer(readme):
            candidate = match.group()
            # Filter out obvious non-model references
            if "/" in candidate and not candidate.startswith(("http", "git@", "ssh://")):
                if any(kw in readme.lower() for kw in ("model", "huggingface", "hf", "transformers")):
                    mentions.append(candidate)
        return list(set(mentions))[:10]

    def _apply_plan_gate(self, task_id: str, analysis: Dict, results: Dict, run_dir: Path, revision: bool = False) -> None:
        """Apply plan decision gate: LLM generates deployment strategy and stage hints."""
        from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
        from auto_harness.agent_runtime.stage_planners import PlanPlanner
        from auto_harness.models.base import write_json

        # Build observation for plan gate
        frameworks = analysis.get("frameworks", [])
        uncertainties = []
        if analysis.get("run_candidates") and len(analysis["run_candidates"]) > 1:
            uncertainties.append("Multiple run candidates; best entrypoint unclear")
        env_solution = analysis.get("env_solution", {})
        if env_solution.get("risk_reasons"):
            uncertainties.extend(env_solution["risk_reasons"])

        observation = {
            "stage": "plan",
            "analysis_summary": {
                "frameworks": frameworks,
                "has_model_assets": bool(analysis.get("model_assets")),
                "env_backend": env_solution.get("backend", "venv"),
            },
            "frameworks": frameworks,
            "previous_results": {k: v.get("status", "") for k, v in results.items() if isinstance(v, dict)},
            "uncertainties": uncertainties,
            "constraints": [
                "Plan gate only generates strategy hints, never executes tools.",
                "Stage names must be valid pipeline stages.",
                "Hints must not change policy or security settings.",
                "Strategy is advisory; deterministic pipeline still runs.",
            ],
        }

        provider = self._agent_provider()
        if not provider:
            return

        planner = PlanPlanner()
        decision = planner.plan(observation, provider=provider)

        # Write plan artifact
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        plan_data = {
            "status": decision.status,
            "deployment_strategy": "",
            "stage_plan": [],
            "uncertainties": uncertainties,
            "fallback": "deterministic_pipeline",
            "hypothesis": decision.hypothesis,
            "confidence": decision.confidence,
            "revision": revision,
        }

        if decision.tool_call and decision.tool_call.get("name") == "set_deployment_strategy":
            plan_data["deployment_strategy"] = decision.tool_call.get("input", {}).get("strategy", "")
            plan_data["stage_plan"] = decision.tool_call.get("input", {}).get("stage_plan", [])
        elif decision.tool_call and decision.tool_call.get("name") == "set_stage_hint":
            plan_data["stage_hints"] = {decision.tool_call.get("input", {}).get("stage", ""): decision.tool_call.get("input", {}).get("hints", {})}

        if revision:
            # Append to revisions JSONL
            revisions_path = run_dir / "agent_plan_revisions.jsonl"
            with revisions_path.open("a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(plan_data, ensure_ascii=False) + "\n")
        else:
            write_json(run_dir / "agent_plan_initial.json", plan_data)

        # Write strategy report
        write_json(reports_dir / "agent_strategy.json", plan_data)

        self.store.events(task_id).append("plan", "decision_gate", {
            "decision_status": decision.status,
            "revision": revision,
            "hypothesis": decision.hypothesis,
        })

    def _maybe_plan_revision(self, task_id: str, stage: str, result) -> None:
        """Generate plan revision at failed/uncertain stage boundary.

        Per design doc: same stage max revision 1 time to avoid infinite loops.
        """
        run_dir = self.store.run_dir(task_id)
        # Check if already revised for this stage
        revisions_path = run_dir / "agent_plan_revisions.jsonl"
        if revisions_path.exists():
            try:
                import json
                existing = revisions_path.read_text(encoding="utf-8").strip().split("\n")
                for line in existing:
                    if not line:
                        continue
                    rev = json.loads(line)
                    if rev.get("source_stage") == stage:
                        return  # Already revised for this stage
            except (OSError, ValueError):
                pass

        # Load current results for context
        results = self._load_previous_results(run_dir)
        analysis = results.get("analyze", {}).get("data", {}) if isinstance(results.get("analyze"), dict) else {}

        self._apply_plan_gate(task_id, analysis, results, run_dir, revision=True)

    def _load_plan_hints(self, run_dir: Path) -> Dict:
        """Load plan hints from agent_plan_initial.json or agent_strategy.json.

        Returns dict with stage_hints and deployment_strategy.
        """
        plan_path = run_dir / "reports" / "agent_strategy.json"
        if not plan_path.exists():
            plan_path = run_dir / "agent_plan_initial.json"
        if not plan_path.exists():
            return {"stage_hints": {}, "deployment_strategy": ""}
        try:
            plan = read_json(plan_path)
            stage_hints = {}
            # Extract stage_hints from stage_plan
            for item in plan.get("stage_plan", []):
                stage = item.get("stage", "")
                hints = item.get("hints", {})
                if stage and hints:
                    stage_hints[stage] = hints
            # Also check direct stage_hints field
            if "stage_hints" in plan:
                stage_hints.update(plan["stage_hints"])
            return {
                "stage_hints": stage_hints,
                "deployment_strategy": plan.get("deployment_strategy", ""),
            }
        except (OSError, ValueError):
            return {"stage_hints": {}, "deployment_strategy": ""}

        # Tag the revision with source stage
        if revisions_path.exists():
            try:
                import json
                lines = revisions_path.read_text(encoding="utf-8").strip().split("\n")
                if lines:
                    last = json.loads(lines[-1])
                    last["source_stage"] = stage
                    last["trigger_status"] = result.status
                    lines[-1] = json.dumps(last, ensure_ascii=False)
                    revisions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except (OSError, ValueError):
                pass

    def _apply_env_gate(self, task_id: str, env_result, deploy_analysis: Dict, run_dir: Path) -> Dict:
        """Apply env decision gate: LLM diagnoses dependency conflicts and proposes constraints."""
        from auto_harness.agent_runtime.decision_gate import AgentDecisionGate
        from auto_harness.agent_runtime.stage_schemas import ENV_TOOLS

        data = env_result.data if isinstance(env_result.data, dict) else {}
        env_solution = deploy_analysis.get("env_solution", {})
        diagnosis = data.get("diagnosis", {})

        observation = {
            "stage": "env_solve",
            "failed_stage": "env_deploy",
            "requirements": deploy_analysis.get("install_plan", []),
            "install_log_tail": str(data.get("error", ""))[:4000],
            "deterministic_constraints": env_solution.get("constraints", []),
            "risk_reasons": env_solution.get("risk_reasons", []),
            "diagnosis": diagnosis,
            "constraints": [
                "Only apply dependency constraints, do not modify source files.",
                "Package names must be valid Python package names.",
                "Version specs must be valid (e.g., '<2', '>=1.0,<2.0', '==1.2.3').",
                "No arbitrary index URLs.",
                "Write constraints to repair_overlay/constraints.txt only.",
            ],
        }

        gate = AgentDecisionGate(provider=self._agent_provider())
        gate_result = gate.decide(
            stage="env_solve",
            observation=observation,
            allowed_tools=list(ENV_TOOLS),
            mode=self.config.agent_mode,
            run_dir=run_dir,
            max_steps=self.config.agent_decision_gate_max_steps,
        )

        self.store.events(task_id).append("env_deploy", "decision_gate", {
            "decision_status": gate_result.decision_status,
            "policy_allowed": gate_result.policy.get("allowed", False),
            "state_changed": gate_result.state_delta.get("changed", False),
        })

        # If constraints were applied, update the analysis with the overlay
        if gate_result.state_delta.get("changed") and gate_result.state_delta.get("constraint"):
            constraint = gate_result.state_delta["constraint"]
            updated_analysis = dict(deploy_analysis)
            env_sol = dict(updated_analysis.get("env_solution", {}))
            existing_constraints = list(env_sol.get("constraints", []))
            new_constraint = "%s%s" % (constraint.get("package", ""), constraint.get("version_spec", ""))
            if new_constraint and new_constraint not in existing_constraints:
                existing_constraints.append(new_constraint)
            env_sol["constraints"] = existing_constraints
            env_sol["constraint_reasons"] = env_sol.get("constraint_reasons", []) + [
                "LLM env gate: %s" % (constraint.get("reason", "dependency conflict"))
            ]
            updated_analysis["env_solution"] = env_sol
            return updated_analysis

        return None

    def _maybe_agent_diagnose(self, task_id: str, stage: str, result, analysis: Dict, runtime: RuntimePolicy, run_dir: Path) -> None:
        if not (self.config.agent_mode in ("planner", "gated_actor") and self.config.agent_enable_log_diagnosis):
            return
        if result.status not in ("failed", "uncertain"):
            return
        data = result.data if isinstance(result.data, dict) else {}
        diagnosis = data.get("diagnosis") if isinstance(data.get("diagnosis"), dict) else {}
        if diagnosis.get("category") not in (None, "", "unknown") and float(diagnosis.get("confidence") or 0) >= 0.75:
            return
        observation = AgentObservation(
            task_id=task_id,
            stage=stage,
            repo_dir=str(run_dir / "workspace" / "repo"),
            deterministic_result=to_plain(result),
            previous_results=self._load_previous_results(run_dir),
            memory_hits=self.memory.query(stage, analysis, limit=self.config.max_memory_items),
            runtime_policy=runtime.__dict__,
            allowed_action_types=["install_package", "update_verify_hint", "request_env_var_name_only", "rerun_from_stage"],
            extra={"analysis": analysis},
        )
        diagnoser = AgentDiagnoser(self._agent_provider(), config=self.config, trace_writer=self._agent_trace_writer(run_dir))
        agent_diagnosis = diagnoser.diagnose(observation)
        result.data.setdefault("agent_diagnosis", agent_diagnosis)

    def _agent_verify_planner(self, trace_writer: AgentTraceWriter) -> AgentVerifyPlanner:
        return AgentVerifyPlanner(self._agent_provider(), config=self.config, trace_writer=trace_writer)

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def _remember(self, task_id: str, stage: str, result, analysis: Dict, results: Dict[str, Dict] = None) -> None:
        entry = self.memory.remember_issue(task_id, stage, result, analysis)
        if entry:
            task = self.store.load_task(task_id)
            run_dir = self.store.run_dir(task_id)
            state = self.store.load_state(task_id)
            loop_result = self._agent_loop_controller().handle_stage_result(
                task_id,
                stage,
                result,
                analysis,
                task.runtime,
                state.last_safe_stage,
                memory_entry=entry,
            )
            if results is not None:
                results[stage] = to_plain(result)
            self._save_stage(task_id, stage, result)
            self.store.events(task_id).append(
                stage,
                "memory_recorded",
                {
                    "memory_id": entry["id"],
                    "signature": entry["signature"],
                    "repair_plan": loop_result.get("repair_plan", {}),
                    "repair_policy": loop_result.get("policy", {}),
                    "repair_apply": loop_result.get("apply_result", {}),
                    "agent_loop": {
                        "next_rerun_from": loop_result.get("next_rerun_from"),
                        "should_auto_resume": loop_result.get("should_auto_resume"),
                        "stop_reason": loop_result.get("stop_reason"),
                    },
                },
            )
