import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from auto_harness.agent import AgentDecisionEngine, AgentDiagnoser, AgentLoopController, AgentMetricsCollector, AgentObservation, AgentTraceWriter, AgentVerifyPlanner
from auto_harness.agent_runtime import AgentContributionAnalyzer, AgentGoal, AgentRuntime
from auto_harness.config import HarnessConfig
from auto_harness.agents.claude_code import ClaudeCodeExecutor
from auto_harness.assets import HuggingFaceDownloader, ModelCache, ModelScopeDownloader
from auto_harness.models.base import read_json, to_plain, write_json
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.memory import MemoryStore, VerifiedMemoryRecorder
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
        current_start_stage = start_stage
        max_iterations = int(self.config.agent_max_loop_iterations or 0)
        for iteration in range(max_iterations + 1):
            self._run_existing_once(task_id, dry_run=dry_run, start_stage=current_start_stage)
            decision = self._next_agent_resume_decision(task_id, iteration)
            if not decision.get("should_resume"):
                break
            self.store.events(task_id).append("task", "agent_auto_resume", decision)
            current_start_stage = decision["start_stage"]
        return task_id

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
            env_solve_result = EnvSolveModule(
                env_backend=self.config.env_backend,
                conda_envs_dir=self.config.conda_envs_dir,
                conda_prefer_mamba=self.config.conda_prefer_mamba,
                conda_allowed_channels=self.config.conda_allowed_channels,
                conda_python_default=self.config.conda_python_default,
                torch_cuda_preference=self.config.torch_cuda_preference,
            ).solve(repo_dir, effective_analysis, resource_data)
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
            results["env_deploy"] = to_plain(env_result)
            self._save_stage(task_id, "env_deploy", env_result)
            self._remember(task_id, "env_deploy", env_result, deploy_analysis, results)

        if should_run("model_prepare"):
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
        verify_result = VerifyModule(
            stage_context=verify_context,
            progress_callback=lambda progress: self.store.update_stage(task_id, "verify", "running_verify", progress=progress),
            verify_planner=self._agent_verify_planner(verify_trace_writer) if self._agent_verify_planner_enabled() else None,
        ).verify(run_dir, deploy_analysis, runner_data)
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
        if any(item.get("executed") and int(item.get("exit_code") or 0) == 0 for item in apply_result.get("action_results", [])):
            return True
        metadata_actions = {"update_verify_hint", "rerun_from_stage"}
        return any(item.get("action_type") in metadata_actions and item.get("status") == "metadata_only" for item in apply_result.get("action_results", []))

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

    def _agent_repair_execute_enabled(self, runtime: RuntimePolicy) -> bool:
        return (
            self.config.agent_mode == "gated_actor"
            and self.config.agent_enable_repair_actions
            and runtime.allow_dependency_install
        )

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
