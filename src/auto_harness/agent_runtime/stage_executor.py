"""AgentStageExecutor: maps Agent structured stage actions to real stage modules.

This is the bridge between the AgentLoop and the deterministic pipeline modules.
The Agent NEVER directly calls shell commands — it outputs structured stage actions,
and this executor maps them to ProjectAnalyzer, ResourcePlanner, EnvSolveModule,
EnvDeployModule, ModelPrepareModule, RunnerModule, VerifyModule.

Key invariants:
- AgentStageExecutor does NOT copy stage module logic
- LLM cannot pass shell commands to executor
- Executor does NOT bypass existing module policy
- Executor does NOT directly write 'passed' — that comes from modules
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.models.result import StageResult
from auto_harness.models.base import to_plain


@dataclass
class StageExecutionResult:
    """Result of executing a single stage through the executor."""
    stage: str
    before_status: str
    after_status: str
    result: dict
    changed: bool
    evidence_paths: list = field(default_factory=list)
    error: str = ""


class AgentStageExecutor:
    """Maps Agent stage actions to real stage module calls.

    Each execute_stage() call invokes the corresponding stage module with
    the current state, and returns a StageExecutionResult with before/after
    status comparison.
    """

    def __init__(
        self,
        *,
        config=None,
        store=None,
        model_prepare=None,
        repair_components: Dict = None,
        provider_factory: Callable = None,
        runtime_policy: Dict = None,
        verify_planner_factory: Callable = None,
        agent_verify_config_factory: Callable = None,
    ) -> None:
        self.config = config
        self.store = store
        self.model_prepare = model_prepare
        self.repair_components = repair_components or {}
        self.provider_factory = provider_factory
        self.runtime_policy = runtime_policy or {}
        # Phase 5: Agent Verify integration — factories let the graph inject
        # the orchestrator's verify planner/provider config into VerifyModule.
        self.verify_planner_factory = verify_planner_factory
        self.agent_verify_config_factory = agent_verify_config_factory

    def execute_stage(
        self,
        *,
        task_id: str,
        run_dir: Path,
        repo_dir: Path,
        stage: str,
        state: dict,
        analysis: dict,
        resource_data: dict,
        deploy_analysis: dict,
        runner_data: dict,
        dry_run: bool,
        stage_hints: dict = None,
        repair_overlay: dict = None,
        runtime_policy: Dict = None,
        skill_context: dict = None,
    ) -> StageExecutionResult:
        """Execute a single pipeline stage and return before/after status.

        Args:
            task_id: Task identifier
            run_dir: Run directory for artifacts
            repo_dir: Repository directory
            stage: Stage name to execute
            state: Current agent state dict
            analysis: Analysis results
            resource_data: Resource plan results
            deploy_analysis: Deploy analysis (env_solve output)
            runner_data: Runner results
            dry_run: If True, don't execute real commands
            stage_hints: Optional hints from plan gate
            repair_overlay: Optional repair overlay
            runtime_policy: Optional per-call runtime policy override (Phase 2).
                When provided, takes precedence over the executor's default.

        Returns:
            StageExecutionResult with before/after status
        """
        stage_hints = stage_hints or {}
        repair_overlay = repair_overlay or {}
        skill_context = skill_context or {}
        # Phase 2: use the per-call runtime policy if provided, else fall back
        # to the executor's default. Never rely on an empty dict + config fallback.
        effective_runtime_policy = dict(runtime_policy or self.runtime_policy or {})
        # Temporarily set effective policy for the duration of this call so
        # the _execute_* helpers pick it up via self.runtime_policy.
        saved_policy = self.runtime_policy
        self.runtime_policy = effective_runtime_policy

        # Get before status from current results
        before_status = self._get_stage_status(stage, state)

        try:
            if stage == "analyze":
                return self._execute_analyze(task_id, run_dir, repo_dir, state, before_status, dry_run)
            elif stage == "resource_plan":
                return self._execute_resource_plan(task_id, run_dir, repo_dir, analysis, before_status)
            elif stage == "host_preflight":
                return self._execute_host_preflight(
                    task_id, run_dir, repo_dir, analysis, resource_data,
                    before_status, dry_run,
                )
            elif stage == "env_solve":
                preflight = (
                    (state.get("stage_results") or {}).get("host_preflight") or {}
                ).get("data") or {}
                return self._execute_env_solve(
                    task_id, run_dir, repo_dir, analysis, resource_data,
                    before_status, stage_hints, preflight,
                )
            elif stage == "env_deploy":
                return self._execute_env_deploy(
                    task_id, run_dir, repo_dir, deploy_analysis,
                    before_status, dry_run, repair_overlay, state,
                )
            elif stage == "model_prepare":
                return self._execute_model_prepare(
                    task_id, run_dir, repo_dir, resource_data, analysis,
                    before_status, dry_run,
                )
            elif stage == "runner":
                return self._execute_runner(task_id, run_dir, repo_dir, deploy_analysis, before_status, dry_run, stage_hints)
            elif stage == "verify":
                return self._execute_verify(
                    task_id,
                    run_dir,
                    repo_dir,
                    deploy_analysis,
                    runner_data,
                    before_status,
                    stage_hints,
                    skill_context,
                )
            elif stage == "repair":
                return self._execute_repair(task_id, run_dir, repo_dir, state, before_status, dry_run)
            else:
                return StageExecutionResult(
                    stage=stage,
                    before_status=before_status,
                    after_status="failed",
                    result={},
                    changed=before_status != "failed",
                    error="unknown stage: %s" % stage,
                )
        except Exception as exc:
            return StageExecutionResult(
                stage=stage,
                before_status=before_status,
                after_status="failed",
                result={},
                changed=before_status != "failed",
                error=str(exc)[:2000],
            )
        finally:
            # Phase 2: restore the executor's default runtime policy
            self.runtime_policy = saved_policy

    def _execute_analyze(self, task_id, run_dir, repo_dir, state, before_status, dry_run):
        from auto_harness.modules.analyzer import ProjectAnalyzer
        from auto_harness.agents.claude_code import ClaudeCodeExecutor
        analyzer = ProjectAnalyzer(
            agent_executor=ClaudeCodeExecutor() if (self.config and self.config.use_agent_analyzer) else None,
            use_agent=bool(self.config and self.config.use_agent_analyzer),
            agent_timeout_seconds=self.config.agent_timeout_seconds if self.config else 900,
            agent_mode=self.config.agent_mode if self.config else "off",
            task_id=task_id,
            agent_max_file_chars=self.config.agent_max_file_chars if self.config else 6000,
        )
        result = analyzer.analyze(repo_dir)
        return StageExecutionResult(
            stage="analyze",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
        )

    def _execute_resource_plan(self, task_id, run_dir, repo_dir, analysis, before_status):
        from auto_harness.modules.resource_plan import ResourcePlanner
        result = ResourcePlanner().plan(repo_dir, analysis)
        model_policy = analysis.get("model_assets")
        if isinstance(model_policy, dict) and model_policy.get("required") is not True:
            # The accepted deployment plan explicitly chose a no-model startup
            # path. Repository-wide detection may still find optional local
            # models in docs, provider catalogs, or tests; those must not turn
            # into a GPU requirement or eager download for this deployment.
            result.data.update({
                "gpu_required": False,
                "cuda_required": "",
                "torch_variant": "",
                "estimated_vram_gb": 0,
                "model_assets": [],
                "external_tokens": [],
                "risk_level": "low",
                "risk_reasons": [
                    "optional model assets deferred by accepted deployment plan",
                ],
            })
        return StageExecutionResult(
            stage="resource_plan",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
        )

    def _execute_host_preflight(
        self, task_id, run_dir, repo_dir, analysis, resource_data,
        before_status, dry_run,
    ):
        from auto_harness.modules.host_preflight import HostPreflightModule
        allow_install = self.runtime_policy.get(
            "allow_dependency_install",
            self.config.allow_dependency_install if self.config else False,
        )
        result = HostPreflightModule(self.config).run(
            repo_dir,
            analysis,
            resource_data,
            run_dir=run_dir,
            allow_mutation=not dry_run and allow_install,
        )
        return StageExecutionResult(
            stage="host_preflight",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
            evidence_paths=list(result.evidence),
        )

    def _execute_env_solve(
        self, task_id, run_dir, repo_dir, analysis, resource_data,
        before_status, stage_hints, preflight,
    ):
        from auto_harness.modules.env_solve import EnvSolveModule
        env_backend = self.config.env_backend if self.config else "auto"
        conda_envs_dir = self.config.conda_envs_dir if self.config else ".conda/envs"
        conda_prefer_mamba = self.config.conda_prefer_mamba if self.config else True
        conda_allowed_channels = self.config.conda_allowed_channels if self.config else ["defaults", "conda-forge", "pytorch"]
        conda_python_default = self.config.conda_python_default if self.config else "3.10"
        torch_cuda_preference = self.config.torch_cuda_preference if self.config else "auto"
        result = EnvSolveModule(
            env_backend=env_backend,
            conda_envs_dir=conda_envs_dir,
            conda_prefer_mamba=conda_prefer_mamba,
            conda_allowed_channels=conda_allowed_channels,
            conda_python_default=conda_python_default,
            torch_cuda_preference=torch_cuda_preference,
        ).solve(
            repo_dir,
            analysis,
            resource_data,
            stage_hints=stage_hints,
            preflight=preflight,
        )
        return StageExecutionResult(
            stage="env_solve",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
        )

    def _execute_env_deploy(
        self, task_id, run_dir, repo_dir, deploy_analysis,
        before_status, dry_run, repair_overlay, state,
    ):
        from auto_harness.modules.env_deploy import EnvDeployModule
        # Merge repair overlay constraints into analysis
        effective_analysis = dict(deploy_analysis)
        if repair_overlay.get("active") and repair_overlay.get("install_commands"):
            existing_plan = list(effective_analysis.get("install_plan", []))
            for cmd in repair_overlay["install_commands"]:
                if cmd not in existing_plan:
                    existing_plan.append(cmd)
            effective_analysis["install_plan"] = existing_plan
        # Use runtime_policy for execute flag (set by CLI), fallback to config
        allow_install = self.runtime_policy.get("allow_dependency_install",
                                                 self.config.allow_dependency_install if self.config else False)
        execute = not dry_run and allow_install
        result = EnvDeployModule().deploy(
            repo_dir,
            effective_analysis,
            execute=execute,
            allowed_commands=self.config.allowed_commands if self.config else None,
            execution_backend=self.config.execution_backend if self.config else "local",
            docker_image=self.config.docker_image if self.config else "python:3.10-slim",
            docker_network=self.config.docker_network if self.config else "bridge",
            docker_gpus=self.config.docker_gpus if self.config else "none",
            docker_model_cache_dir=self.config.docker_model_cache_dir if self.config else "",
            docker_security_options=self._docker_security_options(),
            config=self.config,
            run_dir=run_dir,
            task_id=task_id,
            operation_id=str(state.get("pending_operation_id") or ""),
            operation_prepared=bool(
                state.get("recovery_stage") == "env_deploy"
                and state.get("recovery_decision") in {"execute", "continue", "retry"}
                and state.get("pending_operation_id")
            ),
        )
        return StageExecutionResult(
            stage="env_deploy",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
        )

    def _execute_model_prepare(
        self, task_id, run_dir, repo_dir, resource_data, analysis,
        before_status, dry_run,
    ):
        if not self.model_prepare:
            return StageExecutionResult(
                stage="model_prepare",
                before_status=before_status,
                after_status="uncertain",
                result={},
                changed=False,
                error="model_prepare module not available",
            )
        effective_resources = dict(resource_data)
        declared_assets = analysis.get("model_assets") if isinstance(analysis, dict) else None
        if isinstance(declared_assets, dict) and declared_assets.get("required") is not True:
            effective_resources["model_assets"] = []
            effective_resources["external_tokens"] = []
            effective_resources["model_asset_decision"] = "deployment plan declared assets optional"
        result = self.model_prepare.prepare(
            run_dir,
            effective_resources,
            execute=not dry_run,
            repo_dir=repo_dir,
            allowed_commands=self.config.allowed_commands if self.config else None,
            timeout_seconds=self.config.default_timeout_seconds if self.config else 900,
        )
        return StageExecutionResult(
            stage="model_prepare",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
        )

    def _execute_runner(self, task_id, run_dir, repo_dir, deploy_analysis, before_status, dry_run, stage_hints):
        from auto_harness.modules.runner import RunnerModule
        # Use runtime_policy for execute flag (set by CLI), fallback to config
        allow_start = self.runtime_policy.get("allow_service_start",
                                               self.config.allow_service_start if self.config else False)
        execute = not dry_run and allow_start
        result = RunnerModule().run(
            repo_dir,
            deploy_analysis,
            execute=execute,
            allowed_commands=self.config.allowed_commands if self.config else None,
            execution_backend=self.config.execution_backend if self.config else "local",
            docker_image=self.config.docker_image if self.config else "python:3.10-slim",
            docker_network=self.config.docker_network if self.config else "bridge",
            docker_gpus=self.config.docker_gpus if self.config else "none",
            docker_model_cache_dir=self.config.docker_model_cache_dir if self.config else "",
            docker_security_options=self._docker_security_options(),
            stage_hints=stage_hints,
        )
        return StageExecutionResult(
            stage="runner",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
        )

    def _execute_verify(
        self,
        task_id,
        run_dir,
        repo_dir,
        deploy_analysis,
        runner_data,
        before_status,
        stage_hints,
        skill_context,
    ):
        from auto_harness.modules.verify import VerifyModule
        # runner_data from agent loop may be a StageResult wrapper with {stage, status, summary, data}
        # VerifyModule._service_discovery expects the inner data (pid, service_ready, expected_port, etc.)
        effective_runner_data = runner_data
        if isinstance(runner_data, dict) and "data" in runner_data and isinstance(runner_data.get("data"), dict):
            inner_data = runner_data["data"]
            # If inner_data also has a "data" field (from to_plain), extract that
            if "data" in inner_data and isinstance(inner_data.get("data"), dict):
                effective_runner_data = inner_data["data"]
            else:
                effective_runner_data = inner_data

        # Phase 5: inject verify_planner and agent_verify_config from factories
        verify_planner = None
        agent_verify_config = None
        if self.verify_planner_factory:
            try:
                verify_planner = self.verify_planner_factory()
            except Exception:
                pass
        if self.agent_verify_config_factory:
            try:
                agent_verify_config = self.agent_verify_config_factory()
            except Exception:
                pass
        agent_verify_config = dict(agent_verify_config or {})
        agent_verify_config["skill_context"] = dict(skill_context or {})

        result = VerifyModule(
            verify_planner=verify_planner,
            agent_verify_config=agent_verify_config,
        ).verify(
            run_dir,
            deploy_analysis,
            effective_runner_data,
            stage_hints=stage_hints,
        )
        evidence_paths = list(result.evidence) if hasattr(result, 'evidence') and result.evidence else []
        return StageExecutionResult(
            stage="verify",
            before_status=before_status,
            after_status=result.status,
            result=to_plain(result),
            changed=before_status != result.status,
            evidence_paths=evidence_paths,
        )

    def _execute_repair(self, task_id, run_dir, repo_dir, state, before_status, dry_run):
        """Execute repair: delegate to repair components if available."""
        repair_planner = self.repair_components.get("planner")
        repair_policy = self.repair_components.get("policy")
        repair_applier = self.repair_components.get("applier")
        if not all([repair_planner, repair_policy, repair_applier]):
            return StageExecutionResult(
                stage="repair",
                before_status=before_status,
                after_status=before_status,
                result={},
                changed=False,
                error="repair components not available",
            )
        # Repair execution is handled by the repair loop, not here
        return StageExecutionResult(
            stage="repair",
            before_status=before_status,
            after_status="pending",
            result={"status": "repair_delegated_to_loop"},
            changed=False,
        )

    def _get_stage_status(self, stage: str, state: dict) -> str:
        """Get the current status of a stage from state."""
        stage_results = state.get("stage_results", {})
        if stage in stage_results:
            return stage_results[stage].get("status", "")
        return ""

    def _docker_security_options(self) -> Dict:
        if not self.config:
            return {}
        return {
            "read_only_rootfs": self.config.docker_read_only_rootfs,
            "user": self.config.docker_user,
            "memory": self.config.docker_memory,
            "cpus": self.config.docker_cpus,
            "pids_limit": self.config.docker_pids_limit,
            "tmpfs_size": self.config.docker_tmpfs_size,
            "cap_drop_all": self.config.docker_cap_drop_all,
            "no_new_privileges": self.config.docker_no_new_privileges,
            "repo_mount_mode": self.config.docker_repo_mount_mode,
        }
