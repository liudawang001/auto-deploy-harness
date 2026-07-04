import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from auto_harness.config import HarnessConfig
from auto_harness.agents.claude_code import ClaudeCodeExecutor
from auto_harness.assets import ModelCache
from auto_harness.models.base import read_json, to_plain, write_json
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.memory import MemoryStore
from auto_harness.modules import (
    EnvDeployModule,
    ModelPrepareModule,
    ProjectAnalyzer,
    ReportGenerator,
    ResourcePlanner,
    RunnerModule,
    VerifyModule,
)
from auto_harness.skills import SkillRegistry
from auto_harness.repair import RepairApplier, RepairPlanner, RepairPolicy
from auto_harness.state import StateStore
from auto_harness.utils.files import safe_name, short_hash
from auto_harness.utils.time import compact_timestamp, utc_now_iso


class TaskRunner:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.store = StateStore(config.runs_path)
        self.skills = SkillRegistry(config.skills_path, max_chars=config.max_skill_chars)
        self.memory = MemoryStore(config.memory_path)
        self.model_cache = ModelCache(config.model_cache_path)
        self.repair_planner = RepairPlanner()
        self.repair_policy = RepairPolicy()
        self.repair_applier = RepairApplier()

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

    def run_existing(self, task_id: str, dry_run: bool = True) -> str:
        task = self.store.load_task(task_id)
        run_dir = self.store.run_dir(task_id)
        repo_dir = run_dir / "workspace" / "repo"
        results: Dict[str, Dict] = {}

        analyze_context = self._stage_context("analyze", {})
        analyzer = ProjectAnalyzer(
            agent_executor=ClaudeCodeExecutor() if self.config.use_agent_analyzer else None,
            use_agent=self.config.use_agent_analyzer,
            agent_timeout_seconds=self.config.agent_timeout_seconds,
            stage_context=analyze_context,
        )
        analyze_result = analyzer.analyze(repo_dir)
        results["analyze"] = to_plain(analyze_result)
        self._save_stage(task_id, "analyze", analyze_result)
        self._remember(task_id, "analyze", analyze_result, analyze_result.data)

        resource_context = self._stage_context("resource_plan", analyze_result.data)
        resource_result = ResourcePlanner().plan(repo_dir, analyze_result.data)
        self._attach_context(resource_result, resource_context)
        results["resource_plan"] = to_plain(resource_result)
        self._save_stage(task_id, "resource_plan", resource_result)
        self._remember(task_id, "resource_plan", resource_result, analyze_result.data)

        env_context = self._stage_context("env_deploy", analyze_result.data)
        env_result = EnvDeployModule().deploy(
            repo_dir,
            analyze_result.data,
            execute=not dry_run and task.runtime.allow_dependency_install,
            allowed_commands=self.config.allowed_commands,
        )
        self._attach_context(env_result, env_context)
        results["env_deploy"] = to_plain(env_result)
        self._save_stage(task_id, "env_deploy", env_result)
        self._remember(task_id, "env_deploy", env_result, analyze_result.data)

        model_context = self._stage_context("model_prepare", resource_result.data)
        model_result = ModelPrepareModule(self.model_cache).prepare(
            run_dir,
            resource_result.data,
            execute=not dry_run,
            progress_callback=lambda progress: self.store.update_stage(task_id, "model_prepare", "waiting_download", progress=progress),
        )
        self._attach_context(model_result, model_context)
        results["model_prepare"] = to_plain(model_result)
        self._save_stage(task_id, "model_prepare", model_result)
        self._remember(task_id, "model_prepare", model_result, analyze_result.data)

        runner_context = self._stage_context("runner", analyze_result.data)
        runner_result = RunnerModule().run(
            repo_dir,
            analyze_result.data,
            execute=not dry_run and task.runtime.allow_service_start,
            allowed_commands=self.config.allowed_commands,
        )
        self._attach_context(runner_result, runner_context)
        results["runner"] = to_plain(runner_result)
        self._save_stage(task_id, "runner", runner_result)
        self._remember(task_id, "runner", runner_result, analyze_result.data)

        verify_context = self._stage_context("verify", analyze_result.data)
        verify_result = VerifyModule(stage_context=verify_context).verify(run_dir, analyze_result.data, runner_result.data)
        results["verify"] = to_plain(verify_result)
        self._save_stage(task_id, "verify", verify_result)
        self._remember(task_id, "verify", verify_result, analyze_result.data)

        task_data = read_json(run_dir / "task.json")
        report_result = ReportGenerator().generate(run_dir, task_data, results)
        results["report"] = to_plain(report_result)
        self._save_stage(task_id, "report", report_result)
        state = self.store.load_state(task_id)
        state.report_path = report_result.data.get("report_path")
        self.store.save_state(state)

        write_json(run_dir / "reports" / "pipeline_results.json", results)
        return task_id

    def resume(self, task_id: str, dry_run: bool = True) -> str:
        self.store.events(task_id).append("task", "resume_requested", {"dry_run": dry_run})
        return self.run_existing(task_id, dry_run=dry_run)

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

    def _remember(self, task_id: str, stage: str, result, analysis: Dict) -> None:
        entry = self.memory.remember_issue(task_id, stage, result, analysis)
        if entry:
            task = self.store.load_task(task_id)
            repair_plan = self.repair_planner.propose(stage, result, analysis)
            policy_result = self.repair_policy.check(repair_plan, task.runtime)
            apply_result = self.repair_applier.apply(self.store.run_dir(task_id), repair_plan, policy_result)
            self.store.events(task_id).append(
                stage,
                "memory_recorded",
                {
                    "memory_id": entry["id"],
                    "signature": entry["signature"],
                    "repair_plan": repair_plan,
                    "repair_policy": policy_result,
                    "repair_apply": apply_result,
                },
            )
