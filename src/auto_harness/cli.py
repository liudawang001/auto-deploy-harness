import argparse
import json
from pathlib import Path

from auto_harness.agent import AgentMetricsCollector
from auto_harness.config import HarnessConfig
from auto_harness.artifacts import DeploymentPackageExporter
from auto_harness.benchmarks import BenchmarkRunner, LiveSmokePlanner
from auto_harness.dashboard import DashboardGenerator, DashboardServer
from auto_harness.evals import AgentComparisonReporter
from auto_harness.memory import MemoryPromoter
from auto_harness.memory.evolution import MemoryEvolutionManager
from auto_harness.memory.outcomes import SkillOutcomeRecorder
from auto_harness.skills.rollback import SkillRollbackManager
from auto_harness.live_smoke import LiveAgentSmokeRunner
from auto_harness.orchestrator import TaskRunner
from auto_harness.providers import Message, MockLLMProvider, XunfeiSparkProvider
from auto_harness.queue import DeploymentQueue
from auto_harness.readiness import ReadinessAuditor
from auto_harness.runtime import DockerSmokeChecker
from auto_harness.utils.time import compact_timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-deploy-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create local run directories")

    deploy = sub.add_parser("deploy", help="create and run a deployment task")
    deploy.add_argument("--repo", required=True)
    deploy.add_argument("--name", default="")
    deploy.add_argument("--dry-run", action="store_true", default=False)
    deploy.add_argument("--execute", action="store_true", default=False)
    deploy.add_argument("--allow-install", action="store_true", default=False)
    deploy.add_argument("--allow-start", action="store_true", default=False)
    deploy.add_argument("--skip-clone", action="store_true", default=False)
    deploy.add_argument("--model-download-workers", type=int, default=None)
    deploy.add_argument("--download-retries", type=int, default=None)
    deploy.add_argument("--download-retry-backoff", type=float, default=None)
    deploy.add_argument("--execution-backend", choices=["local", "docker"], default=None)
    deploy.add_argument("--agent-self-heal", action="store_true", default=False)
    deploy.add_argument("--agent-mode", choices=["off", "audit", "planner", "gated_actor"], default=None)
    deploy.add_argument("--agent-enable-verify", action="store_true", default=False)
    deploy.add_argument("--agent-verify-max-steps", type=int, default=None)
    deploy.add_argument("--env-backend", choices=["auto", "venv", "conda", "mamba"], default=None)
    deploy.add_argument("--prefer-mamba", action="store_true", default=False)
    deploy.add_argument("--docker-image", default=None)
    deploy.add_argument("--docker-network", default=None)
    deploy.add_argument("--docker-gpus", default=None)
    deploy.add_argument("--docker-model-cache-dir", default=None)
    deploy.add_argument("--agent-enable-plan-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-runner-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-env-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-model-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-repair-gate", action="store_true", default=False)
    deploy.add_argument("--agent-all-decision-gates", action="store_true", default=False)
    deploy.add_argument("--agent-runtime-loop", action="store_true", default=False)
    deploy.add_argument("--agent-runtime-loop-max-iterations", type=int, default=None)
    deploy.add_argument("--agent-runtime-loop-position", choices=["primary", "post_pipeline"], default=None)

    resume = sub.add_parser("resume", help="resume an existing task")
    resume.add_argument("--task-id", required=True)
    resume.add_argument("--dry-run", action="store_true", default=False)
    resume.add_argument("--execute", action="store_true", default=False)
    resume.add_argument("--model-download-workers", type=int, default=None)
    resume.add_argument("--download-retries", type=int, default=None)
    resume.add_argument("--download-retry-backoff", type=float, default=None)
    resume.add_argument("--execution-backend", choices=["local", "docker"], default=None)
    resume.add_argument("--agent-self-heal", action="store_true", default=False)
    resume.add_argument("--env-backend", choices=["auto", "venv", "conda", "mamba"], default=None)
    resume.add_argument("--prefer-mamba", action="store_true", default=False)
    resume.add_argument("--docker-image", default=None)
    resume.add_argument("--docker-network", default=None)
    resume.add_argument("--docker-gpus", default=None)
    resume.add_argument("--docker-model-cache-dir", default=None)

    status = sub.add_parser("status", help="show task status")
    status.add_argument("--task-id", required=True)

    report = sub.add_parser("report", help="print report path")
    report.add_argument("--task-id", required=True)

    package = sub.add_parser("package", help="export a deployment audit package for one task")
    package.add_argument("--task-id", required=True)
    package.add_argument("--output", default="")
    package.add_argument("--include-logs", action="store_true", default=False)

    repair_approve = sub.add_parser("repair-approve", help="approve the latest repair plan for a task")
    repair_approve.add_argument("--task-id", required=True)
    repair_approve.add_argument("--note", default="")

    memory_promote = sub.add_parser("memory-promote", help="generate or apply skill update proposals from issue memory")
    memory_promote.add_argument("--min-count", type=int, default=2)
    memory_promote.add_argument("--stage", default=None)
    memory_promote.add_argument("--category", default=None)
    memory_promote.add_argument("--output-dir", default="")
    memory_promote.add_argument("--apply", action="store_true", default=False)
    memory_promote.add_argument("--approve", action="store_true", default=False)
    memory_promote.add_argument("--proposal", default="")
    memory_promote.add_argument("--reviewer", default="operator")
    memory_promote.add_argument("--note", default="")
    memory_promote.add_argument("--skip-regression", action="store_true", default=False)

    memory_evolve = sub.add_parser("memory-evolve", help="propose, validate, regress, shadow, and promote skill candidates from verified memory")
    memory_evolve.add_argument("--propose", action="store_true", default=False)
    memory_evolve.add_argument("--regression", action="store_true", default=False)
    memory_evolve.add_argument("--shadow", action="store_true", default=False)
    memory_evolve.add_argument("--promote", action="store_true", default=False)
    memory_evolve.add_argument("--reject", action="store_true", default=False)
    memory_evolve.add_argument("--candidate", default="")
    memory_evolve.add_argument("--min-verified-count", type=int, default=3)
    memory_evolve.add_argument("--stage", default=None)
    memory_evolve.add_argument("--category", default=None)
    memory_evolve.add_argument("--output-dir", default="")
    memory_evolve.add_argument("--provider", choices=["mock", "xunfei"], default=None)
    memory_evolve.add_argument("--run-dir", default="")
    memory_evolve.add_argument("--no-require-shadow", action="store_true", default=False)
    memory_evolve.add_argument("--reason", default="")

    skill_rollback = sub.add_parser("skill-rollback", help="rollback a promoted skill candidate")
    skill_rollback.add_argument("--candidate", required=True)

    skill_outcomes = sub.add_parser("skill-outcomes", help="summarize skill outcome records")
    skill_outcomes.add_argument("--skill", default="")
    skill_outcomes.add_argument("--candidate", default="")

    skill_gain = sub.add_parser("skill-gain", help="evaluate skill candidate gain over baseline")
    skill_gain.add_argument("--candidate", required=True, help="path to candidate JSON file")
    skill_gain.add_argument("--output", default="", help="path to write gain report JSON")

    evidence_package = sub.add_parser("evidence-package", help="export project evidence as tar.gz archive")
    evidence_package.add_argument("--output", default="dist/evidence/agent-project-evidence.tar.gz", help="output tar.gz path")
    evidence_package.add_argument("--project-root", default=".", help="project root directory")

    llm = sub.add_parser("llm-test", help="test LLM provider")
    llm.add_argument("--provider", choices=["mock", "xunfei"], default="mock")
    llm.add_argument("--prompt", default="Return a JSON object with status ok.")

    benchmark = sub.add_parser("benchmark", help="run local benchmark fixtures")
    benchmark.add_argument("--manifest", default="tests/fixtures/benchmarks/manifest.json")
    benchmark.add_argument("--output", default="")
    benchmark.add_argument("--case-id", action="append", default=None)

    dashboard = sub.add_parser("dashboard", help="generate a static HTML dashboard from local runs")
    dashboard.add_argument("--output", default="")
    dashboard.add_argument("--benchmark-report", default="")
    dashboard.add_argument("--serve", action="store_true", default=False)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)

    readiness = sub.add_parser("readiness", help="audit local project readiness and list external smoke gates")
    readiness.add_argument("--benchmark-report", default="")
    readiness.add_argument("--output", default="")

    agent_metrics = sub.add_parser("agent-metrics", help="collect agent metrics from local runs")
    agent_metrics.add_argument("--runs-dir", default="")
    agent_metrics.add_argument("--output", default="")

    eval_compare = sub.add_parser("eval-compare", help="generate baseline vs agent comparison report")
    eval_compare.add_argument("--manifest", default="eval_targets/manifest.json")
    eval_compare.add_argument("--output-dir", default="runs/evals/agent-verify-mvp")
    eval_compare.add_argument("--run", action="store_true", default=False, help="run real off vs gated_actor comparison (not just skeleton)")

    eval_llm = sub.add_parser("eval-llm-necessity", help="evaluate LLM necessity with baseline vs agent comparison")
    eval_llm.add_argument("--manifest", default="eval_targets/llm_necessity_manifest.json")
    eval_llm.add_argument("--output", default="runs/evals/llm-necessity/report.json")

    queue = sub.add_parser("queue", help="submit and run persistent deployment queue jobs")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)
    queue_submit = queue_sub.add_parser("submit", help="submit a deployment job without running it immediately")
    queue_submit.add_argument("--repo", required=True)
    queue_submit.add_argument("--name", default="")
    queue_submit.add_argument("--dry-run", action="store_true", default=False)
    queue_submit.add_argument("--execute", action="store_true", default=False)
    queue_submit.add_argument("--allow-install", action="store_true", default=False)
    queue_submit.add_argument("--allow-start", action="store_true", default=False)
    queue_submit.add_argument("--skip-clone", action="store_true", default=False)
    queue_submit.add_argument("--require-gpu", action="store_true", default=False)
    queue_submit.add_argument("--priority", type=int, default=100)
    queue_list = queue_sub.add_parser("list", help="list queued deployment jobs")
    queue_list.add_argument("--status", default="")
    queue_run = queue_sub.add_parser("run", help="run queued jobs in this foreground process")
    queue_run.add_argument("--max-jobs", type=int, default=None)
    queue_run.add_argument("--gpu-slots", type=int, default=None)

    live_smoke = sub.add_parser("live-smoke-plan", help="print optional networked E2E smoke plan")
    live_smoke.add_argument("--include-long-running", action="store_true", default=False)
    live_smoke.add_argument("--execution-backend", choices=["local", "docker"], default="local")
    live_smoke.add_argument("--execute", action="store_true", default=False)

    agent_live_smoke = sub.add_parser("agent-live-smoke", help="run optional live LLM agent smoke and write redacted manifest")
    agent_live_smoke.add_argument("--repo", default="tests/fixtures/live/llm_repair_missing_dependency")
    agent_live_smoke.add_argument("--provider", choices=["mock", "xunfei"], default="xunfei")
    agent_live_smoke.add_argument("--execute", action="store_true", default=False)
    agent_live_smoke.add_argument("--output", default="")
    agent_live_smoke.add_argument("--disable-analyze-planner", action="store_true", default=False)
    agent_live_smoke.add_argument("--resume-attempts", type=int, default=1)

    docker_smoke = sub.add_parser("docker-smoke", help="plan or probe Docker/GPU runtime readiness")
    docker_smoke.add_argument("--probe", action="store_true", default=False)
    docker_smoke.add_argument("--image", default=None)
    docker_smoke.add_argument("--require-gpu", action="store_true", default=False)

    cache = sub.add_parser("cache", help="inspect or clean model cache")
    cache.add_argument("--cleanup", action="store_true", default=False)
    cache.add_argument("--max-total-bytes", type=int, default=None)
    cache.add_argument("--older-than-days", type=float, default=None)
    cache.add_argument("--source", default=None)
    cache.add_argument("--repo-id", default=None)
    cache.add_argument("--keep-cache-key", action="append", default=None)
    cache.add_argument("--keep-repo-id", action="append", default=None)
    cache.add_argument("--apply", action="store_true", default=False)

    return parser


def _apply_cli_overrides(config: HarnessConfig, args) -> None:
    if getattr(args, "model_download_workers", None) is not None:
        config.model_download_max_workers = max(1, args.model_download_workers)
    if getattr(args, "download_retries", None) is not None:
        config.model_download_retry_count = max(0, args.download_retries)
    if getattr(args, "download_retry_backoff", None) is not None:
        config.model_download_retry_backoff_seconds = max(0.0, args.download_retry_backoff)
    if getattr(args, "execution_backend", None) is not None:
        config.execution_backend = args.execution_backend
    if getattr(args, "docker_image", None):
        config.docker_image = args.docker_image
    if getattr(args, "docker_network", None):
        config.docker_network = args.docker_network
    if getattr(args, "docker_gpus", None):
        config.docker_gpus = args.docker_gpus
    if getattr(args, "docker_model_cache_dir", None):
        config.docker_model_cache_dir = args.docker_model_cache_dir
    if getattr(args, "agent_self_heal", False):
        config.agent_mode = "gated_actor"
        config.agent_enable_log_diagnosis = True
        config.agent_enable_repair_actions = True
        config.agent_enable_verify_planner = True
        config.agent_enable_verify = True
        config.agent_auto_resume_after_repair = True
    if getattr(args, "agent_mode", None):
        config.agent_mode = args.agent_mode
    if getattr(args, "agent_enable_verify", False):
        config.agent_enable_verify = True
    if getattr(args, "agent_verify_max_steps", None) is not None:
        config.agent_verify_max_steps = max(1, args.agent_verify_max_steps)
    if getattr(args, "env_backend", None):
        config.env_backend = args.env_backend
    if getattr(args, "prefer_mamba", False):
        config.conda_prefer_mamba = True
    # Decision gate CLI overrides
    if getattr(args, "agent_enable_plan_gate", False):
        config.agent_enable_plan_gate = True
    if getattr(args, "agent_enable_runner_gate", False):
        config.agent_enable_runner_gate = True
    if getattr(args, "agent_enable_env_gate", False):
        config.agent_enable_env_gate = True
    if getattr(args, "agent_enable_model_gate", False):
        config.agent_enable_model_gate = True
    if getattr(args, "agent_enable_repair_gate", False):
        config.agent_enable_repair_gate = True
    if getattr(args, "agent_all_decision_gates", False):
        config.agent_mode = "gated_actor"
        config.agent_enable_plan_gate = True
        config.agent_enable_runner_gate = True
        config.agent_enable_env_gate = True
        config.agent_enable_model_gate = True
        config.agent_enable_repair_gate = True
        config.agent_enable_verify = True
        config.agent_enable_repair_actions = True
        config.agent_enable_log_diagnosis = True
    # Agent runtime loop CLI overrides
    if getattr(args, "agent_runtime_loop", False):
        config.agent_enable_runtime_loop = True
        config.agent_mode = "gated_actor"
    if getattr(args, "agent_runtime_loop_max_iterations", None) is not None:
        config.agent_runtime_loop_max_iterations = max(1, args.agent_runtime_loop_max_iterations)
    if getattr(args, "agent_runtime_loop_position", None) is not None:
        config.agent_runtime_loop_position = args.agent_runtime_loop_position


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = HarnessConfig.load()
    _apply_cli_overrides(config, args)
    runner = TaskRunner(config)

    if args.command == "init":
        config.runs_path.mkdir(parents=True, exist_ok=True)
        print("initialized: %s" % config.runs_path)
        return 0

    if args.command == "deploy":
        dry_run = args.dry_run or not args.execute
        task_id = runner.deploy(
            args.repo,
            args.name,
            dry_run=dry_run,
            skip_clone=args.skip_clone,
            allow_install=args.allow_install,
            allow_start=args.allow_start,
        )
        print(task_id)
        return 0

    if args.command == "resume":
        dry_run = args.dry_run or not args.execute
        task_id = runner.resume(args.task_id, dry_run=dry_run)
        print(task_id)
        return 0

    if args.command == "status":
        print(json.dumps(runner.store.task_summary(args.task_id), ensure_ascii=False, indent=2))
        return 0

    if args.command == "report":
        state = runner.store.load_state(args.task_id)
        print(state.report_path or "")
        if state.report_path and Path(state.report_path).exists():
            print(Path(state.report_path).read_text(encoding="utf-8"))
        return 0

    if args.command == "package":
        output = Path(args.output) if args.output else Path("dist") / "packages" / ("%s.tar.gz" % args.task_id)
        result = DeploymentPackageExporter().export(
            runner.store.run_dir(args.task_id),
            output,
            include_logs=args.include_logs,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "generated" else 2

    if args.command == "repair-approve":
        approval = runner.approve_repair(args.task_id, note=args.note)
        print(json.dumps(approval, ensure_ascii=False, indent=2))
        return 0

    if args.command == "memory-promote":
        promoter = MemoryPromoter(config.memory_path, config.skills_path)
        if args.approve:
            if not args.proposal:
                print(json.dumps({"status": "failed", "error": "--proposal is required with --approve"}, ensure_ascii=False, indent=2))
                return 2
            result = promoter.approve(Path(args.proposal), reviewer=args.reviewer, note=args.note)
        elif args.apply:
            if not args.proposal:
                print(json.dumps({"status": "failed", "error": "--proposal is required with --apply"}, ensure_ascii=False, indent=2))
                return 2
            result = promoter.apply(Path(args.proposal), run_regression=not args.skip_regression)
        else:
            output_dir = Path(args.output_dir) if args.output_dir else None
            result = promoter.propose(
                min_count=max(1, args.min_count),
                stage=args.stage,
                category=args.category,
                output_dir=output_dir,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        regression = result.get("regression") if isinstance(result.get("regression"), dict) else {}
        return 0 if result.get("status") not in ("failed",) and regression.get("status") not in ("failed",) else 2

    if args.command == "llm-test":
        provider = MockLLMProvider() if args.provider == "mock" else XunfeiSparkProvider()
        result = provider.complete([Message(role="user", content=args.prompt)])
        print(result.text)
        return 0

    if args.command == "benchmark":
        output = Path(args.output) if args.output else None
        report = BenchmarkRunner().run(Path(args.manifest), output, case_ids=args.case_id)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "passed" else 2

    if args.command == "dashboard":
        if args.serve:
            server = DashboardServer().create_server(
                config.runs_path,
                host=args.host,
                port=args.port,
                benchmark_report=Path(args.benchmark_report) if args.benchmark_report else None,
            )
            host, port = server.server_address
            print(json.dumps({"status": "serving", "url": "http://%s:%s/" % (host, port)}, ensure_ascii=False, indent=2))
            server.serve_forever()
            return 0
        output = Path(args.output) if args.output else config.runs_path / "dashboard.html"
        benchmark_report = Path(args.benchmark_report) if args.benchmark_report else None
        result = DashboardGenerator().generate(config.runs_path, output, benchmark_report=benchmark_report)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "generated" else 2

    if args.command == "readiness":
        output = Path(args.output) if args.output else Path("reports") / "readiness_audit.json"
        benchmark_report = Path(args.benchmark_report) if args.benchmark_report else None
        result = ReadinessAuditor().audit(Path.cwd(), benchmark_report=benchmark_report, output_path=output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "ready_for_external_smoke" else 2

    if args.command == "agent-metrics":
        runs_dir = Path(args.runs_dir) if args.runs_dir else config.runs_path
        output = Path(args.output) if args.output else None
        result = AgentMetricsCollector().collect_many(runs_dir, output_path=output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval-compare":
        reporter = AgentComparisonReporter()
        if getattr(args, "run", False):
            result = reporter.run_eval(Path(args.output_dir))
        else:
            result = reporter.from_manifest(Path(args.manifest), Path(args.output_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval-llm-necessity":
        from auto_harness.evals.llm_necessity import LLMNecessityEvaluator
        evaluator = LLMNecessityEvaluator()
        output_path = Path(args.output) if args.output else Path("runs/evals/llm-necessity/report.json")
        result = evaluator.evaluate_manifest(Path(args.manifest), output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "completed" else 2

    if args.command == "queue":
        queue = DeploymentQueue(config.task_queue_path, runner, claim_ttl_seconds=config.queue_claim_ttl_seconds)
        if args.queue_command == "submit":
            dry_run = args.dry_run or not args.execute
            result = queue.submit(
                args.repo,
                name=args.name,
                dry_run=dry_run,
                skip_clone=args.skip_clone,
                allow_install=args.allow_install,
                allow_start=args.allow_start,
                require_gpu=args.require_gpu,
                priority=args.priority,
            )
        elif args.queue_command == "list":
            result = queue.list(status=args.status or None)
        else:
            result = queue.run_next(
                max_jobs=args.max_jobs if args.max_jobs is not None else config.queue_max_concurrent_tasks,
                gpu_slots=args.gpu_slots if args.gpu_slots is not None else config.queue_gpu_slots,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") not in ("failed",) else 2

    if args.command == "live-smoke-plan":
        plan = LiveSmokePlanner(default_execute=args.execute).plan(
            include_long_running=args.include_long_running,
            execution_backend=args.execution_backend,
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.command == "agent-live-smoke":
        output = Path(args.output) if args.output else Path("runs") / "live_smoke" / compact_timestamp()
        result = LiveAgentSmokeRunner().run(
            Path(args.repo),
            args.provider,
            execute=args.execute,
            output_dir=output,
            config=config,
            analyze_planner=not args.disable_analyze_planner,
            resume_attempts=args.resume_attempts,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "completed" else 2

    if args.command == "docker-smoke":
        result = DockerSmokeChecker().check(
            probe=args.probe,
            image=args.image or config.docker_image,
            require_gpu=args.require_gpu,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in ("planned", "passed") else 2

    if args.command == "memory-evolve":
        provider = None
        provider_name = args.provider or config.memory_evolution_provider
        if provider_name == "xunfei":
            from auto_harness.providers import XunfeiSparkProvider
            provider = XunfeiSparkProvider()
        else:
            # Use MemoryEvolutionMockProvider (not generic MockLLMProvider)
            # so memory-evolve --provider mock returns valid curator JSON
            from auto_harness.providers.memory_evolution_mock import MemoryEvolutionMockProvider
            provider = MemoryEvolutionMockProvider()

        manager = MemoryEvolutionManager(
            memory_dir=config.memory_path,
            skills_dir=config.skills_path,
            provider=provider,
        )

        if args.propose:
            output_dir = Path(args.output_dir) if args.output_dir else None
            result = manager.propose(
                min_verified_count=max(1, args.min_verified_count),
                stage=args.stage,
                category=args.category,
                output_dir=output_dir,
            )
        elif args.regression:
            if not args.candidate:
                result = {"status": "failed", "error": "--candidate is required with --regression"}
            else:
                result = manager.run_regression(Path(args.candidate))
        elif args.shadow:
            if not args.candidate or not args.run_dir:
                result = {"status": "failed", "error": "--candidate and --run-dir are required with --shadow"}
            else:
                from auto_harness.skills.shadow import ShadowSkillEvaluator
                evaluator = ShadowSkillEvaluator()
                eval_result = evaluator.evaluate_run(Path(args.run_dir), Path(args.candidate))
                shadow_result = evaluator.record(Path(args.candidate), eval_result)
                result = {"status": "ok", "evaluation": eval_result, "shadow": shadow_result}
        elif args.promote:
            if not args.candidate:
                result = {"status": "failed", "error": "--candidate is required with --promote"}
            else:
                require_shadow = not args.no_require_shadow
                result = manager.promote(Path(args.candidate), require_shadow=require_shadow)
        elif args.reject:
            if not args.candidate:
                result = {"status": "failed", "error": "--candidate is required with --reject"}
            else:
                result = manager.reject(Path(args.candidate), reason=args.reason or "operator rejected")
        else:
            result = {"status": "failed", "error": "one of --propose/--regression/--shadow/--promote/--reject is required"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        exit_code = 0
        if isinstance(result, dict):
            if result.get("status") in ("failed", "rejected", "regression_failed", "base_changed"):
                exit_code = 2
            reg = result.get("regression") if isinstance(result.get("regression"), dict) else {}
            if reg.get("status") == "failed":
                exit_code = 2
        return exit_code

    if args.command == "skill-rollback":
        manager = SkillRollbackManager()
        result = manager.rollback_candidate(Path(args.candidate))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "rolled_back" else 2

    if args.command == "skill-outcomes":
        recorder = SkillOutcomeRecorder(config.memory_path)
        result = recorder.summarize(
            skill_name=args.skill or None,
            candidate_id=args.candidate or None,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "skill-gain":
        from auto_harness.evals.skill_gain import SkillGainEvaluator
        evaluator = SkillGainEvaluator()
        output_path = Path(args.output) if args.output else None
        report = evaluator.evaluate_candidate(
            candidate_path=Path(args.candidate),
            output_path=output_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report.get("status") == "failed":
            return 2
        return 0 if report.get("gain", {}).get("improved") else 1

    if args.command == "evidence-package":
        from auto_harness.evidence import create_evidence_package
        result = create_evidence_package(
            project_root=Path(args.project_root),
            output_path=Path(args.output),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "ok" else 2

    if args.command == "cache":
        if args.cleanup:
            max_total_bytes = args.max_total_bytes
            if max_total_bytes is None:
                max_total_bytes = config.model_cache_cleanup_max_total_bytes
            older_than_days = args.older_than_days
            if older_than_days is None:
                older_than_days = config.model_cache_cleanup_older_than_days
            source = args.source if args.source is not None else config.model_cache_cleanup_source
            repo_id = args.repo_id if args.repo_id is not None else config.model_cache_cleanup_repo_id
            keep_cache_keys = args.keep_cache_key if args.keep_cache_key is not None else config.model_cache_cleanup_keep_cache_keys
            keep_repo_ids = args.keep_repo_id if args.keep_repo_id is not None else config.model_cache_cleanup_keep_repo_ids
            result = runner.model_cache.cleanup(
                max_total_bytes=max_total_bytes,
                older_than_days=older_than_days,
                dry_run=not args.apply,
                source=source,
                repo_id=repo_id,
                keep_cache_keys=keep_cache_keys,
                keep_repo_ids=keep_repo_ids,
            )
        else:
            result = {
                "root": str(config.model_cache_path),
                "entries": runner.model_cache.entries(),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
