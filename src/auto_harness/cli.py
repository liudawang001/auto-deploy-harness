import argparse
import json
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.benchmarks import BenchmarkRunner
from auto_harness.orchestrator import TaskRunner
from auto_harness.providers import Message, MockLLMProvider, XunfeiSparkProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-harness")
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

    resume = sub.add_parser("resume", help="resume an existing task")
    resume.add_argument("--task-id", required=True)
    resume.add_argument("--dry-run", action="store_true", default=False)
    resume.add_argument("--execute", action="store_true", default=False)
    resume.add_argument("--model-download-workers", type=int, default=None)
    resume.add_argument("--download-retries", type=int, default=None)
    resume.add_argument("--download-retry-backoff", type=float, default=None)

    status = sub.add_parser("status", help="show task status")
    status.add_argument("--task-id", required=True)

    report = sub.add_parser("report", help="print report path")
    report.add_argument("--task-id", required=True)

    repair_approve = sub.add_parser("repair-approve", help="approve the latest repair plan for a task")
    repair_approve.add_argument("--task-id", required=True)
    repair_approve.add_argument("--note", default="")

    llm = sub.add_parser("llm-test", help="test LLM provider")
    llm.add_argument("--provider", choices=["mock", "xunfei"], default="mock")
    llm.add_argument("--prompt", default="Return a JSON object with status ok.")

    benchmark = sub.add_parser("benchmark", help="run local benchmark fixtures")
    benchmark.add_argument("--manifest", default="tests/fixtures/benchmarks/manifest.json")
    benchmark.add_argument("--output", default="")

    cache = sub.add_parser("cache", help="inspect or clean model cache")
    cache.add_argument("--cleanup", action="store_true", default=False)
    cache.add_argument("--max-total-bytes", type=int, default=None)
    cache.add_argument("--older-than-days", type=float, default=None)
    cache.add_argument("--apply", action="store_true", default=False)

    return parser


def _apply_cli_overrides(config: HarnessConfig, args) -> None:
    if getattr(args, "model_download_workers", None) is not None:
        config.model_download_max_workers = max(1, args.model_download_workers)
    if getattr(args, "download_retries", None) is not None:
        config.model_download_retry_count = max(0, args.download_retries)
    if getattr(args, "download_retry_backoff", None) is not None:
        config.model_download_retry_backoff_seconds = max(0.0, args.download_retry_backoff)


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

    if args.command == "repair-approve":
        approval = runner.approve_repair(args.task_id, note=args.note)
        print(json.dumps(approval, ensure_ascii=False, indent=2))
        return 0

    if args.command == "llm-test":
        provider = MockLLMProvider() if args.provider == "mock" else XunfeiSparkProvider()
        result = provider.complete([Message(role="user", content=args.prompt)])
        print(result.text)
        return 0

    if args.command == "benchmark":
        output = Path(args.output) if args.output else None
        report = BenchmarkRunner().run(Path(args.manifest), output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "passed" else 2

    if args.command == "cache":
        if args.cleanup:
            max_total_bytes = args.max_total_bytes
            if max_total_bytes is None:
                max_total_bytes = config.model_cache_cleanup_max_total_bytes
            older_than_days = args.older_than_days
            if older_than_days is None:
                older_than_days = config.model_cache_cleanup_older_than_days
            result = runner.model_cache.cleanup(
                max_total_bytes=max_total_bytes,
                older_than_days=older_than_days,
                dry_run=not args.apply,
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
