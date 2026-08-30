import argparse
import json
from pathlib import Path
from typing import Any, Dict

from auto_harness.agent import AgentMetricsCollector
from auto_harness.observability.cost_profile import CostProfileCollector
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
from auto_harness.orchestrator import TaskExecutionError, TaskRunner
from auto_harness.controllers.outcomes import controller_exit_code
from auto_harness.models.base import read_json, to_plain, write_json
from auto_harness.modules import HostPreflightModule, ProjectAnalyzer, ResourcePlanner
from auto_harness.providers import (
    DEFAULT_PROVIDER_REGISTRY,
    InteractiveProviderConfigurator,
    Message,
    ProviderError,
)
from auto_harness.providers.errors import ErrorCategory
from auto_harness.queue import DeploymentQueue
from auto_harness.readiness import ReadinessAuditor
from auto_harness.runtime import DockerSmokeChecker
from auto_harness.utils.time import compact_timestamp


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _gpu_memory_utilization_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("value must be a number")
    if not (0.5 <= parsed <= 0.95):
        raise argparse.ArgumentTypeError("value must be within [0.5, 0.95]")
    return parsed


def _add_llm_runtime_arguments(parser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--context-window-tokens",
        type=_positive_int_arg,
        default=None,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int_arg,
        default=None,
    )


def _add_model_inference_arguments(parser) -> None:
    parser.add_argument(
        "--model-inference",
        action="store_true",
        default=False,
        help="enable the model preparation and inference deployment chain",
    )
    parser.add_argument("--model-runtime", choices=["vllm"], default=None)
    parser.add_argument("--model-runtime-image", default=None)
    parser.add_argument("--model-max-model-len", type=_positive_int_arg, default=None)
    parser.add_argument(
        "--model-gpu-memory-utilization",
        type=_gpu_memory_utilization_arg,
        default=None,
    )
    parser.add_argument("--model-startup-timeout", type=_positive_int_arg, default=None)
    parser.add_argument("--model-request-timeout", type=_positive_int_arg, default=None)
    parser.add_argument("--model-id-override", default=None)
    parser.add_argument("--model-revision-override", default=None)


def _add_retrieval_arguments(parser) -> None:
    parser.add_argument("--retrieval", action="store_true", default=False)
    parser.add_argument("--retrieval-mode", choices=["lexical", "dense", "hybrid"], default=None)
    parser.add_argument(
        "--retrieval-embedding-provider",
        choices=["disabled", "fake", "openai_compatible"],
        default=None,
    )
    parser.add_argument("--retrieval-top-k", type=_positive_int_arg, default=None)
    parser.add_argument("--retrieval-max-context-tokens", type=_positive_int_arg, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-deploy-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="install bundled config and skills without overwriting local files")
    init.add_argument("--force", action="store_true", default=False)

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
    deploy.add_argument("--agent-provider", default=None)
    deploy.add_argument("--interactive-provider", action="store_true", default=False, help="securely prompt for a temporary custom LLM endpoint and API key")
    deploy.add_argument("--agent-enable-verify", action="store_true", default=False)
    deploy.add_argument("--agent-verify-max-steps", type=int, default=None)
    deploy.add_argument("--env-backend", choices=["auto", "venv", "conda", "mamba", "micromamba"], default=None)
    deploy.add_argument("--require-gpu", action="store_true", default=False)
    deploy.add_argument("--min-gpu-memory-mb", type=int, default=None)
    deploy.add_argument("--allow-cpu-fallback", action="store_true", default=False)
    deploy.add_argument("--prefer-mamba", action="store_true", default=False)
    deploy.add_argument("--docker-image", default=None)
    deploy.add_argument("--docker-network", default=None)
    deploy.add_argument("--docker-gpus", default=None)
    deploy.add_argument("--docker-model-cache-dir", default=None)
    _add_model_inference_arguments(deploy)
    deploy.add_argument("--agent-enable-plan-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-runner-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-env-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-model-gate", action="store_true", default=False)
    deploy.add_argument("--agent-enable-repair-gate", action="store_true", default=False)
    deploy.add_argument("--agent-all-decision-gates", action="store_true", default=False)
    deploy.add_argument("--agent-runtime-loop", action="store_true", default=False)
    deploy.add_argument("--agent-runtime-loop-max-iterations", type=int, default=None)
    deploy.add_argument("--agent-runtime-loop-position", choices=["primary", "post_pipeline"], default=None)
    deploy.add_argument("--agent-plan-first", action="store_true", default=False)
    deploy.add_argument("--agent-plan-first-provider", default=None)
    deploy.add_argument("--agent-plan-first-mode", choices=["planner", "gated_actor"], default=None)
    deploy.add_argument("--agent-plan-first-max-replans", type=int, default=None)
    deploy.add_argument("--controller", choices=["legacy", "langgraph"], default=None)
    _add_llm_runtime_arguments(deploy)
    _add_retrieval_arguments(deploy)

    resume = sub.add_parser("resume", help="resume an existing task")
    resume.add_argument("--task-id", required=True)
    resume.add_argument("--dry-run", action="store_true", default=False)
    resume.add_argument("--execute", action="store_true", default=False)
    resume.add_argument("--model-download-workers", type=int, default=None)
    resume.add_argument("--download-retries", type=int, default=None)
    resume.add_argument("--download-retry-backoff", type=float, default=None)
    resume.add_argument("--execution-backend", choices=["local", "docker"], default=None)
    resume.add_argument("--agent-self-heal", action="store_true", default=False)
    resume.add_argument("--agent-provider", default=None)
    resume.add_argument("--interactive-provider", action="store_true", default=False, help="securely prompt for a temporary custom LLM endpoint and API key")
    resume.add_argument("--env-backend", choices=["auto", "venv", "conda", "mamba", "micromamba"], default=None)
    resume.add_argument("--require-gpu", action="store_true", default=False)
    resume.add_argument("--min-gpu-memory-mb", type=int, default=None)
    resume.add_argument("--allow-cpu-fallback", action="store_true", default=False)
    resume.add_argument("--prefer-mamba", action="store_true", default=False)
    resume.add_argument("--docker-image", default=None)
    resume.add_argument("--docker-network", default=None)
    resume.add_argument("--docker-gpus", default=None)
    resume.add_argument("--docker-model-cache-dir", default=None)
    _add_model_inference_arguments(resume)
    resume.add_argument("--controller", choices=["legacy", "langgraph"], default=None)
    _add_llm_runtime_arguments(resume)
    _add_retrieval_arguments(resume)

    status = sub.add_parser("status", help="show task status")
    status.add_argument("--task-id", required=True)

    report = sub.add_parser("report", help="print report path")
    report.add_argument("--task-id", required=True)

    preflight = sub.add_parser("preflight", help="probe GPU and Conda compatibility without mutation")
    preflight.add_argument("--repo", required=True)
    preflight.add_argument("--env-backend", choices=["auto", "venv", "conda", "mamba", "micromamba"], default=None)
    preflight.add_argument("--require-gpu", action="store_true", default=False)
    preflight.add_argument("--min-gpu-memory-mb", type=int, default=None)
    preflight.add_argument("--allow-cpu-fallback", action="store_true", default=False)
    preflight.add_argument("--output", default="")

    package = sub.add_parser("package", help="export a deployment audit package for one task")
    package.add_argument("--task-id", required=True)
    package.add_argument("--output", default="")
    package.add_argument("--include-logs", action="store_true", default=False)

    repair_approve = sub.add_parser("repair-approve", help="approve the latest repair plan for a task")
    repair_approve.add_argument("--task-id", required=True)
    repair_approve.add_argument("--note", default="")

    memory_promote = sub.add_parser("memory-promote", help="legacy read-only proposal workflow; use memory-evolve for skill mutation")
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
    memory_evolve.add_argument("--approve", action="store_true", default=False)
    memory_evolve.add_argument("--reject", action="store_true", default=False)
    memory_evolve.add_argument("--candidate", default="")
    memory_evolve.add_argument("--min-verified-count", type=int, default=3)
    memory_evolve.add_argument("--stage", default=None)
    memory_evolve.add_argument("--category", default=None)
    memory_evolve.add_argument("--output-dir", default="")
    memory_evolve.add_argument("--provider", default=None)
    memory_evolve.add_argument("--interactive-provider", action="store_true", default=False, help="securely prompt for a temporary custom LLM endpoint and API key")
    memory_evolve.add_argument("--run-dir", default="")
    memory_evolve.add_argument("--no-require-shadow", action="store_true", default=False)
    memory_evolve.add_argument("--reason", default="")
    memory_evolve.add_argument("--reviewer", default="operator")
    memory_evolve.add_argument("--note", default="")
    _add_llm_runtime_arguments(memory_evolve)

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
    evidence_package.add_argument("--task-id", default="", help="export one run with a manifest instead of project-level evidence")
    evidence_package.add_argument("--run-dir", default="", help="explicit run directory for --task-id")

    llm = sub.add_parser("llm-test", help="test LLM provider")
    llm.add_argument("--provider", default=None)
    llm.add_argument("--interactive-provider", action="store_true", default=False, help="securely prompt for a temporary custom LLM endpoint and API key")
    llm.add_argument("--prompt", default="Return a JSON object with status ok.")
    _add_llm_runtime_arguments(llm)

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
    readiness.add_argument("--run-local-gates", action="store_true", default=False)

    agent_metrics = sub.add_parser("agent-metrics", help="collect agent metrics from local runs")
    agent_metrics.add_argument("--runs-dir", default="")
    agent_metrics.add_argument("--output", default="")

    cost_profile = sub.add_parser("cost-profile", help="aggregate token cost, latency, stage duration and success rate from local runs")
    cost_profile.add_argument("--runs-dir", default="")
    cost_profile.add_argument("--output", default="")
    cost_profile.add_argument("--task-id", default="", help="profile a single run and write reports/cost_profile.json into it")

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
    _add_llm_runtime_arguments(queue_submit)
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
    agent_live_smoke.add_argument("--provider", default=None)
    agent_live_smoke.add_argument("--interactive-provider", action="store_true", default=False, help="securely prompt for a temporary custom LLM endpoint and API key")
    agent_live_smoke.add_argument("--execute", action="store_true", default=False)
    agent_live_smoke.add_argument("--output", default="")
    agent_live_smoke.add_argument("--disable-analyze-planner", action="store_true", default=False)
    agent_live_smoke.add_argument("--resume-attempts", type=int, default=1)
    _add_llm_runtime_arguments(agent_live_smoke)

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

    approval_show = sub.add_parser("approval-show", help="show pending approval for a task")
    approval_show.add_argument("--task-id", required=True)

    approval_resolve = sub.add_parser("approval-resolve", help="approve or reject a pending approval")
    approval_resolve.add_argument("--task-id", required=True)
    approval_resolve.add_argument("--decision", choices=["approve", "reject"], required=True)
    approval_resolve.add_argument("--note", default="")
    approval_resolve.add_argument("--reviewer", default="operator")
    approval_resolve.add_argument("--execute", action="store_true", default=False)
    approval_resolve.add_argument("--approval-id", default="")

    return parser


def _apply_cli_overrides(config: HarnessConfig, args) -> None:
    retrieval = dict(config.retrieval)
    if getattr(args, "retrieval", False):
        retrieval["enabled"] = True
    if getattr(args, "retrieval_mode", None):
        retrieval["mode"] = args.retrieval_mode
        retrieval["dense_enabled"] = args.retrieval_mode in {"dense", "hybrid"}
    if getattr(args, "retrieval_embedding_provider", None):
        retrieval["embedding_provider"] = args.retrieval_embedding_provider
    if getattr(args, "retrieval_top_k", None) is not None:
        retrieval["default_top_k"] = min(args.retrieval_top_k, retrieval["max_top_k"])
    if getattr(args, "retrieval_max_context_tokens", None) is not None:
        retrieval["max_context_tokens"] = min(args.retrieval_max_context_tokens, 32000)
    config.retrieval = retrieval
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
    if getattr(args, "model_inference", False):
        config.model_inference_enabled = True
    if getattr(args, "model_runtime", None):
        config.model_runtime = args.model_runtime
    if getattr(args, "model_runtime_image", None):
        config.model_runtime_image = args.model_runtime_image
    if getattr(args, "model_max_model_len", None) is not None:
        config.model_runtime_max_model_len = args.model_max_model_len
    if getattr(args, "model_gpu_memory_utilization", None) is not None:
        config.model_runtime_gpu_memory_utilization = args.model_gpu_memory_utilization
    if getattr(args, "model_startup_timeout", None) is not None:
        config.model_runtime_startup_timeout_seconds = args.model_startup_timeout
    if getattr(args, "model_request_timeout", None) is not None:
        config.model_runtime_request_timeout_seconds = args.model_request_timeout
    if getattr(args, "model_id_override", None):
        config.model_id_override = args.model_id_override
    if getattr(args, "model_revision_override", None):
        config.model_revision_override = args.model_revision_override
    if getattr(args, "agent_self_heal", False):
        config.agent_mode = "gated_actor"
        config.agent_enable_log_diagnosis = True
        config.agent_enable_repair_actions = True
        config.agent_enable_verify_planner = True
        config.agent_enable_verify = True
        config.agent_auto_resume_after_repair = True
    if getattr(args, "agent_mode", None):
        config.agent_mode = args.agent_mode
    if getattr(args, "agent_provider", None):
        config.agent_provider = args.agent_provider
    if getattr(args, "agent_enable_verify", False):
        config.agent_enable_verify = True
    if getattr(args, "agent_verify_max_steps", None) is not None:
        config.agent_verify_max_steps = max(1, args.agent_verify_max_steps)
    if getattr(args, "env_backend", None):
        config.env_backend = args.env_backend
    if getattr(args, "prefer_mamba", False):
        config.conda_prefer_mamba = True
    if getattr(args, "require_gpu", False):
        config.preflight_require_gpu = True
    if getattr(args, "min_gpu_memory_mb", None) is not None:
        config.min_gpu_memory_mb = max(0, args.min_gpu_memory_mb)
    if getattr(args, "allow_cpu_fallback", False):
        config.conda_allow_cpu_fallback = True
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
    # Plan-first CLI overrides
    if getattr(args, "agent_plan_first", False):
        config.agent_plan_first = True
        config.agent_mode = "gated_actor"
    if getattr(args, "agent_plan_first_provider", None):
        config.agent_plan_first_provider = args.agent_plan_first_provider
    if getattr(args, "agent_plan_first_mode", None):
        config.agent_plan_first_mode = args.agent_plan_first_mode
    if getattr(args, "agent_plan_first_max_replans", None) is not None:
        config.agent_plan_first_max_replans = max(0, args.agent_plan_first_max_replans)
    # Controller selection is handled at deploy/resume time, not stored in config


def _interactive_default_name(args, config: HarnessConfig) -> str:
    if args.command in {"llm-test", "memory-evolve", "agent-live-smoke"}:
        current = getattr(args, "provider", "") or config.agent_provider
    else:
        current = getattr(args, "agent_provider", "")
    if not current or current in {"mock", "xunfei"}:
        return "custom"
    return str(current)


def _apply_interactive_session(
    config: HarnessConfig,
    args,
    provider_name: str,
) -> None:
    if args.command in {"deploy", "resume"}:
        config.agent_provider = provider_name
        config.agent_plan_first_provider = provider_name
        args.agent_provider = provider_name
        if hasattr(args, "agent_plan_first_provider"):
            args.agent_plan_first_provider = provider_name
        return
    if args.command == "memory-evolve":
        config.memory_evolution_provider = provider_name
        args.provider = provider_name
        return
    if args.command == "agent-live-smoke":
        config.agent_provider = provider_name
        args.provider = provider_name
        return
    args.provider = provider_name


def _providers_for_command(config: HarnessConfig, args):
    if args.command in {"deploy", "resume"}:
        return [
            (config.agent_provider, "agent"),
            (config.agent_plan_first_provider, "plan_first"),
        ]
    if args.command == "llm-test":
        return [(args.provider or config.agent_provider, "llm_test")]
    if args.command == "memory-evolve":
        if getattr(args, "propose", False):
            return [
                (
                    args.provider or config.memory_evolution_provider,
                    "memory_evolution",
                )
            ]
        return []
    if args.command == "agent-live-smoke":
        return [(args.provider or config.agent_provider, "live_smoke")]
    return []


def _deployment_exit_code(runner, task_id: str) -> int:
    """Read the persisted outcome while keeping test doubles compatible."""
    try:
        run_dir = runner.store.run_dir(task_id)
        if isinstance(run_dir, (str, Path)):
            result_path = Path(run_dir) / "reports" / "controller_result.json"
            if result_path.exists():
                result = read_json(result_path)
                return controller_exit_code(result.get("status", ""))
        state = runner.store.load_state(task_id)
        status = getattr(state, "status", None)
        if isinstance(status, str) and status:
            return controller_exit_code(status)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = HarnessConfig.load()
    _apply_cli_overrides(config, args)

    if args.command == "init":
        from auto_harness.resources.installer import initialize_workspace
        result = initialize_workspace(Path.cwd(), force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "preflight":
        repo_dir = Path(args.repo).resolve()
        if not repo_dir.is_dir():
            print(json.dumps({"status": "failed", "error": "repo directory does not exist"}, indent=2))
            return 2
        analysis_result = ProjectAnalyzer(use_agent=False).analyze(repo_dir)
        resource_result = ResourcePlanner().plan(repo_dir, analysis_result.data)
        result = HostPreflightModule(config).run(
            repo_dir,
            analysis_result.data,
            resource_result.data,
            allow_mutation=False,
        )
        payload = to_plain(result)
        if args.output:
            write_json(Path(args.output), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.status == "passed" else 2

    if getattr(args, "interactive_provider", False):
        try:
            session = InteractiveProviderConfigurator(
                DEFAULT_PROVIDER_REGISTRY
            ).configure(
                config=config,
                default_name=_interactive_default_name(args, config),
            )
        except (EOFError, KeyboardInterrupt, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": str(exc) or "interactive input cancelled",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        _apply_interactive_session(config, args, session.provider_name)
        print(
            json.dumps(
                {
                    "status": "provider_configured",
                    "provider": session.safe_summary(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    # Resolve effective providers for this command
    command_providers = _providers_for_command(config, args)

    # Check mixed-provider + uniform override conflict (23.2)
    uniform_override_present = any(
        getattr(args, key, None) is not None
        for key in ("model", "context_window_tokens", "max_output_tokens")
    )
    if uniform_override_present and len(command_providers) > 1:
        unique_providers = {p for p, _ in command_providers}
        if len(unique_providers) > 1:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": "uniform LLM overrides require a single effective provider",
                        "providers": sorted(unique_providers),
                        "user_action": "remove the uniform override or configure each provider in provider_configs",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2

    # Apply LLM runtime overrides (after interactive config for correct priority)
    from auto_harness.providers.settings import set_runtime_overrides
    llm_model = getattr(args, "model", None)
    llm_ctx = getattr(args, "context_window_tokens", None)
    llm_max = getattr(args, "max_output_tokens", None)
    if uniform_override_present:
        provider_names = [p for p, _ in command_providers]
        try:
            set_runtime_overrides(
                config,
                provider_names,
                model=llm_model,
                context_window_tokens=llm_ctx,
                max_output_tokens=llm_max,
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {"status": "failed", "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2

    # Sync Context Governance when all purposes use one effective provider.
    # deploy/resume normally contain two entries (agent + plan_first), so the
    # number of unique provider names — not tuple count — is authoritative.
    unique_command_providers = {
        DEFAULT_PROVIDER_REGISTRY.normalize_name(provider_name)
        for provider_name, _ in command_providers
    }
    if len(unique_command_providers) == 1:
        from auto_harness.context.capabilities import _positive_int as _cap_pos_int
        provider_name, purpose = command_providers[0]
        try:
            temp_provider = DEFAULT_PROVIDER_REGISTRY.create(
                provider_name,
                config=config,
                purpose=purpose,
            )
            effective_ctx = _cap_pos_int(getattr(temp_provider, "context_window_tokens", None))
            effective_max = _cap_pos_int(getattr(temp_provider, "max_tokens", None))
        except Exception:
            effective_ctx = None
            effective_max = None
        if effective_ctx is not None:
            config.agent_context_window_tokens = effective_ctx
        if effective_max is not None:
            config.agent_context_reserved_output_tokens = effective_max

    try:
        for provider_name, purpose in command_providers:
            provider = DEFAULT_PROVIDER_REGISTRY.create(
                provider_name,
                config=config,
                purpose=purpose,
            )
            missing_checker = getattr(provider, "missing_configuration", None)
            missing = list(missing_checker()) if callable(missing_checker) else []
            if missing:
                raise ProviderError(
                    "%s provider configuration is incomplete" % provider_name,
                    provider_name=str(provider_name),
                    category=ErrorCategory.CONFIGURATION_ERROR,
                    safe_detail="missing: %s" % ", ".join(missing),
                )
    except (TypeError, ValueError, ProviderError) as exc:
        detail = exc.to_dict() if isinstance(exc, ProviderError) else None
        error_payload: Dict[str, Any] = {
            "status": "failed",
            "error": str(exc),
            "provider_error": detail,
        }
        # Add user_action for DeepSeek missing key
        if isinstance(exc, ProviderError) and exc.category == ErrorCategory.CONFIGURATION_ERROR:
            safe = (detail or {}).get("safe_detail", "")
            if "deepseek" in str(exc.provider_name).lower() and "DEEPSEEK_API_KEY" in safe:
                error_payload["user_action"] = (
                    '请先执行：export DEEPSEEK_API_KEY="你的 API Key"'
                )
        print(
            json.dumps(error_payload, ensure_ascii=False, indent=2)
        )
        return 2

    runner = TaskRunner(config)

    if args.command == "deploy":
        dry_run = args.dry_run or not args.execute
        controller = getattr(args, "controller", None)
        try:
            task_id = runner.deploy(
                args.repo,
                args.name,
                dry_run=dry_run,
                skip_clone=args.skip_clone,
                allow_install=args.allow_install,
                allow_start=args.allow_start,
                controller=controller,
            )
        except TaskExecutionError as exc:
            print(json.dumps({
                "task_id": exc.task_id,
                "status": "failed",
                "stop_reason": "internal_controller_error",
                "error_type": exc.cause_type,
                "exit_code": 3,
            }, ensure_ascii=False))
            return 3
        except KeyboardInterrupt:
            return 130
        print(task_id)
        return _deployment_exit_code(runner, task_id)

    if args.command == "resume":
        dry_run = args.dry_run or not args.execute
        controller = getattr(args, "controller", None)
        try:
            task_id = runner.resume(
                args.task_id, dry_run=dry_run, controller=controller
            )
        except TaskExecutionError as exc:
            print(json.dumps({
                "task_id": exc.task_id,
                "status": "failed",
                "stop_reason": "internal_controller_error",
                "error_type": exc.cause_type,
                "exit_code": 3,
            }, ensure_ascii=False))
            return 3
        except KeyboardInterrupt:
            return 130
        print(task_id)
        return _deployment_exit_code(runner, task_id)

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
        provider = DEFAULT_PROVIDER_REGISTRY.create(
            args.provider or config.agent_provider,
            config=config,
            purpose="llm_test",
        )
        try:
            result = provider.complete([Message(role="user", content=args.prompt)])
        except ProviderError as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "provider_error": exc.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(result.text)
        return 0

    if args.command == "benchmark":
        output = Path(args.output) if args.output else None
        report = BenchmarkRunner().run(Path(args.manifest), output, case_ids=args.case_id)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return {"passed": 0, "partial": 1, "failed": 1}.get(
            report.get("status"), 3
        )

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
        if args.run_local_gates:
            from auto_harness.release_gates import run_local_gates
            gate_result = run_local_gates(Path.cwd())
            if gate_result.get("status") != "passed":
                print(json.dumps(gate_result, ensure_ascii=False, indent=2))
                return 2
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

    if args.command == "cost-profile":
        collector = CostProfileCollector(config.cost_profile)
        runs_dir = Path(args.runs_dir) if args.runs_dir else config.runs_path
        if args.task_id:
            run_dir = runs_dir / args.task_id
            if not run_dir.exists():
                print(json.dumps({"error": "run dir not found: %s" % run_dir}, ensure_ascii=False))
                return 2
            result = collector.collect(run_dir)
            write_json(run_dir / "reports" / "cost_profile.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        output = Path(args.output) if args.output else Path("reports") / "cost_profile.json"
        result = collector.collect_many(runs_dir, output_path=output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("run_count") else 2

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
            # Build a resolved, non-sensitive LLM snapshot for reproducible
            # execution by a later queue worker. API keys are never included.
            model = getattr(args, "model", None)
            ctx = getattr(args, "context_window_tokens", None)
            max_out = getattr(args, "max_output_tokens", None)
            queue_provider_names = {
                DEFAULT_PROVIDER_REGISTRY.normalize_name(config.agent_provider),
                DEFAULT_PROVIDER_REGISTRY.normalize_name(
                    config.agent_plan_first_provider
                ),
            }
            if (model is not None or ctx is not None or max_out is not None) and len(
                queue_provider_names
            ) > 1:
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "error": "uniform LLM overrides require a single effective provider",
                            "providers": sorted(queue_provider_names),
                            "user_action": "remove the uniform override or configure each provider in provider_configs",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 2

            if len(queue_provider_names) == 1:
                try:
                    set_runtime_overrides(
                        config,
                        queue_provider_names,
                        model=model,
                        context_window_tokens=ctx,
                        max_output_tokens=max_out,
                    )
                    snapshot_provider = DEFAULT_PROVIDER_REGISTRY.create(
                        config.agent_provider,
                        config=config,
                        purpose="agent",
                    )
                except (TypeError, ValueError, ProviderError) as exc:
                    print(
                        json.dumps(
                            {"status": "failed", "error": str(exc)},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    return 2
                snapshot_model = getattr(snapshot_provider, "model", "") or None
                snapshot_ctx = getattr(
                    snapshot_provider, "context_window_tokens", None
                )
                snapshot_max = getattr(snapshot_provider, "max_tokens", None)
            else:
                snapshot_model = None
                snapshot_ctx = None
                snapshot_max = None

            llm_snapshot = {
                "agent_provider": config.agent_provider,
                "plan_first_provider": config.agent_plan_first_provider,
                "model": snapshot_model,
                "context_window_tokens": snapshot_ctx,
                "max_output_tokens": snapshot_max,
                "agent_repo_context_mode": config.agent_repo_context_mode,
                "agent_repo_inventory_budget_tokens": config.agent_repo_inventory_budget_tokens,
                "agent_repo_core_budget_tokens": config.agent_repo_core_budget_tokens,
                "agent_repo_observation_budget_tokens": config.agent_repo_observation_budget_tokens,
                "agent_repo_max_observation_rounds": config.agent_repo_max_observation_rounds,
                "agent_repo_max_requests_per_round": config.agent_repo_max_requests_per_round,
                "agent_repo_max_observed_files": config.agent_repo_max_observed_files,
                "agent_repo_max_chars_per_read": config.agent_repo_max_chars_per_read,
                "agent_repo_max_lines_per_read": config.agent_repo_max_lines_per_read,
                "agent_repo_search_max_results": config.agent_repo_search_max_results,
                "agent_repo_search_max_files": config.agent_repo_search_max_files,
                "agent_repo_search_max_bytes": config.agent_repo_search_max_bytes,
                "agent_repo_tree_max_entries": config.agent_repo_tree_max_entries,
            }
            result = queue.submit(
                args.repo,
                name=args.name,
                dry_run=dry_run,
                skip_clone=args.skip_clone,
                allow_install=args.allow_install,
                allow_start=args.allow_start,
                require_gpu=args.require_gpu,
                priority=args.priority,
                llm=llm_snapshot,
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
            args.provider or config.agent_provider,
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
        # Only --propose requires an LLM Provider (Section 23.4)
        provider = None
        if args.propose:
            provider_name = args.provider or config.memory_evolution_provider
            provider = DEFAULT_PROVIDER_REGISTRY.create(
                provider_name,
                config=config,
                purpose="memory_evolution",
            )

        manager = MemoryEvolutionManager(
            memory_dir=config.memory_path,
            skills_dir=config.skills_path,
            provider=provider,
            config=config,
        )

        if args.propose:
            output_dir = Path(args.output_dir) if args.output_dir else None
            result = manager.propose(
                min_verified_count=max(1, args.min_verified_count),
                stage=args.stage,
                category=args.category,
                output_dir=output_dir,
            )
        elif args.approve:
            if not args.candidate:
                result = {"status": "failed", "error": "--candidate is required with --approve"}
            else:
                result = manager.approve(Path(args.candidate), reviewer=args.reviewer, note=args.note)
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
                result = {
                    "status": "blocked" if shadow_result.get("status") in ("blocked", "failed") else "ok",
                    "evaluation": eval_result,
                    "shadow": shadow_result,
                }
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
            result = {"status": "failed", "error": "one of --propose/--approve/--regression/--shadow/--promote/--reject is required"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        exit_code = 0
        if isinstance(result, dict):
            if result.get("status") in ("failed", "rejected", "regression_failed", "base_changed", "approval_required", "blocked"):
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
        if args.task_id:
            from auto_harness.evidence import EvidenceExporter
            project_root = Path(args.project_root).resolve()
            configured_runs = config.runs_path
            default_run_dir = configured_runs if configured_runs.is_absolute() else project_root / configured_runs
            run_dir = Path(args.run_dir) if args.run_dir else default_run_dir / args.task_id
            result = EvidenceExporter(project_root=project_root).export(
                run_dir=run_dir,
                task_id=args.task_id,
                output_path=Path(args.output),
            )
        else:
            from auto_harness.evidence import create_evidence_package
            result = create_evidence_package(
                project_root=Path(args.project_root),
                output_path=Path(args.output),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in ("ok", "complete") else 2

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

    if args.command == "approval-show":
        from auto_harness.graph.approval import ApprovalStore
        from auto_harness.state.store import StateStore
        store = StateStore(config.runs_path)
        run_dir = store.run_dir(args.task_id)
        # Find pending approvals
        approvals_dir = Path(run_dir) / "approvals"
        results = []
        if approvals_dir.exists():
            for f in sorted(approvals_dir.glob("*.json")):
                try:
                    record = json.loads(f.read_text(encoding="utf-8"))
                    if record.get("status") == "pending":
                        results.append(record)
                except (OSError, ValueError):
                    pass
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.command == "approval-resolve":
        from auto_harness.graph.approval import ApprovalStore
        from auto_harness.state.store import StateStore
        from auto_harness.utils.time import utc_now_iso
        store = StateStore(config.runs_path)
        run_dir = store.run_dir(args.task_id)
        # Find the pending approval(s)
        approvals_dir = Path(run_dir) / "approvals"
        pending_approvals = []
        if approvals_dir.exists():
            for f in sorted(approvals_dir.glob("*.json")):
                try:
                    record = json.loads(f.read_text(encoding="utf-8"))
                    if record.get("status") == "pending":
                        pending_approvals.append(record)
                except (OSError, ValueError):
                    pass

        if not pending_approvals:
            print(json.dumps({"status": "no_pending_approval"}, ensure_ascii=False, indent=2))
            return 2

        if len(pending_approvals) > 1 and not args.approval_id:
            print(json.dumps({
                "status": "error",
                "reason": "multiple_pending_approvals",
                "pending_count": len(pending_approvals),
                "message": "specify --approval-id to disambiguate",
            }, ensure_ascii=False, indent=2))
            return 2

        # Select the right approval record
        if args.approval_id:
            target = None
            for record in pending_approvals:
                if record.get("request", {}).get("approval_id") == args.approval_id:
                    target = record
                    break
            if not target:
                print(json.dumps({"status": "error", "reason": "approval_id_not_found"}, ensure_ascii=False, indent=2))
                return 2
        else:
            target = pending_approvals[0]

        request = target.get("request", {})
        approval_id = request.get("approval_id", "")
        if not approval_id:
            print(json.dumps({"status": "error", "reason": "approval_id_missing"}, ensure_ascii=False, indent=2))
            return 2

        # Build the decision to pass to runner.resume
        decision = {
            "schema_version": 1,
            "approval_id": approval_id,
            "operation_id": request.get("operation_id", ""),
            "decision": args.decision,
            "reviewer": args.reviewer,
            "note": args.note,
            "request_hash": request.get("request_hash", ""),
            "resolved_at": utc_now_iso(),
        }

        # Resume the graph with the approval decision
        task_id = runner.resume(
            args.task_id,
            dry_run=not args.execute,
            controller=None,
            resume_input=decision,
        )

        print(json.dumps({
            "status": "resume_requested",
            "task_id": task_id,
            "decision": args.decision,
        }, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
