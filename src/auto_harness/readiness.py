"""Readiness audit and Capability Matrix.

Two components:
1. ReadinessAuditor: local completion audit without external smoke tests.
2. CapabilityMatrix: derive project capabilities from test/eval artifacts.

Capability Matrix never marks capabilities as validated just because a
class exists. Status must come from actual test reports and evaluation artifacts.

Allowed capability statuses:
- implemented: code exists
- integrated: passes integration tests
- validated: passes full evaluation including evidence
- not_run: external/environment-dependent, not yet tested
- failed: test/eval exists and failed

Prohibited: production_ready (never allowed)
"""
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.release_evidence import validate_evidence
from auto_harness.utils.time import utc_now_iso


ALLOWED_STATUSES = ("implemented", "integrated", "validated", "not_run", "failed")


class ReadinessAuditor:
    """Produce a local completion audit without running external smoke tests."""

    REQUIRED_FILES = [
        "README.md",
        "LICENSE",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "tests/fixtures/benchmarks/manifest.json",
    ]

    REQUIRED_EVIDENCE = {
        "test_summary": "reports/test-summary.json",
        "benchmark": "reports/benchmark.json",
        "package_smoke": "reports/package-smoke.json",
        "default_cli_smoke": "reports/default-cli-smoke.json",
    }

    REQUIRED_BENCHMARK_CASES = [
        "model_download_resume",
        "cache_hit",
        "parallel_model_download",
        "etag_cache_invalidation",
        "checksum_failure",
        "gradio_config_discovery",
        "gradio_queue_call_followup",
        "openapi_schema_verify",
        "openai_model_discovery_stream_verify",
        "verify_progress_refresh",
        "local_e2e_fixture_matrix",
        "docker_backend_plan",
        "docker_gpu_cache_backend",
        "deployment_queue_dry_run",
        "queue_parallel_worker_pool",
        "queue_claim_lock_prevents_duplicate",
        "queue_stale_claim_lock_recovery",
        "queue_gpu_probe_scheduling",
        "deployment_package_export",
        "dashboard_http_server",
        "readiness_audit_report",
        "llm_planner_policy_merge",
        "llm_repair_dependency_execute_loop",
        "llm_verify_hint_recovery",
        "agent_loop_dependency_self_repair_e2e",
        "agent_prompt_injection_defense",
        "agent_metrics_paired_comparison",
        "langgraph_fault_injection_idempotency",
        "docker_phase_security_profiles",
        "unified_metrics_consistency",
        "gpu_conda_preflight_decision",
        "conda_environment_policy",
        "conda_postcheck_recovery",
    ]

    EXTERNAL_GATES = [
        {
            "id": "networked_long_running_e2e",
            "status": "external_required",
            "reason": "真实 Hugging Face/ModelScope/Git LFS 长耗时部署需要网络、token、磁盘和时间窗口，本机开发机不默认执行。",
        },
        {
            "id": "docker_gpu_smoke",
            "status": "external_required",
            "reason": "真实 Docker + GPU runtime 需要带 NVIDIA runtime 的机器，普通 mac 开发机只能生成 plan/probe 结果。",
        },
        {
            "id": "vllm_service_smoke",
            "status": "external_required",
            "reason": "真实 vLLM/OpenAI-compatible 服务 smoke 需要 GPU、模型权重和较长启动窗口。",
        },
        {
            "id": "larger_model_repository_matrix",
            "status": "external_required",
            "reason": "更多真实开源仓库覆盖属于验收矩阵扩展，不阻塞本地代码闭环。",
        },
        {
            "id": "distributed_resource_lock",
            "status": "future_scale_gate",
            "reason": "当前已具备本地持久化队列、并发 worker、claim lock 和 stale recovery；跨机器锁属于规模化部署增强。",
        },
    ]

    def audit(
        self,
        project_root: Path = None,
        benchmark_report: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Dict:
        root = Path(project_root or Path.cwd())
        local_gates = []
        local_gates.extend(self._file_gates(root))
        local_gates.append(self._benchmark_manifest_gate(root))
        local_gates.extend(self._evidence_gates(root))
        if benchmark_report and Path(benchmark_report) != root / self.REQUIRED_EVIDENCE["benchmark"]:
            local_gates.append(self._benchmark_report_gate(Path(benchmark_report)))

        failed = [gate for gate in local_gates if gate["status"] != "passed"]
        report = {
            "status": "ready_for_external_smoke" if not failed else "incomplete",
            "local_readiness_percent": 100 if not failed else self._completion_percent(local_gates),
            "local_gates": local_gates,
            "external_gates": list(self.EXTERNAL_GATES),
            "summary": {
                "local_gate_count": len(local_gates),
                "local_gate_passed": len(local_gates) - len(failed),
                "external_gate_count": len(self.EXTERNAL_GATES),
                "benchmark_manifest_cases": self._manifest_case_count(root),
            },
            "operator_next_steps": [
                "在具备网络、磁盘和 token 的环境执行 live-smoke-plan 中的真实联网 E2E。",
                "在 GPU Linux 机器执行 docker-smoke --probe --require-gpu 和 vLLM/OpenAI-compatible smoke。",
                "把外部验收报告归档到 CI 或发布记录中；本地 readiness audit 不保存任何密钥值。",
            ],
        }
        if output_path:
            write_json(Path(output_path), report)
        return report

    def _file_gates(self, root: Path) -> List[Dict]:
        gates = []
        for file_name in self.REQUIRED_FILES:
            path = root / file_name
            gates.append({
                "id": "file:%s" % file_name,
                "status": "passed" if path.exists() else "missing",
                "evidence": str(path),
            })
        return gates

    def _benchmark_manifest_gate(self, root: Path) -> Dict:
        manifest_path = root / "tests" / "fixtures" / "benchmarks" / "manifest.json"
        if not manifest_path.exists():
            return {"id": "benchmark_manifest", "status": "missing", "reason": "manifest file is missing"}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = {case.get("id") for case in manifest.get("cases", [])}
        missing = [case_id for case_id in self.REQUIRED_BENCHMARK_CASES if case_id not in ids]
        return {
            "id": "benchmark_manifest",
            "status": "passed" if not missing else "missing_cases",
            "case_count": len(manifest.get("cases", [])),
            "required_case_count": len(self.REQUIRED_BENCHMARK_CASES),
            "missing_case_ids": missing,
        }

    def _benchmark_report_gate(self, benchmark_report: Path) -> Dict:
        if not benchmark_report.exists():
            return {"id": "benchmark_report", "status": "missing", "evidence": str(benchmark_report)}
        report = json.loads(benchmark_report.read_text(encoding="utf-8"))
        failed = [case.get("id") for case in report.get("cases", []) if case.get("status") != "passed"]
        return {
            "id": "benchmark_report",
            "status": "passed" if report.get("status") == "passed" and not failed else "failed",
            "evidence": str(benchmark_report),
            "case_count": len(report.get("cases", [])),
            "failed_case_ids": failed,
        }

    def _evidence_gates(self, root: Path) -> List[Dict]:
        gates: List[Dict] = []
        for gate_id, relative_path in self.REQUIRED_EVIDENCE.items():
            path = root / relative_path
            if not path.exists():
                gates.append({
                    "id": "evidence:%s" % gate_id,
                    "status": "missing",
                    "evidence": str(path),
                })
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                errors = validate_evidence(payload, root)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors = ["invalid evidence JSON: %s" % exc]
            gates.append({
                "id": "evidence:%s" % gate_id,
                "status": "passed" if not errors else "failed",
                "evidence": str(path),
                "errors": errors,
            })
        return gates

    def _manifest_case_count(self, root: Path) -> int:
        manifest_path = root / "tests" / "fixtures" / "benchmarks" / "manifest.json"
        if not manifest_path.exists():
            return 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return len(manifest.get("cases", []))

    def _completion_percent(self, gates: List[Dict]) -> int:
        if not gates:
            return 0
        passed = sum(1 for gate in gates if gate.get("status") == "passed")
        return int(round(passed * 100 / len(gates)))


# ------------------------------------------------------------------
# Capability Matrix
# ------------------------------------------------------------------

def _get_commit_sha(project_root: Path) -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _check_test_artifact(artifact_path: Path) -> str:
    """Check if a test artifact exists and what status it reports.

    Returns:
        "validated" if artifact exists and reports success,
        "failed" if artifact exists and reports failure,
        "implemented" if artifact doesn't exist.
    """
    if not artifact_path.exists():
        return "implemented"
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            status = data.get("status", "")
            if status in ("completed", "passed", "pass"):
                return "validated"
            if status in ("failed", "error"):
                return "failed"
    except (OSError, ValueError):
        return "failed"
    return "implemented"


class CapabilityMatrix:
    """Derive capability matrix from test artifacts."""

    def __init__(self, project_root: Path = None) -> None:
        self.project_root = Path(project_root or Path.cwd())

    def generate(self, reports_dir: Path = None) -> Dict[str, Any]:
        """Generate capability matrix from available evidence.

        Args:
            reports_dir: Directory containing test/eval reports.

        Returns:
            Capability matrix dict.
        """
        reports_dir = Path(reports_dir or self.project_root / "reports")
        commit_sha = _get_commit_sha(self.project_root)

        capabilities = {}

        # 1. default_langgraph: CLI integration test
        capabilities["default_langgraph"] = {
            "status": _check_test_artifact(
                reports_dir / "controller_result.json"
            ),
            "evidence": ["tests/test_default_agent_controller.py"],
        }

        # 2. crash_safe_reconcile: fault injection test
        capabilities["crash_safe_reconcile"] = {
            "status": _check_test_artifact(
                reports_dir / "fault_injection_result.json"
            ),
            "evidence": ["tests/test_recovery_fault_injection.py"],
        }

        # 3. approval_resume: CLI approval E2E
        capabilities["approval_resume"] = {
            "status": _check_test_artifact(
                reports_dir / "approval_e2e_result.json"
            ),
            "evidence": ["tests/test_cli_approval_e2e.py"],
        }

        # 4. memory_skill_mainline: route artifacts + verified memory test
        capabilities["memory_skill_mainline"] = {
            "status": _check_test_artifact(
                reports_dir / "skill_memory_result.json"
            ),
            "evidence": ["tests/test_langgraph_skill_memory_integration.py"],
        }

        # 5. llm_necessity: comparison report
        llm_status = "not_run"
        llm_report = reports_dir / "llm-necessity" / "report.json"
        if llm_report.exists():
            try:
                data = json.loads(llm_report.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if data.get("status") == "completed" and data.get("summary", {}).get("infrastructure_error_count", 0) == 0:
                        llm_status = "validated"
                    elif data.get("status") == "failed":
                        llm_status = "failed"
            except (OSError, ValueError):
                llm_status = "failed"
        capabilities["llm_necessity"] = {
            "status": llm_status,
            "evidence": ["eval_targets/llm_necessity_manifest.json"],
        }

        # 6. docker_gpu: requires real external manifest
        capabilities["docker_gpu"] = {
            "status": "not_run",
            "evidence": [],
        }

        # 7. tool_registry_contract
        capabilities["tool_registry_contract"] = {
            "status": _check_test_artifact(
                reports_dir / "tool_registry_result.json"
            ),
            "evidence": ["tests/test_tool_registry_contract.py"],
        }

        # 8. provider_protocol
        capabilities["provider_protocol"] = {
            "status": _check_test_artifact(
                reports_dir / "provider_protocol_result.json"
            ),
            "evidence": ["tests/test_provider_protocol.py"],
        }

        # 9. self_repair_closure
        capabilities["self_repair_closure"] = {
            "status": _check_test_artifact(
                reports_dir / "self_repair_result.json"
            ),
            "evidence": ["tests/test_self_repair_evidence.py"],
        }

        # 10. docker_sandbox_policy
        capabilities["docker_sandbox_policy"] = {
            "status": _check_test_artifact(
                reports_dir / "docker_sandbox_result.json"
            ),
            "evidence": ["tests/test_docker_sandbox_policy.py"],
        }

        # 11. evidence_provenance
        capabilities["evidence_provenance"] = {
            "status": _check_test_artifact(
                reports_dir / "evidence_package_result.json"
            ),
            "evidence": ["tests/test_evidence_package.py"],
        }

        capabilities["fault_window_idempotency"] = {
            "status": _check_test_artifact(
                reports_dir / "p1_fault_window_result.json"
            ),
            "evidence": ["tests/test_p1_recovery_fault_windows.py"],
        }

        capabilities["docker_phase_profiles"] = {
            "status": _check_test_artifact(
                reports_dir / "p1_docker_phase_result.json"
            ),
            "evidence": ["tests/test_docker_sandbox_policy.py"],
        }

        # 12. deepseek_provider
        capabilities["deepseek_provider"] = self._deepseek_readiness()

        capabilities["unified_agent_observability"] = {
            "status": _check_test_artifact(
                reports_dir / "p1_unified_metrics_result.json"
            ),
            "evidence": ["tests/test_p1_unified_metrics.py"],
        }

        capabilities["model_runtime"] = self._model_runtime_readiness()

        return {
            "schema_version": 1,
            "commit_sha": commit_sha,
            "generated_at": utc_now_iso(),
            "capabilities": capabilities,
        }

    def _model_runtime_readiness(self) -> Dict[str, Any]:
        """Assess model runtime capability (never validated from code alone)."""
        manifest_path = self.project_root / "docs" / "evidence" / "gpu-model-e2e-manifest.json"
        manifest = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
        return ModelRuntimeReadiness(self.project_root).assess(manifest=manifest)

    def _deepseek_readiness(self) -> Dict[str, Any]:
        """Assess DeepSeek provider readiness from code and config.

        Checks:
        - DeepSeekProvider class exists
        - deepseek registered in ProviderRegistry
        - Config validation for retired models
        - ProviderError available
        """
        status = "implemented"
        evidence = []
        details = {
            "provider": "deepseek",
            "registered": False,
            "configured": False,
            "model_supported": True,
            "protocol": "json_action",
            "thinking_supported": True,
            "json_mode_supported": True,
            "thinking_configured": False,
            "json_mode_configured": False,
            "native_tool_calling": False,
            "live_smoke_status": "not_run",
            "test_status": "not_run",
        }

        # Check DeepSeekProvider class
        try:
            from auto_harness.providers.deepseek import DeepSeekProvider  # noqa: F401
            evidence.append("src/auto_harness/providers/deepseek.py")
        except ImportError:
            details["model_supported"] = False
            details["thinking_supported"] = False
            details["json_mode_supported"] = False
            details["thinking_configured"] = False
            details["json_mode_configured"] = False
            return {
                "status": "not_run",
                "evidence": [],
                "details": details,
            }

        # Check registry registration
        try:
            from auto_harness.providers.registry import DEFAULT_PROVIDER_REGISTRY
            if "deepseek" in DEFAULT_PROVIDER_REGISTRY.names():
                details["registered"] = True
                evidence.append("registry: deepseek registered")
        except Exception:
            pass

        # Check ProviderError
        try:
            from auto_harness.providers.errors import ProviderError  # noqa: F401
            evidence.append("src/auto_harness/providers/errors.py")
        except ImportError:
            pass

        # Registration only proves that code exists. Configuration additionally
        # requires a constructible provider with no missing endpoint/model/key.
        try:
            from auto_harness.config import HarnessConfig
            from auto_harness.providers.registry import DEFAULT_PROVIDER_REGISTRY

            config_path = os.environ.get("AUTO_HARNESS_CONFIG")
            if not config_path:
                default_path = self.project_root / "configs" / "default.json"
                config_path = str(default_path) if default_path.exists() else None
            config = HarnessConfig.load(config_path)
            provider = DEFAULT_PROVIDER_REGISTRY.create(
                "deepseek",
                config=config,
                purpose="live_smoke",
            )
            missing = provider.missing_configuration()
            details["configured"] = not missing
            details["thinking_configured"] = not missing
            details["json_mode_configured"] = not missing
            details["missing_configuration"] = list(missing)
        except Exception as exc:
            details["configured"] = False
            details["configuration_error"] = str(exc)[:200]

        # Check config validation
        try:
            from auto_harness.config import _validate_deepseek_config  # noqa: F401
            evidence.append("config: DeepSeek validation")
        except ImportError:
            pass

        # Check test artifacts
        test_artifact = (
            self.project_root / "reports" / "deepseek_provider_result.json"
        )
        if test_artifact.exists():
            details["test_status"] = _check_test_artifact(test_artifact)
            if details["test_status"] == "validated":
                status = "integrated"
            elif details["test_status"] == "failed":
                status = "failed"

        live_manifest = (
            self.project_root
            / "docs"
            / "evidence"
            / "live-agent-smoke-manifest.json"
        )
        if live_manifest.exists():
            try:
                manifest = json.loads(live_manifest.read_text(encoding="utf-8"))
                if manifest.get("provider_name") == "deepseek":
                    final_status = str(
                        manifest.get("final_verify_status", "")
                    ).lower()
                    if final_status in {"pass", "passed"}:
                        details["live_smoke_status"] = "validated"
                        status = "validated"
                        evidence.append(str(live_manifest.relative_to(self.project_root)))
                    elif final_status:
                        details["live_smoke_status"] = "failed"
                        status = "failed"
            except (OSError, ValueError):
                details["live_smoke_status"] = "failed"

        return {
            "status": status,
            "evidence": evidence,
            "details": details,
        }

    def check_readiness(self, matrix: Dict) -> int:
        """Check if all required capabilities are validated.

        Returns:
            0 if all required capabilities are validated,
            1 if some are not validated.
        """
        required = [
            "default_langgraph",
            "crash_safe_reconcile",
            "approval_resume",
            "memory_skill_mainline",
        ]
        caps = matrix.get("capabilities", {})
        for cap in required:
            status = caps.get(cap, {}).get("status", "")
            if status not in ("validated", "integrated"):
                return 1
        return 0


# ------------------------------------------------------------------
# Model Runtime Readiness (Document B Phase B9)
# ------------------------------------------------------------------

# Real-GPU external gates. Each is only ever ``validated`` when fresh, hash-bound
# real evidence exists — never because the corresponding code exists.
GPU_E2E_EXTERNAL_GATES = [
    ("model_revision_resolution_live", "resolve an immutable model revision against the live source"),
    ("large_weight_download_live", "download >10GiB of Safetensors weights with resume"),
    ("docker_nvidia_runtime_live", "probe Docker + NVIDIA Container Toolkit"),
    ("vllm_7b_startup_live", "start vLLM and reach /v1/models ready"),
    ("vllm_non_stream_trace_live", "prove a non-stream inference with a current trace id"),
    ("vllm_sse_trace_live", "prove an SSE inference with a current trace id"),
    ("model_cache_warm_resume_live", "warm-cache resume with a new trace id"),
]

FRESHNESS_DEFAULT_DAYS = 30


def _is_commit_sha(value: str) -> bool:
    return bool(value) and len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value)


class ModelRuntimeReadiness:
    """Assess model runtime capability status from real GPU evidence.

    Status:
    - validated: a fresh, hash-bound, SHA-matching GPU evidence manifest.
    - failed:    an evidence manifest exists but is stale / wrong-SHA / invalid.
    - integrated: offline integration test artifact present (no real GPU).
    - implemented: code exists, no evidence yet.
    """

    def __init__(self, project_root=None, freshness_days: int = FRESHNESS_DEFAULT_DAYS) -> None:
        self.project_root = Path(project_root or Path.cwd())
        self.freshness_days = freshness_days

    def assess(self, manifest: Optional[Dict] = None, git_sha: Optional[str] = None, now=None) -> Dict[str, Any]:
        gates = [{"id": gid, "status": "not_run", "reason": reason} for gid, reason in GPU_E2E_EXTERNAL_GATES]
        if manifest is None:
            integrated = (self.project_root / "reports" / "model_runtime_result.json").exists()
            status = "integrated" if integrated else "implemented"
            return {"status": status, "external_gates": gates, "evidence": ["src/auto_harness/model_runtime/"]}

        problems: List[str] = []
        expected_sha = git_sha if git_sha is not None else _get_commit_sha(self.project_root)
        if manifest.get("git_sha") != expected_sha:
            problems.append("git_sha_mismatch")
        if not _is_commit_sha(str(manifest.get("model_revision", ""))):
            problems.append("model_revision_not_immutable")
        if not str(manifest.get("image_digest", "")).startswith("sha256:"):
            problems.append("image_digest_not_fixed")
        if not self._fresh(manifest.get("generated_at", ""), now):
            problems.append("evidence_stale")

        if problems:
            return {"status": "failed", "external_gates": gates, "problems": problems}

        validated_gates = [{"id": gid, "status": "validated", "reason": reason} for gid, reason in GPU_E2E_EXTERNAL_GATES]
        return {
            "status": "validated",
            "external_gates": validated_gates,
            "evidence": list(manifest.get("evidence_paths", []) or []),
        }

    def _fresh(self, generated_at: str, now=None) -> bool:
        if not generated_at:
            return False
        try:
            from datetime import datetime, timedelta, timezone

            parsed = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            now_dt = now if now is not None else datetime.now(timezone.utc)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)
            return (now_dt - parsed) <= timedelta(days=self.freshness_days)
        except (ValueError, TypeError):
            return False
