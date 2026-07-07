import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json


class ReadinessAuditor:
    """Produce a local completion audit without running external smoke tests."""

    REQUIRED_FILES = [
        "README.md",
        "docs/progress.md",
        "docs/optimization-roadmap.md",
        "docs/llm-agent-upgrade-execution-plan.md",
        "src/auto_harness/agent/schemas.py",
        "src/auto_harness/agent/engine.py",
        "src/auto_harness/agent/policy.py",
        "src/auto_harness/agent/traces.py",
        "src/auto_harness/agent/diagnoser.py",
        "src/auto_harness/agent/verify_planner.py",
        "src/auto_harness/orchestrator.py",
        "src/auto_harness/modules/verify.py",
        "src/auto_harness/modules/model_prepare.py",
        "src/auto_harness/modules/env_solve.py",
        "src/auto_harness/runtime/docker_smoke.py",
        "src/auto_harness/runtime/gpu.py",
        "src/auto_harness/queue.py",
        "src/auto_harness/dashboard.py",
        "src/auto_harness/artifacts.py",
        "tests/fixtures/benchmarks/manifest.json",
    ]

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
        if benchmark_report:
            local_gates.append(self._benchmark_report_gate(Path(benchmark_report)))

        failed = [gate for gate in local_gates if gate["status"] != "passed"]
        report = {
            "status": "ready_for_external_smoke" if not failed else "incomplete",
            "local_readiness_percent": 100 if not failed else self._completion_percent(local_gates),
            "project_progress_percent": self._progress_percent(root / "docs" / "progress.md"),
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

    def _manifest_case_count(self, root: Path) -> int:
        manifest_path = root / "tests" / "fixtures" / "benchmarks" / "manifest.json"
        if not manifest_path.exists():
            return 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return len(manifest.get("cases", []))

    def _progress_percent(self, progress_path: Path) -> Optional[int]:
        if not progress_path.exists():
            return None
        text = progress_path.read_text(encoding="utf-8")
        matches = re.findall(r"进度(?:约为|达到)?\s*\*\*(\d+)%\*\*", text)
        if matches:
            return int(matches[-1])
        matches = re.findall(r"\*\*(\d+)%\*\*", text)
        return int(matches[-1]) if matches else None

    def _completion_percent(self, gates: List[Dict]) -> int:
        if not gates:
            return 0
        passed = sum(1 for gate in gates if gate.get("status") == "passed")
        return int(round(passed * 100 / len(gates)))
