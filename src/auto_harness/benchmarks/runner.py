import json
import os
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Dict, List

from auto_harness.artifacts import DeploymentPackageExporter
from auto_harness.assets import GitLFSDetector, GitSubmoduleDetector, HuggingFaceDownloader, ModelCache
from auto_harness.assets.manifest import ModelAsset
from auto_harness.config import HarnessConfig
from auto_harness.dashboard import DashboardGenerator
from auto_harness.diagnostics import LogClassifier
from auto_harness.memory import MemoryPromoter
from auto_harness.models.base import read_json, write_json
from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.env_solve import EnvSolveModule
from auto_harness.modules.model_prepare import ModelPrepareModule
from auto_harness.modules.reporter import ReportGenerator
from auto_harness.modules.resource_plan import ResourcePlanner
from auto_harness.modules.runner import RunnerModule
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.result import StageResult
from auto_harness.models.task import RuntimePolicy
from auto_harness.repair import RepairLoopController, RepairPlanner, RepairPolicy
from auto_harness.orchestrator import TaskRunner
from auto_harness.queue import DeploymentQueue
from auto_harness.verify import BrowserVerifier, StreamlitVerifier
from auto_harness.utils.shell import CommandResult


class _FakeResponse:
    def __init__(self, body, status: int = 200):
        self.body = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.status = status
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class _FakeBrowserBackend:
    def load(self, url: str, timeout_ms: int = 15000, screenshot_path: Path = None) -> Dict:
        trace = url.split("_auto_harness_trace=", 1)[1] if "_auto_harness_trace=" in url else ""
        return {
            "status": "loaded",
            "url": url,
            "title": "fake gradio",
            "status_code": 200,
            "html": '<html><body><div class="gradio-container">%s</div></body></html>' % trace,
        }


class BenchmarkRunner:
    def run(self, manifest_path: Path, output_path: Path = None, case_ids: List[str] = None) -> Dict:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_ids = list(case_ids or [])
        manifest_cases = manifest.get("cases", [])
        if selected_ids:
            case_by_id = {case.get("id"): case for case in manifest_cases}
            cases_to_run = []
            for case_id in selected_ids:
                case = case_by_id.get(case_id)
                if case:
                    cases_to_run.append(case)
                else:
                    cases_to_run.append({
                        "id": case_id,
                        "purpose": "selected benchmark case is missing from manifest",
                        "expected_signal": "case id must exist",
                        "_missing": True,
                    })
        else:
            cases_to_run = manifest_cases
        cases: List[Dict] = []
        for case in cases_to_run:
            if case.get("_missing"):
                result = self._result(case, "failed", "benchmark case is not present in manifest")
            else:
                result = self._run_case(case, manifest_path.parent)
            cases.append(result)
        report = {
            "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
            "selected_case_ids": selected_ids,
            "selected": bool(selected_ids),
            "cases": cases,
        }
        if output_path:
            write_json(Path(output_path), report)
        return report

    def _run_case(self, case: Dict, fixture_dir: Path) -> Dict:
        case_id = case.get("id")
        try:
            if case_id == "model_download_resume":
                return self._case_model_download_resume(case)
            if case_id == "cache_hit":
                return self._case_cache_hit(case)
            if case_id == "streamlit_error_page":
                return self._case_streamlit_error_page(case, fixture_dir)
            if case_id == "verify_false_positive_http_200":
                return self._case_verify_false_positive(case, fixture_dir)
            if case_id == "gradio_config_discovery":
                return self._case_gradio_config_discovery(case)
            if case_id == "repair_policy_reject":
                return self._case_repair_policy_reject(case)
            if case_id == "checksum_failure":
                return self._case_checksum_failure(case)
            if case_id == "browser_dom_trace":
                return self._case_browser_dom_trace(case)
            if case_id == "parallel_model_download":
                return self._case_parallel_model_download(case)
            if case_id == "etag_cache_invalidation":
                return self._case_etag_cache_invalidation(case)
            if case_id == "cache_cleanup_plan":
                return self._case_cache_cleanup_plan(case)
            if case_id == "cache_cleanup_scoped_keep":
                return self._case_cache_cleanup_scoped_keep(case)
            if case_id == "repair_loop_attempt_limit":
                return self._case_repair_loop_attempt_limit(case)
            if case_id == "operator_repair_approval":
                return self._case_operator_repair_approval(case)
            if case_id == "service_exits_after_start":
                return self._case_service_exits_after_start(case)
            if case_id == "stale_artifact_ignored":
                return self._case_stale_artifact_ignored(case)
            if case_id == "artifact_download_validation":
                return self._case_artifact_download_validation(case)
            if case_id == "git_lfs_detection":
                return self._case_git_lfs_detection(case)
            if case_id == "git_lfs_prepare_execute":
                return self._case_git_lfs_prepare_execute(case)
            if case_id == "git_lfs_progress_parse":
                return self._case_git_lfs_progress_parse(case)
            if case_id == "git_submodule_prepare_execute":
                return self._case_git_submodule_prepare_execute(case)
            if case_id == "docker_backend_plan":
                return self._case_docker_backend_plan(case)
            if case_id == "env_solve_legacy_gradio_constraints":
                return self._case_env_solve_legacy_gradio_constraints(case)
            if case_id == "env_solve_torch_cuda_wheel":
                return self._case_env_solve_torch_cuda_wheel(case)
            if case_id == "gpu_package_matrix_rules":
                return self._case_gpu_package_matrix_rules(case)
            if case_id == "docker_gpu_cache_backend":
                return self._case_docker_gpu_cache_backend(case)
            if case_id == "memory_promotion_approval_regression":
                return self._case_memory_promotion_approval_regression(case)
            if case_id == "memory_promotion_apply_regression_run":
                return self._case_memory_promotion_apply_regression_run(case)
            if case_id == "verify_progress_refresh":
                return self._case_verify_progress_refresh(case)
            if case_id == "openai_compatible_verify":
                return self._case_openai_compatible_verify(case)
            if case_id == "openai_model_discovery_stream_verify":
                return self._case_openai_model_discovery_stream_verify(case)
            if case_id == "openapi_schema_verify":
                return self._case_openapi_schema_verify(case)
            if case_id == "local_e2e_fixture_matrix":
                return self._case_local_e2e_fixture_matrix(case, fixture_dir)
            if case_id == "memory_promotion_proposal":
                return self._case_memory_promotion_proposal(case)
            if case_id == "gradio_api_shape_variation":
                return self._case_gradio_api_shape_variation(case)
            if case_id == "gradio_queue_call_followup":
                return self._case_gradio_queue_call_followup(case)
            if case_id == "token_missing_diagnosis":
                return self._case_token_missing_diagnosis(case)
            if case_id == "structured_dependency_diagnosis":
                return self._case_structured_dependency_diagnosis(case)
            if case_id == "repair_resume_stage_jump":
                return self._case_repair_resume_stage_jump(case)
            if case_id == "repair_resume_audit_report":
                return self._case_repair_resume_audit_report(case)
            if case_id == "token_report_required_env":
                return self._case_token_report_required_env(case)
            if case_id == "static_dashboard_export":
                return self._case_static_dashboard_export(case)
            if case_id == "deployment_queue_dry_run":
                return self._case_deployment_queue_dry_run(case)
            if case_id == "deployment_package_export":
                return self._case_deployment_package_export(case)
            if case_id == "queue_parallel_worker_pool":
                return self._case_queue_parallel_worker_pool(case)
            if case_id == "queue_gpu_probe_scheduling":
                return self._case_queue_gpu_probe_scheduling(case)
            if case_id == "queue_claim_lock_prevents_duplicate":
                return self._case_queue_claim_lock_prevents_duplicate(case)
            if case_id == "queue_stale_claim_lock_recovery":
                return self._case_queue_stale_claim_lock_recovery(case)
            return self._result(case, "skipped", "unknown benchmark case")
        except Exception as exc:  # noqa: BLE001 - benchmark report should continue
            return self._result(case, "failed", str(exc))

    def _case_model_download_resume(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))
            target_dir = Path(asset.cache_path)
            partial = target_dir / "model.safetensors.part"
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"12")
            calls = []

            def fake_urlopen(req, timeout):
                calls.append(req)
                if "/api/models/" in req.full_url:
                    return _FakeResponse(json.dumps([{"type": "file", "path": "model.safetensors", "size": 4}]))
                return _FakeResponse(b"34", status=206)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1).download(asset)
            ok = result.status == "downloaded" and calls[1].headers.get("Range") == "bytes=2-"
            return self._result(case, "passed" if ok else "failed", "Range resume verified")

    def _case_cache_hit(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))
            target = Path(asset.cache_path) / "config.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"abc")

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return _FakeResponse(json.dumps([{"type": "file", "path": "config.json", "size": 3}]))
                raise AssertionError("download should not be called for cached file")

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1).download(asset)
            ok = result.files and result.files[0]["status"] == "cached"
            return self._result(case, "passed" if ok else "failed", "cache hit verified")

    def _case_streamlit_error_page(self, case: Dict, fixture_dir: Path) -> Dict:
        page = (fixture_dir / "streamlit_error_page.html").read_text(encoding="utf-8")

        def fake_urlopen(req, timeout):
            return _FakeResponse(page)

        check = StreamlitVerifier(urlopen=fake_urlopen).probe("http://127.0.0.1:8501", "trace_bench")
        ok = check["status"] == "fail"
        return self._result(case, "passed" if ok else "failed", check["reason"])

    def _case_verify_false_positive(self, case: Dict, fixture_dir: Path) -> Dict:
        body = (fixture_dir / "http_200_no_trace.txt").read_text(encoding="utf-8")

        def fake_urlopen(req, timeout):
            return _FakeResponse(body)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/health"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
        ok = result.status == "uncertain"
        return self._result(case, "passed" if ok else "failed", "HTTP 200 without trace did not pass")

    def _case_gradio_config_discovery(self, case: Dict) -> Dict:
        captured = {}

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/config"):
                return _FakeResponse(json.dumps({
                    "dependencies": [
                        {"id": 2, "api_name": "predict", "backend_fn": True}
                    ]
                }))
            captured["url"] = req.full_url
            captured["body"] = req.data.decode("utf-8")
            trace = json.loads(captured["body"])["data"][0]
            return _FakeResponse("handled %s" % trace)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
        ok = (
            result.status == "passed"
            and captured.get("url") == "http://127.0.0.1:7860/api/predict"
            and json.loads(captured.get("body", "{}")).get("fn_index") == 2
        )
        return self._result(case, "passed" if ok else "failed", "Gradio /config request discovery verified")

    def _case_repair_policy_reject(self, case: Dict) -> Dict:
        policy = RepairPolicy().check(
            {
                "actions": [
                    {
                        "type": "install_package",
                        "requires": {"dependency_install": True},
                        "payload": {"package": "gradio"},
                    }
                ]
            },
            RuntimePolicy(workspace_root="/tmp/auto-harness-benchmark", allow_dependency_install=False),
        )
        ok = not policy["allowed"] and policy["decisions"][0]["reasons"]
        return self._result(case, "passed" if ok else "failed", "repair policy rejected dependency install")

    def _case_checksum_failure(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return _FakeResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 3, "sha256": "0" * 64}
                    ]))
                return _FakeResponse(b"abc", status=200)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1).download(asset)
        ok = result.status == "failed" and "sha256 mismatch" in (result.last_error or "")
        return self._result(case, "passed" if ok else "failed", "checksum mismatch failed the download")

    def _case_browser_dom_trace(self, case: Dict) -> Dict:
        def fake_urlopen(req, timeout):
            return _FakeResponse("ok without trace")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(
                urlopen=fake_urlopen,
                browser_verifier=BrowserVerifier(browser_backend=_FakeBrowserBackend()),
            ).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
        checks = {check["name"]: check for check in result.data["checks"]}
        ok = result.status == "passed" and checks["browser_dom_probe"]["status"] == "pass"
        return self._result(case, "passed" if ok else "failed", "browser DOM trace evidence verified")

    def _case_parallel_model_download(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))
            download_urls = []

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return _FakeResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 1},
                        {"type": "file", "path": "tokenizer.json", "size": 1},
                    ]))
                download_urls.append(req.full_url)
                return _FakeResponse(b"a" if req.full_url.endswith("config.json") else b"b", status=200)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1, max_workers=2).download(asset)
            ok = result.status == "downloaded" and len(download_urls) == 2 and [file["path"] for file in result.files] == ["config.json", "tokenizer.json"]
            return self._result(case, "passed" if ok else "failed", "parallel file download verified")

    def _case_etag_cache_invalidation(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))
            target = Path(asset.cache_path) / "config.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"old")
            (target.parent / "config.json.auto_harness_meta.json").write_text(
                json.dumps({"size_bytes": 3, "etag": "old-etag"}),
                encoding="utf-8",
            )
            download_count = 0

            def fake_urlopen(req, timeout):
                nonlocal download_count
                if "/api/models/" in req.full_url:
                    return _FakeResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 3, "oid": "new-etag"}
                    ]))
                download_count += 1
                return _FakeResponse(b"new", status=200)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1).download(asset)
            ok = result.status == "downloaded" and target.read_bytes() == b"new" and download_count == 1
            return self._result(case, "passed" if ok else "failed", "etag mismatch invalidated cached file")

    def _case_cache_cleanup_plan(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            old_dir = cache.root / "huggingface" / "old"
            new_dir = cache.root / "huggingface" / "new"
            old_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)
            (old_dir / "model.bin").write_bytes(b"old")
            (new_dir / "model.bin").write_bytes(b"newer")
            plan = cache.cleanup(max_total_bytes=5, dry_run=True)
            applied = cache.cleanup(max_total_bytes=5, dry_run=False)
            ok = plan["candidate_count"] == 1 and len(applied["deleted"]) == 1 and not old_dir.exists() and new_dir.exists()
            return self._result(case, "passed" if ok else "failed", "cache cleanup dry-run and delete verified")

    def _case_cache_cleanup_scoped_keep(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            keep = cache.reserve(ModelAsset(asset_id="hf:keep", source="huggingface", repo_id="org/keep"))
            delete = cache.reserve(ModelAsset(asset_id="hf:delete", source="huggingface", repo_id="org/delete"))
            other_source = cache.reserve(ModelAsset(asset_id="ms:delete", source="modelscope", repo_id="org/delete"))
            Path(keep.cache_path, "model.bin").write_bytes(b"keep")
            Path(delete.cache_path, "model.bin").write_bytes(b"delete")
            Path(other_source.cache_path, "model.bin").write_bytes(b"modelscope")
            plan = cache.cleanup(max_total_bytes=0, source="huggingface", keep_repo_ids=["org/keep"], dry_run=True)
            applied = cache.cleanup(max_total_bytes=0, source="huggingface", keep_repo_ids=["org/keep"], dry_run=False)
            ok = (
                plan["candidate_count"] == 1
                and plan["candidates"][0]["repo_id"] == "org/delete"
                and len(applied["deleted"]) == 1
                and Path(keep.cache_path).exists()
                and not Path(delete.cache_path).exists()
                and Path(other_source.cache_path).exists()
            )
            return self._result(case, "passed" if ok else "failed", "scoped cache cleanup with keep-list verified")

    def _case_repair_loop_attempt_limit(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            controller = RepairLoopController(max_attempts=1)
            plan = {"root_cause": "trace missing", "rerun_from": "unsafe_stage", "actions": []}
            entry = {"signature": "bench-repair-loop"}
            policy = {"allowed": True, "decisions": []}
            first = controller.gate(Path(tmp), "verify", entry, dict(plan), policy)
            second = controller.gate(Path(tmp), "verify", entry, dict(plan), policy)
        ok = first["allowed"] is True and second["allowed"] is False and second["loop"]["rerun_from_effective"] == "verify"
        return self._result(case, "passed" if ok else "failed", "repair loop attempt limit verified")

    def _case_operator_repair_approval(self, case: Dict) -> Dict:
        action = {
            "type": "change_cache_dir",
            "requires": {"operator_approval": True},
            "payload": {"config": "model_cache_dir"},
        }
        runtime = RuntimePolicy(workspace_root="/tmp/auto-harness-benchmark")
        rejected = RepairPolicy().check({"actions": [action]}, runtime)
        approved = RepairPolicy().check(
            {"actions": [action]},
            runtime,
            operator_approval={"approved": True, "approved_action_types": ["change_cache_dir"]},
        )
        ok = rejected["allowed"] is False and approved["allowed"] is True
        return self._result(case, "passed" if ok else "failed", "operator approval gate verified")

    def _case_service_exits_after_start(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "workspace" / "repo"
            repo.mkdir(parents=True)
            (repo / "app.py").write_text("print('boot'); raise SystemExit(3)\n", encoding="utf-8")
            result = RunnerModule().run(
                repo,
                {"run_candidates": [{"cmd": ["python3", "app.py"], "expected_port": 7860}]},
                execute=True,
                wait_seconds=0.2,
                allowed_commands=["python3"],
            )
        ok = result.status == "failed" and result.summary == "service process exited"
        return self._result(case, "passed" if ok else "failed", "runner detects early service exit")

    def _case_stale_artifact_ignored(self, case: Dict) -> Dict:
        def fake_urlopen(req, timeout):
            return _FakeResponse("ok without current trace")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)
            (repo / "old_output.txt").write_text("stale successful output from previous run", encoding="utf-8")
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/health"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
        checks = {check["name"]: check for check in result.data["checks"]}
        ok = result.status == "uncertain" and checks["artifact_freshness"]["status"] == "uncertain"
        return self._result(case, "passed" if ok else "failed", "stale artifact did not pass verify")

    def _case_artifact_download_validation(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)

            def fake_urlopen(req, timeout):
                (repo / "outputs").mkdir(exist_ok=True)
                (repo / "outputs" / "result.bin").write_bytes(b"generated artifact")
                return _FakeResponse("ok without current trace")

            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/generate"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
        checks = {check["name"]: check for check in result.data["checks"]}
        validation = checks.get("artifact_download_validation", {})
        evidence = validation.get("evidence", {})
        ok = (
            result.status == "passed"
            and validation.get("status") == "pass"
            and evidence.get("validated", [{}])[0].get("path") == "outputs/result.bin"
            and len(evidence.get("validated", [{}])[0].get("sha256", "")) == 64
        )
        return self._result(case, "passed" if ok else "failed", "artifact download validation verified")

    def _case_git_lfs_detection(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
            (repo / "model.safetensors").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:%s\n"
                "size 2147483648\n" % ("b" * 64),
                encoding="utf-8",
            )
            result = ResourcePlanner(git_lfs_detector=GitLFSDetector(available=False)).plan(repo, {"frameworks": []})
        lfs = result.data.get("git_lfs", {})
        ok = (
            result.status == "uncertain"
            and result.data.get("diagnosis", {}).get("category") == "git_lfs_missing"
            and lfs.get("pointer_count") == 1
            and lfs.get("total_pointer_size_bytes") == 2147483648
            and lfs.get("prepare_commands") == [["git", "lfs", "install"], ["git", "lfs", "pull"]]
        )
        return self._result(case, "passed" if ok else "failed", "Git LFS detection verified")

    def _case_git_lfs_prepare_execute(self, case: Dict) -> Dict:
        calls = []

        def fake_runner(cmd, cwd, timeout_seconds=900):
            calls.append(cmd)
            return CommandResult(cmd, str(cwd), 0, "ok", "", False)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)
            (run_dir / "reports").mkdir(parents=True)
            result = ModelPrepareModule(ModelCache(Path(tmp) / "cache"), command_runner=fake_runner).prepare(
                run_dir,
                {
                    "model_assets": [],
                    "git_lfs": {
                        "required": True,
                        "available": True,
                        "pointer_count": 1,
                        "total_pointer_size_bytes": 2048,
                        "prepare_commands": [["git", "lfs", "install"], ["git", "lfs", "pull"]],
                    },
                },
                execute=True,
                repo_dir=repo,
                allowed_commands=["git"],
            )
        ok = (
            result.status == "passed"
            and result.data.get("git_lfs", {}).get("status") == "ready"
            and calls == [["git", "lfs", "install"], ["git", "lfs", "pull"]]
        )
        return self._result(case, "passed" if ok else "failed", "Git LFS controlled execution verified")

    def _case_git_lfs_progress_parse(self, case: Dict) -> Dict:
        def fake_runner(cmd, cwd, timeout_seconds=900):
            if cmd == ["git", "lfs", "pull"]:
                return CommandResult(
                    cmd,
                    str(cwd),
                    0,
                    "Downloading LFS objects:  50% (1/2), 20 MB | 2 MB/s\nGit LFS: (1 of 2 files) 10 MB / 20 MB\n",
                    "",
                    False,
                )
            return CommandResult(cmd, str(cwd), 0, "ok", "", False)

        updates = []
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)
            (run_dir / "reports").mkdir(parents=True)
            result = ModelPrepareModule(ModelCache(Path(tmp) / "cache"), command_runner=fake_runner).prepare(
                run_dir,
                {
                    "model_assets": [],
                    "git_lfs": {
                        "required": True,
                        "available": True,
                        "pointer_count": 2,
                        "total_pointer_size_bytes": 20 * 1024 * 1024,
                        "prepare_commands": [["git", "lfs", "install"], ["git", "lfs", "pull"]],
                    },
                },
                execute=True,
                repo_dir=repo,
                allowed_commands=["git"],
                progress_callback=lambda progress: updates.append(progress),
            )
        lfs_progress = result.data.get("git_lfs", {}).get("commands", [{}, {}])[1].get("progress", {})
        ok = (
            result.status == "passed"
            and lfs_progress.get("percent") == 50
            and lfs_progress.get("files_done") == 1
            and lfs_progress.get("files_total") == 2
            and lfs_progress.get("downloaded_bytes") == 10 * 1024 * 1024
            and any(update.get("status") == "git_lfs_downloading" for update in updates)
            and result.data.get("git_lfs", {}).get("progress", {}).get("percent") == 100
        )
        return self._result(case, "passed" if ok else "failed", "Git LFS progress parsing verified")

    def _case_git_submodule_prepare_execute(self, case: Dict) -> Dict:
        calls = []

        def fake_runner(cmd, cwd, timeout_seconds=900):
            calls.append(cmd)
            return CommandResult(cmd, str(cwd), 0, "ok", "", False)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / ".gitmodules").write_text(
                '[submodule "extensions/foo"]\n'
                "\tpath = extensions/foo\n"
                "\turl = https://github.com/example/foo.git\n",
                encoding="utf-8",
            )
            resource = ResourcePlanner(git_submodule_detector=GitSubmoduleDetector(available=True)).plan(repo, {"frameworks": []})
            run_dir = Path(tmp) / "run"
            (run_dir / "reports").mkdir(parents=True)
            result = ModelPrepareModule(ModelCache(Path(tmp) / "cache"), command_runner=fake_runner).prepare(
                run_dir,
                resource.data,
                execute=True,
                repo_dir=repo,
                allowed_commands=["git"],
            )
        submodules = result.data.get("git_submodules", {})
        ok = (
            resource.status == "passed"
            and resource.data.get("git_submodules", {}).get("submodule_count") == 1
            and result.status == "passed"
            and submodules.get("status") == "ready"
            and calls == [
                ["git", "submodule", "sync", "--recursive"],
                ["git", "submodule", "update", "--init", "--recursive"],
            ]
        )
        return self._result(case, "passed" if ok else "failed", "Git submodule controlled preparation verified")

    def _case_docker_backend_plan(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            env_result = EnvDeployModule().deploy(
                repo,
                {"install_plan": [["python3", "-m", "pip", "install", "-r", "requirements.txt"]]},
                execute=False,
                execution_backend="docker",
                docker_image="python:3.11-slim",
                docker_network="none",
            )
            runner_result = RunnerModule().run(
                repo,
                {"run_candidates": [{"cmd": ["python3", "app.py"], "expected_port": 7860}]},
                execute=True,
                allowed_commands=["python3"],
                execution_backend="docker",
                docker_image="python:3.11-slim",
                docker_network="none",
            )
        ok = (
            env_result.status == "passed"
            and env_result.data.get("effective_commands", [[]])[0][0] == "docker"
            and "python:3.11-slim" in env_result.data.get("effective_commands", [[]])[0]
            and runner_result.status == "failed"
            and runner_result.data.get("cmd", [None])[0] == "docker"
            and "-p" in runner_result.data.get("cmd", [])
            and "disallowed command: docker" in (runner_result.error or "")
        )
        return self._result(case, "passed" if ok else "failed", "Docker backend planning verified")

    def _case_env_solve_legacy_gradio_constraints(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "requirements.txt").write_text("gradio\nopencv-python\n", encoding="utf-8")
            result = EnvSolveModule().solve(
                repo,
                {
                    "frameworks": ["gradio"],
                    "install_plan": [
                        ["python3", "-m", "venv", ".venv"],
                        [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                    ],
                },
                {"python_range": ">=3.10,<3.12"},
            )
        ok = (
            result.status == "passed"
            and {"numpy<2", "pydantic<2", "opencv-python-headless"}.issubset(set(result.data.get("constraints", [])))
            and "numpy<2" in result.data.get("install_plan", [[], []])[1]
            and result.data.get("analysis", {}).get("env_solution", {}).get("python") == "3.10"
        )
        return self._result(case, "passed" if ok else "failed", "env_solve legacy gradio constraints verified")

    def _case_env_solve_torch_cuda_wheel(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "requirements.txt").write_text("torch\ntorchvision\ntransformers\n", encoding="utf-8")
            result = EnvSolveModule(local_environment={
                "python_version": "3.10",
                "platform": "linux",
                "machine": "x86_64",
                "cuda": {"available": True, "version": "12.1", "source": "benchmark"},
            }).solve(
                repo,
                {
                    "frameworks": ["torch", "transformers"],
                    "install_plan": [
                        ["python3", "-m", "venv", ".venv"],
                        [".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"],
                        [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                    ],
                },
                {"gpu_required": True, "python_range": ">=3.10,<3.12", "torch_variant": "cuda_or_cpu"},
            )
        torch_solution = result.data.get("torch_solution", {})
        fallback_urls = [item.get("index_url") for item in torch_solution.get("fallbacks", [])]
        ok = (
            result.status == "passed"
            and torch_solution.get("selected", {}).get("variant") == "cu121"
            and torch_solution.get("selected", {}).get("index_url") == "https://download.pytorch.org/whl/cu121"
            and "https://download.pytorch.org/whl/cpu" in fallback_urls
            and any(cmd and cmd[0] == ".venv/bin/python" and "https://download.pytorch.org/whl/cu121" in cmd for cmd in result.data.get("install_plan", []))
        )
        return self._result(case, "passed" if ok else "failed", "env_solve torch CUDA wheel verified")

    def _case_gpu_package_matrix_rules(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "requirements.txt").write_text("torch\nxformers\nflash-attn\nbitsandbytes\ntriton\n", encoding="utf-8")
            result = EnvSolveModule(local_environment={
                "python_version": "3.11",
                "platform": "darwin",
                "machine": "arm64",
                "cuda": {"available": False, "version": "", "source": "benchmark"},
            }).solve(
                repo,
                {"frameworks": ["torch"], "install_plan": [["python3", "-m", "venv", ".venv"]]},
                {"gpu_required": True, "torch_variant": "cuda_or_cpu"},
            )
        matrix = {item["name"]: item for item in result.data.get("gpu_package_matrix", {}).get("packages", [])}
        ok = (
            matrix.get("flash-attn", {}).get("status") == "blocked"
            and matrix.get("xformers", {}).get("status") == "blocked"
            and matrix.get("bitsandbytes", {}).get("status") == "blocked"
            and matrix.get("triton", {}).get("status") == "blocked"
            and any("flash-attn:" in reason for reason in result.data.get("risk_reasons", []))
        )
        return self._result(case, "passed" if ok else "failed", "GPU package matrix rules verified")

    def _case_docker_gpu_cache_backend(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            cache = Path(tmp) / "model_cache"
            repo.mkdir()
            env_result = EnvDeployModule().deploy(
                repo,
                {"install_plan": [["python3", "-m", "pip", "install", "-r", "requirements.txt"]]},
                execute=False,
                execution_backend="docker",
                docker_image="python:3.11-slim",
                docker_network="bridge",
                docker_gpus="all",
                docker_model_cache_dir=str(cache),
            )
            runner_result = RunnerModule().run(
                repo,
                {"run_candidates": [{"cmd": ["python3", "app.py"], "expected_port": 7860}]},
                execute=True,
                allowed_commands=["python3"],
                execution_backend="docker",
                docker_image="python:3.11-slim",
                docker_gpus="all",
                docker_model_cache_dir=str(cache),
            )
        env_cmd = env_result.data.get("effective_commands", [[]])[0]
        sandbox = runner_result.data.get("sandbox", {})
        ok = (
            "--gpus" in env_cmd
            and "%s:/workspace/model_cache" % cache.resolve() in env_cmd
            and sandbox.get("gpus") == "all"
            and sandbox.get("model_cache_mount", {}).get("container_path") == "/workspace/model_cache"
            and sandbox.get("log_command", [None])[0] == "docker"
            and sandbox.get("cleanup_command", [None])[0] == "docker"
        )
        return self._result(case, "passed" if ok else "failed", "Docker GPU/cache/log metadata verified")

    def _case_memory_promotion_approval_regression(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            skills_dir = root / "skills"
            memory_dir.mkdir()
            (skills_dir / "verify-evidence").mkdir(parents=True)
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            skill_path.write_text("---\nname: verify-evidence\n---\n# Verify\n", encoding="utf-8")
            entries = [
                {
                    "id": "mem_1",
                    "stage": "verify",
                    "category": "trace_not_observed",
                    "frameworks": ["gradio"],
                    "symptom": "trace missing",
                    "root_cause": "api shape changed",
                    "suggested_next_action": "use config discovery",
                },
                {
                    "id": "mem_2",
                    "stage": "verify",
                    "category": "trace_not_observed",
                    "frameworks": ["gradio"],
                    "symptom": "trace missing again",
                    "root_cause": "api shape changed",
                    "suggested_next_action": "bind gradio regression",
                },
            ]
            (memory_dir / "deployment_issues.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n",
                encoding="utf-8",
            )
            promoter = MemoryPromoter(memory_dir, skills_dir)
            proposal = promoter.propose(min_count=2)["proposals"][0]
            proposal_path = memory_dir / "promotions" / ("%s.json" % proposal["proposal_id"])
            rejected = promoter.apply(proposal_path)
            approved = promoter.approve(proposal_path, reviewer="benchmark", note="regression cases selected")
            applied = promoter.apply(proposal_path)
            skill_text = skill_path.read_text(encoding="utf-8")
            ok = (
                rejected.get("status") == "approval_required"
                and approved.get("status") == "approved"
                and "gradio_config_discovery" in approved.get("regression_binding", {}).get("case_ids", [])
                and applied.get("status") == "applied"
                and "Memory Promotion" in skill_text
            )
        return self._result(case, "passed" if ok else "failed", "memory promotion approval and regression binding verified")

    def _case_memory_promotion_apply_regression_run(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            skills_dir = root / "skills"
            memory_dir.mkdir()
            (skills_dir / "verify-evidence").mkdir(parents=True)
            skill_path = skills_dir / "verify-evidence" / "SKILL.md"
            skill_path.write_text("---\nname: verify-evidence\n---\n# Verify\n", encoding="utf-8")
            entries = [
                {
                    "id": "mem_apply_1",
                    "stage": "verify",
                    "category": "trace_not_observed",
                    "frameworks": ["gradio"],
                    "symptom": "trace missing",
                    "root_cause": "api shape changed",
                    "suggested_next_action": "run bound gradio regressions",
                },
                {
                    "id": "mem_apply_2",
                    "stage": "verify",
                    "category": "trace_not_observed",
                    "frameworks": ["gradio"],
                    "symptom": "trace missing again",
                    "root_cause": "api shape changed",
                    "suggested_next_action": "run bound gradio regressions",
                },
            ]
            (memory_dir / "deployment_issues.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n",
                encoding="utf-8",
            )
            promoter = MemoryPromoter(memory_dir, skills_dir)
            proposal = promoter.propose(min_count=2)["proposals"][0]
            proposal_path = memory_dir / "promotions" / ("%s.json" % proposal["proposal_id"])
            promoter.approve(proposal_path, reviewer="benchmark", note="run regression after apply")
            applied = promoter.apply(proposal_path)
            regression_path = Path(applied.get("regression", {}).get("output_path", ""))
            regression = read_json(regression_path) if regression_path.exists() else {}
            ok = (
                applied.get("status") == "applied"
                and applied.get("regression", {}).get("status") == "passed"
                and regression.get("selected") is True
                and regression.get("selected_case_ids") == proposal.get("regression_binding", {}).get("case_ids")
                and all(item.get("status") == "passed" for item in regression.get("cases", []))
            )
        return self._result(case, "passed" if ok else "failed", "memory promotion apply regression execution verified")

    def _case_verify_progress_refresh(self, case: Dict) -> Dict:
        updates = []

        def fake_urlopen(req, timeout):
            trace = req.full_url.split("_auto_harness_trace=")[1]
            return _FakeResponse("handled trace %s" % trace)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen, progress_callback=lambda progress: updates.append(progress)).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/echo"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
        statuses = [update.get("status") for update in updates]
        ok = (
            result.status == "passed"
            and "service_discovered" in statuses
            and "first_inference_probe_started" in statuses
            and "http_trace_request_sent" in statuses
            and statuses[-1] == "verify_completed"
        )
        return self._result(case, "passed" if ok else "failed", "verify progress refresh verified")

    def _case_openai_compatible_verify(self, case: Dict) -> Dict:
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            trace = captured["body"]["messages"][0]["content"].split("trace ", 1)[1]
            return _FakeResponse(json.dumps({"choices": [{"message": {"content": "ok %s" % trace}}]}))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={
                    "frameworks": ["vllm", "openai_compatible"],
                    "verify_hint": {
                        "service_type": "openai_compatible",
                        "endpoint": "http://127.0.0.1:8000",
                        "model": "bench-model",
                    },
                },
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
        ok = (
            result.status == "passed"
            and captured.get("url") == "http://127.0.0.1:8000/v1/chat/completions"
            and captured.get("body", {}).get("model") == "bench-model"
        )
        return self._result(case, "passed" if ok else "failed", "OpenAI-compatible chat completion verify verified")

    def _case_openai_model_discovery_stream_verify(self, case: Dict) -> Dict:
        captured = {"urls": []}

        def fake_urlopen(req, timeout):
            captured["urls"].append(req.full_url)
            if req.full_url.endswith("/v1/models"):
                return _FakeResponse(json.dumps({"data": [{"id": "bench-served-model"}]}))
            captured["body"] = json.loads(req.data.decode("utf-8"))
            trace = captured["body"]["messages"][0]["content"].split("trace ", 1)[1]
            return _FakeResponse(
                "data: {\"choices\":[{\"delta\":{\"content\":\"stream %s\"}}]}\n\n"
                "data: [DONE]\n\n" % trace
            )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={
                    "frameworks": ["openai_compatible"],
                    "verify_hint": {
                        "service_type": "openai_compatible",
                        "endpoint": "http://127.0.0.1:8000",
                        "request": {
                            "method": "POST",
                            "path": "/v1/chat/completions",
                            "json": {
                                "model": "{{model}}",
                                "messages": [{"role": "user", "content": "auto harness trace {{trace_id}}"}],
                                "stream": True,
                            },
                        },
                    },
                },
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
            evidence = read_json(Path(result.evidence[1]))
        ok = (
            result.status == "passed"
            and "http://127.0.0.1:8000/v1/models" in captured["urls"]
            and captured.get("body", {}).get("model") == "bench-served-model"
            and captured.get("body", {}).get("stream") is True
            and evidence.get("response", {}).get("stream_detected") is True
        )
        return self._result(case, "passed" if ok else "failed", "OpenAI-compatible model discovery and stream verify verified")

    def _case_openapi_schema_verify(self, case: Dict) -> Dict:
        captured = {}

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/openapi.json"):
                return _FakeResponse(json.dumps({
                    "openapi": "3.0.0",
                    "paths": {
                        "/predict": {
                            "post": {
                                "operationId": "predict",
                                "requestBody": {
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "required": ["prompt"],
                                                "properties": {
                                                    "prompt": {"type": "string"},
                                                    "seed": {"type": "integer"},
                                                },
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    },
                }))
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            trace = captured["body"]["prompt"].split("trace ", 1)[1]
            return _FakeResponse(json.dumps({"trace": trace}))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["fastapi"], "verify_hint": {"endpoint": "http://127.0.0.1:8000", "service_type": "api"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
        ok = (
            result.status == "passed"
            and captured.get("url") == "http://127.0.0.1:8000/predict"
            and "prompt" in captured.get("body", {})
        )
        return self._result(case, "passed" if ok else "failed", "OpenAPI schema trace verify verified")

    def _case_local_e2e_fixture_matrix(self, case: Dict, fixture_dir: Path) -> Dict:
        fixture_root = fixture_dir.parent / "e2e"
        fixture_specs = [
            {
                "name": "gradio_tiny_model",
                "framework": "gradio",
                "port": 7860,
                "expect_constraints": {"numpy<2", "pydantic<2"},
            },
            {
                "name": "streamlit_tiny_demo",
                "framework": "streamlit",
                "port": 8501,
            },
            {
                "name": "git_lfs_weight_repo",
                "framework": "gradio",
                "port": 7860,
                "expect_git_lfs": True,
                "expect_torch_solution": True,
            },
        ]
        details = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = TaskRunner(HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                skills_dir=str(Path("skills").resolve()),
            ))
            for spec in fixture_specs:
                repo = fixture_root / spec["name"]
                task_id = runner.deploy(str(repo), spec["name"], dry_run=True)
                run_dir = root / "runs" / task_id
                results = read_json(run_dir / "reports" / "pipeline_results.json")
                ok, detail = self._assert_local_e2e_fixture(results, spec, run_dir)
                details.append(detail)
                if not ok:
                    return self._result(case, "failed", "local E2E fixture failed: %s" % detail)
        return self._result(case, "passed", "local E2E fixture matrix verified: %s" % ", ".join(details))

    def _assert_local_e2e_fixture(self, results: Dict, spec: Dict, run_dir: Path) -> tuple:
        stages = ("analyze", "resource_plan", "env_solve", "env_deploy", "model_prepare", "runner", "verify", "report")
        missing = [stage for stage in stages if stage not in results]
        if missing:
            return False, "%s missing stages %s" % (spec["name"], missing)
        analyze = results["analyze"]["data"]
        resource_plan = results["resource_plan"]["data"]
        env_solve = results["env_solve"]["data"]
        runner = results["runner"]["data"]
        model_prepare = results["model_prepare"]["data"]
        if spec["framework"] not in analyze.get("frameworks", []):
            return False, "%s framework not detected" % spec["name"]
        if not any(candidate.get("expected_port") == spec["port"] for candidate in analyze.get("run_candidates", [])):
            return False, "%s runner candidate missing expected port" % spec["name"]
        if results["env_deploy"]["status"] != "passed":
            return False, "%s env_deploy did not pass dry-run" % spec["name"]
        if results["runner"]["status"] != "passed" or runner.get("executed") is not False:
            return False, "%s runner dry-run did not pass" % spec["name"]
        if not Path(model_prepare.get("manifest_path", "")).exists():
            return False, "%s model manifest missing" % spec["name"]
        expected_constraints = spec.get("expect_constraints")
        if expected_constraints and not expected_constraints.issubset(set(env_solve.get("constraints", []))):
            return False, "%s env_solve constraints missing" % spec["name"]
        if spec.get("expect_git_lfs"):
            git_lfs = resource_plan.get("git_lfs", {})
            if not git_lfs.get("required") or not git_lfs.get("pointers"):
                return False, "%s Git LFS pointer not detected" % spec["name"]
            if not git_lfs.get("total_pointer_size_bytes"):
                return False, "%s Git LFS size not counted" % spec["name"]
        if spec.get("expect_torch_solution"):
            torch_solution = env_solve.get("torch_solution", {})
            if not torch_solution.get("required") or not torch_solution.get("selected"):
                return False, "%s torch solution missing" % spec["name"]
        if not (run_dir / "reports" / "report.md").exists():
            return False, "%s report missing" % spec["name"]
        return True, spec["name"]

    def _case_memory_promotion_proposal(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            memory_dir.mkdir()
            skills_dir = root / "skills"
            target_dir = skills_dir / "verify-evidence"
            target_dir.mkdir(parents=True)
            skill_path = target_dir / "SKILL.md"
            skill_path.write_text("---\nname: verify-evidence\n---\n# Verify\n", encoding="utf-8")
            memories = [
                {
                    "id": "mem_gradio_trace_1",
                    "stage": "verify",
                    "category": "trace_not_observed",
                    "frameworks": ["gradio"],
                    "symptom": "HTTP response did not contain trace id",
                    "root_cause": "Gradio API shape differs from fallback",
                    "suggested_next_action": "Read /config before selecting verify request.",
                },
                {
                    "id": "mem_gradio_trace_2",
                    "stage": "verify",
                    "category": "trace_not_observed",
                    "frameworks": ["gradio"],
                    "symptom": "artifact did not include current trace id",
                    "root_cause": "Default /api/predict endpoint does not match app",
                    "suggested_next_action": "Generate a verify_hint from discovered dependency api_name.",
                },
            ]
            (memory_dir / "deployment_issues.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in memories) + "\n",
                encoding="utf-8",
            )
            result = MemoryPromoter(memory_dir, skills_dir).propose(min_count=2)
            proposal = result.get("proposals", [{}])[0]
            proposal_path = memory_dir / "promotions" / ("%s.json" % proposal.get("proposal_id", "missing"))
            md_path = memory_dir / "promotions" / ("%s.md" % proposal.get("proposal_id", "missing"))
            ok = (
                result.get("status") == "proposed"
                and proposal.get("review_required") is True
                and proposal.get("target_skill") == "verify-evidence/SKILL.md"
                and proposal_path.exists()
                and md_path.exists()
                and "Memory Promotion" in md_path.read_text(encoding="utf-8")
                and "Memory Promotion" not in skill_path.read_text(encoding="utf-8")
            )
        return self._result(case, "passed" if ok else "failed", "memory promotion proposal verified")

    def _case_gradio_api_shape_variation(self, case: Dict) -> Dict:
        captured = {}

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/config"):
                return _FakeResponse(json.dumps({
                    "dependencies": [
                        {"id": 1, "api_name": False, "backend_fn": False},
                        {"id": 7, "api_name": "/predict", "backend_fn": True},
                    ]
                }))
            captured["url"] = req.full_url
            captured["body"] = req.data.decode("utf-8")
            trace = json.loads(captured["body"])["data"][0]
            return _FakeResponse(json.dumps({"data": [trace]}))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
        body = json.loads(captured.get("body", "{}"))
        ok = result.status == "passed" and captured.get("url") == "http://127.0.0.1:7860/api/predict" and body.get("fn_index") == 7
        return self._result(case, "passed" if ok else "failed", "Gradio API shape variation verified")

    def _case_gradio_queue_call_followup(self, case: Dict) -> Dict:
        captured = {"urls": []}

        def fake_urlopen(req, timeout):
            captured["urls"].append(req.full_url)
            if req.full_url.endswith("/config"):
                return _FakeResponse(json.dumps({
                    "enable_queue": True,
                    "dependencies": [
                        {"id": 7, "api_name": "/predict", "backend_fn": True, "queue": True},
                    ],
                }))
            if req.full_url.endswith("/call/predict"):
                captured["body"] = req.data.decode("utf-8")
                return _FakeResponse(json.dumps({"event_id": "evt-123"}))
            if req.full_url.endswith("/call/predict/evt-123"):
                trace = json.loads(captured["body"])["data"][0]
                return _FakeResponse("event: complete\ndata: {\"data\": [\"%s\"]}\n\n" % trace)
            raise AssertionError("unexpected url %s" % req.full_url)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
            evidence = read_json(Path(result.evidence[1]))
            ok = (
                result.status == "passed"
                and "http://127.0.0.1:7860/call/predict" in captured["urls"]
                and "http://127.0.0.1:7860/call/predict/evt-123" in captured["urls"]
                and evidence.get("follow_up_response", {}).get("trace_found") is True
            )
        return self._result(case, "passed" if ok else "failed", "Gradio queue /call follow-up verified")

    def _case_token_missing_diagnosis(self, case: Dict) -> Dict:
        diagnosis = LogClassifier().classify("401 Unauthorized: Repository Not Found. Please set HF_TOKEN.")
        ok = diagnosis["category"] == "auth_required" and diagnosis["confidence"] >= 0.9
        return self._result(case, "passed" if ok else "failed", "token missing diagnosis verified")

    def _case_structured_dependency_diagnosis(self, case: Dict) -> Dict:
        diagnosis = LogClassifier().classify("ValueError: numpy.dtype size changed, may indicate binary incompatibility")
        plan = RepairPlanner().propose(
            "env_deploy",
            StageResult("env_deploy", "failed", "dependency failed", {"diagnosis": diagnosis}),
            {"frameworks": ["gradio"]},
        )
        ok = (
            diagnosis.get("category") == "numpy_abi_conflict"
            and diagnosis.get("package_constraints") == ["numpy<2"]
            and plan.get("actions", [{}])[0].get("payload", {}).get("package") == "numpy<2"
            and plan.get("rerun_from") == "env_deploy"
        )
        return self._result(case, "passed" if ok else "failed", "structured dependency diagnosis verified")

    def _case_repair_resume_stage_jump(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "app.py").write_text("import gradio as gr\n", encoding="utf-8")
            runner = TaskRunner(HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
            ))
            task_id = runner.deploy(str(repo), "demo", dry_run=True)
            run_dir = root / "runs" / task_id
            before = self._stage_update_count(run_dir)
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(exist_ok=True)
            write_json(repair_dir / "repair_apply_result.json", {
                "status": "applied",
                "policy": {"allowed": True, "loop": {"rerun_from_effective": "verify"}},
            })
            write_json(repair_dir / "repair_verify_hints.json", {
                "verify_hints": [{"endpoint": "http://127.0.0.1:9"}],
            })
            runner.resume(task_id, dry_run=True)
            after = self._stage_update_count(run_dir)
            ok = (
                after.get("verify", 0) > before.get("verify", 0)
                and after.get("report", 0) > before.get("report", 0)
                and all(after.get(stage, 0) == before.get(stage, 0) for stage in ("analyze", "resource_plan", "env_solve", "env_deploy", "model_prepare", "runner"))
            )
        return self._result(case, "passed" if ok else "failed", "repair resume stage jump verified")

    def _case_repair_resume_audit_report(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "app.py").write_text("import gradio as gr\n", encoding="utf-8")
            runner = TaskRunner(HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
            ))
            task_id = runner.deploy(str(repo), "demo", dry_run=True)
            run_dir = root / "runs" / task_id
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(exist_ok=True)
            write_json(repair_dir / "repair_apply_result.json", {
                "status": "applied",
                "policy": {"allowed": True, "loop": {"rerun_from_effective": "verify"}},
            })
            write_json(repair_dir / "repair_verify_hints.json", {
                "verify_hints": [{"endpoint": "http://127.0.0.1:9"}],
            })
            runner.resume(task_id, dry_run=True)
            audit = read_json(run_dir / "reports" / "execution_audit.json")
            report = (run_dir / "reports" / "report.md").read_text(encoding="utf-8")
            ok = (
                audit.get("effective_start_stage") == "verify"
                and audit.get("reused_stages") == ["analyze", "resource_plan", "env_solve", "env_deploy", "model_prepare", "runner"]
                and audit.get("rerun_stages") == ["verify", "report"]
                and "## Execution Audit" in report
                and "- Rerun stages: `verify`, `report`" in report
            )
        return self._result(case, "passed" if ok else "failed", "repair resume audit report verified")

    def _case_token_report_required_env(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(parents=True)
            write_json(repair_dir / "repair_plan.json", {
                "actions": [
                    {
                        "type": "set_env_var_name_only",
                        "payload": {"env_vars": ["HF_TOKEN"], "token_value": "should_not_be_recorded"},
                    }
                ]
            })
            result = ReportGenerator().generate(
                run_dir,
                {"project": {"name": "demo", "repo_url": "local"}},
                {
                    "model_prepare": {
                        "status": "failed",
                        "summary": "auth required",
                        "data": {"diagnosis": {"category": "auth_required", "required_env_vars": ["HF_TOKEN"]}},
                    }
                },
            )
            report = Path(result.data["report_path"]).read_text(encoding="utf-8")
            ok = "`HF_TOKEN`" in report and "should_not_be_recorded" not in report and "Values are not recorded" in report
        return self._result(case, "passed" if ok else "failed", "token env var report hint verified")

    def _case_static_dashboard_export(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            task_dir = runs / "task-dashboard"
            reports = task_dir / "reports"
            reports.mkdir(parents=True)
            write_json(task_dir / "task.json", {
                "task_id": "task-dashboard",
                "project": {"name": "dashboard-demo", "repo_url": "local://demo"},
                "runtime": {"workspace_root": str(task_dir / "workspace")},
                "created_at": "2026-07-05T00:00:00Z",
            })
            write_json(task_dir / "state.json", {
                "task_id": "task-dashboard",
                "status": "completed",
                "current_stage": "report",
                "last_safe_stage": "report",
                "report_path": str(reports / "report.md"),
                "stages": {
                    "analyze": {"status": "passed", "updated_at": "2026-07-05T00:00:01Z"},
                    "verify": {"status": "passed", "updated_at": "2026-07-05T00:00:02Z"},
                    "report": {"status": "passed", "updated_at": "2026-07-05T00:00:03Z"},
                },
            })
            benchmark_path = root / "benchmark.json"
            write_json(benchmark_path, {"status": "passed", "cases": [{"id": "x", "status": "passed"}]})
            output = root / "dashboard.html"
            result = DashboardGenerator().generate(runs, output, benchmark_report=benchmark_path)
            html = output.read_text(encoding="utf-8")
            summary = read_json(output.with_suffix(".json"))
            ok = (
                result.get("status") == "generated"
                and output.exists()
                and summary.get("task_count") == 1
                and summary.get("benchmark", {}).get("case_count") == 1
                and "dashboard-demo" in html
                and "AI-Auto-Harness Dashboard" in html
            )
        return self._result(case, "passed" if ok else "failed", "static dashboard export verified")

    def _case_deployment_queue_dry_run(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("FastAPI local queue demo", encoding="utf-8")
            (repo / "app.py").write_text("print('queued')\n", encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                task_queue_dir=str(root / "queue"),
            )
            queue = DeploymentQueue(config.task_queue_path, TaskRunner(config))
            submitted = queue.submit(str(repo), name="queue-demo", dry_run=True)
            before = queue.list()
            run = queue.run_next(max_jobs=1)
            after = queue.list()
            task_id = run.get("results", [{}])[0].get("task_id", "")
            ok = (
                submitted.get("status") == "queued"
                and before.get("status_counts", {}).get("queued") == 1
                and run.get("started") == 1
                and run.get("results", [{}])[0].get("status") == "completed"
                and after.get("status_counts", {}).get("completed") == 1
                and bool(task_id)
                and (config.runs_path / task_id / "reports" / "pipeline_results.json").exists()
            )
        return self._result(case, "passed" if ok else "failed", "deployment queue dry-run scheduling verified")

    def _case_deployment_package_export(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("FastAPI package demo", encoding="utf-8")
            (repo / "app.py").write_text("print('package')\n", encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
                task_queue_dir=str(root / "queue"),
            )
            task_id = TaskRunner(config).deploy(str(repo), "package-demo", dry_run=True)
            output = root / "deployment-package.tar.gz"
            result = DeploymentPackageExporter().export(config.runs_path / task_id, output)
            with tarfile.open(output, "r:gz") as tar:
                names = tar.getnames()
            manifest = read_json(Path(result["manifest_path"]))
            ok = (
                result.get("status") == "generated"
                and output.exists()
                and manifest.get("package_sha256") == result.get("package_sha256")
                and ("%s/task.json" % task_id) in names
                and ("%s/state.json" % task_id) in names
                and ("%s/deployment_package_manifest.json" % task_id) in names
                and any(name.startswith("%s/reports/" % task_id) for name in names)
                and not any("/workspace/" in name for name in names)
            )
        return self._result(case, "passed" if ok else "failed", "deployment package export verified")

    def _case_queue_parallel_worker_pool(self, case: Dict) -> Dict:
        class BarrierRunner:
            def __init__(self):
                self.barrier = threading.Barrier(2)
                self.calls = []

            def deploy(self, repo_url, name, dry_run=True, skip_clone=False, allow_install=False, allow_start=False):
                self.calls.append(name)
                self.barrier.wait(timeout=2)
                return "task_%s" % name

        with tempfile.TemporaryDirectory() as tmp:
            runner = BarrierRunner()
            queue = DeploymentQueue(Path(tmp) / "queue", runner)
            queue.submit("local://one", name="one")
            queue.submit("local://two", name="two")
            result = queue.run_next(max_jobs=2)
            listed = queue.list()
            ok = (
                result.get("worker_count") == 2
                and result.get("started") == 2
                and [item.get("status") for item in result.get("results", [])] == ["completed", "completed"]
                and [item.get("task_id") for item in result.get("results", [])] == ["task_one", "task_two"]
                and listed.get("status_counts", {}).get("completed") == 2
                and sorted(runner.calls) == ["one", "two"]
            )
        return self._result(case, "passed" if ok else "failed", "queue parallel worker pool verified")

    def _case_queue_gpu_probe_scheduling(self, case: Dict) -> Dict:
        class FakeRunner:
            def deploy(self, repo_url, name, dry_run=True, skip_clone=False, allow_install=False, allow_start=False):
                return "task_%s" % name

        class FakeProbe:
            def probe(self):
                return {"status": "detected", "source": "fixture", "available_slots": 1, "gpus": [{"index": 0}]}

        with tempfile.TemporaryDirectory() as tmp:
            queue = DeploymentQueue(Path(tmp) / "queue", FakeRunner(), gpu_probe=FakeProbe())
            queue.submit("local://gpu-one", name="gpu-one", require_gpu=True)
            queue.submit("local://gpu-two", name="gpu-two", require_gpu=True)
            result = queue.run_next(max_jobs=2)
            listed = queue.list()
            ok = (
                result.get("gpu_probe", {}).get("source") == "fixture"
                and result.get("gpu_slots") == 1
                and result.get("started") == 1
                and result.get("results", [{}])[0].get("task_id") == "task_gpu-one"
                and result.get("skipped", [{}])[0].get("reason") == "gpu slot unavailable"
                and listed.get("status_counts", {}).get("completed") == 1
                and listed.get("status_counts", {}).get("queued") == 1
            )
        return self._result(case, "passed" if ok else "failed", "queue GPU probe scheduling verified")

    def _case_queue_claim_lock_prevents_duplicate(self, case: Dict) -> Dict:
        class FakeRunner:
            def __init__(self):
                self.calls = []

            def deploy(self, repo_url, name, dry_run=True, skip_clone=False, allow_install=False, allow_start=False):
                self.calls.append(name)
                return "task_%s" % name

        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            queue = DeploymentQueue(Path(tmp) / "queue", runner)
            submitted = queue.submit("local://locked", name="locked")
            queue._lock_path(submitted["job_id"]).write_text("pid=other\n", encoding="utf-8")
            result = queue.run_next(max_jobs=1)
            listed = queue.list()
            ok = (
                result.get("started") == 0
                and result.get("skipped", [{}])[0].get("reason") == "job already claimed"
                and listed.get("status_counts", {}).get("queued") == 1
                and runner.calls == []
            )
        return self._result(case, "passed" if ok else "failed", "queue claim lock duplicate prevention verified")

    def _case_queue_stale_claim_lock_recovery(self, case: Dict) -> Dict:
        class FakeRunner:
            def deploy(self, repo_url, name, dry_run=True, skip_clone=False, allow_install=False, allow_start=False):
                return "task_%s" % name

        with tempfile.TemporaryDirectory() as tmp:
            queue = DeploymentQueue(Path(tmp) / "queue", FakeRunner(), claim_ttl_seconds=1)
            submitted = queue.submit("local://stale", name="stale")
            lock_path = queue._lock_path(submitted["job_id"])
            lock_path.write_text("pid=dead\n", encoding="utf-8")
            old = 1
            os.utime(lock_path, (old, old))
            result = queue.run_next(max_jobs=1)
            listed = queue.list()
            ok = (
                result.get("started") == 1
                and result.get("results", [{}])[0].get("task_id") == "task_stale"
                and result.get("recovered_locks", [{}])[0].get("job_id") == submitted["job_id"]
                and not lock_path.exists()
                and listed.get("status_counts", {}).get("completed") == 1
            )
        return self._result(case, "passed" if ok else "failed", "queue stale claim lock recovery verified")

    def _stage_update_count(self, run_dir: Path) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("type") == "stage_update":
                counts[event["stage"]] = counts.get(event["stage"], 0) + 1
        return counts

    def _result(self, case: Dict, status: str, reason: str) -> Dict:
        return {
            "id": case.get("id"),
            "status": status,
            "purpose": case.get("purpose", ""),
            "expected_signal": case.get("expected_signal", ""),
            "reason": reason,
        }
