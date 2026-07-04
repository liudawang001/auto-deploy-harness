import json
import tempfile
from pathlib import Path
from typing import Dict, List

from auto_harness.assets import HuggingFaceDownloader, ModelCache
from auto_harness.assets.manifest import ModelAsset
from auto_harness.config import HarnessConfig
from auto_harness.diagnostics import LogClassifier
from auto_harness.models.base import read_json, write_json
from auto_harness.modules.reporter import ReportGenerator
from auto_harness.modules.runner import RunnerModule
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.task import RuntimePolicy
from auto_harness.repair import RepairLoopController, RepairPolicy
from auto_harness.orchestrator import TaskRunner
from auto_harness.verify import BrowserVerifier, StreamlitVerifier


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
    def run(self, manifest_path: Path, output_path: Path = None) -> Dict:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases: List[Dict] = []
        for case in manifest.get("cases", []):
            result = self._run_case(case, manifest_path.parent)
            cases.append(result)
        report = {
            "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
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
            if case_id == "gradio_api_shape_variation":
                return self._case_gradio_api_shape_variation(case)
            if case_id == "gradio_queue_call_followup":
                return self._case_gradio_queue_call_followup(case)
            if case_id == "token_missing_diagnosis":
                return self._case_token_missing_diagnosis(case)
            if case_id == "repair_resume_stage_jump":
                return self._case_repair_resume_stage_jump(case)
            if case_id == "repair_resume_audit_report":
                return self._case_repair_resume_audit_report(case)
            if case_id == "token_report_required_env":
                return self._case_token_report_required_env(case)
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
                and all(after.get(stage, 0) == before.get(stage, 0) for stage in ("analyze", "resource_plan", "env_deploy", "model_prepare", "runner"))
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
                and audit.get("reused_stages") == ["analyze", "resource_plan", "env_deploy", "model_prepare", "runner"]
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
