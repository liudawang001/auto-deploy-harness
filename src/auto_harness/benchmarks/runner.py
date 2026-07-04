import json
import tempfile
from pathlib import Path
from typing import Dict, List

from auto_harness.assets import HuggingFaceDownloader, ModelCache
from auto_harness.assets.manifest import ModelAsset
from auto_harness.models.base import write_json
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.task import RuntimePolicy
from auto_harness.repair import RepairPolicy
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
            return self._result(case, "skipped", "unknown benchmark case")
        except Exception as exc:  # noqa: BLE001 - benchmark report should continue
            return self._result(case, "failed", str(exc))

    def _case_model_download_resume(self, case: Dict) -> Dict:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))
            target_dir = Path(asset.cache_path)
            partial = target_dir / "model.safetensors.part"
            partial.parent.mkdir(parents=True)
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
            target.parent.mkdir(parents=True)
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
            target.parent.mkdir(parents=True)
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

    def _result(self, case: Dict, status: str, reason: str) -> Dict:
        return {
            "id": case.get("id"),
            "status": status,
            "purpose": case.get("purpose", ""),
            "expected_signal": case.get("expected_signal", ""),
            "reason": reason,
        }
