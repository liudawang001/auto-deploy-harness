import json
import tempfile
from pathlib import Path
from typing import Dict, List

from auto_harness.assets import HuggingFaceDownloader, ModelCache
from auto_harness.assets.manifest import ModelAsset
from auto_harness.models.base import write_json
from auto_harness.modules.verify import VerifyModule
from auto_harness.verify import StreamlitVerifier


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

    def _result(self, case: Dict, status: str, reason: str) -> Dict:
        return {
            "id": case.get("id"),
            "status": status,
            "purpose": case.get("purpose", ""),
            "expected_signal": case.get("expected_signal", ""),
            "reason": reason,
        }
