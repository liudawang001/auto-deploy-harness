import json
import hashlib
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.benchmarks import BenchmarkRunner
from auto_harness.cli import main as cli_main
from auto_harness.assets.huggingface import HuggingFaceDownloader
from auto_harness.assets.modelscope import ModelScopeDownloader
from auto_harness.diagnostics import LogClassifier
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.agents.base import AgentResult
from auto_harness.memory import MemoryStore
from auto_harness.assets import ModelCache, ModelAssetDetector, ModelFileSelector
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.model_prepare import ModelPrepareModule
from auto_harness.modules.resource_plan import ResourcePlanner
from auto_harness.modules.runner import RunnerModule
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.result import StageResult
from auto_harness.providers import Message, MockLLMProvider
from auto_harness.skills import SkillRegistry
from auto_harness.state import StateStore
from auto_harness.orchestrator import TaskRunner
from auto_harness.repair import RepairApplier, RepairLoopController, RepairOverlay, RepairPlanner, RepairPolicy
from auto_harness.verify import BrowserVerifier
from auto_harness.utils.time import utc_now_iso


class FakeHttpResponse:
    def __init__(self, body: str, status: int = 200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body.encode("utf-8")


class FakeStreamingResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
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


class FakeAgentExecutor:
    def __init__(self, text='{"risk":"low"}'):
        self.text = text
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return AgentResult(status="passed", text=self.text)

    def resume(self, session_id, request):
        return self.run(request)


class FakeBrowserBackend:
    def __init__(self, html_template: str):
        self.html_template = html_template
        self.urls = []

    def load(self, url: str, timeout_ms: int = 15000, screenshot_path: Path = None):
        self.urls.append(url)
        trace = url.split("_auto_harness_trace=", 1)[1] if "_auto_harness_trace=" in url else ""
        return {
            "status": "loaded",
            "url": url,
            "title": "fake browser",
            "status_code": 200,
            "html": self.html_template.replace("{{trace_id}}", trace),
        }


class CoreTests(unittest.TestCase):
    def test_state_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp))
            spec = TaskSpec(
                task_id="task1",
                project=ProjectSpec(name="demo", repo_url="local"),
                runtime=RuntimePolicy(workspace_root=str(Path(tmp) / "task1" / "workspace")),
                created_at=utc_now_iso(),
            )
            store.create_task(spec)
            store.update_stage("task1", "analyze", "passed", result_path="analysis.json", progress={"step": "done"})
            state = store.load_state("task1")
            self.assertEqual(state.stages["analyze"].status, "passed")
            self.assertEqual(state.stages["analyze"].progress["step"], "done")
            self.assertEqual(state.last_safe_stage, "analyze")
            self.assertIn("resource_plan", state.stages)
            self.assertIn("model_prepare", state.stages)

    def test_analyzer_detects_gradio_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "requirements.txt").write_text("gradio\n", encoding="utf-8")
            (repo / "app.py").write_text("import gradio as gr\n", encoding="utf-8")
            result = ProjectAnalyzer().analyze(repo)
            self.assertEqual(result.status, "passed")
            self.assertIn("gradio", result.data["frameworks"])
            self.assertTrue(result.data["install_plan"])
            self.assertTrue(result.data["run_candidates"])
            self.assertEqual(result.data["verify_hint"]["request"]["method"], "POST")

    def test_analyzer_can_call_optional_agent_advisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "README.md").write_text("demo", encoding="utf-8")
            executor = FakeAgentExecutor()
            result = ProjectAnalyzer(agent_executor=executor, use_agent=True).analyze(repo)
            self.assertIn("agent_advice", result.data)
            self.assertEqual(result.data["agent_advice"]["risk"], "low")
            self.assertEqual(len(executor.requests), 1)

    def test_verify_does_not_pass_without_artifact_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule().verify(
                run_dir,
                analysis={},
                runner_result={"pid": 1234, "expected_port": 0, "service_ready": False},
            )
            self.assertEqual(result.status, "uncertain")
            self.assertEqual(result.data["status"], "uncertain")

    def test_verify_passes_when_http_response_contains_trace(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            trace = req.full_url.split("_auto_harness_trace=")[1]
            return FakeHttpResponse("handled trace %s" % trace)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/echo"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.data["status"], "pass")
            self.assertIn("_auto_harness_trace=", captured["url"])

    def test_verify_post_json_trace(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = req.data.decode("utf-8")
            trace = json.loads(captured["body"])["prompt"].replace("verify ", "")
            return FakeHttpResponse("accepted verify %s" % trace)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={
                    "verify_hint": {
                        "endpoint": "http://127.0.0.1:8000",
                        "request": {
                            "method": "POST",
                            "path": "/api/check",
                            "json": {"prompt": "verify {{trace_id}}"},
                        },
                    }
                },
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
            self.assertEqual(result.status, "passed")
            self.assertEqual(captured["method"], "POST")
            self.assertEqual(captured["url"], "http://127.0.0.1:8000/api/check")
            self.assertIn("verify_", captured["body"])

    def test_verify_discovers_gradio_config_request(self):
        captured = {"config_called": False}

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/config"):
                captured["config_called"] = True
                return FakeHttpResponse(json.dumps({
                    "dependencies": [
                        {"id": 3, "api_name": "predict", "backend_fn": True}
                    ]
                }))
            captured["url"] = req.full_url
            captured["body"] = req.data.decode("utf-8")
            trace = json.loads(captured["body"])["data"][0]
            return FakeHttpResponse("gradio handled %s" % trace)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
            self.assertEqual(result.status, "passed")
            self.assertTrue(captured["config_called"])
            self.assertEqual(captured["url"], "http://127.0.0.1:7860/api/predict")
            self.assertEqual(json.loads(captured["body"])["fn_index"], 3)

    def test_verify_handles_gradio_api_shape_variation(self):
        captured = {}

        def fake_urlopen(req, timeout):
            if req.full_url.endswith("/config"):
                return FakeHttpResponse(json.dumps({
                    "dependencies": [
                        {"id": 1, "api_name": False, "backend_fn": False},
                        {"id": 7, "api_name": "/predict", "backend_fn": True},
                    ]
                }))
            captured["url"] = req.full_url
            captured["body"] = req.data.decode("utf-8")
            trace = json.loads(captured["body"])["data"][0]
            return FakeHttpResponse(json.dumps({"data": [trace]}))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
            self.assertEqual(result.status, "passed")
            self.assertEqual(captured["url"], "http://127.0.0.1:7860/api/predict")
            self.assertEqual(json.loads(captured["body"])["fn_index"], 7)

    def test_mock_provider(self):
        result = MockLLMProvider().complete([Message(role="user", content="hello")])
        self.assertIn("mock provider response", result.text)

    def test_env_deploy_rejects_disallowed_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = EnvDeployModule().deploy(
                Path(tmp),
                {"install_plan": [["bash", "-c", "echo unsafe"]]},
                execute=True,
                allowed_commands=["python3"],
            )
            self.assertEqual(result.status, "failed")
            self.assertIn("disallowed command", result.error)

    def test_runner_detects_service_exit_after_start(self):
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
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.summary, "service process exited")

    def test_skill_registry_selects_verify_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "verify-evidence"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: verify-evidence\n"
                "description: Verify Gradio API trace evidence during verify stage.\n"
                "---\n"
                "# Verify\n"
                "Use POST /api/predict for gradio trace verification.\n",
                encoding="utf-8",
            )
            selected = SkillRegistry(Path(tmp) / "skills").select_for_stage(
                "verify",
                {"frameworks": ["gradio"], "verify_hint": {"service_type": "webui"}},
            )
            self.assertEqual(selected[0].name, "verify-evidence")
            self.assertIn("api/predict", selected[0].content)
            self.assertEqual(len(selected[0].sha256), 64)

    def test_memory_store_records_and_queries_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory")
            analysis = {"frameworks": ["gradio"]}
            entry = store.remember_issue(
                "task1",
                "verify",
                StageResult(
                    "verify",
                    "uncertain",
                    "verify completed with uncertain",
                    {"diagnosis": {"category": "trace_not_observed", "root_cause": "response did not contain trace"}},
                ),
                analysis,
            )
            self.assertIsNotNone(entry)
            hits = store.query("verify", analysis)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["category"], "trace_not_observed")
            store.remember_issue("task1", "verify", StageResult("verify", "uncertain", "verify completed with uncertain"), analysis)
            self.assertEqual(len(store.query("verify", analysis)), 2)

    def test_model_asset_detector_finds_huggingface_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "README.md").write_text(
                "Download from https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct. Size 2 GB.",
                encoding="utf-8",
            )
            (repo / "app.py").write_text(
                'model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")\n',
                encoding="utf-8",
            )
            assets = ModelAssetDetector().detect(repo)
            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0].source, "huggingface")
            self.assertEqual(assets[0].repo_id, "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertGreater(assets[0].expected_size_bytes, 0)

    def test_resource_planner_includes_model_assets_and_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "README.md").write_text(
                "Use CUDA GPU with https://huggingface.co/org/demo-model",
                encoding="utf-8",
            )
            result = ResourcePlanner().plan(repo, {"frameworks": ["torch", "transformers"]})
            self.assertEqual(result.status, "passed")
            self.assertTrue(result.data["gpu_required"])
            self.assertEqual(result.data["risk_level"], "high")
            self.assertEqual(result.data["model_assets"][0]["repo_id"], "org/demo-model")
            self.assertIn("HF_TOKEN", result.data["external_tokens"])

    def test_model_prepare_writes_manifest_and_cache_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            resource_plan = {
                "model_assets": [
                    {
                        "asset_id": "huggingface:org/demo-model",
                        "source": "huggingface",
                        "repo_id": "org/demo-model",
                        "revision": "main",
                        "origin": "README.md",
                        "status": "planned",
                    }
                ]
            }
            result = ModelPrepareModule(ModelCache(Path(tmp) / "model_cache")).prepare(run_dir, resource_plan)
            self.assertEqual(result.status, "passed")
            self.assertTrue(Path(result.data["manifest_path"]).exists())
            self.assertEqual(result.data["assets"][0]["source"], "huggingface")
            self.assertIn("model_cache", result.data["assets"][0]["cache_path"])

    def test_huggingface_downloader_resumes_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAssetDetector().detect(self._repo_with_hf_model(Path(tmp) / "repo"))[0])
            target_dir = Path(asset.cache_path)
            partial = target_dir / "model.safetensors.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"abc")
            calls = []

            def fake_urlopen(req, timeout):
                calls.append(req)
                if "/api/models/" in req.full_url:
                    return FakeStreamingResponse(json.dumps([
                        {"type": "file", "path": "model.safetensors", "size": 6}
                    ]).encode("utf-8"))
                return FakeStreamingResponse(b"def", status=206)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=2).download(asset)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual((target_dir / "model.safetensors").read_bytes(), b"abcdef")
            self.assertEqual(calls[1].headers.get("Range"), "bytes=3-")

    def test_huggingface_downloader_records_etag_and_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAssetDetector().detect(self._repo_with_hf_model(Path(tmp) / "repo"))[0])
            digest = hashlib.sha256(b"abc").hexdigest()

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return FakeStreamingResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 3, "sha256": digest, "oid": "etag123"}
                    ]).encode("utf-8"))
                return FakeStreamingResponse(b"abc", status=200)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=2).download(asset)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.files[0]["etag"], "etag123")
            self.assertTrue(result.files[0]["verified"])

    def test_huggingface_downloader_redownloads_on_etag_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAssetDetector().detect(self._repo_with_hf_model(Path(tmp) / "repo"))[0])
            target = Path(asset.cache_path) / "config.json"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"old")
            (target.parent / "config.json.auto_harness_meta.json").write_text(
                json.dumps({"size_bytes": 3, "etag": "old-etag"}),
                encoding="utf-8",
            )
            download_calls = []

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return FakeStreamingResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 3, "oid": "new-etag"}
                    ]).encode("utf-8"))
                download_calls.append(req)
                return FakeStreamingResponse(b"new", status=200)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=2).download(asset)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(len(download_calls), 1)
            self.assertTrue(result.files[0]["etag_verified"])

    def test_huggingface_downloader_can_download_files_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAssetDetector().detect(self._repo_with_hf_model(Path(tmp) / "repo"))[0])
            download_urls = []

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return FakeStreamingResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 1},
                        {"type": "file", "path": "tokenizer.json", "size": 1},
                    ]).encode("utf-8"))
                download_urls.append(req.full_url)
                if req.full_url.endswith("config.json"):
                    return FakeStreamingResponse(b"a", status=200)
                return FakeStreamingResponse(b"b", status=200)

            result = HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1, max_workers=2).download(asset)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual([file["path"] for file in result.files], ["config.json", "tokenizer.json"])
            self.assertEqual(len(download_urls), 2)
            self.assertEqual((Path(asset.cache_path) / "config.json").read_bytes(), b"a")
            self.assertEqual((Path(asset.cache_path) / "tokenizer.json").read_bytes(), b"b")

    def test_model_file_selector_skips_readme_and_scripts(self):
        selector = ModelFileSelector()
        self.assertTrue(selector.should_download("model.safetensors"))
        self.assertTrue(selector.should_download("tokenizer.json"))
        self.assertFalse(selector.should_download("README.md"))
        self.assertFalse(selector.should_download("app.py"))

    def test_modelscope_downloader_downloads_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("https://modelscope.cn/models/org/demo-model", encoding="utf-8")
            asset = cache.reserve(ModelAssetDetector().detect(repo)[0])
            target_dir = Path(asset.cache_path)
            partial = target_dir / "weights.bin.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"12")
            calls = []

            def fake_urlopen(req, timeout):
                calls.append(req)
                if "/repo/files" in req.full_url:
                    return FakeStreamingResponse(json.dumps({
                        "Data": [
                            {"Path": "weights.bin", "Size": 4}
                        ]
                    }).encode("utf-8"))
                return FakeStreamingResponse(b"34", status=206)

            result = ModelScopeDownloader(
                urlopen=fake_urlopen,
                token="",
                api_base="https://mock.modelscope/api/v1/models",
                download_base="https://mock.modelscope/models",
                chunk_size=1,
            ).download(asset)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual((target_dir / "weights.bin").read_bytes(), b"1234")
            self.assertEqual(calls[1].headers.get("Range"), "bytes=2-")

    def test_model_cache_cleanup_plans_and_deletes_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            old_dir = cache.root / "huggingface" / "old"
            new_dir = cache.root / "huggingface" / "new"
            old_dir.mkdir(parents=True)
            new_dir.mkdir(parents=True)
            (old_dir / "model.bin").write_bytes(b"old")
            (new_dir / "model.bin").write_bytes(b"newer")
            plan = cache.cleanup(max_total_bytes=5, dry_run=True)
            self.assertEqual(plan["candidate_count"], 1)
            self.assertTrue(old_dir.exists())
            applied = cache.cleanup(max_total_bytes=5, dry_run=False)
            self.assertEqual(len(applied["deleted"]), 1)
            self.assertFalse(old_dir.exists())
            self.assertTrue(new_dir.exists())

    def _repo_with_hf_model(self, repo: Path) -> Path:
        repo.mkdir()
        (repo / "README.md").write_text("https://huggingface.co/org/demo-model", encoding="utf-8")
        return repo

    def test_log_classifier_detects_common_failures(self):
        result = LogClassifier().classify("ModuleNotFoundError: No module named 'gradio'")
        self.assertEqual(result["category"], "dependency_missing")
        self.assertGreater(result["confidence"], 0.8)

    def test_log_classifier_detects_missing_token(self):
        result = LogClassifier().classify("401 Unauthorized: Repository Not Found. Please set HF_TOKEN.")
        self.assertEqual(result["category"], "auth_required")
        self.assertGreaterEqual(result["confidence"], 0.9)

    def test_repair_planner_proposes_dependency_action(self):
        plan = RepairPlanner().propose(
            "env_deploy",
            StageResult(
                "env_deploy",
                "failed",
                "dependency installation failed",
                {"diagnosis": {"category": "dependency_missing", "signal": "gradio", "confidence": 0.9}},
            ),
            {"frameworks": ["gradio"]},
        )
        self.assertEqual(plan["rerun_from"], "env_deploy")
        self.assertEqual(plan["actions"][0]["type"], "install_package")
        self.assertEqual(plan["actions"][0]["payload"]["package"], "gradio")

    def test_repair_policy_rejects_dependency_install_without_permission(self):
        result = RepairPolicy().check(
            {
                "actions": [
                    {
                        "type": "install_package",
                        "requires": {"dependency_install": True},
                        "payload": {"package": "gradio"},
                    }
                ]
            },
            RuntimePolicy(workspace_root="/tmp/demo", allow_dependency_install=False),
        )
        self.assertFalse(result["allowed"])
        self.assertIn("dependency install is not allowed", result["decisions"][0]["reasons"])

    def test_repair_policy_allows_operator_approved_action(self):
        action = {
            "type": "change_cache_dir",
            "requires": {"operator_approval": True},
            "payload": {"config": "model_cache_dir"},
        }
        runtime = RuntimePolicy(workspace_root="/tmp/demo")
        rejected = RepairPolicy().check({"actions": [action]}, runtime)
        approved = RepairPolicy().check(
            {"actions": [action]},
            runtime,
            operator_approval={"approved": True, "approved_action_types": ["change_cache_dir"]},
        )
        self.assertFalse(rejected["allowed"])
        self.assertTrue(approved["allowed"])

    def test_repair_loop_limits_attempts_and_safely_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = RepairLoopController(max_attempts=1)
            entry = {"signature": "sig-repair"}
            policy = {"allowed": True, "decisions": []}
            first_plan = {"root_cause": "missing trace", "rerun_from": "unsafe_stage", "actions": []}
            first = controller.gate(Path(tmp), "verify", entry, first_plan, policy)
            second_plan = {"root_cause": "missing trace", "rerun_from": "unsafe_stage", "actions": []}
            second = controller.gate(Path(tmp), "verify", entry, second_plan, policy)
            self.assertTrue(first["allowed"])
            self.assertEqual(first["loop"]["rerun_from_effective"], "verify")
            self.assertFalse(second["allowed"])
            self.assertIn("repair attempt limit reached", second["loop"]["reasons"])
            self.assertTrue((Path(tmp) / "repairs" / "repair_loop_state.json").exists())

    def test_repair_overlay_merges_allowed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            plan = {
                "actions": [
                    {
                        "type": "install_package",
                        "requires": {"dependency_install": True},
                        "payload": {"package": "gradio"},
                    },
                    {
                        "type": "update_verify_hint",
                        "requires": {"source_edit": False},
                        "payload": {
                            "endpoint": "http://127.0.0.1:7860",
                            "request": {"method": "POST", "path": "/api/predict", "json": {"data": ["{{trace_id}}"]}},
                        },
                    },
                ]
            }
            policy = {"allowed": True, "decisions": [{"allowed": True}, {"allowed": True}]}
            RepairApplier().apply(run_dir, plan, policy)
            overlay = RepairOverlay().load(run_dir)
            merged = RepairOverlay().merge_analysis(
                {"install_plan": [["python3", "-m", "venv", ".venv"]], "verify_hint": {}},
                overlay,
            )
            self.assertTrue(overlay["active"])
            self.assertIn([".venv/bin/python", "-m", "pip", "install", "gradio"], merged["install_plan"])
            self.assertEqual(merged["verify_hint"]["endpoint"], "http://127.0.0.1:7860")
            self.assertEqual(merged["verify_hint"]["request"]["path"], "/api/predict")

    def test_rejected_repair_apply_result_disables_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            plan = {"actions": [{"type": "update_verify_hint", "payload": {"endpoint": "http://127.0.0.1:7860"}}]}
            RepairApplier().apply(run_dir, plan, {"allowed": True, "decisions": []})
            self.assertTrue(RepairOverlay().load(run_dir)["active"])
            RepairApplier().apply(run_dir, plan, {"allowed": False, "decisions": [{"allowed": False}]})
            self.assertFalse(RepairOverlay().load(run_dir)["active"])

    def test_cli_repair_approve_writes_operator_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "runs_dir": str(root / "runs"),
                "memory_dir": str(root / "memory"),
                "model_cache_dir": str(root / "model_cache"),
            }), encoding="utf-8")
            repair_dir = root / "runs" / "task1" / "repairs"
            repair_dir.mkdir(parents=True)
            (repair_dir / "repair_plan.json").write_text(json.dumps({
                "root_cause": "disk full",
                "rerun_from": "model_prepare",
                "actions": [{"type": "change_cache_dir", "requires": {"operator_approval": True}}],
            }), encoding="utf-8")
            old_config = os.environ.get("AUTO_HARNESS_CONFIG")
            os.environ["AUTO_HARNESS_CONFIG"] = str(config_path)
            try:
                with redirect_stdout(io.StringIO()):
                    code = cli_main(["repair-approve", "--task-id", "task1", "--note", "approved in test"])
            finally:
                if old_config is None:
                    os.environ.pop("AUTO_HARNESS_CONFIG", None)
                else:
                    os.environ["AUTO_HARNESS_CONFIG"] = old_config
            self.assertEqual(code, 0)
            approval = json.loads((repair_dir / "operator_approval.json").read_text(encoding="utf-8"))
            self.assertTrue(approval["approved"])
            self.assertEqual(approval["approved_action_types"], ["change_cache_dir"])

    def test_verify_streamlit_dom_probe_can_pass_with_trace(self):
        def fake_urlopen(req, timeout):
            if "_auto_harness_trace=" in req.full_url:
                trace = req.full_url.split("_auto_harness_trace=")[1]
                return FakeHttpResponse('<html><script>streamlit</script><div>%s</div></html>' % trace)
            return FakeHttpResponse("ok")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["streamlit"], "verify_hint": {"endpoint": "http://127.0.0.1:8501"}},
                runner_result={"pid": 1234, "expected_port": 8501, "service_ready": True},
            )
            self.assertEqual(result.status, "passed")
            self.assertTrue(any(check["name"] == "streamlit_dom_probe" for check in result.data["checks"]))

    def test_verify_streamlit_dom_probe_fails_on_error_page(self):
        page = Path("tests/fixtures/benchmarks/streamlit_error_page.html").read_text(encoding="utf-8")

        def fake_urlopen(req, timeout):
            return FakeHttpResponse(page)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["streamlit"], "verify_hint": {"endpoint": "http://127.0.0.1:8501"}},
                runner_result={"pid": 1234, "expected_port": 8501, "service_ready": True},
            )
            checks = {check["name"]: check for check in result.data["checks"]}
            self.assertEqual(checks["streamlit_dom_probe"]["status"], "fail")
            self.assertEqual(result.status, "uncertain")

    def test_verify_ignores_stale_artifact(self):
        def fake_urlopen(req, timeout):
            return FakeHttpResponse("ok without current trace")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)
            (repo / "old_output.txt").write_text("old successful output", encoding="utf-8")
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/health"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
            checks = {check["name"]: check for check in result.data["checks"]}
            self.assertEqual(result.status, "uncertain")
            self.assertEqual(checks["artifact_freshness"]["status"], "uncertain")

    def test_verify_browser_dom_probe_can_pass_with_trace(self):
        def fake_urlopen(req, timeout):
            return FakeHttpResponse("http response without trace")

        browser = BrowserVerifier(
            browser_backend=FakeBrowserBackend('<html><body><div class="gradio-container">{{trace_id}}</div></body></html>')
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen, browser_verifier=browser).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
            checks = {check["name"]: check for check in result.data["checks"]}
            self.assertEqual(checks["browser_dom_probe"]["status"], "pass")
            self.assertEqual(result.status, "passed")

    def test_browser_dom_probe_fails_on_error_marker(self):
        browser = BrowserVerifier(
            browser_backend=FakeBrowserBackend("<html><body>Traceback (most recent call last)</body></html>")
        )
        check = browser.probe("http://127.0.0.1:7860", "trace123", frameworks=["gradio"])
        self.assertEqual(check["status"], "fail")
        self.assertEqual(check["name"], "browser_dom_probe")

    def test_model_prepare_progress_callback_receives_download_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            updates = []

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return FakeStreamingResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 3}
                    ]).encode("utf-8"))
                return FakeStreamingResponse(b"abc", status=200)

            result = ModelPrepareModule(
                cache,
                huggingface_downloader=HuggingFaceDownloader(urlopen=fake_urlopen, token="", chunk_size=1),
            ).prepare(
                run_dir,
                {
                    "model_assets": [
                        {
                            "asset_id": "huggingface:org/demo-model",
                            "source": "huggingface",
                            "repo_id": "org/demo-model",
                        }
                    ]
                },
                execute=True,
                progress_callback=lambda progress: updates.append(progress),
            )
            self.assertEqual(result.status, "passed")
            self.assertTrue(any(update.get("status") == "downloading" for update in updates))
            self.assertEqual(result.data["progress"]["downloaded_bytes"], 3)

    def test_benchmark_fixture_manifest_is_present(self):
        manifest = json.loads(Path("tests/fixtures/benchmarks/manifest.json").read_text(encoding="utf-8"))
        ids = {case["id"] for case in manifest["cases"]}
        self.assertIn("model_download_resume", ids)
        self.assertIn("verify_false_positive_http_200", ids)
        self.assertIn("gradio_config_discovery", ids)
        self.assertIn("repair_policy_reject", ids)
        self.assertIn("checksum_failure", ids)
        self.assertIn("browser_dom_trace", ids)
        self.assertIn("parallel_model_download", ids)
        self.assertIn("etag_cache_invalidation", ids)
        self.assertIn("cache_cleanup_plan", ids)
        self.assertIn("repair_loop_attempt_limit", ids)
        self.assertIn("operator_repair_approval", ids)
        self.assertIn("service_exits_after_start", ids)
        self.assertIn("stale_artifact_ignored", ids)
        self.assertIn("gradio_api_shape_variation", ids)
        self.assertIn("token_missing_diagnosis", ids)

    def test_benchmark_runner_executes_all_fixture_cases(self):
        report = BenchmarkRunner().run(Path("tests/fixtures/benchmarks/manifest.json"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 17)
        self.assertTrue(all(case["status"] == "passed" for case in report["cases"]))

    def test_benchmark_cli_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark_report.json"
            with redirect_stdout(io.StringIO()):
                code = cli_main([
                    "benchmark",
                    "--manifest",
                    "tests/fixtures/benchmarks/manifest.json",
                    "--output",
                    str(output),
                ])
            self.assertEqual(code, 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "passed")

    def test_task_runner_dry_run_includes_resource_and_model_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "requirements.txt").write_text("gradio\ntransformers\n", encoding="utf-8")
            (repo / "README.md").write_text("Model: https://huggingface.co/org/demo-model", encoding="utf-8")
            (repo / "app.py").write_text("import gradio as gr\n", encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
            )
            task_id = TaskRunner(config).deploy(str(repo), "demo", dry_run=True)
            pipeline_path = root / "runs" / task_id / "reports" / "pipeline_results.json"
            data = json.loads(pipeline_path.read_text(encoding="utf-8"))
            self.assertIn("resource_plan", data)
            self.assertIn("model_prepare", data)
            self.assertEqual(data["resource_plan"]["data"]["model_assets"][0]["repo_id"], "org/demo-model")
            self.assertTrue(Path(data["model_prepare"]["data"]["manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
