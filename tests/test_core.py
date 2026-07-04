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
from auto_harness.cli import _apply_cli_overrides, build_parser, main as cli_main
from auto_harness.assets.huggingface import HuggingFaceDownloader
from auto_harness.assets.modelscope import ModelScopeDownloader
from auto_harness.assets.manifest import ModelAsset
from auto_harness.diagnostics import LogClassifier
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.agents.base import AgentResult
from auto_harness.memory import MemoryStore
from auto_harness.assets import GitLFSDetector, ModelCache, ModelAssetDetector, ModelFileSelector
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.env_solve import EnvSolveModule
from auto_harness.modules.model_prepare import ModelPrepareModule
from auto_harness.modules.resource_plan import ResourcePlanner
from auto_harness.modules.reporter import ReportGenerator
from auto_harness.modules.runner import RunnerModule
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.result import StageResult
from auto_harness.providers import Message, MockLLMProvider
from auto_harness.skills import SkillRegistry
from auto_harness.state import StateStore
from auto_harness.orchestrator import TaskRunner
from auto_harness.repair import RepairApplier, RepairLoopController, RepairOverlay, RepairPlanner, RepairPolicy
from auto_harness.verify import BrowserVerifier
from auto_harness.utils.shell import CommandResult
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

    def test_config_loads_download_and_cache_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "model_download_max_workers": 4,
                "model_download_retry_count": 3,
                "model_download_retry_backoff_seconds": 0.25,
                "model_cache_cleanup_max_total_bytes": 1024,
                "model_cache_cleanup_older_than_days": 7,
                "model_cache_cleanup_source": "huggingface",
                "model_cache_cleanup_repo_id": "org/demo",
                "model_cache_cleanup_keep_cache_keys": ["keep-key"],
                "model_cache_cleanup_keep_repo_ids": ["org/keep"],
            }), encoding="utf-8")
            config = HarnessConfig.load(str(path))
            self.assertEqual(config.model_download_max_workers, 4)
            self.assertEqual(config.model_download_retry_count, 3)
            self.assertEqual(config.model_download_retry_backoff_seconds, 0.25)
            self.assertEqual(config.model_cache_cleanup_max_total_bytes, 1024)
            self.assertEqual(config.model_cache_cleanup_older_than_days, 7)
            self.assertEqual(config.model_cache_cleanup_source, "huggingface")
            self.assertEqual(config.model_cache_cleanup_repo_id, "org/demo")
            self.assertEqual(config.model_cache_cleanup_keep_cache_keys, ["keep-key"])
            self.assertEqual(config.model_cache_cleanup_keep_repo_ids, ["org/keep"])

    def test_cli_download_overrides_update_config(self):
        args = build_parser().parse_args([
            "deploy",
            "--repo",
            "local",
            "--model-download-workers",
            "3",
            "--download-retries",
            "5",
            "--download-retry-backoff",
            "0",
        ])
        config = HarnessConfig()
        _apply_cli_overrides(config, args)
        self.assertEqual(config.model_download_max_workers, 3)
        self.assertEqual(config.model_download_retry_count, 5)
        self.assertEqual(config.model_download_retry_backoff_seconds, 0.0)

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

    def test_verify_supports_gradio_queue_call_followup(self):
        captured = {"urls": []}

        def fake_urlopen(req, timeout):
            captured["urls"].append(req.full_url)
            if req.full_url.endswith("/config"):
                return FakeHttpResponse(json.dumps({
                    "enable_queue": True,
                    "dependencies": [
                        {"id": 7, "api_name": "/predict", "backend_fn": True, "queue": True},
                    ],
                }))
            if req.full_url.endswith("/call/predict"):
                captured["body"] = req.data.decode("utf-8")
                return FakeHttpResponse(json.dumps({"event_id": "evt-123"}))
            if req.full_url.endswith("/call/predict/evt-123"):
                trace = json.loads(captured["body"])["data"][0]
                return FakeHttpResponse("event: complete\ndata: {\"data\": [\"%s\"]}\n\n" % trace)
            raise AssertionError("unexpected url %s" % req.full_url)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "workspace" / "repo").mkdir(parents=True)
            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"frameworks": ["gradio"], "verify_hint": {"endpoint": "http://127.0.0.1:7860"}},
                runner_result={"pid": 1234, "expected_port": 7860, "service_ready": True},
            )
            self.assertEqual(result.status, "passed")
            self.assertIn("http://127.0.0.1:7860/call/predict", captured["urls"])
            self.assertIn("http://127.0.0.1:7860/call/predict/evt-123", captured["urls"])
            evidence = json.loads(Path(result.evidence[1]).read_text(encoding="utf-8"))
            self.assertTrue(evidence["request"]["discovery"]["queue_enabled"])
            self.assertTrue(evidence["follow_up_response"]["trace_found"])

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

    def test_git_lfs_detector_reads_attributes_and_pointers(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
            (repo / "model.safetensors").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:%s\n"
                "size 123456\n" % ("a" * 64),
                encoding="utf-8",
            )
            result = GitLFSDetector(available=False).detect(repo)
            self.assertTrue(result["required"])
            self.assertFalse(result["available"])
            self.assertEqual(result["patterns"], ["*.safetensors"])
            self.assertEqual(result["pointers"][0]["path"], "model.safetensors")
            self.assertEqual(result["pointers"][0]["size_bytes"], 123456)
            self.assertEqual(result["diagnosis"]["category"], "git_lfs_missing")

    def test_resource_planner_marks_missing_git_lfs_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
            result = ResourcePlanner(git_lfs_detector=GitLFSDetector(available=False)).plan(repo, {"frameworks": []})
            self.assertEqual(result.status, "uncertain")
            self.assertEqual(result.data["diagnosis"]["category"], "git_lfs_missing")
            self.assertTrue(result.data["git_lfs"]["prepare_commands"])
            self.assertIn("Git LFS model files detected", result.data["risk_reasons"])

    def test_env_solve_adds_legacy_gradio_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "requirements.txt").write_text("gradio\nopencv-python\n", encoding="utf-8")
            analysis = {
                "frameworks": ["gradio"],
                "install_plan": [
                    ["python3", "-m", "venv", ".venv"],
                    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
                ],
            }
            result = EnvSolveModule().solve(repo, analysis, {"python_range": ">=3.10,<3.12"})
            self.assertEqual(result.status, "passed")
            self.assertIn("numpy<2", result.data["constraints"])
            self.assertIn("pydantic<2", result.data["constraints"])
            self.assertIn("opencv-python-headless", result.data["constraints"])
            self.assertEqual(result.data["python"], "3.10")
            self.assertIn("numpy<2", result.data["install_plan"][1])
            self.assertIn("env_solution", result.data["analysis"])

    def test_env_solve_marks_cuda_torch_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "requirements.txt").write_text("transformers\nflash-attn\n", encoding="utf-8")
            result = EnvSolveModule().solve(
                repo,
                {
                    "frameworks": ["torch", "transformers"],
                    "install_plan": [["python3", "-m", "venv", ".venv"]],
                },
                {"gpu_required": True, "torch_variant": "cuda_or_cpu"},
            )
            self.assertIn("GPU/CUDA signals detected; torch wheel variant must match local CUDA runtime", result.data["risk_reasons"])
            self.assertIn("torch framework detected but requirements do not pin torch", result.data["risk_reasons"])
            self.assertIn("flash-attn may require CUDA toolkit and build isolation tuning", result.data["risk_reasons"])

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

    def test_model_prepare_executes_git_lfs_when_allowed(self):
        calls = []

        def fake_runner(cmd, cwd, timeout_seconds=900):
            calls.append((cmd, str(cwd), timeout_seconds))
            return CommandResult(cmd, str(cwd), 0, "ok", "", False)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)
            (run_dir / "reports").mkdir(parents=True)
            module = ModelPrepareModule(ModelCache(Path(tmp) / "cache"), command_runner=fake_runner)
            result = module.prepare(
                run_dir,
                {
                    "model_assets": [],
                    "git_lfs": {
                        "required": True,
                        "available": True,
                        "pointer_count": 1,
                        "total_pointer_size_bytes": 123,
                        "prepare_commands": [["git", "lfs", "install"], ["git", "lfs", "pull"]],
                    },
                },
                execute=True,
                repo_dir=repo,
                allowed_commands=["git"],
                timeout_seconds=7,
            )
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.summary, "git lfs model assets prepared")
            self.assertEqual([call[0] for call in calls], [["git", "lfs", "install"], ["git", "lfs", "pull"]])
            self.assertEqual(calls[0][2], 7)
            self.assertEqual(result.data["git_lfs"]["status"], "ready")

    def test_model_prepare_rejects_git_lfs_when_command_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)
            (run_dir / "reports").mkdir(parents=True)
            result = ModelPrepareModule(ModelCache(Path(tmp) / "cache")).prepare(
                run_dir,
                {
                    "model_assets": [],
                    "git_lfs": {
                        "required": True,
                        "available": True,
                        "prepare_commands": [["git", "lfs", "pull"]],
                    },
                },
                execute=True,
                repo_dir=repo,
                allowed_commands=["python3"],
            )
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.data["git_lfs"]["diagnosis"]["category"], "command_rejected")
            self.assertIn("disallowed command", result.error)

    def test_huggingface_downloader_resumes_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAssetDetector().detect(self._repo_with_hf_model(Path(tmp) / "repo"))[0])
            target_dir = Path(asset.cache_path)
            partial = target_dir / "model.safetensors.part"
            partial.parent.mkdir(parents=True, exist_ok=True)
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
            target.parent.mkdir(parents=True, exist_ok=True)
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

    def test_huggingface_downloader_retries_transient_download_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            asset = cache.reserve(ModelAsset(asset_id="huggingface:org/demo", source="huggingface", repo_id="org/demo"))
            attempts = {"download": 0}

            def fake_urlopen(req, timeout):
                if "/api/models/" in req.full_url:
                    return FakeStreamingResponse(json.dumps([
                        {"type": "file", "path": "config.json", "size": 3}
                    ]).encode("utf-8"))
                attempts["download"] += 1
                if attempts["download"] == 1:
                    raise OSError("temporary network reset")
                return FakeStreamingResponse(b"abc", status=200)

            result = HuggingFaceDownloader(
                urlopen=fake_urlopen,
                token="",
                chunk_size=1,
                retry_count=1,
                retry_backoff_seconds=0,
            ).download(asset)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(attempts["download"], 2)
            self.assertEqual((Path(asset.cache_path) / "config.json").read_bytes(), b"abc")

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
            partial.parent.mkdir(parents=True, exist_ok=True)
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

    def test_model_cache_cleanup_filters_by_source_repo_and_keep_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            keep = cache.reserve(ModelAsset(asset_id="hf:keep", source="huggingface", repo_id="org/keep"))
            delete = cache.reserve(ModelAsset(asset_id="hf:delete", source="huggingface", repo_id="org/delete"))
            other_source = cache.reserve(ModelAsset(asset_id="ms:delete", source="modelscope", repo_id="org/delete"))
            Path(keep.cache_path, "model.bin").write_bytes(b"keep")
            Path(delete.cache_path, "model.bin").write_bytes(b"delete")
            Path(other_source.cache_path, "model.bin").write_bytes(b"modelscope")

            entries = cache.entries()
            self.assertIn("repo_id", entries[0])
            plan = cache.cleanup(
                max_total_bytes=0,
                source="huggingface",
                keep_repo_ids=["org/keep"],
                dry_run=True,
            )
            self.assertEqual(plan["scoped_count"], 2)
            self.assertEqual(plan["candidate_count"], 1)
            self.assertEqual(plan["candidates"][0]["repo_id"], "org/delete")
            applied = cache.cleanup(
                max_total_bytes=0,
                source="huggingface",
                keep_repo_ids=["org/keep"],
                dry_run=False,
            )
            self.assertEqual(len(applied["deleted"]), 1)
            self.assertTrue(Path(keep.cache_path).exists())
            self.assertFalse(Path(delete.cache_path).exists())
            self.assertTrue(Path(other_source.cache_path).exists())

    def test_model_cache_cleanup_filters_specific_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            target = cache.reserve(ModelAsset(asset_id="hf:target", source="huggingface", repo_id="org/target"))
            other = cache.reserve(ModelAsset(asset_id="hf:other", source="huggingface", repo_id="org/other"))
            Path(target.cache_path, "model.bin").write_bytes(b"target")
            Path(other.cache_path, "model.bin").write_bytes(b"other")

            plan = cache.cleanup(max_total_bytes=0, repo_id="org/target", dry_run=True)
            self.assertEqual(plan["scoped_count"], 1)
            self.assertEqual(plan["candidate_count"], 1)
            self.assertEqual(plan["candidates"][0]["repo_id"], "org/target")

    def test_model_cache_cleanup_matches_legacy_cache_key_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ModelCache(Path(tmp) / "model_cache")
            legacy = cache.root / "huggingface" / "org-legacy_abc123"
            legacy.mkdir(parents=True)
            (legacy / "model.bin").write_bytes(b"legacy")
            plan = cache.cleanup(max_total_bytes=0, repo_id="org/legacy", dry_run=True)
            self.assertEqual(plan["scoped_count"], 1)
            self.assertEqual(plan["candidate_count"], 1)
            self.assertEqual(plan["candidates"][0]["cache_key"], "org-legacy_abc123")

    def _repo_with_hf_model(self, repo: Path) -> Path:
        repo.mkdir()
        (repo / "README.md").write_text("https://huggingface.co/org/demo-model", encoding="utf-8")
        return repo

    def test_log_classifier_detects_common_failures(self):
        result = LogClassifier().classify("ModuleNotFoundError: No module named 'gradio'")
        self.assertEqual(result["category"], "dependency_missing")
        self.assertGreater(result["confidence"], 0.8)

    def test_log_classifier_detects_missing_token(self):
        result = LogClassifier().classify("401 Unauthorized: Repository Not Found. Please set HF_TOKEN=should_not_be_recorded.")
        self.assertEqual(result["category"], "auth_required")
        self.assertGreaterEqual(result["confidence"], 0.9)
        self.assertEqual(result["required_env_vars"], ["HF_TOKEN"])
        self.assertFalse(result["values_recorded"])
        self.assertNotIn("should_not_be_recorded", json.dumps(result))

    def test_repair_planner_uses_diagnosed_required_env_vars(self):
        plan = RepairPlanner().propose(
            "model_prepare",
            StageResult(
                "model_prepare",
                "failed",
                "model download failed",
                {
                    "diagnosis": {
                        "category": "auth_required",
                        "required_env_vars": ["HF_TOKEN"],
                        "confidence": 0.9,
                    }
                },
            ),
            {},
        )
        self.assertEqual(plan["actions"][0]["type"], "set_env_var_name_only")
        self.assertEqual(plan["actions"][0]["payload"]["env_vars"], ["HF_TOKEN"])
        self.assertFalse(plan["actions"][0]["payload"]["values_recorded"])

    def test_report_lists_required_env_var_names_without_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(parents=True)
            (repair_dir / "repair_plan.json").write_text(json.dumps({
                "actions": [
                    {
                        "type": "set_env_var_name_only",
                        "payload": {
                            "env_vars": ["HF_TOKEN"],
                            "token_value": "should_not_be_recorded",
                        },
                    }
                ]
            }), encoding="utf-8")
            result = ReportGenerator().generate(
                run_dir,
                {"project": {"name": "demo", "repo_url": "local"}},
                {
                    "model_prepare": {
                        "status": "failed",
                        "summary": "download failed",
                        "data": {
                            "diagnosis": {
                                "category": "auth_required",
                                "required_env_vars": ["HF_TOKEN"],
                            }
                        },
                    },
                    "resource_plan": {
                        "status": "passed",
                        "summary": "resource plan generated",
                        "data": {"external_tokens": ["HF_TOKEN"]},
                    },
                },
            )
            report = Path(result.data["report_path"]).read_text(encoding="utf-8")
            self.assertIn("Required Environment Variables", report)
            self.assertIn("`HF_TOKEN`", report)
            self.assertIn("Values are not recorded", report)
            self.assertNotIn("should_not_be_recorded", report)

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
            env_plan = {"root_cause": "dependency solve", "rerun_from": "env_solve", "actions": []}
            env_gate = RepairLoopController(max_attempts=1).gate(Path(tmp) / "fresh", "env_deploy", {"signature": "env"}, env_plan, policy)
            self.assertEqual(env_gate["loop"]["rerun_from_effective"], "env_solve")

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

    def test_verify_validates_new_download_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)

            def fake_urlopen(req, timeout):
                (repo / "outputs" / "image.png").parent.mkdir(parents=True, exist_ok=True)
                (repo / "outputs" / "image.png").write_bytes(b"fake image bytes")
                return FakeHttpResponse("ok without trace")

            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/generate"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
            checks = {check["name"]: check for check in result.data["checks"]}
            self.assertEqual(result.status, "passed")
            self.assertEqual(checks["artifact_download_validation"]["status"], "pass")
            validated = checks["artifact_download_validation"]["evidence"]["validated"]
            self.assertEqual(validated[0]["path"], "outputs/image.png")
            self.assertEqual(validated[0]["size_bytes"], len(b"fake image bytes"))
            self.assertEqual(len(validated[0]["sha256"]), 64)

    def test_verify_rejects_empty_new_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            repo = run_dir / "workspace" / "repo"
            repo.mkdir(parents=True)

            def fake_urlopen(req, timeout):
                (repo / "outputs").mkdir(exist_ok=True)
                (repo / "outputs" / "empty.txt").write_bytes(b"")
                return FakeHttpResponse("ok without trace")

            result = VerifyModule(urlopen=fake_urlopen).verify(
                run_dir,
                analysis={"verify_hint": {"endpoint": "http://127.0.0.1:8000/generate"}},
                runner_result={"pid": 1234, "expected_port": 8000, "service_ready": True},
            )
            checks = {check["name"]: check for check in result.data["checks"]}
            self.assertEqual(result.status, "uncertain")
            self.assertEqual(checks["artifact_download_validation"]["status"], "fail")

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
        self.assertIn("cache_cleanup_scoped_keep", ids)
        self.assertIn("repair_loop_attempt_limit", ids)
        self.assertIn("operator_repair_approval", ids)
        self.assertIn("service_exits_after_start", ids)
        self.assertIn("stale_artifact_ignored", ids)
        self.assertIn("artifact_download_validation", ids)
        self.assertIn("git_lfs_detection", ids)
        self.assertIn("git_lfs_prepare_execute", ids)
        self.assertIn("env_solve_legacy_gradio_constraints", ids)
        self.assertIn("gradio_api_shape_variation", ids)
        self.assertIn("gradio_queue_call_followup", ids)
        self.assertIn("token_missing_diagnosis", ids)
        self.assertIn("repair_resume_stage_jump", ids)
        self.assertIn("repair_resume_audit_report", ids)
        self.assertIn("token_report_required_env", ids)

    def test_benchmark_runner_executes_all_fixture_cases(self):
        report = BenchmarkRunner().run(Path("tests/fixtures/benchmarks/manifest.json"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 26)
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

    def test_cache_cli_cleanup_defaults_to_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_entry = root / "model_cache" / "huggingface" / "demo"
            cache_entry.mkdir(parents=True)
            (cache_entry / "weights.bin").write_bytes(b"0123456789")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "runs_dir": str(root / "runs"),
                "memory_dir": str(root / "memory"),
                "model_cache_dir": str(root / "model_cache"),
                "model_cache_cleanup_max_total_bytes": 5,
            }), encoding="utf-8")
            old_config = os.environ.get("AUTO_HARNESS_CONFIG")
            os.environ["AUTO_HARNESS_CONFIG"] = str(config_path)
            try:
                output = io.StringIO()
                with redirect_stdout(output):
                    code = cli_main(["cache", "--cleanup"])
            finally:
                if old_config is None:
                    os.environ.pop("AUTO_HARNESS_CONFIG", None)
                else:
                    os.environ["AUTO_HARNESS_CONFIG"] = old_config
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["candidate_count"], 1)
            self.assertTrue((cache_entry / "weights.bin").exists())

    def test_cache_cli_cleanup_accepts_source_repo_and_keep_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = ModelCache(root / "model_cache")
            keep = cache.reserve(ModelAsset(asset_id="hf:keep", source="huggingface", repo_id="org/keep"))
            delete = cache.reserve(ModelAsset(asset_id="hf:delete", source="huggingface", repo_id="org/delete"))
            Path(keep.cache_path, "model.bin").write_bytes(b"keep")
            Path(delete.cache_path, "model.bin").write_bytes(b"delete")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "runs_dir": str(root / "runs"),
                "memory_dir": str(root / "memory"),
                "model_cache_dir": str(root / "model_cache"),
            }), encoding="utf-8")
            old_config = os.environ.get("AUTO_HARNESS_CONFIG")
            os.environ["AUTO_HARNESS_CONFIG"] = str(config_path)
            try:
                output = io.StringIO()
                with redirect_stdout(output):
                    code = cli_main([
                        "cache",
                        "--cleanup",
                        "--source",
                        "huggingface",
                        "--max-total-bytes",
                        "0",
                        "--keep-repo-id",
                        "org/keep",
                    ])
            finally:
                if old_config is None:
                    os.environ.pop("AUTO_HARNESS_CONFIG", None)
                else:
                    os.environ["AUTO_HARNESS_CONFIG"] = old_config
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["filters"]["source"], "huggingface")
            self.assertEqual(result["filters"]["keep_repo_ids"], ["org/keep"])
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["candidates"][0]["repo_id"], "org/delete")

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
            self.assertIn("env_solve", data)
            self.assertIn("model_prepare", data)
            self.assertEqual(data["resource_plan"]["data"]["model_assets"][0]["repo_id"], "org/demo-model")
            self.assertIn("numpy<2", data["env_solve"]["data"]["constraints"])
            self.assertTrue(Path(data["model_prepare"]["data"]["manifest_path"]).exists())

    def test_resume_uses_repair_rerun_from_effective_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "app.py").write_text("import gradio as gr\n", encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
            )
            runner = TaskRunner(config)
            task_id = runner.deploy(str(repo), "demo", dry_run=True)
            run_dir = root / "runs" / task_id

            def stage_updates():
                events = []
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
                    event = json.loads(line)
                    if event["type"] == "stage_update":
                        events.append((event["stage"], event["data"]["status"]))
                return events

            before = stage_updates()
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(exist_ok=True)
            (repair_dir / "repair_apply_result.json").write_text(json.dumps({
                "status": "applied",
                "policy": {
                    "allowed": True,
                    "loop": {
                        "rerun_from_effective": "verify",
                        "rerun_from_requested": "verify",
                    },
                },
            }), encoding="utf-8")
            (repair_dir / "repair_verify_hints.json").write_text(json.dumps({
                "verify_hints": [
                    {
                        "endpoint": "http://127.0.0.1:9",
                        "request": {"method": "GET", "path": "/health"},
                    }
                ]
            }), encoding="utf-8")

            runner.resume(task_id, dry_run=True)
            after = stage_updates()
            new_events = after[len(before):]
            self.assertNotIn(("analyze", "passed"), new_events)
            self.assertNotIn(("resource_plan", "passed"), new_events)
            self.assertNotIn(("env_solve", "passed"), new_events)
            self.assertNotIn(("env_deploy", "passed"), new_events)
            self.assertNotIn(("model_prepare", "passed"), new_events)
            self.assertNotIn(("runner", "passed"), new_events)
            self.assertTrue(any(stage == "verify" for stage, _ in new_events))
            self.assertTrue(any(stage == "report" for stage, _ in new_events))
            pipeline = json.loads((run_dir / "reports" / "pipeline_results.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline["verify"]["data"]["repair_overlay"]["verify_hint_count"], 1)
            audit = json.loads((run_dir / "reports" / "execution_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["effective_start_stage"], "verify")
            self.assertIn("runner", audit["reused_stages"])
            self.assertEqual(audit["rerun_stages"], ["verify", "report"])
            report = (run_dir / "reports" / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Execution Audit", report)
            self.assertIn("- Reused stages: `analyze`, `resource_plan`, `env_solve`, `env_deploy`, `model_prepare`, `runner`", report)
            self.assertIn("- Rerun stages: `verify`, `report`", report)

    def test_resume_falls_back_when_previous_stage_results_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
            )
            runner = TaskRunner(config)
            task_id = runner.deploy(str(repo), "demo", dry_run=True)
            run_dir = root / "runs" / task_id
            (run_dir / "reports" / "pipeline_results.json").unlink()
            (run_dir / "reports" / "resource_plan_result.json").unlink()
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(exist_ok=True)
            (repair_dir / "repair_apply_result.json").write_text(json.dumps({
                "status": "applied",
                "policy": {"allowed": True, "loop": {"rerun_from_effective": "verify"}},
            }), encoding="utf-8")

            runner.resume(task_id, dry_run=True)
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            fallback = [event for event in events if event["type"] == "resume_stage_fallback"]
            self.assertTrue(fallback)
            self.assertIn("resource_plan", fallback[-1]["data"]["missing_previous_results"])

    def test_resume_ignores_rejected_repair_stage_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "app.py").write_text("print('hello')\n", encoding="utf-8")
            config = HarnessConfig(
                runs_dir=str(root / "runs"),
                memory_dir=str(root / "memory"),
                model_cache_dir=str(root / "model_cache"),
            )
            runner = TaskRunner(config)
            task_id = runner.deploy(str(repo), "demo", dry_run=True)
            run_dir = root / "runs" / task_id
            repair_dir = run_dir / "repairs"
            repair_dir.mkdir(exist_ok=True)
            (repair_dir / "repair_plan.json").write_text(json.dumps({
                "rerun_from_effective": "verify",
            }), encoding="utf-8")
            (repair_dir / "repair_apply_result.json").write_text(json.dumps({
                "status": "rejected",
                "policy": {"allowed": False},
            }), encoding="utf-8")

            runner.resume(task_id, dry_run=True)
            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            resume_events = [event for event in events if event["type"] == "resume_requested"]
            self.assertEqual(resume_events[-1]["data"]["start_stage"], "analyze")


if __name__ == "__main__":
    unittest.main()
