import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.config import HarnessConfig
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.agents.base import AgentResult
from auto_harness.memory import MemoryStore
from auto_harness.assets import ModelCache, ModelAssetDetector
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.model_prepare import ModelPrepareModule
from auto_harness.modules.resource_plan import ResourcePlanner
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.result import StageResult
from auto_harness.providers import Message, MockLLMProvider
from auto_harness.skills import SkillRegistry
from auto_harness.state import StateStore
from auto_harness.orchestrator import TaskRunner
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


class FakeAgentExecutor:
    def __init__(self, text='{"risk":"low"}'):
        self.text = text
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return AgentResult(status="passed", text=self.text)

    def resume(self, session_id, request):
        return self.run(request)


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
            store.update_stage("task1", "analyze", "passed", result_path="analysis.json")
            state = store.load_state("task1")
            self.assertEqual(state.stages["analyze"].status, "passed")
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
