import json
import tempfile
import unittest
from pathlib import Path

from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.agents.base import AgentResult
from auto_harness.memory import MemoryStore
from auto_harness.modules.analyzer import ProjectAnalyzer
from auto_harness.modules.env_deploy import EnvDeployModule
from auto_harness.modules.verify import VerifyModule
from auto_harness.models.result import StageResult
from auto_harness.providers import Message, MockLLMProvider
from auto_harness.skills import SkillRegistry
from auto_harness.state import StateStore
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


if __name__ == "__main__":
    unittest.main()
