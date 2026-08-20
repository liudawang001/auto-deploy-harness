import json

from auto_harness.agent_runtime.native_tool_loop import NativeToolTurnLoop
from auto_harness.agent_runtime.observation_ledger import (
    ObservationLedger,
    RepositoryObservationService,
    enrich_plan_grounding,
)
from auto_harness.agent_runtime.plan_policy import PlanPolicyGate
from auto_harness.agent_runtime.schemas import ToolCall
from auto_harness.config import HarnessConfig
from auto_harness.providers.base import LLMResult, Message
from auto_harness.providers.mock import FakeNativeToolProvider
from auto_harness.tools.registry import ToolRegistry
from auto_harness.tools.retrieval_executor import RetrievalToolExecutor


def _config():
    return HarnessConfig(retrieval={
        "enabled": True,
        "mode": "lexical",
        "sources": ["repository"],
        "default_top_k": 2,
    })


def _native_call(call_id, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "retrieve_deployment_context",
            "arguments": json.dumps(arguments),
        },
    }


def test_retrieval_tool_is_hidden_by_default_and_visible_when_enabled():
    hidden = ToolRegistry().executable_for_stage("plan", agent_mode="planner")
    visible = ToolRegistry(config=_config()).executable_for_stage("plan", agent_mode="planner")
    assert "retrieve_deployment_context" not in {item["name"] for item in hidden}
    assert "retrieve_deployment_context" in {item["name"] for item in visible}


def test_disabled_retrieval_executor_fails_closed(tmp_path):
    result = RetrievalToolExecutor(config=HarnessConfig()).execute(
        ToolCall(name="retrieve_deployment_context", input={
            "query": "entrypoint", "purpose": "plan_repository",
        }),
        {"stage": "plan", "repo_dir": str(tmp_path)},
    )
    assert result.status == "rejected"
    assert "disabled" in result.error


def test_json_action_retrieval_requires_exact_read_before_grounding(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "serve.py").write_text(
        "def launch_service():\n    return 'deployment_entrypoint'\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    ledger_path = run_dir / "reports" / "observation_ledger.jsonl"
    service = RepositoryObservationService(config=_config())
    budget = {"remaining_tokens": 10000, "remaining_files": 10}
    first = service.execute_round(
        [{"request_id": "r1", "tool": "retrieve_deployment_context", "input": {
            "query": "deployment_entrypoint", "purpose": "plan_repository",
            "sources": ["repository"], "top_k": 2,
        }}],
        repo_dir=repo, ledger_path=ledger_path, repository_fingerprint="fp",
        round_number=1, budget=budget, stage="plan", task_id="task-1",
        run_dir=run_dir,
    )
    retrieval_record = first["results"][0]
    assert retrieval_record["status"] == "passed"
    assert retrieval_record["evidence"]["authority"] == "candidate_only"
    exact_request = retrieval_record["evidence"]["exact_read_requests"][0]

    second = service.execute_round(
        [{"request_id": "r2", **exact_request}],
        repo_dir=repo, ledger_path=ledger_path, repository_fingerprint="fp",
        round_number=2, budget=first["budget"], stage="plan", task_id="task-1",
        run_dir=run_dir,
    )
    exact_record = second["results"][0]
    assert exact_record["status"] == "passed"
    assert exact_record["retrieved_from_query_id"] == retrieval_record["evidence"]["query_id"]
    observed = exact_record["evidence"]["files"][0]
    plan = enrich_plan_grounding(
        {"grounding": [{"file": "serve.py", "claim": "entry", "reason": "exact read"}]},
        {"selected_files": {}}, ObservationLedger(ledger_path).load(),
    )
    snapshot = {
        "context_mode": "layered", "file_tree": ["serve.py"],
        "selected_files": {}, "observation_ledger": ObservationLedger(ledger_path).load(),
        "repo_dir": str(repo),
    }
    assert plan["grounding"][0]["sha256"] == observed["sha256"]
    assert PlanPolicyGate()._validate_grounding(plan, snapshot)["allowed"] is True


def test_native_protocol_can_call_retrieval_tool(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("gradio deployment endpoint\n", encoding="utf-8")
    provider = FakeNativeToolProvider([
        LLMResult(
            text="", protocol="native_tools", finish_reason="tool_calls",
            tool_calls=[_native_call("r1", {
                "query": "gradio endpoint", "purpose": "plan_repository",
                "sources": ["repository"], "top_k": 1,
            })],
        ),
        LLMResult(text="grounded next step", protocol="native_tools", finish_reason="stop"),
    ])
    outcome = NativeToolTurnLoop(
        provider, config=_config(), run_dir=tmp_path / "run",
    ).run(
        [Message(role="user", content="find the deployment endpoint")],
        context={"repo_dir": str(repo)}, task_id="task-1",
        repository_fingerprint="fp",
    )
    assert outcome.status == "completed"
    assert "retrieve_deployment_context" in outcome.visible_tool_names
    assert outcome.tool_results[0].status == "passed"
    assert outcome.tool_results[0].result["evidence"]["authority"] == "candidate_only"


def test_tool_indexes_verified_memory_but_not_unverified_by_default(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("plain entrypoint\n", encoding="utf-8")
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "deployment_issues.jsonl").write_text(
        json.dumps({
            "id": "verified-1", "verified_success": True,
            "root_cause": "queue_trace_bridge", "stage": "verify",
            "verification_trace_id": "trace-1", "repair_action_hash": "sha256:repair",
            "repair_action_status": "executed",
        }) + "\n" + json.dumps({
            "id": "unverified-1", "verified_success": False,
            "root_cause": "poisoned_install_advice", "stage": "verify",
        }) + "\n",
        encoding="utf-8",
    )
    config = _config()
    config.retrieval["sources"] = ["repository", "verified_memory"]
    config.memory_dir = str(memory)
    result = RetrievalToolExecutor(config=config).execute(
        ToolCall(name="retrieve_deployment_context", input={
            "query": "queue_trace_bridge", "purpose": "select_verify_strategy",
            "sources": ["verified_memory"], "top_k": 2,
        }),
        {
            "stage": "verify", "repo_dir": str(repo), "run_dir": str(tmp_path / "run"),
            "task_id": "task-1", "repository_fingerprint": "fp",
        },
    )
    assert result.status == "passed"
    assert result.evidence["hits"][0]["source_type"] == "verified_memory"
    assert "poisoned_install_advice" not in json.dumps(result.evidence)


def test_fake_hybrid_tool_rehydrates_persisted_vectors(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "gpu.py").write_text("cuda memory allocation repair\n", encoding="utf-8")
    config = HarnessConfig(retrieval={
        "enabled": True, "mode": "hybrid", "dense_enabled": True,
        "embedding_provider": "fake", "sources": ["repository"],
    })
    context = {
        "stage": "repair", "repo_dir": str(repo), "run_dir": str(tmp_path / "run"),
        "task_id": "task-1", "repository_fingerprint": "fp",
    }
    call = ToolCall(name="retrieve_deployment_context", input={
        "query": "cuda memory", "purpose": "select_repair", "sources": ["repository"],
    })
    first = RetrievalToolExecutor(config=config).execute(call, context)
    second = RetrievalToolExecutor(config=config).execute(call, context)
    assert first.status == second.status == "passed"
    assert second.evidence["trace"]["candidate_counts"]["dense"] >= 1
