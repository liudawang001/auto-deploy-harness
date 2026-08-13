import json

import pytest

from auto_harness.agent_runtime.observation_ledger import ObservationLedger, RepositoryObservationService, enrich_plan_grounding
from auto_harness.agent_runtime.planner_turn import PlannerTurnParser
from auto_harness.agent_runtime.plan_first_loop import DEPLOYMENT_PLAN_SCHEMA, LLMDeploymentPlanner
from auto_harness.providers.mock import MockLLMProvider
from auto_harness.tools.registry import ToolRegistry


def test_planner_turn_parses_observe_and_final():
    parser = PlannerTurnParser(max_requests=2)
    observe = parser.parse(json.dumps({
        "protocol_version": 1,
        "kind": "observe",
        "reason": "inspect entrypoint",
        "requests": [{"request_id": "r1", "tool": "search_repo", "input": {"query": "uvicorn"}}],
    }))
    assert observe.kind == "observe"
    assert observe.requests[0].tool == "search_repo"
    final = parser.parse(json.dumps({"protocol_version": 1, "kind": "final", "plan": {"status": "no_safe_plan"}}))
    assert final.kind == "final"
    assert final.plan["status"] == "no_safe_plan"


def test_planner_turn_accepts_legacy_direct_plan():
    turn = PlannerTurnParser().parse('{"status":"needs_human_input","summary":"missing config"}')
    assert turn.kind == "final"


def test_planner_turn_defaults_missing_protocol_version_but_rejects_unknown():
    turn = PlannerTurnParser().parse(json.dumps({
        "kind": "final",
        "plan": {"status": "no_safe_plan"},
    }))
    assert turn.protocol_version == 1
    with pytest.raises(ValueError, match="unsupported planner turn protocol_version"):
        PlannerTurnParser().parse(json.dumps({
            "protocol_version": 2,
            "kind": "final",
            "plan": {"status": "no_safe_plan"},
        }))


def test_planner_turn_rejects_duplicate_and_too_many_requests():
    parser = PlannerTurnParser(max_requests=1)
    with pytest.raises(ValueError):
        parser.parse(json.dumps({
            "protocol_version": 1, "kind": "observe",
            "requests": [
                {"request_id": "r", "tool": "search_repo", "input": {}},
                {"request_id": "r", "tool": "search_repo", "input": {}},
            ],
        }))


def test_planner_turn_derives_stable_ids_when_provider_omits_them():
    turn = PlannerTurnParser(max_requests=3).parse(json.dumps({
        "protocol_version": 1,
        "kind": "observe",
        "requests": [
            {"tool": "search_repo", "input": {"query": "uvicorn"}},
            {
                "request_id": "auto_request_2",
                "tool": "read_selected_files",
                "input": {"paths": ["server.py"]},
            },
        ],
    }))

    assert [request.request_id for request in turn.requests] == [
        "auto_request_1",
        "auto_request_2",
    ]


def test_planner_turn_rejects_side_effect_tool():
    with pytest.raises(ValueError, match="not allowed"):
        PlannerTurnParser().parse(json.dumps({
            "protocol_version": 1,
            "kind": "observe",
            "requests": [{
                "request_id": "r1",
                "tool": "install_package",
                "input": {"package": "example"},
            }],
        }))


def test_observation_service_persists_and_deduplicates(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("uvicorn.run(app, port=8000)\n", encoding="utf-8")
    ledger_path = tmp_path / "observations.jsonl"
    request = {"request_id": "r1", "tool": "search_repo", "input": {"query": "uvicorn", "path_glob": "**/*.py"}}
    service = RepositoryObservationService()
    budget = {"remaining_tokens": 1000, "remaining_files": 10}
    first = service.execute_round([request], repo_dir=repo, ledger_path=ledger_path, repository_fingerprint="fp", round_number=1, budget=budget)
    second = service.execute_round([request], repo_dir=repo, ledger_path=ledger_path, repository_fingerprint="fp", round_number=2, budget=first["budget"])
    assert first["results"][0]["status"] == "passed"
    assert second["results"][0]["cache_hit"] is True
    records = ObservationLedger(ledger_path).load()
    assert [item["status"] for item in records] == ["passed", "cache_hit"]
    assert second["budget"]["remaining_tokens"] == first["budget"]["remaining_tokens"]
    assert second["budget"]["remaining_files"] == first["budget"]["remaining_files"]


def test_grounding_is_enriched_only_from_observed_file():
    snapshot = {"selected_files": {"app.py": {"observation_id": "core_1", "sha256": "abc", "line_start": 1, "line_end": 10}}}
    plan = {"grounding": [
        {"claim": "entry", "file": "app.py", "reason": "code"},
        {"claim": "guess", "file": "missing.py", "reason": "unknown"},
    ]}
    result = enrich_plan_grounding(plan, snapshot, [])
    assert result["grounding"][0]["observation_id"] == "core_1"
    assert "observation_id" not in result["grounding"][1]


def test_grounding_replaces_model_alias_with_authoritative_observation():
    snapshot = {
        "selected_files": {
            "pyproject.toml": {
                "observation_id": "core_real",
                "sha256": "real-digest",
                "line_start": 1,
                "line_end": 40,
            }
        }
    }
    plan = {"grounding": [{
        "claim": "dependencies",
        "file": "pyproject.toml",
        "reason": "project metadata",
        "observation_id": "selected_files",
        "sha256": "stale",
        "line_start": 1,
        "line_end": 20,
    }]}
    result = enrich_plan_grounding(plan, snapshot, [])
    grounding = result["grounding"][0]
    assert grounding["observation_id"] == "core_real"
    assert grounding["sha256"] == "real-digest"
    assert grounding["line_start"] == 1
    assert grounding["line_end"] == 20


def test_layered_planner_turn_uses_json_action_provider():
    planner = LLMDeploymentPlanner(MockLLMProvider(), config={})
    raw = planner.turn(
        {"context_mode": "layered", "file_tree": ["app.py"], "selected_files": {}},
        observations=[],
        observation_budget={"remaining_rounds": 4, "remaining_tokens": 1000},
    )
    turn = PlannerTurnParser().parse(raw.text)
    assert turn.kind == "final"


def test_layered_planner_prompt_exposes_repository_tool_input_contracts():
    planner = LLMDeploymentPlanner(MockLLMProvider(), config={})
    tools = ToolRegistry().executable_for_stage("plan", agent_mode="planner")
    messages = planner._turn_messages({}, [], {}, tools, {})
    prompt = messages[-1].content

    assert '"name":"search_repo"' in prompt
    assert '"required":["query"]' in prompt
    assert '"name":"read_selected_files"' in prompt
    assert '"required":["files"]' in prompt
    assert '"risk_level"' not in prompt


def test_prompt_deployment_schema_matches_parser_candidate_and_verify_fields():
    candidate = DEPLOYMENT_PLAN_SCHEMA["properties"]["run"]["properties"][
        "candidates"
    ]["items"]
    request = DEPLOYMENT_PLAN_SCHEMA["properties"]["verify"]["properties"][
        "request"
    ]

    assert set(candidate["required"]) == {"id", "cmd", "expected_port", "reason"}
    assert "command" not in candidate["properties"]
    assert set(request["required"]) == {"method", "path"}
    assert request["properties"]["path"]["pattern"] == r"\{\{trace_id\}\}"
