import json
from pathlib import Path

import pytest

from auto_harness.cli import main as cli_main
from auto_harness.config import HarnessConfig
from auto_harness.models.base import write_json
from auto_harness.observability.cost_profile import CostProfileCollector


T0 = "2026-08-23T11:39:15.907117+00:00"
T1 = "2026-08-23T11:39:40.907117+00:00"
T2 = "2026-08-23T11:40:10.907117+00:00"


def _write_event(run_dir, ts, stage, event_type, data=None):
    events_path = run_dir / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": ts, "stage": stage, "type": event_type, "data": data or {}}) + "\n")


def _write_agent_call(run_dir, name, record):
    path = run_dir / "logs" / "agent_calls"
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / name, record)


def _usage(source, input_tokens, output_tokens=0, total_tokens=None, cache_hit=0, cache_miss=0):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
        "source": source,
        "cache_hit_tokens": cache_hit,
        "cache_miss_tokens": cache_miss,
    }
    return usage


def _write_controller_result(run_dir, status="completed", verify_status="passed"):
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "reports" / "controller_result.json", {
        "task_id": run_dir.name,
        "status": status,
        "verify_status": verify_status,
        "stop_reason": "",
    })


def _build_run(run_dir):
    """A completed run with reported usage, an estimated-only call and stage events."""
    _write_agent_call(run_dir, "verify_001.json", {
        "stage": "verify",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "latency_ms": 2319,
        "context": {"usage": _usage("provider_reported", 2302, 195, cache_hit=384, cache_miss=1918)},
    })
    _write_agent_call(run_dir, "replan_001.json", {
        "stage": "replan",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "latency_ms": 100,
        "context": {"usage": _usage("estimated", 1000)},
    })
    turns = run_dir / "reports" / "planner_turns"
    turns.mkdir(parents=True, exist_ok=True)
    write_json(turns / "turn_001.json", {
        "kind": "plan",
        "context": {
            "model": "deepseek-v4-pro",
            "usage": _usage("provider_reported", 2736, 2160, cache_hit=256, cache_miss=2480),
            "provider_response": {"latency_ms": 1500},
        },
    })
    _write_event(run_dir, T0, "task", "created", {"status": "created"})
    _write_event(run_dir, T0, "analyze", "stage_update", {"status": "running"})
    _write_event(run_dir, T1, "analyze", "stage_update", {"status": "passed"})
    _write_event(run_dir, T1, "verify", "stage_update", {"status": "running"})
    _write_event(run_dir, T2, "verify", "stage_update", {"status": "passed"})
    _write_event(run_dir, T2, "controller", "controller_terminal", {
        "status": "completed",
        "verify_status": "passed",
    })
    _write_controller_result(run_dir)


def test_per_run_profile_aggregates_usage_latency_stages_and_success(tmp_path):
    run_dir = tmp_path / "profiled-run"
    _build_run(run_dir)

    profile = CostProfileCollector().collect(run_dir)

    assert profile["tokens"]["provider_reported"] == {
        "input_tokens": 5038,
        "output_tokens": 2355,
        "total_tokens": 7393,
        "cache_hit_tokens": 640,
        "cache_miss_tokens": 4398,
        "call_count": 2,
    }
    assert profile["tokens"]["estimated"] == {
        "input_tokens": 1000,
        "output_tokens": 0,
        "total_tokens": 1000,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "call_count": 1,
    }
    assert profile["tokens"]["coverage"] == {
        "calls_total": 3,
        "calls_with_usage": 3,
        "calls_without_usage": 0,
    }
    assert profile["llm_latency"] == {
        "count": 3,
        "total_ms": 3919,
        "avg_ms": 1306,
        "p50_ms": 1500,
        "p95_ms": 2319,
        "max_ms": 2319,
    }
    assert [(stage["stage"], stage["duration_ms"]) for stage in profile["stages"]] == [
        ("analyze", 25000),
        ("verify", 30000),
    ]
    assert profile["run"]["duration_ms"] == 55000
    assert profile["success"] == {
        "final_status": "completed",
        "verify_status": "passed",
        "verify_passed": True,
        "success": True,
    }
    assert profile["cost"]["status"] == "unpriced"
    assert profile["cost"]["reason"] == "no pricing configured"


def test_estimated_tokens_are_never_priced(tmp_path):
    run_dir = tmp_path / "priced-run"
    _write_agent_call(run_dir, "runner_001.json", {
        "stage": "runner",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "latency_ms": 500,
        "context": {"usage": _usage(
            "provider_reported", 300000, 200000, cache_hit=100000, cache_miss=200000,
        )},
    })
    _write_agent_call(run_dir, "runner_002.json", {
        "stage": "runner",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "context": {"usage": _usage("estimated", 500000)},
    })

    collector = CostProfileCollector({
        "currency": "USD",
        "pricing_as_of": "2026-08-30",
        "pricing": {
            "deepseek-v4-flash": {
                "input_per_million": 0.28,
                "output_per_million": 0.42,
                "cache_hit_input_per_million": 0.028,
            },
        },
    })
    profile = collector.collect(run_dir)

    assert profile["tokens"]["estimated"]["input_tokens"] == 500000
    assert profile["tokens"]["by_model"] == [{
        "model": "deepseek-v4-flash",
        "input_tokens": 300000,
        "output_tokens": 200000,
        "total_tokens": 500000,
        "cache_hit_tokens": 100000,
        "cache_miss_tokens": 200000,
        "call_count": 1,
    }]
    # 100000 cache-hit * 0.028/1M + 200000 input * 0.28/1M + 200000 output * 0.42/1M
    assert profile["cost"]["status"] == "priced"
    assert profile["cost"]["models"] == [{
        "model": "deepseek-v4-flash",
        "input_tokens": 300000,
        "output_tokens": 200000,
        "cache_hit_tokens": 100000,
        "cost": pytest.approx(0.1428, abs=1e-9),
    }]
    assert profile["cost"]["total_cost"] == pytest.approx(0.1428, abs=1e-9)


def test_unknown_model_tokens_are_unpriced_not_invented(tmp_path):
    run_dir = tmp_path / "mystery-run"
    _write_agent_call(run_dir, "verify_001.json", {
        "stage": "verify",
        "provider": "custom",
        "model": "mystery-model",
        "latency_ms": 42,
        "context": {"usage": _usage("provider_reported", 1000, 100)},
    })

    collector = CostProfileCollector({
        "pricing": {"deepseek-v4-flash": {"input_per_million": 0.28}},
    })
    profile = collector.collect(run_dir)

    assert profile["cost"]["status"] == "unpriced"
    assert profile["cost"]["reason"] == "no pricing entry for observed models"
    assert profile["cost"]["unpriced_models"] == [{"model": "mystery-model", "total_tokens": 1100}]
    assert profile["cost"]["total_cost"] == 0.0


def test_raw_plan_fallback_counts_only_without_planner_turns(tmp_path):
    usage = _usage("provider_reported", 5637, 1113, cache_hit=256, cache_miss=5381)
    context = {
        "model": "deepseek-v4-flash",
        "usage": usage,
        "provider_response": {"latency_ms": 4000},
    }

    with_turns = tmp_path / "with-turns"
    (with_turns / "reports" / "planner_turns").mkdir(parents=True)
    write_json(with_turns / "reports" / "planner_turns" / "turn_001.json", {"context": context})
    write_json(with_turns / "reports" / "llm_deployment_plan.raw.json", {"context": context})

    without_turns = tmp_path / "without-turns"
    (without_turns / "reports").mkdir(parents=True)
    write_json(without_turns / "reports" / "llm_deployment_plan.raw.json", {"context": context})

    collector = CostProfileCollector()
    assert collector.collect(with_turns)["tokens"]["coverage"]["calls_total"] == 1
    assert collector.collect(without_turns)["tokens"]["coverage"]["calls_total"] == 1


def test_legacy_trace_without_context_is_tolerated(tmp_path):
    run_dir = tmp_path / "legacy-run"
    _write_agent_call(run_dir, "analyze_001.json", {
        "stage": "analyze",
        "provider": "mock",
        "model": "deterministic-fixture",
        "latency_ms": 30,
    })

    profile = CostProfileCollector().collect(run_dir)

    assert profile["tokens"]["coverage"] == {
        "calls_total": 1,
        "calls_with_usage": 0,
        "calls_without_usage": 1,
    }
    assert profile["llm_latency"]["count"] == 1
    assert profile["tokens"]["provider_reported"]["total_tokens"] == 0


def test_stage_rerun_keeps_latest_attempt(tmp_path):
    run_dir = tmp_path / "rerun-run"
    _write_event(run_dir, T0, "env_deploy", "stage_update", {"status": "running"})
    _write_event(run_dir, T1, "env_deploy", "stage_update", {"status": "failed"})
    _write_event(run_dir, T1, "env_deploy", "stage_update", {"status": "running"})
    _write_event(run_dir, T2, "env_deploy", "stage_update", {"status": "passed"})

    profile = CostProfileCollector().collect(run_dir)

    assert profile["stages"] == [{
        "stage": "env_deploy",
        "status": "passed",
        "duration_ms": 30000,
        "attempts": 2,
    }]


def test_portfolio_aggregates_status_rates_and_coverage(tmp_path):
    runs_dir = tmp_path / "runs"
    good = runs_dir / "good-run"
    _build_run(good)

    bad = runs_dir / "bad-run"
    _write_agent_call(bad, "verify_001.json", {
        "stage": "verify",
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "latency_ms": 700,
        "context": {"usage": _usage("provider_reported", 100, 50)},
    })
    _write_event(bad, T0, "task", "created")
    _write_event(bad, T0, "verify", "stage_update", {"status": "running"})
    _write_event(bad, T1, "verify", "stage_update", {"status": "failed"})
    _write_event(bad, T1, "controller", "controller_terminal", {"status": "stopped"})
    _write_controller_result(bad, status="stopped", verify_status="failed")

    legacy = runs_dir / "legacy-run"
    legacy.mkdir(parents=True)
    _write_event(legacy, T0, "task", "created")

    collector = CostProfileCollector()
    report = collector.collect_many(runs_dir)

    assert report["run_count"] == 3
    assert report["status_counts"] == {"completed": 1, "stopped": 1, "unknown": 1}
    assert report["success_rate"] == round(1 / 3, 4)
    assert report["runs_with_known_status"] == 2
    assert report["success_rate_known_status"] == 0.5
    assert report["data_coverage"]["runs_with_provider_reported_usage"] == 2
    assert report["data_coverage"]["runs_without_usage"] == 1
    assert report["tokens"]["provider_reported"]["call_count"] == 3
    assert report["tokens"]["provider_reported"]["total_tokens"] == 7393 + 150
    assert report["llm_latency"]["count"] == 4
    assert report["stage_durations"]["analyze"]["count"] == 1
    assert report["stage_durations"]["verify"]["count"] == 2
    assert [row["task_id"] for row in report["runs"]] == ["bad-run", "good-run", "legacy-run"]


def test_collect_many_writes_json_and_markdown(tmp_path):
    runs_dir = tmp_path / "runs"
    _build_run(runs_dir / "profiled-run")
    output = tmp_path / "reports" / "cost_profile.json"

    report = CostProfileCollector().collect_many(runs_dir, output_path=output)

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["run_count"] == 1
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert markdown.startswith("# Performance & Cost Profile")
    assert "| model | input | output | cache hit | cost (USD) |" not in markdown
    assert "Provider reported tokens" in markdown


def test_cli_single_run_writes_profile_into_run_dir(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    _build_run(runs_dir / "profiled-run")

    exit_code = cli_main([
        "cost-profile", "--runs-dir", str(runs_dir),
        "--task-id", "profiled-run",
    ])

    assert exit_code == 0
    profile = json.loads(
        (runs_dir / "profiled-run" / "reports" / "cost_profile.json").read_text(encoding="utf-8")
    )
    assert profile["success"]["final_status"] == "completed"
    assert json.loads(capsys.readouterr().out)["task_id"] == "profiled-run"


def test_cli_portfolio_writes_reports_and_empty_runs_exit_code(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    _build_run(runs_dir / "profiled-run")
    output = tmp_path / "out" / "cost_profile.json"

    exit_code = cli_main([
        "cost-profile", "--runs-dir", str(runs_dir), "--output", str(output),
    ])
    assert exit_code == 0
    assert output.exists()
    assert output.with_suffix(".md").exists()
    assert json.loads(capsys.readouterr().out)["run_count"] == 1

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    exit_code = cli_main(["cost-profile", "--runs-dir", str(empty_dir)])
    assert exit_code == 2
    assert json.loads(capsys.readouterr().out)["run_count"] == 0


def test_cli_missing_run_returns_config_error(tmp_path):
    exit_code = cli_main([
        "cost-profile", "--runs-dir", str(tmp_path), "--task-id", "nope",
    ])
    assert exit_code == 2


def test_cost_profile_config_defaults_and_validation():
    config = HarnessConfig(agent_provider="mock")
    assert config.cost_profile["currency"] == "USD"
    assert config.cost_profile["pricing"] == {}

    merged = HarnessConfig(agent_provider="mock", cost_profile={"pricing_as_of": "2026-08-30"})
    assert merged.cost_profile["currency"] == "USD"
    assert merged.cost_profile["pricing_as_of"] == "2026-08-30"

    with pytest.raises(ValueError):
        HarnessConfig(agent_provider="mock", cost_profile="yes")
    with pytest.raises(ValueError):
        HarnessConfig(agent_provider="mock", cost_profile={"currency": ""})
    with pytest.raises(ValueError):
        HarnessConfig(agent_provider="mock", cost_profile={
            "pricing": {"m": {"input_per_million": -1}},
        })
