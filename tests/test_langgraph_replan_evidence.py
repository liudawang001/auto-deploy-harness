from auto_harness.controllers.langgraph_deps import _build_replan_failure


def test_replan_uses_observed_failure_evidence_and_diagnosis():
    signal = "Bind for 0.0.0.0:8000 failed: port is already allocated"
    state = {
        "failure_context": {
            "failed_stage": "runner",
            "error": signal,
            "stderr_tail": signal,
            "runtime_backend": "docker",
        },
        "diagnosis": {
            "status": "ok",
            "summary": "host port conflict",
            "diagnosis": {"category": "port_conflict"},
            "rerun_from": "runner",
            "rerun_reason": "select a free port",
        },
    }

    failure = _build_replan_failure(
        state,
        "runner",
        {"status": "failed", "summary": "service process exited", "error": None},
    )

    assert failure["error"] == signal
    assert failure["stderr_tail"] == signal
    assert failure["diagnosis"]["diagnosis"]["category"] == "port_conflict"
