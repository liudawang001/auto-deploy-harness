from auto_harness.evals.native_protocol import (
    NativeProtocolMatrix,
    REQUIRED_NATIVE_PROTOCOL_CASES,
)


def test_matrix_keeps_missing_external_cases_not_run(tmp_path):
    result = NativeProtocolMatrix().evaluate([
        {"case_id": "json_action_mock", "status": "passed", "provider_name": "mock"},
        {"case_id": "native_tools_fake_provider", "status": "passed", "provider_name": "fake_native"},
        {"case_id": "native_tools_unsupported_provider", "status": "passed"},
        {"case_id": "native_tools_prompt_injection", "status": "passed"},
        {"case_id": "native_tools_crash_recovery", "status": "passed"},
    ], output_path=tmp_path / "matrix.json")
    assert len(result["cases"]) == len(REQUIRED_NATIVE_PROTOCOL_CASES)
    assert result["release_ready"] is False
    real = next(
        item for item in result["cases"]
        if item["case_id"] == "native_tools_real_provider"
    )
    assert real["status"] == "not_run"


def test_fake_provider_cannot_pass_real_provider_matrix_row():
    result = NativeProtocolMatrix().evaluate([{
        "case_id": "native_tools_real_provider",
        "status": "passed",
        "provider_name": "fake_native",
    }])
    real = next(
        item for item in result["cases"]
        if item["case_id"] == "native_tools_real_provider"
    )
    assert real["status"] == "failed"
