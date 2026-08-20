"""Opt-in real DeepSeek native-tool smoke; never faked or auto-promoted."""

import json
import os
from pathlib import Path

import pytest

from auto_harness.agent_runtime.native_tool_loop import NativeToolTurnLoop
from auto_harness.providers.base import Message
from auto_harness.providers.deepseek import DeepSeekProvider


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DEEPSEEK_LIVE_TESTS") != "1"
    or not os.environ.get("DEEPSEEK_API_KEY"),
    reason="requires RUN_DEEPSEEK_LIVE_TESTS=1 and DEEPSEEK_API_KEY",
)


def test_real_deepseek_performs_two_read_only_native_calls(tmp_path):
    repo = tmp_path / "fixture"
    run_dir = tmp_path / "run"
    repo.mkdir()
    (repo / "README.md").write_text(
        "Run `python app.py`; the service listens on port 8917.\n",
        encoding="utf-8",
    )
    (repo / "app.py").write_text(
        "from http.server import HTTPServer, BaseHTTPRequestHandler\n"
        "HTTPServer(('127.0.0.1', 8917), BaseHTTPRequestHandler).serve_forever()\n",
        encoding="utf-8",
    )
    provider = DeepSeekProvider(
        purpose="agent",
        settings_override={
            "native_tool_calling": True,
            "require_api_key": True,
            "json_mode": {"agent": False},
        },
    )
    outcome = NativeToolTurnLoop(
        provider,
        run_dir=run_dir,
        max_turns=6,
    ).run(
        [Message(
            role="user",
            content=(
                "Analyze this tiny repository. You must first call inspect_repo_tree, "
                "then call read_selected_files for README.md and app.py, then return "
                "a concise deployment plan. Do not use any other tool."
            ),
        )],
        context={"repo_dir": str(repo)},
        task_id="deepseek-native-live",
    )
    names = [item.tool_name for item in outcome.tool_results if item.policy_allowed]
    assert outcome.status == "completed"
    assert "inspect_repo_tree" in names
    assert "read_selected_files" in names
    assert len(outcome.provider_request_ids) >= 3

    evidence_dir = os.environ.get("AUTO_HARNESS_LIVE_EVIDENCE_DIR")
    if evidence_dir:
        target = Path(evidence_dir) / "native-tool-live-smoke-manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "provider_protocol": "native_tools",
            "provider_name": "deepseek",
            "network_transport": "live",
            "tool_call_count": len(names),
            "final_status": "completed",
            "credential_values_persisted": False,
            "request_ids": outcome.provider_request_ids,
        }, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
