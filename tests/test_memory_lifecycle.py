import hashlib
import json
from pathlib import Path

from auto_harness.memory.lifecycle import SkillCandidateLifecycle


def _candidate(path: Path):
    path.write_text(
        json.dumps({"candidate_id": "candidate-1", "status": "proposed"}),
        encoding="utf-8",
    )


def test_lifecycle_rejects_skipping_approval(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path)
    lifecycle = SkillCandidateLifecycle()
    lifecycle.initialize(path)

    result = lifecycle.transition(path, "regression_passed", "test")

    assert result["status"] == "failed"
    assert "proposed -> regression_passed" in result["error"]


def test_lifecycle_writes_hash_chained_audit(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path)
    lifecycle = SkillCandidateLifecycle()
    lifecycle.initialize(path)
    assert lifecycle.transition(path, "approved", "reviewer")["status"] == "transitioned"
    assert lifecycle.transition(path, "regression_passed", "regression")["status"] == "transitioned"

    events = [
        json.loads(line)
        for line in path.with_suffix(".lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["to_status"] for event in events] == [
        "proposed",
        "approved",
        "regression_passed",
    ]
    assert events[1]["previous_event_hash"] == events[0]["event_hash"]
    assert events[2]["previous_event_hash"] == events[1]["event_hash"]
    for event in events:
        event_hash = event.pop("event_hash")
        previous_hash = event["previous_event_hash"]
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256((previous_hash + canonical).encode("utf-8")).hexdigest() == event_hash
