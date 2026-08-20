"""Evidence-first comparison matrix for independently versioned protocols."""

from pathlib import Path
from typing import Any, Dict, Iterable, List

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


REQUIRED_NATIVE_PROTOCOL_CASES = (
    "json_action_mock",
    "json_action_real_provider",
    "native_tools_fake_provider",
    "native_tools_real_provider",
    "native_tools_unsupported_provider",
    "native_tools_prompt_injection",
    "native_tools_crash_recovery",
)


class NativeProtocolMatrix:
    """Normalize protocol evaluation evidence without inventing missing runs."""

    def evaluate(
        self,
        cases: Iterable[Dict[str, Any]],
        output_path: Path = None,
    ) -> Dict[str, Any]:
        indexed = {
            str(item.get("case_id", "")): dict(item)
            for item in cases
            if isinstance(item, dict) and item.get("case_id")
        }
        rows: List[Dict[str, Any]] = []
        for case_id in REQUIRED_NATIVE_PROTOCOL_CASES:
            source = indexed.get(case_id, {})
            status = str(source.get("status", "not_run"))
            if status not in {"passed", "failed", "not_run", "blocked"}:
                status = "failed"
            live = case_id == "native_tools_real_provider"
            provider_name = str(source.get("provider_name", ""))
            if live and status == "passed" and provider_name.lower() in {
                "", "fake", "fake_native", "mock",
            }:
                status = "failed"
            rows.append({
                "case_id": case_id,
                "status": status,
                "provider_protocol": source.get(
                    "provider_protocol",
                    "native_tools" if case_id.startswith("native_tools") else "json_action",
                ),
                "provider_name": provider_name,
                "evidence_paths": list(source.get("evidence_paths") or []),
                "reason": str(source.get("reason", "")),
            })
        payload = {
            "schema_version": 1,
            "generated_at": utc_now_iso(),
            "cases": rows,
            "passed_count": sum(1 for item in rows if item["status"] == "passed"),
            "failed_count": sum(1 for item in rows if item["status"] == "failed"),
            "not_run_count": sum(
                1 for item in rows if item["status"] in {"not_run", "blocked"}
            ),
            "release_ready": all(item["status"] == "passed" for item in rows),
        }
        if output_path:
            write_json(Path(output_path), payload)
        return payload


__all__ = ["NativeProtocolMatrix", "REQUIRED_NATIVE_PROTOCOL_CASES"]
