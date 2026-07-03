import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.result import StageResult
from auto_harness.models.verify import VerifyResult
from auto_harness.utils.files import diff_snapshot, snapshot_files, short_hash
from auto_harness.utils.ports import is_port_open
from auto_harness.utils.time import compact_timestamp


class VerifyModule:
    def __init__(self, urlopen=None) -> None:
        self.urlopen = urlopen or urllib.request.urlopen

    def verify(self, run_dir: Path, analysis: Dict, runner_result: Dict) -> StageResult:
        trace_id = "verify_%s_%s" % (compact_timestamp(), short_hash(str(run_dir), 6))
        service = self._service_discovery(runner_result)
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        verify_workspace = run_dir / "workspace" / "verify_workspace"
        before = snapshot_files(run_dir / "workspace" / "repo")
        http_evidence = self._execute_http_trace(trace_id, service, analysis, evidence_dir)
        after = snapshot_files(run_dir / "workspace" / "repo")
        changed = diff_snapshot(before, after)
        checks = self._artifact_checks(changed)
        if http_evidence:
            checks.append(http_evidence["check"])
        status = "pass" if self._can_pass(service, checks) else "uncertain"
        diagnosis = {
            "category": "none" if status == "pass" else "unknown",
            "root_cause": "" if status == "pass" else "dry-run or missing end-to-end evidence",
            "confidence": 0.5 if status == "uncertain" else 0.9,
        }
        result = VerifyResult(
            status=status,
            trace_id=trace_id,
            service=service,
            checks=checks,
            diagnosis=diagnosis,
            evidence=[http_evidence["path"]] if http_evidence else [],
            next_action="report",
        )
        evidence_path = evidence_dir / ("%s_verify.json" % trace_id)
        evidence_path.write_text(json.dumps(result.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(
            "verify",
            "passed" if status == "pass" else "uncertain",
            "verify completed with %s" % status,
            data=result.__dict__,
            evidence=[str(evidence_path)] + ([http_evidence["path"]] if http_evidence else []),
        )

    def _service_discovery(self, runner_result: Dict) -> Dict:
        port = int(runner_result.get("expected_port") or 0)
        service_ready = bool(runner_result.get("service_ready"))
        if port and not service_ready:
            service_ready = is_port_open("127.0.0.1", port)
        return {
            "type": "unknown",
            "endpoint_candidates": ["http://127.0.0.1:%s" % port] if port else [],
            "process_alive": bool(runner_result.get("pid")),
            "port_ready": service_ready,
            "log_path": runner_result.get("log_path"),
        }

    def _execute_http_trace(self, trace_id: str, service: Dict, analysis: Dict, evidence_dir: Path) -> Optional[Dict]:
        request_plan = self._build_request_plan(trace_id, service, analysis)
        if not request_plan:
            return None
        traced_url = request_plan["url"]
        method = request_plan["method"]
        body = request_plan.get("body")
        headers = request_plan.get("headers", {})
        request_record = {
            "method": method,
            "url": traced_url,
            "trace_id": trace_id,
            "headers": headers,
            "body": request_plan.get("body_json") if request_plan.get("body_json") is not None else None,
        }
        response_record = {}
        status = "uncertain"
        reason = "HTTP response did not contain trace id"
        try:
            req = urllib.request.Request(traced_url, data=body, method=method)
            for name, value in headers.items():
                req.add_header(name, value)
            with self.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                response_record = {
                    "status_code": getattr(resp, "status", None) or getattr(resp, "code", None),
                    "body_tail": body[-4000:],
                }
                if trace_id in body:
                    status = "pass"
                    reason = "HTTP response contains trace id"
        except Exception as exc:  # noqa: BLE001 - stored as evidence, not re-raised
            response_record = {"error": str(exc)}
            reason = "HTTP trace request failed"
        evidence = {
            "request": request_record,
            "response": response_record,
            "check": {
                "name": "http_trace_response",
                "status": status,
                "evidence": "trace_id=%s" % trace_id,
                "reason": reason,
            },
        }
        evidence_path = evidence_dir / ("%s_http_trace.json" % trace_id)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(evidence_path), "check": evidence["check"]}

    def _build_request_plan(self, trace_id: str, service: Dict, analysis: Dict) -> Optional[Dict]:
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        request_hint = verify_hint.get("request", {}) if isinstance(verify_hint, dict) else {}
        endpoint = self._select_endpoint(service, analysis)
        if not endpoint:
            return None

        method = str(request_hint.get("method") or "GET").upper()
        path = request_hint.get("path")
        if path:
            endpoint = urllib.parse.urljoin(endpoint.rstrip("/") + "/", path.lstrip("/"))

        body_json = None
        body = None
        headers = {}
        if method == "GET":
            endpoint = self._append_trace_query(endpoint, trace_id)
        elif method == "POST":
            template = request_hint.get("json")
            if template is None:
                template = {"trace_id": "{{trace_id}}", "prompt": "auto harness trace {{trace_id}}"}
            body_json = self._replace_trace(template, trace_id)
            body = json.dumps(body_json).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            return None
        return {
            "method": method,
            "url": endpoint,
            "body": body,
            "body_json": body_json,
            "headers": headers,
        }

    def _append_trace_query(self, endpoint: str, trace_id: str) -> str:
        parsed = urllib.parse.urlparse(endpoint)
        query = urllib.parse.parse_qs(parsed.query)
        query["_auto_harness_trace"] = [trace_id]
        return urllib.parse.urlunparse(
            parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
        )

    def _replace_trace(self, value, trace_id: str):
        if isinstance(value, str):
            return value.replace("{{trace_id}}", trace_id)
        if isinstance(value, list):
            return [self._replace_trace(item, trace_id) for item in value]
        if isinstance(value, dict):
            return {key: self._replace_trace(item, trace_id) for key, item in value.items()}
        return value

    def _select_endpoint(self, service: Dict, analysis: Dict) -> Optional[str]:
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        endpoint = verify_hint.get("endpoint")
        if endpoint:
            return endpoint
        candidates = service.get("endpoint_candidates") or []
        return candidates[0] if candidates else None

    def _artifact_checks(self, changed_files: List[str]) -> List[Dict]:
        return [
            {
                "name": "artifact_freshness",
                "status": "pass" if changed_files else "uncertain",
                "evidence": changed_files,
                "reason": "new or changed files after trace execution are required for strong pass",
            }
        ]

    def _can_pass(self, service: Dict, checks: List[Dict]) -> bool:
        if not service.get("process_alive"):
            return False
        if any(check.get("status") == "fail" for check in checks):
            return False
        strong_pass_names = {"artifact_freshness", "http_trace_response"}
        return any(
            check.get("name") in strong_pass_names and check.get("status") == "pass"
            for check in checks
        )
