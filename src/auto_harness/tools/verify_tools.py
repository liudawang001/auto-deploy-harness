"""Verify probe tools for the LLM-driven verify agent.

Each tool:
- Receives normalized input (from policy, not raw LLM output)
- Performs a real local probe against the service
- Writes evidence file
- Returns a ToolResult with strong_verify_pass only when current trace_id is found
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from auto_harness.agent_runtime.schemas import ToolResult
from auto_harness.utils.time import utc_now_iso

# Defense-in-depth: allowed hosts for verify probes
_ALLOWED_VERIFY_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _check_host_allowed(endpoint: str) -> Optional[str]:
    """Return error message if endpoint host is not in allowed list, else None."""
    try:
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or ""
    except Exception:
        return "invalid endpoint URL: %s" % endpoint[:100]
    if host and host not in _ALLOWED_VERIFY_HOSTS:
        return "external host not allowed: %s" % host
    return None


def probe_http(tool_input: Dict, context: Dict) -> ToolResult:
    """Send an HTTP request to the local service and check for trace_id in response.

    Input: endpoint, method, path, body, headers, trace_template
    """
    started = utc_now_iso()
    endpoint = tool_input.get("endpoint", "")
    method = str(tool_input.get("method") or "GET").upper()
    path = tool_input.get("path", "/")
    trace_id = context.get("trace_id", "")
    evidence_dir = context.get("evidence_dir")
    run_dir = context.get("run_dir")
    urlopen = context.get("urlopen") or urllib.request.urlopen

    # Defense-in-depth: reject external hosts
    host_error = _check_host_allowed(endpoint)
    if host_error:
        return ToolResult(
            status="rejected",
            tool_name="probe_http",
            evidence={},
            error=host_error,
            started_at=started,
            ended_at=utc_now_iso(),
        )

    # Build URL
    if path and path.startswith("/"):
        url = endpoint.rstrip("/") + path
    else:
        url = endpoint

    # Append trace to query for GET
    if method == "GET" and trace_id:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        query["_auto_harness_trace"] = [trace_id]
        url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))

    body_data = None
    headers = dict(tool_input.get("headers") or {})
    body_json = tool_input.get("body")

    if method == "POST":
        if body_json is not None and isinstance(body_json, dict):
            body_str = json.dumps(body_json)
            if trace_id:
                body_str = body_str.replace("{{trace_id}}", trace_id)
            body_data = body_str.encode("utf-8")
        elif body_json is not None and isinstance(body_json, str):
            b = body_json
            if trace_id:
                b = b.replace("{{trace_id}}", trace_id)
            body_data = b.encode("utf-8")
        else:
            # Default POST body with trace_id
            default_body = {"data": [trace_id], "trace_id": trace_id}
            body_data = json.dumps(default_body).encode("utf-8")
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

    status = "uncertain"
    reason = "HTTP response did not contain trace id"
    response_record = {}
    strong_verify_pass = False
    error = None

    try:
        req = urllib.request.Request(url, data=body_data, method=method)
        for name, value in headers.items():
            req.add_header(name, value)
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status_code = getattr(resp, "status", None) or getattr(resp, "code", None)
            response_record = {"status_code": status_code, "body_tail": body[-4000:]}
            if trace_id and trace_id in body:
                status = "passed"
                reason = "HTTP response contains current trace id"
                strong_verify_pass = True
            else:
                reason = "HTTP %s did not contain current trace id" % status_code
    except Exception as exc:
        error = str(exc)
        reason = "HTTP request failed: %s" % str(exc)[:200]

    ended = utc_now_iso()

    # Write evidence
    evidence_path = None
    if evidence_dir:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence = {
            "tool": "probe_http",
            "request": {"method": method, "url": url, "trace_id": trace_id},
            "response": response_record,
            "status": status,
            "reason": reason,
        }
        fname = "%s_agent_probe_%s.json" % (trace_id, context.get("step_index", 0))
        ep = evidence_dir / fname
        ep.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_path = str(ep)

    return ToolResult(
        status=status,
        tool_name="probe_http",
        evidence={"request_url": url, "method": method, "reason": reason},
        evidence_path=evidence_path,
        strong_verify_pass=strong_verify_pass,
        error=error,
        started_at=started,
        ended_at=ended,
    )


def discover_gradio_api(tool_input: Dict, context: Dict) -> ToolResult:
    """Read Gradio /config, find a callable endpoint, and probe it with trace_id.

    Input: endpoint, trace_template
    """
    started = utc_now_iso()
    endpoint = tool_input.get("endpoint", "")
    trace_id = context.get("trace_id", "")
    evidence_dir = context.get("evidence_dir")
    urlopen = context.get("urlopen") or urllib.request.urlopen

    # Defense-in-depth: reject external hosts
    host_error = _check_host_allowed(endpoint)
    if host_error:
        return ToolResult(
            status="rejected",
            tool_name="discover_gradio_api",
            evidence={},
            error=host_error,
            started_at=started,
            ended_at=utc_now_iso(),
        )

    config_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "config")
    status = "uncertain"
    reason = "Gradio config not accessible or no callable endpoint found"
    strong_verify_pass = False
    error = None
    discovery = {}

    try:
        req = urllib.request.Request(config_url, method="GET")
        with urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        config = json.loads(raw)
        dependencies = config.get("dependencies") if isinstance(config, dict) else None
        if isinstance(dependencies, list):
            for dep in dependencies:
                if not isinstance(dep, dict):
                    continue
                if dep.get("backend_fn") is False or dep.get("api_name") is False:
                    continue
                api_name = dep.get("api_name")
                if not isinstance(api_name, str) or not api_name.strip("/"):
                    continue
                normalized_api = api_name.strip("/")
                # Build probe URL
                probe_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "call/%s" % normalized_api)
                probe_body = json.dumps({"data": [trace_id]}).encode("utf-8")
                probe_req = urllib.request.Request(probe_url, data=probe_body, method="POST")
                probe_req.add_header("Content-Type", "application/json")
                with urlopen(probe_req, timeout=10) as probe_resp:
                    probe_body_text = probe_resp.read().decode("utf-8", errors="replace")

                discovery = {"config_url": config_url, "api_name": api_name, "probe_url": probe_url}

                # Check initial response for trace
                if trace_id in probe_body_text:
                    status = "passed"
                    reason = "Gradio API response contains current trace id"
                    strong_verify_pass = True
                    break

                # Try follow-up for queue mode
                event_id = _extract_event_id(probe_body_text)
                if event_id:
                    follow_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "call/%s/%s" % (normalized_api, urllib.parse.quote(event_id, safe="")))
                    follow_req = urllib.request.Request(follow_url, method="GET")
                    with urlopen(follow_req, timeout=10) as follow_resp:
                        follow_body = follow_resp.read().decode("utf-8", errors="replace")
                    if trace_id in follow_body:
                        status = "passed"
                        reason = "Gradio queue follow-up response contains current trace id"
                        strong_verify_pass = True
                    discovery["follow_up_url"] = follow_url
                    break
    except Exception as exc:
        error = str(exc)
        reason = "Gradio discovery failed: %s" % str(exc)[:200]

    ended = utc_now_iso()

    evidence_path = None
    if evidence_dir:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence = {"tool": "discover_gradio_api", "discovery": discovery, "status": status, "reason": reason, "trace_id": trace_id}
        fname = "%s_agent_probe_%s.json" % (trace_id, context.get("step_index", 0))
        ep = evidence_dir / fname
        ep.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_path = str(ep)

    return ToolResult(
        status=status,
        tool_name="discover_gradio_api",
        evidence={"discovery": discovery, "reason": reason},
        evidence_path=evidence_path,
        strong_verify_pass=strong_verify_pass,
        error=error,
        started_at=started,
        ended_at=ended,
    )


def discover_openapi_schema(tool_input: Dict, context: Dict) -> ToolResult:
    """Read /openapi.json, find a safe POST endpoint, and probe with trace_id.

    Input: endpoint, trace_template
    """
    started = utc_now_iso()
    endpoint = tool_input.get("endpoint", "")
    trace_id = context.get("trace_id", "")
    evidence_dir = context.get("evidence_dir")
    urlopen = context.get("urlopen") or urllib.request.urlopen

    # Defense-in-depth: reject external hosts
    host_error = _check_host_allowed(endpoint)
    if host_error:
        return ToolResult(
            status="rejected",
            tool_name="discover_openapi_schema",
            evidence={},
            error=host_error,
            started_at=started,
            ended_at=utc_now_iso(),
        )

    openapi_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "openapi.json")
    status = "uncertain"
    reason = "OpenAPI schema not accessible or no suitable endpoint found"
    strong_verify_pass = False
    error = None
    discovery = {}

    try:
        req = urllib.request.Request(openapi_url, method="GET")
        with urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        spec = json.loads(raw)

        paths = spec.get("paths") if isinstance(spec, dict) else None
        if isinstance(paths, dict):
            for path in sorted(paths):
                if "{" in path:
                    continue
                path_item = paths.get(path)
                if not isinstance(path_item, dict):
                    continue
                operation = path_item.get("post")
                if not isinstance(operation, dict):
                    continue

                # Found a safe POST endpoint - probe it
                probe_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", path.lstrip("/"))
                probe_body = json.dumps({"trace_id": trace_id, "prompt": "auto harness trace %s" % trace_id}).encode("utf-8")
                probe_req = urllib.request.Request(probe_url, data=probe_body, method="POST")
                probe_req.add_header("Content-Type", "application/json")
                with urlopen(probe_req, timeout=10) as probe_resp:
                    probe_body_text = probe_resp.read().decode("utf-8", errors="replace")

                discovery = {"openapi_url": openapi_url, "probe_path": path, "probe_url": probe_url}
                if trace_id in probe_body_text:
                    status = "passed"
                    reason = "OpenAPI endpoint response contains current trace id"
                    strong_verify_pass = True
                else:
                    reason = "OpenAPI endpoint responded but did not contain current trace id"
                break
    except Exception as exc:
        error = str(exc)
        reason = "OpenAPI discovery failed: %s" % str(exc)[:200]

    ended = utc_now_iso()

    evidence_path = None
    if evidence_dir:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence = {"tool": "discover_openapi_schema", "discovery": discovery, "status": status, "reason": reason, "trace_id": trace_id}
        fname = "%s_agent_probe_%s.json" % (trace_id, context.get("step_index", 0))
        ep = evidence_dir / fname
        ep.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_path = str(ep)

    return ToolResult(
        status=status,
        tool_name="discover_openapi_schema",
        evidence={"discovery": discovery, "reason": reason},
        evidence_path=evidence_path,
        strong_verify_pass=strong_verify_pass,
        error=error,
        started_at=started,
        ended_at=ended,
    )


def probe_browser_dom(tool_input: Dict, context: Dict) -> ToolResult:
    """Probe a local page DOM for trace_id presence.

    This is a lightweight implementation that fetches the page HTML and
    checks for trace_id in the response body. A full browser-based probe
    (e.g. using Playwright/Selenium) is deferred to a future phase.

    Input: endpoint, trace_template
    """
    started = utc_now_iso()
    endpoint = tool_input.get("endpoint", "")
    trace_id = context.get("trace_id", "")
    evidence_dir = context.get("evidence_dir")
    urlopen = context.get("urlopen") or urllib.request.urlopen

    status = "uncertain"
    reason = "Browser DOM probe not yet fully implemented; falling back to HTML fetch"
    strong_verify_pass = False
    error = None
    response_record = {}

    # Defense-in-depth: reject external hosts
    host = ""
    try:
        parsed = urllib.parse.urlparse(endpoint)
        host = parsed.hostname or ""
    except Exception:
        pass
    from auto_harness.agent_runtime.policy import DEFAULT_ALLOWED_HOSTS
    if host and host not in DEFAULT_ALLOWED_HOSTS:
        return ToolResult(
            status="rejected",
            tool_name="probe_browser_dom",
            evidence={},
            error="external host not allowed: %s" % host,
            started_at=started,
            ended_at=utc_now_iso(),
        )

    # Fetch HTML and check for trace_id
    try:
        req = urllib.request.Request(endpoint, method="GET")
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status_code = getattr(resp, "status", None) or getattr(resp, "code", None)
            response_record = {"status_code": status_code, "body_tail": body[-4000:]}
            if trace_id and trace_id in body:
                status = "passed"
                reason = "Page HTML contains current trace id"
                strong_verify_pass = True
            else:
                reason = "HTTP %s page did not contain current trace id" % status_code
    except Exception as exc:
        error = str(exc)
        reason = "Browser DOM probe failed: %s" % str(exc)[:200]

    ended = utc_now_iso()

    # Write evidence
    evidence_path = None
    if evidence_dir:
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence = {
            "tool": "probe_browser_dom",
            "request": {"url": endpoint, "trace_id": trace_id},
            "response": response_record,
            "status": status,
            "reason": reason,
        }
        fname = "%s_agent_probe_%s.json" % (trace_id, context.get("step_index", 0))
        ep = evidence_dir / fname
        ep.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_path = str(ep)

    return ToolResult(
        status=status,
        tool_name="probe_browser_dom",
        evidence={"request_url": endpoint, "reason": reason},
        evidence_path=evidence_path,
        strong_verify_pass=strong_verify_pass,
        error=error,
        started_at=started,
        ended_at=ended,
    )


def _extract_event_id(body: str) -> Optional[str]:
    """Extract event_id from Gradio queue initial response."""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("event_id"), str):
        return parsed["event_id"]
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            parsed = json.loads(line[5:].strip())
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("event_id"), str):
            return parsed["event_id"]
    return None
