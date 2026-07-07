import json
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent.schemas import AgentObservation
from auto_harness.models.result import StageResult
from auto_harness.models.verify import VerifyResult
from auto_harness.utils.files import diff_snapshot, snapshot_files, short_hash
from auto_harness.utils.ports import is_port_open
from auto_harness.utils.time import compact_timestamp
from auto_harness.verify import BrowserVerifier, StreamlitVerifier


class VerifyModule:
    def __init__(
        self,
        urlopen=None,
        stage_context: Optional[Dict] = None,
        browser_verifier: BrowserVerifier = None,
        progress_callback=None,
        verify_planner=None,
    ) -> None:
        self.urlopen = urlopen or urllib.request.urlopen
        self.stage_context = stage_context or {}
        self.streamlit_verifier = StreamlitVerifier(urlopen=self.urlopen)
        self.browser_verifier = browser_verifier or BrowserVerifier()
        self.progress_callback = progress_callback
        self.verify_planner = verify_planner

    def verify(self, run_dir: Path, analysis: Dict, runner_result: Dict) -> StageResult:
        trace_id = "verify_%s_%s" % (compact_timestamp(), short_hash(str(run_dir), 6))
        service = self._service_discovery(runner_result)
        self._progress("service_discovered", {"service": service})
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        verify_workspace = run_dir / "workspace" / "verify_workspace"
        repo_dir = run_dir / "workspace" / "repo"
        before = snapshot_files(repo_dir)
        self._progress("first_inference_probe_started", {"probe": "http_trace"})
        http_evidence = self._execute_http_trace(trace_id, service, analysis, evidence_dir, attempt_label="initial")
        self._progress("first_inference_probe_completed", {"probe": "http_trace", "check": http_evidence["check"] if http_evidence else {}})
        after = snapshot_files(repo_dir)
        changed = diff_snapshot(before, after)
        checks = self._artifact_checks(repo_dir, changed)
        if http_evidence:
            checks.append(http_evidence["check"])
        streamlit_evidence = self._execute_streamlit_probe(trace_id, service, analysis, evidence_dir)
        self._progress("streamlit_probe_completed", {"check": streamlit_evidence["check"] if streamlit_evidence else {}})
        if streamlit_evidence:
            checks.append(streamlit_evidence["check"])
        browser_evidence = self._execute_browser_probe(trace_id, service, analysis, evidence_dir)
        self._progress("browser_probe_completed", {"check": browser_evidence["check"] if browser_evidence else {}})
        if browser_evidence:
            checks.append(browser_evidence["check"])
        status = "pass" if self._can_pass(service, checks) else "uncertain"
        planner_evidence = None
        planner_evidences = []
        planner_result = None
        if status == "uncertain":
            planner_result = self._plan_and_execute_verify_hint(
                trace_id,
                run_dir,
                service,
                analysis,
                evidence_dir,
                checks,
            )
            if planner_result and planner_result.get("evidences"):
                planner_evidences = planner_result["evidences"]
                for evidence in planner_evidences:
                    checks.append(evidence["check"])
                    if self._can_pass(service, checks):
                        planner_evidence = evidence
                        status = "pass"
                        break
                if status != "pass":
                    status = "pass" if self._can_pass(service, checks) else "uncertain"
        self._progress("verify_completed", {"result_status": status, "trace_id": trace_id})
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
            evidence=(
                ([http_evidence["path"]] if http_evidence else [])
                + ([streamlit_evidence["path"]] if streamlit_evidence else [])
                + ([browser_evidence["path"]] if browser_evidence else [])
                + [evidence["path"] for evidence in planner_evidences]
            ),
            next_action="report",
        )
        result_data = result.__dict__
        if planner_result:
            result_data["llm_verify_planner"] = planner_result.get("planner", {})
        if self.stage_context:
            result_data["control_context"] = self.stage_context
        evidence_path = evidence_dir / ("%s_verify.json" % trace_id)
        evidence_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(
            "verify",
            "passed" if status == "pass" else "uncertain",
            "verify completed with %s" % status,
            data=result_data,
            evidence=(
                [str(evidence_path)]
                + ([http_evidence["path"]] if http_evidence else [])
                + ([streamlit_evidence["path"]] if streamlit_evidence else [])
                + ([browser_evidence["path"]] if browser_evidence else [])
                + [evidence["path"] for evidence in planner_evidences]
            ),
        )

    def _plan_and_execute_verify_hint(
        self,
        trace_id: str,
        run_dir: Path,
        service: Dict,
        analysis: Dict,
        evidence_dir: Path,
        checks: List[Dict],
    ) -> Optional[Dict]:
        if not self.verify_planner:
            return None
        if not service.get("process_alive") or not service.get("port_ready"):
            return None
        if any(check.get("status") == "pass" for check in checks if check.get("name") in {"http_trace_response", "browser_dom_probe", "artifact_download_validation"}):
            return None
        repo_dir = run_dir / "workspace" / "repo"
        observation = AgentObservation(
            task_id=run_dir.name,
            stage="verify",
            repo_dir=str(repo_dir),
            file_tree=[],
            selected_files=self._verify_selected_files(repo_dir),
            deterministic_result={"analysis": analysis, "service": service, "checks": checks},
            runtime_policy={},
            allowed_action_types=["update_verify_hint"],
        )
        planner = self.verify_planner.plan(observation)
        if planner.get("status") != "ok":
            return {"planner": planner}
        evidences = []
        candidates = planner.get("verify_candidates") or ([planner["verify_hint"]] if planner.get("verify_hint") else [])
        for index, verify_hint in enumerate(candidates[:3]):
            planned_analysis = dict(analysis)
            planned_analysis["verify_hint"] = verify_hint
            self._progress("llm_verify_hint_generated", {"confidence": planner.get("confidence"), "candidate_index": index})
            evidence = self._execute_http_trace(trace_id, service, planned_analysis, evidence_dir, attempt_label="llm_planner_%s" % index)
            if evidence:
                evidences.append(evidence)
            if evidence and evidence.get("check", {}).get("status") == "pass":
                break
        return {"planner": planner, "evidences": evidences, "evidence": evidences[-1] if evidences else None}

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

    def _execute_http_trace(self, trace_id: str, service: Dict, analysis: Dict, evidence_dir: Path, attempt_label: str = "initial") -> Optional[Dict]:
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
            "discovery": request_plan.get("discovery"),
        }
        response_record = {}
        follow_up_record = None
        status = "uncertain"
        reason = "HTTP response did not contain trace id"
        try:
            req = urllib.request.Request(traced_url, data=body, method=method)
            for name, value in headers.items():
                req.add_header(name, value)
            self._progress("http_trace_request_sent", {"method": method, "url": traced_url})
            with self.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                response_record = {
                    "status_code": getattr(resp, "status", None) or getattr(resp, "code", None),
                    "body_tail": body[-4000:],
                    "stream_detected": self._body_looks_like_sse(body),
                }
                if trace_id in body:
                    status = "pass"
                    reason = "HTTP response contains trace id"
                elif request_plan.get("follow_up_url_template"):
                    self._progress("http_trace_follow_up_started", {"trace_id": trace_id})
                    follow_up_record = self._execute_follow_up_trace(
                        request_plan["follow_up_url_template"],
                        body,
                        trace_id,
                    )
                    if follow_up_record.get("trace_found"):
                        status = "pass"
                        reason = "HTTP follow-up response contains trace id"
                    elif follow_up_record.get("error"):
                        reason = "HTTP follow-up request failed"
        except Exception as exc:  # noqa: BLE001 - stored as evidence, not re-raised
            response_record = {"error": str(exc)}
            reason = "HTTP trace request failed"
        evidence = {
            "request": request_record,
            "response": response_record,
            "follow_up_response": follow_up_record,
            "attempt_label": attempt_label,
            "check": {
                "name": "http_trace_response",
                "status": status,
                "evidence": "trace_id=%s" % trace_id,
                "reason": reason,
            },
        }
        safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in attempt_label) or "initial"
        evidence_path = evidence_dir / ("%s_http_trace_%s.json" % (trace_id, safe_label))
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(evidence_path), "check": evidence["check"]}

    def _execute_streamlit_probe(self, trace_id: str, service: Dict, analysis: Dict, evidence_dir: Path) -> Optional[Dict]:
        frameworks = set(analysis.get("frameworks") or []) if isinstance(analysis, dict) else set()
        if "streamlit" not in frameworks:
            return None
        endpoint = self._select_endpoint(service, analysis)
        if not endpoint:
            return None
        check = self.streamlit_verifier.probe(endpoint, trace_id)
        evidence = {"check": check}
        evidence_path = evidence_dir / ("%s_streamlit_probe.json" % trace_id)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(evidence_path), "check": check}

    def _execute_browser_probe(self, trace_id: str, service: Dict, analysis: Dict, evidence_dir: Path) -> Optional[Dict]:
        frameworks = set(analysis.get("frameworks") or []) if isinstance(analysis, dict) else set()
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        if not frameworks.intersection({"gradio", "streamlit"}) and verify_hint.get("service_type") != "webui":
            return None
        endpoint = self._select_endpoint(service, analysis)
        if not endpoint:
            return None
        screenshot_path = evidence_dir / ("%s_browser.png" % trace_id)
        check = self.browser_verifier.probe(endpoint, trace_id, frameworks=frameworks, screenshot_path=screenshot_path)
        evidence = {"check": check}
        evidence_path = evidence_dir / ("%s_browser_probe.json" % trace_id)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(evidence_path), "check": check}

    def _build_request_plan(self, trace_id: str, service: Dict, analysis: Dict) -> Optional[Dict]:
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        request_hint = verify_hint.get("request", {}) if isinstance(verify_hint, dict) else {}
        endpoint = self._select_endpoint(service, analysis)
        if not endpoint:
            return None

        discovered = self._discover_gradio_request(endpoint, analysis)
        if discovered:
            request_hint = discovered["request"]
            endpoint = discovered["endpoint"]
        elif self._is_openai_compatible(analysis, verify_hint):
            request_hint = self._openai_compatible_request_hint(verify_hint)
            model_discovery = self._discover_openai_model(endpoint, verify_hint)
            discovered = {
                "type": "openai_compatible",
                "models_url": model_discovery.get("models_url", ""),
                "model_id": model_discovery.get("model_id", ""),
                "model_source": model_discovery.get("source", ""),
                "stream": bool((request_hint.get("json") or {}).get("stream")),
            }
        else:
            discovered = self._discover_openapi_request(endpoint, analysis)
            if discovered:
                request_hint = discovered["request"]
                endpoint = discovered["endpoint"]

        base_endpoint = endpoint
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
            body_json = self._replace_model_placeholder(body_json, verify_hint, (discovered or {}).get("model_id", ""))
            body = json.dumps(body_json).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            return None
        follow_up_url_template = None
        follow_up_hint = request_hint.get("follow_up") if isinstance(request_hint, dict) else None
        if isinstance(follow_up_hint, dict) and follow_up_hint.get("method", "GET").upper() == "GET":
            follow_up_path = follow_up_hint.get("path")
            if follow_up_path:
                follow_up_url_template = urllib.parse.urljoin(base_endpoint.rstrip("/") + "/", follow_up_path.lstrip("/"))
        return {
            "method": method,
            "url": endpoint,
            "body": body,
            "body_json": body_json,
            "headers": headers,
            "discovery": discovered,
            "follow_up_url_template": follow_up_url_template,
        }

    def _is_openai_compatible(self, analysis: Dict, verify_hint: Dict) -> bool:
        frameworks = set(analysis.get("frameworks") or []) if isinstance(analysis, dict) else set()
        return verify_hint.get("service_type") == "openai_compatible" or bool(frameworks.intersection({"vllm", "openai_compatible"}))

    def _openai_compatible_request_hint(self, verify_hint: Dict) -> Dict:
        request = verify_hint.get("request") if isinstance(verify_hint, dict) else None
        if isinstance(request, dict) and request.get("path"):
            return request
        return {
            "method": "POST",
            "path": "/v1/chat/completions",
            "json": {
                "model": "{{model}}",
                "messages": [{"role": "user", "content": "auto harness trace {{trace_id}}"}],
                "temperature": 0,
                "max_tokens": 16,
            },
        }

    def _discover_openai_model(self, endpoint: str, verify_hint: Dict) -> Dict:
        hinted = verify_hint.get("model") or verify_hint.get("model_id")
        if hinted:
            return {"model_id": hinted, "source": "verify_hint"}
        models_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "v1/models")
        try:
            req = urllib.request.Request(models_url, method="GET")
            with self.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 - optional discovery
            return {"model_id": "auto-harness-smoke-model", "source": "fallback", "models_url": models_url}
        discovered = self._first_openai_model_id(data)
        if discovered:
            return {"model_id": discovered, "source": "v1/models", "models_url": models_url}
        return {"model_id": "auto-harness-smoke-model", "source": "fallback", "models_url": models_url}

    def _first_openai_model_id(self, data) -> str:
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return ""
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                return item["id"]
        return ""

    def _discover_openapi_request(self, endpoint: str, analysis: Dict) -> Optional[Dict]:
        frameworks = set(analysis.get("frameworks") or []) if isinstance(analysis, dict) else set()
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        if not frameworks.intersection({"fastapi", "flask"}) and verify_hint.get("service_type") != "api":
            return None
        openapi_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "openapi.json")
        try:
            req = urllib.request.Request(openapi_url, method="GET")
            with self.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            spec = json.loads(raw)
        except Exception:  # noqa: BLE001 - discovery is optional
            return None
        operation = self._select_openapi_operation(spec)
        if not operation:
            return None
        schema = operation.get("schema") or {"type": "object"}
        body = self._sample_json_from_schema(schema, spec, trace_token="{{trace_id}}")
        return {
            "type": "openapi_schema",
            "openapi_url": openapi_url,
            "endpoint": endpoint,
            "operation_id": operation.get("operation_id", ""),
            "request": {
                "method": "POST",
                "path": operation["path"],
                "json": body,
            },
        }

    def _select_openapi_operation(self, spec: Dict) -> Optional[Dict]:
        paths = spec.get("paths") if isinstance(spec, dict) else None
        if not isinstance(paths, dict):
            return None
        for path in sorted(paths):
            if "{" in path or "}" in path:
                continue
            path_item = paths.get(path)
            if not isinstance(path_item, dict):
                continue
            operation = path_item.get("post")
            if not isinstance(operation, dict):
                continue
            request_body = operation.get("requestBody") if isinstance(operation.get("requestBody"), dict) else {}
            content = request_body.get("content") if isinstance(request_body, dict) else {}
            json_content = content.get("application/json") if isinstance(content, dict) else None
            schema = json_content.get("schema") if isinstance(json_content, dict) else None
            return {
                "path": path,
                "operation_id": operation.get("operationId", ""),
                "schema": schema or {"type": "object"},
            }
        return None

    def _sample_json_from_schema(self, schema, spec: Dict, trace_token: str):
        schema = self._resolve_schema_ref(schema, spec)
        if not isinstance(schema, dict):
            return {"trace_id": trace_token, "prompt": "auto harness trace %s" % trace_token}
        schema_type = schema.get("type")
        if schema_type == "array":
            return [self._sample_json_from_schema(schema.get("items") or {"type": "string"}, spec, trace_token)]
        if schema_type == "integer":
            return 0
        if schema_type == "number":
            return 0
        if schema_type == "boolean":
            return True
        if schema_type == "string" or "enum" in schema:
            return self._sample_string(schema, trace_token)
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if properties:
            required = schema.get("required") if isinstance(schema.get("required"), list) else []
            keys = required or list(properties.keys())[:3]
            result = {}
            for key in keys:
                if key in properties:
                    result[key] = self._sample_json_from_schema(properties[key], spec, trace_token)
            if not any(self._value_contains_trace(value, trace_token) for value in result.values()):
                result["trace_id"] = trace_token
            return result
        return {"trace_id": trace_token, "prompt": "auto harness trace %s" % trace_token}

    def _resolve_schema_ref(self, schema, spec: Dict):
        if not isinstance(schema, dict) or "$ref" not in schema:
            return schema
        ref = str(schema.get("$ref") or "")
        if not ref.startswith("#/"):
            return schema
        current = spec
        for part in ref[2:].split("/"):
            if not isinstance(current, dict):
                return schema
            current = current.get(part)
        return current if isinstance(current, dict) else schema

    def _sample_string(self, schema: Dict, trace_token: str) -> str:
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return str(enum[0])
        return "auto harness trace %s" % trace_token

    def _value_contains_trace(self, value, trace_token: str) -> bool:
        if isinstance(value, str):
            return trace_token in value
        if isinstance(value, list):
            return any(self._value_contains_trace(item, trace_token) for item in value)
        if isinstance(value, dict):
            return any(self._value_contains_trace(item, trace_token) for item in value.values())
        return False

    def _discover_gradio_request(self, endpoint: str, analysis: Dict) -> Optional[Dict]:
        frameworks = set(analysis.get("frameworks") or []) if isinstance(analysis, dict) else set()
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        if "gradio" not in frameworks and verify_hint.get("service_type") != "webui":
            return None
        config_url = urllib.parse.urljoin(endpoint.rstrip("/") + "/", "config")
        try:
            req = urllib.request.Request(config_url, method="GET")
            with self.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            config = json.loads(raw)
        except Exception:  # noqa: BLE001 - discovery is optional
            return None
        dependency = self._select_gradio_dependency(config)
        if not dependency:
            return None
        fn_index = dependency.get("id")
        api_name = dependency.get("api_name")
        path = "/api/predict"
        if isinstance(api_name, str) and api_name and api_name not in ("false", "None"):
            path = "/api/%s" % api_name.strip("/")
        body = {"data": ["{{trace_id}}"]}
        if fn_index is not None:
            body["fn_index"] = fn_index
        follow_up = None
        if self._gradio_queue_enabled(config, dependency) and isinstance(api_name, str) and api_name.strip("/"):
            normalized_api = api_name.strip("/")
            path = "/call/%s" % normalized_api
            body = {"data": ["{{trace_id}}"]}
            follow_up = {
                "method": "GET",
                "path": "/call/%s/{{event_id}}" % normalized_api,
            }
            fn_index = dependency.get("id")
        return {
            "type": "gradio_config",
            "config_url": config_url,
            "endpoint": endpoint,
            "dependency_id": fn_index,
            "api_name": api_name,
            "queue_enabled": bool(follow_up),
            "request": {
                "method": "POST",
                "path": path,
                "json": body,
                "follow_up": follow_up,
            },
        }

    def _select_gradio_dependency(self, config: Dict) -> Optional[Dict]:
        dependencies = config.get("dependencies") if isinstance(config, dict) else None
        if not isinstance(dependencies, list):
            return None
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                continue
            if dependency.get("backend_fn") is False:
                continue
            if dependency.get("api_name") is False:
                continue
            return dependency
        return None

    def _gradio_queue_enabled(self, config: Dict, dependency: Dict) -> bool:
        if dependency.get("queue") is True:
            return True
        if config.get("enable_queue") is True:
            return True
        if config.get("queue") is True:
            return True
        return False

    def _execute_follow_up_trace(self, url_template: str, initial_body: str, trace_id: str) -> Dict:
        event_id = self._extract_event_id(initial_body)
        if not event_id:
            return {
                "status": "skipped",
                "reason": "Gradio queue event_id was not present in initial response",
                "trace_found": False,
            }
        follow_up_url = url_template.replace("{{event_id}}", urllib.parse.quote(event_id, safe=""))
        try:
            req = urllib.request.Request(follow_up_url, method="GET")
            with self.urlopen(req, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return {
                "status": "completed",
                "url": follow_up_url,
                "event_id": event_id,
                "status_code": getattr(resp, "status", None) or getattr(resp, "code", None),
                "body_tail": body[-4000:],
                "trace_found": trace_id in body,
            }
        except Exception as exc:  # noqa: BLE001 - stored as evidence, not re-raised
            return {
                "status": "failed",
                "url": follow_up_url,
                "event_id": event_id,
                "error": str(exc),
                "trace_found": False,
            }

    def _extract_event_id(self, body: str) -> Optional[str]:
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

    def _progress(self, status: str, detail: Optional[Dict] = None) -> None:
        if not self.progress_callback:
            return
        payload = {"status": status}
        if detail:
            payload.update(detail)
        try:
            self.progress_callback(payload)
        except Exception:
            return

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

    def _replace_model_placeholder(self, value, verify_hint: Dict, discovered_model: str = ""):
        model = verify_hint.get("model") or verify_hint.get("model_id") or discovered_model or "auto-harness-smoke-model"
        if isinstance(value, str):
            return value.replace("{{model}}", model)
        if isinstance(value, list):
            return [self._replace_model_placeholder(item, verify_hint, discovered_model) for item in value]
        if isinstance(value, dict):
            return {key: self._replace_model_placeholder(item, verify_hint, discovered_model) for key, item in value.items()}
        return value

    def _body_looks_like_sse(self, body: str) -> bool:
        return any(line.startswith("data:") for line in body.splitlines())

    def _select_endpoint(self, service: Dict, analysis: Dict) -> Optional[str]:
        verify_hint = analysis.get("verify_hint", {}) if isinstance(analysis, dict) else {}
        endpoint = verify_hint.get("endpoint")
        if endpoint:
            return endpoint
        candidates = service.get("endpoint_candidates") or []
        return candidates[0] if candidates else None

    def _artifact_checks(self, repo_dir: Path, changed_files: List[str]) -> List[Dict]:
        validated = []
        invalid = []
        for rel_path in changed_files:
            path = repo_dir / rel_path
            try:
                if not path.is_file():
                    invalid.append({"path": rel_path, "reason": "not a file"})
                    continue
                size = path.stat().st_size
                if size <= 0:
                    invalid.append({"path": rel_path, "reason": "empty file"})
                    continue
                validated.append({
                    "path": rel_path,
                    "size_bytes": size,
                    "sha256": self._sha256_file(path),
                })
            except OSError as exc:
                invalid.append({"path": rel_path, "reason": str(exc)})
        if validated:
            validation_status = "pass"
            validation_reason = "new artifact files are readable and non-empty"
        elif changed_files:
            validation_status = "fail"
            validation_reason = "changed artifact files were empty, missing, or unreadable"
        else:
            validation_status = "uncertain"
            validation_reason = "no new artifact files were observed"
        return [
            {
                "name": "artifact_freshness",
                "status": "pass" if changed_files else "uncertain",
                "evidence": changed_files,
                "reason": "new or changed files after trace execution are required for strong pass",
            },
            {
                "name": "artifact_download_validation",
                "status": validation_status,
                "evidence": {
                    "validated": validated,
                    "invalid": invalid,
                },
                "reason": validation_reason,
            }
        ]

    def _verify_selected_files(self, repo_dir: Path) -> Dict[str, str]:
        selected = {}
        for name in ("README.md", "readme.md", "app.py", "main.py", "server.py", "routes.py"):
            path = repo_dir / name
            lowered = name.lower()
            if any(marker in lowered for marker in (".env", "secret", "credential", "token", "key")):
                continue
            try:
                if path.is_file():
                    selected[name] = "UNTRUSTED REPO CONTENT:\n" + path.read_text(encoding="utf-8", errors="ignore")[:6000]
            except OSError:
                continue
        return selected

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _can_pass(self, service: Dict, checks: List[Dict]) -> bool:
        if not service.get("process_alive"):
            return False
        if any(check.get("status") == "fail" for check in checks):
            return False
        strong_pass_names = {"artifact_download_validation", "http_trace_response", "browser_dom_probe"}
        return any(
            check.get("name") in strong_pass_names and check.get("status") == "pass"
            for check in checks
        )
