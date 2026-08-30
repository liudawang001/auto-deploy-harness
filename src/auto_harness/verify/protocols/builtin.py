"""Built-in protocol verifiers and strong-evidence evaluation."""

import urllib.parse

from auto_harness.verify.protocols.schemas import (
    Probe,
    ProbeEvidence,
    VerifyDecision,
)


class TraceProtocolVerifier:
    verifier_id = ""
    protocols = ()
    priority = 0

    def supports(self, candidate, analysis):
        protocol = str((candidate or {}).get("protocol") or "")
        return protocol in self.protocols

    def build_probe(self, trace_id, candidate, verify_spec):
        endpoint = str((candidate or {}).get("endpoint") or "")
        request = dict((verify_spec or {}).get("request") or {})
        return Probe(
            verifier_id=self.verifier_id,
            protocol=str((candidate or {}).get("protocol") or self.protocols[0]),
            trace_id=str(trace_id),
            method=str(request.get("method") or "GET").upper(),
            endpoint=endpoint,
            expected_port=int((candidate or {}).get("expected_port") or 0),
            request=request,
        )

    def execute_probe(self, probe, runtime):
        if runtime is None or not hasattr(runtime, "execute_probe"):
            return ProbeEvidence(
                verifier_id=self.verifier_id,
                protocol=probe.protocol,
                trace_id=probe.trace_id,
                endpoint=probe.endpoint,
                expected_port=probe.expected_port,
                process_alive=False,
                port_ready=False,
                status="unavailable",
                details={"reason": "probe runtime unavailable"},
            )
        return runtime.execute_probe(probe)

    def evaluate(self, evidence, expectation=None):
        expected_trace = str((expectation or {}).get("trace_id") or evidence.trace_id)
        endpoint = urllib.parse.urlparse(str(evidence.endpoint or ""))
        if endpoint.scheme not in {"http", "https"}:
            return self._decision("failed", "endpoint_scheme_rejected", evidence)
        if endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return self._decision("failed", "external_endpoint_rejected", evidence)
        try:
            endpoint_port = endpoint.port
        except ValueError:
            return self._decision("failed", "service_port_invalid", evidence)
        if evidence.expected_port and endpoint_port != evidence.expected_port:
            return self._decision("failed", "service_port_mismatch", evidence)
        if not evidence.process_alive:
            return self._decision("failed", "service_process_not_alive", evidence)
        if not evidence.port_ready:
            return self._decision("uncertain", "service_port_not_ready", evidence)
        if evidence.trace_id != expected_trace:
            return self._decision("failed", "stale_trace_rejected", evidence)
        if evidence.status != "pass" or not evidence.trace_observed:
            return self._decision("uncertain", "current_trace_not_observed", evidence)
        return VerifyDecision(
            status="passed",
            reason_code="current_trace_strong_evidence",
            verifier_id=self.verifier_id,
            trace_id=expected_trace,
            strong_evidence=True,
            reasons=["current trace is bound to the live local service"],
        )

    def _decision(self, status, reason, evidence):
        return VerifyDecision(
            status=status,
            reason_code=reason,
            verifier_id=self.verifier_id,
            trace_id=evidence.trace_id,
            strong_evidence=False,
            reasons=[reason],
        )


class HttpTraceVerifier(TraceProtocolVerifier):
    verifier_id = "builtin.http_trace"
    protocols = ("http",)
    priority = 100


class OpenAPITraceVerifier(TraceProtocolVerifier):
    verifier_id = "builtin.openapi_trace"
    protocols = ("openapi",)
    priority = 95


class OpenAICompatibleVerifier(TraceProtocolVerifier):
    verifier_id = "builtin.openai_compatible"
    protocols = ("openai_compatible",)
    priority = 90


class GradioVerifier(TraceProtocolVerifier):
    verifier_id = "builtin.gradio"
    protocols = ("gradio",)
    priority = 85


class StreamlitBrowserVerifier(TraceProtocolVerifier):
    verifier_id = "builtin.streamlit_browser"
    protocols = ("streamlit",)
    priority = 80


class BrowserDomTraceVerifier(TraceProtocolVerifier):
    verifier_id = "builtin.browser_dom_trace"
    protocols = ("browser_dom",)
    priority = 75


BUILTIN_VERIFIERS = (
    HttpTraceVerifier,
    OpenAPITraceVerifier,
    OpenAICompatibleVerifier,
    GradioVerifier,
    StreamlitBrowserVerifier,
    BrowserDomTraceVerifier,
)
