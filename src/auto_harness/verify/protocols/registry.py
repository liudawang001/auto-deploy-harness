"""Stable selection registry for protocol-specific verification."""

from auto_harness.verify.protocols.builtin import BUILTIN_VERIFIERS
from auto_harness.verify.protocols.schemas import ProtocolVerifierSelection


class ProtocolVerifierRegistry:
    def __init__(self, verifiers=()):
        self._verifiers = []
        for verifier in verifiers:
            self.register(verifier)

    @classmethod
    def builtins(cls):
        return cls(verifier_type() for verifier_type in BUILTIN_VERIFIERS)

    def register(self, verifier):
        if any(item.verifier_id == verifier.verifier_id for item in self._verifiers):
            raise ValueError("duplicate_verifier_id:%s" % verifier.verifier_id)
        self._verifiers.append(verifier)
        self._verifiers.sort(key=lambda item: (-item.priority, item.verifier_id))

    def all(self):
        return list(self._verifiers)

    def get(self, verifier_id):
        return next(
            (item for item in self._verifiers if item.verifier_id == verifier_id),
            None,
        )

    def select(self, analysis, candidate=None, operator_protocol=""):
        protocol, source, reason = self._requested_protocol(
            analysis or {}, candidate or {}, operator_protocol,
        )
        candidates = [
            item for item in self._verifiers if protocol in item.protocols
        ]
        if not candidates:
            protocol = "http"
            source = "registry_fallback"
            reason = "no specialized protocol signal; use local HTTP trace"
            candidates = [
                item for item in self._verifiers if protocol in item.protocols
            ]
        selected = candidates[0]
        return selected, ProtocolVerifierSelection(
            verifier_id=selected.verifier_id,
            protocol=protocol,
            source=source,
            reason=reason,
            candidates=[item.verifier_id for item in candidates],
        )

    @staticmethod
    def _requested_protocol(analysis, candidate, operator_protocol):
        if operator_protocol:
            return str(operator_protocol), "operator", "operator selected protocol"
        contract = analysis.get("deployment_contract") or {}
        if contract.get("valid"):
            protocol = str((contract.get("verify") or {}).get("protocol") or "")
            if protocol:
                return protocol, "deployment_contract", "validated manifest verify protocol"
        if candidate.get("protocol"):
            return str(candidate["protocol"]), "deployment_candidate", "candidate protocol hint"
        hint = analysis.get("verify_hint") or {}
        service_type = str(hint.get("service_type") or "")
        frameworks = set(analysis.get("frameworks") or [])
        if service_type == "openai_compatible" or frameworks.intersection({"vllm", "openai_compatible"}):
            return "openai_compatible", "service_metadata", "OpenAI-compatible service metadata"
        if "streamlit" in frameworks:
            return "streamlit", "adapter_default", "Streamlit adapter default"
        if "gradio" in frameworks or service_type == "webui":
            return "gradio", "adapter_default", "Gradio/web UI adapter default"
        if "fastapi" in frameworks:
            return "openapi", "service_metadata", "FastAPI OpenAPI metadata"
        if service_type in {"api", "http"} or "flask" in frameworks or "http.server" in frameworks:
            return "http", "repository_evidence", "local HTTP service evidence"
        return "http", "registry_fallback", "no specialized protocol signal"
