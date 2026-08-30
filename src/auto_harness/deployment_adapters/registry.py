"""Deterministic registry for built-in deployment adapters."""

from auto_harness.deployment_adapters.builtin import BUILTIN_ADAPTERS
from auto_harness.deployment_adapters.composer import CandidateComposer


class DeploymentAdapterRegistry:
    def __init__(self, adapters=()):
        self._adapters = []
        for adapter in adapters:
            self.register(adapter)

    @classmethod
    def builtins(cls):
        return cls(adapter_type() for adapter_type in BUILTIN_ADAPTERS)

    def register(self, adapter):
        if any(item.adapter_id == adapter.adapter_id for item in self._adapters):
            raise ValueError("duplicate_adapter_id:%s" % adapter.adapter_id)
        self._adapters.append(adapter)
        self._adapters.sort(key=lambda item: (-item.priority, item.adapter_id))

    def all(self):
        return list(self._adapters)

    def detect_all(self, context):
        return [adapter.detect(context) for adapter in self._adapters]

    def proposals(self, context, detections=None):
        detections = detections or self.detect_all(context)
        detection_by_id = {item.adapter_id: item for item in detections}
        environments = []
        runs = []
        verifies = []
        for adapter in self._adapters:
            detection = detection_by_id[adapter.adapter_id]
            environments.extend(adapter.propose_environment(context, detection))
            runs.extend(adapter.propose_run_candidates(context, detection))
            verifies.extend(adapter.propose_verify_candidates(context, detection))
        return {
            "environment": environments,
            "run": CandidateComposer().merge_run_proposals(runs),
            "verify": verifies,
        }

    def legacy_frameworks(self, context, detections=None):
        detections = detections or self.detect_all(context)
        labels = set(context.legacy_frameworks)
        for field in (
            "service_frameworks", "ui_frameworks", "ml_libraries",
            "inference_runtimes",
        ):
            labels.update(getattr(context.capabilities, field, []))
        if "openai_compatible" in context.capabilities.protocols:
            labels.add("openai_compatible")
        if "node" in context.capabilities.languages:
            labels.add("node")
        if not labels:
            labels.add("unknown")
        return sorted(labels)

    def legacy_run_candidates(self, context, detections=None):
        proposals = self.proposals(context, detections)["run"]
        return [{
            "cmd": list(item.argv),
            "expected_port": item.expected_port,
            "confidence": item.confidence,
        } for item in proposals]

    def legacy_verify_hint(self, context, detections=None):
        proposals = self.proposals(context, detections)["verify"]
        if proposals:
            return dict(proposals[0].verify_hint)
        return {"service_type": "unknown", "expected_output": "unknown"}
