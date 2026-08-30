"""Built-in deployment adapters for the legacy framework behaviours."""

import hashlib
from pathlib import Path
from typing import List

from auto_harness.deployment_adapters.schemas import (
    AdapterDetection,
    EnvironmentProposal,
    RunProposal,
    VerifyProposal,
)


def _evidence_ids(context, category, value):
    matched = [
        item for item in getattr(context.capabilities, "evidence", [])
        if item.capability_type == category and item.capability_value == value
    ]
    return (
        sorted({item.evidence_id for item in matched}),
        [item.to_dict() for item in matched],
    )


def _legacy_evidence(context, signal):
    root = Path(context.repo_dir).resolve()
    normalized_signal = str(signal).lower()
    for relative in context.files:
        path = root / relative
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
            content = resolved.read_text(encoding="utf-8", errors="ignore")
            payload = resolved.read_bytes()
        except (OSError, ValueError):
            continue
        if normalized_signal not in content.lower():
            continue
        sha256 = hashlib.sha256(payload).hexdigest()
        evidence_id = "adapterev_%s" % hashlib.sha256(
            ("%s\0%s\0%s" % (relative, sha256, signal)).encode("utf-8")
        ).hexdigest()[:20]
        evidence = {
            "evidence_id": evidence_id,
            "source_type": "legacy_file_signal",
            "path": str(relative),
            "sha256": sha256,
            "signal": str(signal),
        }
        return [evidence_id], [evidence]
    return [], []


class BuiltinAdapter:
    adapter_id = ""
    priority = 0
    capability_category = ""
    capability_value = ""
    legacy_label = ""

    def detect(self, context):
        values = getattr(context.capabilities, self.capability_category, [])
        matched = (
            self.capability_value in values
            or self.legacy_label in context.legacy_frameworks
        )
        evidence_ids, evidence = _evidence_ids(
            context, self.capability_category, self.capability_value,
        )
        if matched and not evidence_ids:
            evidence_ids, evidence = _legacy_evidence(
                context, self.legacy_label,
            )
        reasons = []
        if self.capability_value in values:
            reasons.append("structured capability %s:%s" % (
                self.capability_category, self.capability_value,
            ))
        elif self.legacy_label in context.legacy_frameworks:
            reasons.append("legacy compatibility signal %s" % self.legacy_label)
        return AdapterDetection(
            adapter_id=self.adapter_id,
            matched=matched,
            confidence=0.9 if evidence_ids else (0.6 if matched else 0.0),
            evidence_ids=evidence_ids,
            evidence=evidence,
            reasons=reasons,
        )

    def propose_environment(self, context, detection):
        return []

    def propose_run_candidates(self, context, detection):
        return []

    def propose_verify_candidates(self, context, detection):
        return []


class GenericPythonAdapter(BuiltinAdapter):
    adapter_id = "builtin.generic_python"
    priority = 100
    capability_category = "languages"
    capability_value = "python"
    legacy_label = "python"

    def propose_environment(self, context, detection):
        if not detection.matched:
            return []
        backend = "conda" if any(
            name in context.files for name in ("environment.yml", "environment.yaml")
        ) else "venv"
        return [EnvironmentProposal(
            adapter_id=self.adapter_id,
            backend=backend,
            confidence=0.8,
            evidence_ids=list(detection.evidence_ids),
            reasons=["Python project environment"],
        )]

    def propose_run_candidates(self, context, detection):
        if not detection.matched:
            return []
        result = []
        for entry in ("app.py", "main.py", "server.py", "webui.py", "demo.py"):
            if entry not in context.files:
                continue
            port = _python_entry_port(context.repo_dir / entry) or 7860
            confidence = 0.75 if port != 7860 else 0.7
            result.append(RunProposal(
                adapter_id=self.adapter_id,
                argv=[".venv/bin/python", entry],
                expected_port=port,
                confidence=confidence,
                evidence_ids=list(detection.evidence_ids),
                reasons=["Python entrypoint %s" % entry],
            ))
        return result


class GradioAdapter(BuiltinAdapter):
    adapter_id = "builtin.gradio"
    priority = 95
    capability_category = "ui_frameworks"
    capability_value = "gradio"
    legacy_label = "gradio"

    def propose_verify_candidates(self, context, detection):
        if not detection.matched:
            return []
        return [VerifyProposal(
            adapter_id=self.adapter_id,
            protocol="gradio",
            confidence=detection.confidence,
            evidence_ids=list(detection.evidence_ids),
            reasons=["Gradio prediction endpoint"],
            verify_hint={
                "service_type": "webui",
                "expected_output": "web_result",
                "request": {
                    "method": "POST",
                    "path": "/api/predict",
                    "json": {"data": ["{{trace_id}}"]},
                },
            },
        )]


class StreamlitAdapter(BuiltinAdapter):
    adapter_id = "builtin.streamlit"
    priority = 90
    capability_category = "ui_frameworks"
    capability_value = "streamlit"
    legacy_label = "streamlit"

    def propose_run_candidates(self, context, detection):
        if not detection.matched:
            return []
        return [
            RunProposal(
                adapter_id=self.adapter_id,
                argv=[".venv/bin/streamlit", "run", entry],
                expected_port=8501,
                confidence=0.8,
                evidence_ids=list(detection.evidence_ids),
                reasons=["Streamlit entrypoint %s" % entry],
            )
            for entry in ("app.py", "main.py", "demo.py")
            if entry in context.files
        ]


class FastAPIAdapter(BuiltinAdapter):
    adapter_id = "builtin.fastapi"
    priority = 85
    capability_category = "service_frameworks"
    capability_value = "fastapi"
    legacy_label = "fastapi"

    def propose_verify_candidates(self, context, detection):
        return _generic_api_verify(self, detection, "openapi")


class FlaskAdapter(BuiltinAdapter):
    adapter_id = "builtin.flask"
    priority = 84
    capability_category = "service_frameworks"
    capability_value = "flask"
    legacy_label = "flask"

    def propose_verify_candidates(self, context, detection):
        return _generic_api_verify(self, detection, "http")


class StdlibHttpAdapter(BuiltinAdapter):
    adapter_id = "builtin.stdlib_http"
    priority = 80
    capability_category = "service_frameworks"
    capability_value = "http.server"
    legacy_label = "http.server"

    def propose_verify_candidates(self, context, detection):
        if not detection.matched:
            return []
        return [VerifyProposal(
            adapter_id=self.adapter_id,
            protocol="http",
            confidence=detection.confidence,
            evidence_ids=list(detection.evidence_ids),
            reasons=["stdlib HTTP trace echo"],
            verify_hint={
                "service_type": "http",
                "expected_output": "trace_echo",
                "request": {
                    "method": "GET",
                    "path": "/?_auto_harness_trace={{trace_id}}",
                },
            },
        )]


class VllmAdapter(BuiltinAdapter):
    adapter_id = "builtin.vllm"
    priority = 75
    capability_category = "inference_runtimes"
    capability_value = "vllm"
    legacy_label = "vllm"

    def propose_run_candidates(self, context, detection):
        if not detection.matched:
            return []
        return [RunProposal(
            adapter_id=self.adapter_id,
            argv=[
                ".venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
                "--host", "127.0.0.1", "--port", "8000",
            ],
            expected_port=8000,
            confidence=0.5,
            evidence_ids=list(detection.evidence_ids),
            reasons=["vLLM OpenAI-compatible runtime"],
        )]

    def propose_verify_candidates(self, context, detection):
        return _openai_verify(self, detection)


class OpenAICompatibleAdapter(BuiltinAdapter):
    adapter_id = "builtin.openai_compatible"
    priority = 74
    capability_category = "protocols"
    capability_value = "openai_compatible"
    legacy_label = "openai_compatible"

    def propose_verify_candidates(self, context, detection):
        return _openai_verify(self, detection)


class DjangoAdapter(BuiltinAdapter):
    adapter_id = "builtin.django"
    priority = 82
    capability_category = "service_frameworks"
    capability_value = "django"
    legacy_label = "django"

    def propose_run_candidates(self, context, detection):
        if not detection.matched:
            return []
        from auto_harness.command_auth.adapters.common import readme_commands
        from auto_harness.command_auth.adapters.entrypoint import discover_python_services

        found_evidence, found_candidates, _ = discover_python_services(
            Path(context.repo_dir),
            list(context.files),
            readme_commands(Path(context.repo_dir), list(context.files)),
            "",
        )
        return [
            RunProposal(
                adapter_id=self.adapter_id,
                argv=list(item.argv),
                expected_port=int(getattr(item, "expected_port", 0) or 0),
                confidence=0.85,
                evidence_ids=list(detection.evidence_ids),
                reasons=["Django service entrypoint %s" % (item.argv[-1] if item.argv else item.source_kind)],
            )
            for item in found_candidates
            if item.source_kind in {"django_manage", "asgi_wsgi_entrypoint", "procfile_web"}
        ]

    def propose_verify_candidates(self, context, detection):
        if not detection.matched:
            return []
        return [VerifyProposal(
            adapter_id=self.adapter_id,
            protocol="http",
            confidence=detection.confidence,
            evidence_ids=list(detection.evidence_ids),
            reasons=["Django HTTP service"],
            verify_hint={
                "service_type": "http",
                "expected_output": "trace_echo",
                "request": {
                    "method": "GET",
                    "path": "/?_auto_harness_trace={{trace_id}}",
                },
            },
        )]


class NodePackageAdapter(BuiltinAdapter):
    adapter_id = "builtin.node_package"
    priority = 70
    capability_category = "languages"
    capability_value = "node"
    legacy_label = "node"

    def propose_environment(self, context, detection):
        if not detection.matched:
            return []
        return [EnvironmentProposal(
            adapter_id=self.adapter_id,
            backend="node",
            confidence=detection.confidence,
            evidence_ids=list(detection.evidence_ids),
            reasons=["package.json Node environment"],
        )]

    def propose_run_candidates(self, context, detection):
        if not detection.matched:
            return []
        from auto_harness.command_auth.adapters.entrypoint import (
            declared_node_run_scripts,
            source_listen_port,
        )

        try:
            import json

            scripts = {}
            for item in declared_node_run_scripts(Path(context.repo_dir), list(context.files)):
                try:
                    package = json.loads(
                        (Path(context.repo_dir) / item["package_path"]).read_text(
                            encoding="utf-8", errors="ignore",
                        )
                    )
                except (OSError, TypeError, ValueError):
                    package = {}
                raw_script = ((package.get("scripts") or {}).get(item["script"]) or "")
                port = source_listen_port(Path(context.repo_dir), list(context.files), raw_script)
                scripts.setdefault(
                    tuple(item["argv"]),
                    RunProposal(
                        adapter_id=self.adapter_id,
                        argv=list(item["argv"]),
                        expected_port=port,
                        confidence=0.7,
                        evidence_ids=list(detection.evidence_ids),
                        reasons=["declared package script %s with matching lockfile" % item["script"]],
                    ),
                )
            return list(scripts.values())
        except OSError:
            return []


from auto_harness.deployment_adapters.native import (
    CargoAdapter,
    GoModuleAdapter,
    GradleWrapperAdapter,
    MavenWrapperAdapter,
)


def _generic_api_verify(adapter, detection, protocol):
    if not detection.matched:
        return []
    return [VerifyProposal(
        adapter_id=adapter.adapter_id,
        protocol=protocol,
        confidence=detection.confidence,
        evidence_ids=list(detection.evidence_ids),
        reasons=["%s local API" % adapter.capability_value],
        verify_hint={
            "service_type": "api",
            "expected_output": "json_or_text",
            "request": {"method": "GET"},
        },
    )]


def _openai_verify(adapter, detection):
    if not detection.matched:
        return []
    return [VerifyProposal(
        adapter_id=adapter.adapter_id,
        protocol="openai_compatible",
        confidence=detection.confidence,
        evidence_ids=list(detection.evidence_ids),
        reasons=["OpenAI-compatible chat endpoint"],
        verify_hint={
            "service_type": "openai_compatible",
            "expected_output": "chat_completion",
            "request": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "json": {
                    "model": "{{model}}",
                    "messages": [{
                        "role": "user",
                        "content": "auto harness trace {{trace_id}}",
                    }],
                    "temperature": 0,
                    "max_tokens": 16,
                },
            },
        },
    )]


def _python_entry_port(path: Path) -> int:
    import re

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    patterns = (
        r"HTTPServer\(\s*\(\s*['\"][^'\"]*['\"]\s*,\s*(\d+)\s*\)",
        r"uvicorn\.run\([^)]*port\s*=\s*(\d+)",
        r"\.run\([^)]*port\s*=\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return int(match.group(1))
    return 0


BUILTIN_ADAPTERS = (
    GenericPythonAdapter,
    GradioAdapter,
    StreamlitAdapter,
    FastAPIAdapter,
    FlaskAdapter,
    DjangoAdapter,
    StdlibHttpAdapter,
    VllmAdapter,
    OpenAICompatibleAdapter,
    NodePackageAdapter,
    MavenWrapperAdapter,
    GradleWrapperAdapter,
    GoModuleAdapter,
    CargoAdapter,
)
