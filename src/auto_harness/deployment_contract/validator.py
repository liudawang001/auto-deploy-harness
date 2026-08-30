"""Fail-closed validation for repository deployment contracts."""

import json
import re
from pathlib import Path
from typing import Any, Dict

from auto_harness.command_auth import CommandAuthorizationEngine
from auto_harness.command_auth.schemas import CommandCandidate


_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_KEYS = re.compile(r"(?:token|secret|password|api[_-]?key|credential)", re.IGNORECASE)


class DeploymentContractValidationError(ValueError):
    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class DeploymentContractValidator:
    def validate(self, contract, *, raw: Dict = None) -> None:
        self._reject_secret_values(raw or {})
        self._raw_schema(raw or {})
        if isinstance(contract.schema_version, bool) or contract.schema_version != 1:
            self._fail("unsupported_schema_version")
        if contract.workload_type != "service":
            self._fail("unsupported_workload_type")
        if contract.environment.backend not in {"venv", "conda", "mamba", "uv"}:
            self._fail("unsupported_environment_backend")
        for relative in contract.environment.dependency_files:
            self._relative_path(relative, "dependency_path_invalid")
        for index, command in enumerate(contract.environment.install_commands):
            self._command(command, "install_command_%d_invalid" % index)
        self._command(contract.service.command, "service_command_invalid")
        self._relative_path(contract.service.cwd, "service_cwd_invalid", allow_dot=True)
        if isinstance(contract.service.port, bool) or not isinstance(contract.service.port, int) or not 1 <= contract.service.port <= 65535:
            self._fail("service_port_invalid")
        if (
            isinstance(contract.service.startup_timeout_seconds, bool)
            or not isinstance(contract.service.startup_timeout_seconds, int)
            or not 1 <= contract.service.startup_timeout_seconds <= 3600
        ):
            self._fail("startup_timeout_invalid")
        for name in contract.service.required_env_names:
            if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
                self._fail("required_env_name_invalid")
        if contract.verify.protocol not in {"http", "openapi", "openai_compatible", "gradio", "streamlit", "browser_dom"}:
            self._fail("verify_protocol_unsupported")
        if not isinstance(contract.verify.request, dict) or not isinstance(contract.verify.success, dict):
            self._fail("verify_contract_invalid")
        method = str(contract.verify.request.get("method") or "").upper()
        if method not in {"GET", "POST"}:
            self._fail("verify_method_invalid")
        path = str(contract.verify.request.get("path") or "")
        if not path.startswith("/") or path.startswith("//") or path.startswith(("http://", "https://")):
            self._fail("verify_path_invalid")
        verify_payload = json.dumps(
            {"request": contract.verify.request, "success": contract.verify.success},
            ensure_ascii=False, sort_keys=True,
        )
        if "{{trace_id}}" not in verify_payload:
            self._fail("verify_trace_required")
        if (
            isinstance(contract.verify.timeout_seconds, bool)
            or not isinstance(contract.verify.timeout_seconds, int)
            or not 1 <= contract.verify.timeout_seconds <= 120
        ):
            self._fail("verify_timeout_invalid")
        if contract.security.required_backend != "docker":
            self._fail("contract_requires_docker")
        if contract.security.network_profile not in {"none", "registry_only"}:
            self._fail("network_profile_invalid")
        if contract.security.allow_source_edit is not False:
            self._fail("source_edit_not_allowed")

    def _command(self, command, reason_code: str) -> None:
        if not isinstance(command, list) or not command or any(
            not isinstance(item, str) or not item or "\x00" in item for item in command
        ):
            self._fail(reason_code)
        if any(Path(item).is_absolute() for item in command):
            self._fail("manifest_absolute_path_not_allowed")
        candidate = CommandCandidate.build(
            phase="run", argv=list(command), source_kind="manifest_command",
        )
        hard = CommandAuthorizationEngine.hard_deny_reason(candidate)
        if hard:
            self._fail(hard)

    def _relative_path(self, value: Any, reason_code: str, allow_dot: bool = False) -> None:
        if not isinstance(value, str) or not value or "\x00" in value:
            self._fail(reason_code)
        path = Path(value)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            self._fail(reason_code)
        if not allow_dot and value in {".", "./"}:
            self._fail(reason_code)

    def _reject_secret_values(self, value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = "%s.%s" % (path, key) if path else str(key)
                if _SECRET_KEYS.search(str(key)) and item not in (None, "", [], {}):
                    self._fail("secret_value_not_allowed:%s" % key_path)
                self._reject_secret_values(item, key_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._reject_secret_values(item, "%s[%d]" % (path, index))

    def _raw_schema(self, raw: Dict) -> None:
        fields = {
            "project": {"workload_type", "runtime_family"},
            "environment": {
                "backend", "python", "dependency_files", "install_commands",
            },
            "service": {
                "command", "cwd", "host", "port", "startup_timeout_seconds",
                "required_env_names",
            },
            "verify": {"protocol", "request", "success", "timeout_seconds"},
            "security": {
                "required_backend", "network_profile", "allow_source_edit",
            },
        }
        unknown_root = set(raw) - ({"schema_version"} | set(fields))
        if unknown_root:
            self._fail("contract_unknown_field:%s" % sorted(unknown_root)[0])
        for section, allowed in fields.items():
            value = raw.get(section, {})
            if not isinstance(value, dict):
                self._fail("contract_section_invalid:%s" % section)
            unknown = set(value) - allowed
            if unknown:
                self._fail(
                    "contract_unknown_field:%s.%s"
                    % (section, sorted(unknown)[0])
                )
        environment = raw.get("environment") or {}
        service = raw.get("service") or {}
        verify = raw.get("verify") or {}
        security = raw.get("security") or {}
        if not self._list_of_strings(environment.get("dependency_files", [])):
            self._fail("dependency_files_schema_invalid")
        install_commands = environment.get("install_commands", [])
        if not isinstance(install_commands, list) or any(
            not self._list_of_strings(item) for item in install_commands
        ):
            self._fail("install_commands_schema_invalid")
        if not self._list_of_strings(service.get("command", [])):
            self._fail("service_command_invalid")
        if not self._list_of_strings(service.get("required_env_names", [])):
            self._fail("required_env_names_schema_invalid")
        if not isinstance(verify.get("request", {}), dict) or not isinstance(
            verify.get("success", {}), dict,
        ):
            self._fail("verify_contract_invalid")
        if "allow_source_edit" in security and not isinstance(
            security["allow_source_edit"], bool,
        ):
            self._fail("allow_source_edit_schema_invalid")

    @staticmethod
    def _list_of_strings(value) -> bool:
        return isinstance(value, list) and all(
            isinstance(item, str) and bool(item) for item in value
        )

    @staticmethod
    def _fail(reason_code: str):
        raise DeploymentContractValidationError(reason_code)
