"""Single deterministic authorization engine for every command entry point."""

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

from auto_harness.command_auth.approval import approval_valid, command_operation_id
from auto_harness.command_auth.evidence import revalidate_evidence
from auto_harness.command_auth.resolver import ExecutableResolver
from auto_harness.command_auth.schemas import (
    CommandCandidate,
    CommandDecision,
    CommandRegistry,
    canonical_hash,
)


POLICY_VERSION = "1"
SHELL_META = re.compile(r"[;&|>`$()]|\n|\r")
DANGEROUS_ROOTS = frozenset({
    "rm", "sudo", "su", "chmod", "chown", "curl", "wget", "nc", "ssh", "scp",
    "osascript", "open", "eval", "powershell", "cmd",
})
SHELL_ROOTS = frozenset({"sh", "bash", "zsh", "fish", "csh", "tcsh"})
SAFE_SYSTEM_ROOTS = frozenset({
    "python", "python3", "pip", "streamlit", "uvicorn", "gradio", "gunicorn", "flask",
    "conda", "mamba", "micromamba", "uv", "npm", "pnpm", "yarn", "make", "docker", "git",
})


class CommandAuthorizationEngine:
    def __init__(self, policy_version: str = POLICY_VERSION):
        self.policy_version = policy_version
        self.resolver = ExecutableResolver()

    def authorize(
        self,
        candidate: CommandCandidate,
        registry: CommandRegistry,
        *,
        repo_dir: Optional[Path] = None,
        execution_backend: str = "local",
        sandbox_policy_fingerprint: str = "",
        approval: Optional[Dict] = None,
        require_executable: bool = False,
        environment_ownership_marker: Optional[Path] = None,
    ) -> CommandDecision:
        policy_fingerprint = canonical_hash({
            "version": self.policy_version,
            "candidate": candidate.to_dict(),
            "repository_fingerprint": registry.repository_fingerprint,
            "sandbox": sandbox_policy_fingerprint,
        })
        base = {
            "candidate_id": candidate.candidate_id,
            "normalized_argv": list(candidate.argv),
            "effective_backend": candidate.required_backend or execution_backend,
            "operation_id": command_operation_id(candidate, registry.repository_fingerprint),
            "policy_version": self.policy_version,
            "policy_fingerprint": policy_fingerprint,
        }
        # The managed inference runtime source is reserved for the built-in
        # adapter. A repository or LLM candidate claiming it is impersonating
        # the adapter and is always hard-denied.
        if candidate.source_kind == "managed_inference_runtime":
            return CommandDecision(
                verdict="hard_denied",
                reason_code="managed_inference_runtime_reserved_for_adapter",
                reasons=["managed_inference_runtime is only producible by the built-in adapter"],
                **base,
            )
        hard = self.hard_deny_reason(candidate)
        if hard:
            return CommandDecision(verdict="hard_denied", reason_code=hard, reasons=[hard], **base)

        evidence_by_id = registry.evidence_by_id()
        evidence = [evidence_by_id.get(item) for item in candidate.evidence_ids]
        if not evidence or any(item is None for item in evidence):
            return CommandDecision(
                verdict="candidate_rejected", reason_code="repository_evidence_missing",
                reasons=["candidate does not reference complete repository evidence"], **base,
            )
        if any(item.repository_fingerprint != registry.repository_fingerprint for item in evidence):
            return CommandDecision(
                verdict="candidate_rejected", reason_code="repository_fingerprint_changed",
                reasons=["evidence belongs to another repository snapshot"], **base,
            )
        if repo_dir is not None and any(not revalidate_evidence(repo_dir, item) for item in evidence):
            return CommandDecision(
                verdict="candidate_rejected", reason_code="evidence_hash_mismatch",
                reasons=["repository evidence changed after discovery"], **base,
            )
        evidence_types = {item.source_type for item in evidence}
        evidence_reason = self._evidence_reason(candidate, evidence_types)
        if evidence_reason:
            return CommandDecision(
                verdict="candidate_rejected", reason_code=evidence_reason,
                reasons=[evidence_reason], **base,
            )
        if require_executable and repo_dir is not None:
            resolution = self.resolver.resolve(
                repo_dir, candidate, require_exists=True,
                repository_fingerprint=registry.repository_fingerprint,
                ownership_marker_path=environment_ownership_marker,
            )
            if not resolution["resolved"]:
                verdict = "hard_denied" if resolution["reason_code"].endswith("hard_denied") else "candidate_rejected"
                return CommandDecision(
                    verdict=verdict, reason_code=resolution["reason_code"],
                    reasons=[resolution["reason_code"]], **base,
                )

        needs_approval = candidate.source_kind in {
            "make_target", "repository_script", "python_entrypoint", "manifest_command",
            "llm_candidate_request",
        }
        if needs_approval:
            approval_reason = approval_valid(
                approval or {}, candidate, registry.repository_fingerprint,
                sandbox_policy_fingerprint,
            )
            if approval_reason:
                return CommandDecision(
                    verdict="approval_required", reason_code=(
                        "make_target_requires_approval" if candidate.source_kind == "make_target"
                        else "python_entrypoint_requires_approval" if candidate.source_kind == "python_entrypoint"
                        else "manifest_command_requires_approval" if candidate.source_kind == "manifest_command"
                        else "llm_candidate_request_requires_approval" if candidate.source_kind == "llm_candidate_request"
                        else "repository_script_requires_approval"
                    ),
                    reasons=[approval_reason], required_approval=True, **base,
                )
        reason_code = (
            "locked_package_script" if candidate.source_kind == "package_json_script"
            else "locked_source_build" if candidate.source_kind == "source_build"
            else "locked_dependency_install" if candidate.source_kind == "node_install"
            else "declared_cli_bound_to_owned_env" if candidate.source_kind in {"pep621_script", "poetry_script"}
            else "declared_node_run_script" if candidate.source_kind == "node_run_script"
            else "django_manage_entrypoint" if candidate.source_kind == "django_manage"
            else "declared_asgi_wsgi_entrypoint" if candidate.source_kind == "asgi_wsgi_entrypoint"
            else "declared_procfile_web" if candidate.source_kind == "procfile_web"
            else "approved_repository_command"
        )
        return CommandDecision(
            verdict="auto_allowed", reason_code=reason_code,
            reasons=[reason_code], required_approval=needs_approval, **base,
        )

    def authorize_argv(
        self,
        argv,
        *,
        allowed_commands: Iterable[str] = (),
        section: str = "",
        strict_allowlist: bool = False,
    ) -> Dict:
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            return {"allowed": False, "verdict": "hard_denied", "reason_code": "command_schema_invalid", "section": section}
        candidate = CommandCandidate.build(phase="run", argv=argv, source_kind="unresolved")
        hard = self.hard_deny_reason(candidate)
        if hard:
            return {"allowed": False, "verdict": "hard_denied", "reason_code": hard, "section": section}
        root = Path(argv[0]).name
        allowed = set(allowed_commands or ())
        if not strict_allowlist:
            allowed |= SAFE_SYSTEM_ROOTS
        if root in allowed or argv[0] in allowed:
            return {"allowed": True, "verdict": "auto_allowed", "reason_code": "harness_tool_allowlist", "section": section}
        return {"allowed": False, "verdict": "candidate_rejected", "reason_code": "readme_only_unbound_command", "section": section}

    @staticmethod
    def hard_deny_reason(candidate: CommandCandidate) -> str:
        if not isinstance(candidate.argv, list) or not candidate.argv:
            return "command_schema_invalid"
        if any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in candidate.argv):
            return "command_schema_invalid"
        root = Path(candidate.argv[0]).name
        if root in DANGEROUS_ROOTS:
            return "dangerous_command_hard_denied"
        if root in SHELL_ROOTS:
            if candidate.source_kind != "repository_script" or len(candidate.argv) < 2:
                return "shell_wrapper_hard_denied"
            if any(arg in {"-c", "-lc"} for arg in candidate.argv[1:]):
                return "shell_wrapper_hard_denied"
        for arg in candidate.argv:
            if SHELL_META.search(arg):
                return "shell_metacharacter_hard_denied"
            if "../" in arg or "..\\" in arg:
                return "path_escape_hard_denied"
        if any(token in " ".join(candidate.argv).lower() for token in ("--privileged", "docker.sock", "--network=host")):
            return "host_escape_hard_denied"
        return ""

    @staticmethod
    def _evidence_reason(candidate, evidence_types):
        if candidate.source_kind not in {
            "manifest_command", "node_install", "python_entrypoint", "source_build",
            "pep621_script", "poetry_script", "package_json_script", "node_run_script",
            "django_manage", "asgi_wsgi_entrypoint", "procfile_web",
            "llm_candidate_request",
        } and "readme_reference" not in evidence_types:
            return "readme_reference_missing"
        if candidate.source_kind == "manifest_command" and "manifest_command" not in evidence_types:
            return "manifest_command_evidence_missing"
        if candidate.source_kind == "node_install" and not {"package_manifest", "lockfile"}.issubset(evidence_types):
            return "node_manifest_or_lockfile_missing"
        if candidate.source_kind == "pep621_script" and "pep621_script" not in evidence_types:
            return "project_cli_declaration_missing"
        if candidate.source_kind == "poetry_script" and "poetry_script" not in evidence_types:
            return "project_cli_declaration_missing"
        if candidate.source_kind == "package_json_script" and not {"package_json_script", "lockfile"}.issubset(evidence_types):
            return "node_script_or_lockfile_missing"
        if candidate.source_kind == "node_run_script" and not {"package_json_script", "lockfile"}.issubset(evidence_types):
            return "node_script_or_lockfile_missing"
        if candidate.source_kind == "django_manage" and not {"django_manage", "python_dependency"}.issubset(evidence_types):
            return "django_entrypoint_evidence_missing"
        if candidate.source_kind == "asgi_wsgi_entrypoint" and not {"asgi_wsgi_module", "python_dependency"}.issubset(evidence_types):
            return "asgi_wsgi_evidence_missing"
        if candidate.source_kind == "procfile_web" and not {"procfile_web", "repository_file"}.issubset(evidence_types):
            return "procfile_corroboration_missing"
        if candidate.source_kind == "source_build" and not {
            "package_json_script", "lockfile", "make_reference",
        }.issubset(evidence_types):
            return "source_build_evidence_missing"
        if candidate.source_kind == "make_target" and "make_target" not in evidence_types:
            return "make_target_missing"
        if candidate.source_kind == "repository_script" and "repository_script" not in evidence_types:
            return "repository_script_missing"
        if candidate.source_kind == "python_entrypoint" and "repository_script" not in evidence_types:
            return "repository_script_missing"
        if candidate.source_kind == "llm_candidate_request" and not (
            evidence_types & {"readme_reference", "python_dependency", "package_json_script",
                              "pep621_script", "poetry_script", "repository_file", "django_manage",
                              "asgi_wsgi_module", "procfile_web"}
        ):
            return "llm_candidate_grounded_evidence_missing"
        return ""
