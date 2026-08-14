"""Plan Policy Gate for LLM plan-first deployment.

Validates LLM-generated deployment plans before they can influence execution.
All LLM plan content is treated as untrusted proposals. The policy gate
ensures commands are safe, paths are bounded, verify includes trace evidence,
and grounding requirements are met.

This is a hard gate: if policy rejects, the plan must NOT be compiled or executed.
"""
import re
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.agent.safety import AgentInputSanitizer
from auto_harness.context.repository import safe_repo_path
from auto_harness.command_auth import (
    CommandAuthorizationEngine,
    CommandCandidateSelector,
    CommandRegistry,
)
from auto_harness.command_auth.approval import build_command_approval_request
from auto_harness.command_auth.schemas import sandbox_policy_fingerprint


# Command root allowlist - only these programs may be proposed
COMMAND_ROOT_ALLOWLIST = frozenset({
    "python", "python3", "pip",
    ".venv/bin/python", ".venv/bin/pip",
    ".venv/bin/streamlit", ".venv/bin/uvicorn",
    "streamlit", "uvicorn", "gradio",
    "conda", "mamba", "micromamba",
    "uv", "npm",
})

# High-risk commands that are always rejected
DANGEROUS_COMMANDS = frozenset({
    "rm", "sudo", "chmod", "chown",
    "curl", "wget", "nc", "ssh", "scp",
    "osascript", "open",
    "sh", "bash",
})

# Shell metacharacters that must never appear in commands
SHELL_META_PATTERN = re.compile(r'[;&|>`\$\(\)]')

# Shell wrapper patterns (checked as consecutive args)
SHELL_WRAPPER_PATTERNS = [
    ("bash", "-c"), ("bash", "-lc"), ("sh", "-c"), ("sh", "-lc"),
]

# Path traversal pattern
PATH_TRAVERSAL_PATTERN = re.compile(r'\.\./|\.\.\\\\')

# Forbidden path components
FORBIDDEN_PATH_COMPONENTS = frozenset({".ssh", ".env", "/etc/", "/Users/"})

# Verify success evidence that's too weak (bare HTTP status)
WEAK_EVIDENCE_PATTERNS = re.compile(
    r"^HTTP\s*\d{3}$|^status\s*code\s*\d{3}$|^2\d{2}$",
    re.IGNORECASE,
)


class PlanPolicyGate:
    """Validates LLM deployment plans against safety rules.

    Returns a policy result dict with:
    - allowed: bool - whether the plan (or its non-rejected portions) may proceed
    - status: str - "accepted", "partially_accepted", or "rejected"
    - accepted_sections: list of section names that passed
    - rejected_items: list of {section, item_index, reason} for rejections
    - normalized_plan: dict - the plan with rejected items removed
    - risk_summary: dict - side_effects, network, requires_human_input
    """

    def validate(
        self,
        plan: Dict,
        snapshot: Dict,
        runtime_policy: Optional[Dict] = None,
        config: Any = None,
        approval: Optional[Dict] = None,
        excluded_candidate_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Validate a parsed DeploymentPlan dict.

        Args:
            plan: The DeploymentPlan.to_dict() output
            snapshot: The ProjectSnapshotBuilder output
            runtime_policy: Optional dict with allow flags
            config: Optional HarnessConfig for grounding checks
        """
        runtime_policy = runtime_policy or {}
        rejected_items: List[Dict] = []
        accepted_sections: List[str] = []
        normalized_plan = dict(plan)
        registry_present = "command_registry" in snapshot
        registry = CommandRegistry.from_dict(snapshot.get("command_registry", {}))
        command_decisions = []
        approval_candidates = []
        approval_preview_candidates = []
        required_approval_candidates = []
        excluded_candidate_ids = set(excluded_candidate_ids or ())
        execution_backend = (
            getattr(config, "execution_backend", "local")
            if config is not None else "local"
        )
        if not isinstance(execution_backend, str):
            execution_backend = "local"
        sandbox_policy_fingerprint = self._sandbox_fingerprint(config)
        command_policy_config = (
            getattr(config, "repository_command_policy", {})
            if config is not None else {}
        )
        if not isinstance(command_policy_config, dict):
            command_policy_config = {}

        # If status is not ok, no execution happens
        if plan.get("status") != "ok":
            return {
                "allowed": False,
                "status": "rejected",
                "accepted_sections": [],
                "rejected_items": [{"section": "status", "item_index": -1, "reason": "plan status is %s, not ok" % plan.get("status")}],
                "normalized_plan": {},
                "risk_summary": {
                    "side_effects": [],
                    "network": "none",
                    "requires_human_input": plan.get("status") == "needs_human_input",
                },
            }

        # Validate environment.install_commands
        env = plan.get("environment", {})
        install_commands = env.get("install_commands", [])
        safe_install_commands = []
        for i, cmd in enumerate(install_commands):
            # Reject command as string (must be list)
            if isinstance(cmd, str):
                rejected_items.append({"section": "environment.install_commands", "item_index": i, "reason": "command must be a list of strings, not a shell string"})
                continue
            cmd = self._normalize_declared_console_script(cmd, snapshot)
            registry_candidate = (
                self._registry_candidate_for_command(registry, cmd)
                if registry_present else None
            )
            if registry_candidate is not None:
                cmd = list(registry_candidate.argv)
                decision = CommandAuthorizationEngine().authorize(
                    registry_candidate,
                    registry,
                    repo_dir=Path(snapshot["repo_dir"])
                    if snapshot.get("repo_dir") else None,
                    execution_backend=execution_backend,
                    sandbox_policy_fingerprint=sandbox_policy_fingerprint,
                    approval=approval,
                )
                command_decisions.append(decision)
                if decision.verdict == "approval_required":
                    required_approval_candidates.append(registry_candidate)
                    continue
                result = (
                    {"allowed": True}
                    if decision.verdict == "auto_allowed" else
                    {"allowed": False, "rejection": {
                        "section": "environment.install_commands",
                        "item_index": i,
                        "reason": decision.reason_code,
                        "reason_code": decision.reason_code,
                    }}
                )
            else:
                result = self._validate_command(cmd, "environment.install_commands", i)
                if result["allowed"] and registry_present and self._looks_repository_command(cmd):
                    result = {"allowed": False, "rejection": {
                        "section": "environment.install_commands",
                        "item_index": i,
                        "reason": "repository_command_not_declared",
                        "reason_code": "repository_command_not_declared",
                    }}
            if result["allowed"]:
                safe_install_commands.append(cmd)
            else:
                rejected_items.append(result["rejection"])
        # A model may select the documented application launcher but omit its
        # separate non-interactive initializer. Append only setup commands
        # deterministically extracted from repository evidence and validated
        # by the same command policy.
        documented_setup = (
            snapshot.get("detected_signals", {}).get("documented_setup_commands", [])
            or []
        )
        for item in documented_setup[:1]:
            if not isinstance(item, dict) or not item.get("cmd"):
                continue
            cmd = self._normalize_declared_console_script(item["cmd"], snapshot)
            if cmd in safe_install_commands:
                continue
            registry_candidate = (
                self._registry_candidate_for_command(registry, cmd)
                if registry_present else None
            )
            if registry_candidate is not None:
                cmd = list(registry_candidate.argv)
                decision = CommandAuthorizationEngine().authorize(
                    registry_candidate,
                    registry,
                    repo_dir=Path(snapshot["repo_dir"])
                    if snapshot.get("repo_dir") else None,
                    execution_backend=execution_backend,
                    sandbox_policy_fingerprint=sandbox_policy_fingerprint,
                    approval=approval,
                )
                command_decisions.append(decision)
                if decision.verdict == "approval_required":
                    required_approval_candidates.append(registry_candidate)
                    continue
                result = (
                    {"allowed": True}
                    if decision.verdict == "auto_allowed" else
                    {"allowed": False, "rejection": {
                        "section": "environment.documented_setup_commands",
                        "item_index": len(safe_install_commands),
                        "reason": decision.reason_code,
                    }}
                )
            else:
                result = self._validate_command(
                    cmd, "environment.documented_setup_commands", len(safe_install_commands),
                )
            if result["allowed"]:
                safe_install_commands.append(cmd)
            else:
                rejected_items.append(result["rejection"])
        safe_install_commands = self._prefer_existing_source_artifacts(
            safe_install_commands,
            snapshot,
        )
        if safe_install_commands:
            accepted_sections.append("environment")
        normalized_env = dict(env)
        normalized_env["install_commands"] = safe_install_commands
        normalized_plan["environment"] = normalized_env

        # Validate run.candidates
        run = plan.get("run", {})
        candidates = run.get("candidates", [])
        safe_candidates = []
        for i, cand in enumerate(candidates):
            cand = dict(cand)
            cand["cmd"] = self._normalize_declared_console_script(
                cand.get("cmd", []), snapshot,
            )
            registry_candidate = (
                self._registry_candidate_for_command(registry, cand.get("cmd", []))
                if registry_present else None
            )
            if registry_candidate is not None:
                cand["cmd"] = list(registry_candidate.argv)
            result = (
                self._validate_candidate_port(cand, i)
                if registry_candidate is not None
                else self._validate_candidate(cand, snapshot, i)
            )
            if registry_present:
                if registry_candidate is None:
                    hard = CommandAuthorizationEngine().authorize_argv(cand.get("cmd", []))
                    reason = (
                        hard["reason_code"] if hard.get("verdict") == "hard_denied"
                        else "repository_command_not_declared"
                    )
                    result = {
                        "allowed": False,
                        "rejection": {
                            "section": "run.candidates[%d].cmd" % i,
                            "item_index": i,
                            "reason": reason,
                            "reason_code": reason,
                        },
                    }
                elif result["allowed"] and registry_candidate.candidate_id in excluded_candidate_ids:
                    result = {
                        "allowed": False,
                        "rejection": {
                            "section": "run.candidates[%d].cmd" % i,
                            "item_index": i,
                            "reason": "command_candidate_rejected_by_operator",
                            "reason_code": "command_candidate_rejected_by_operator",
                        },
                    }
                elif result["allowed"]:
                    decision = CommandAuthorizationEngine().authorize(
                        registry_candidate,
                        registry,
                        repo_dir=Path(snapshot["repo_dir"])
                        if snapshot.get("repo_dir") else None,
                        execution_backend=execution_backend,
                        sandbox_policy_fingerprint=sandbox_policy_fingerprint,
                        approval=approval,
                    )
                    command_decisions.append(decision)
                    cand["command_candidate_id"] = registry_candidate.candidate_id
                    cand["command_decision"] = decision.to_dict()
                    cand["required_backend"] = decision.effective_backend
                    if decision.verdict == "approval_required":
                        approval_candidates.append(registry_candidate)
                        approval_preview_candidates.append(cand)
                        result = {"allowed": False, "approval_required": True}
                    elif decision.verdict != "auto_allowed":
                        result = {
                            "allowed": False,
                            "rejection": {
                                "section": "run.candidates[%d].cmd" % i,
                                "item_index": i,
                                "reason": decision.reason_code,
                                "reason_code": decision.reason_code,
                            },
                        }
            if result["allowed"]:
                safe_candidates.append(cand)
            elif not result.get("approval_required"):
                rejected_items.append(result["rejection"])
        if safe_candidates:
            accepted_sections.append("run")
        normalized_run = dict(run)
        normalized_run["candidates"] = safe_candidates
        # Fix selected_candidate_id if the selected one was rejected
        selected_id = run.get("selected_candidate_id", "")
        safe_ids = {c.get("id") for c in safe_candidates}
        if selected_id not in safe_ids and safe_candidates:
            normalized_run["selected_candidate_id"] = safe_candidates[0].get("id", "")
        normalized_plan["run"] = normalized_run
        normalized_plan["command_registry"] = registry.to_dict()
        normalized_plan["command_decisions"] = [
            item.to_dict() for item in command_decisions
        ]
        normalized_plan["sandbox_policy_fingerprint"] = sandbox_policy_fingerprint
        if approval:
            normalized_plan["command_approval"] = approval

        # Validate verify
        verify = plan.get("verify", {})
        verify_result = self._validate_verify(verify)
        if verify_result["allowed"]:
            accepted_sections.append("verify")
            normalized_plan["verify"] = verify
        else:
            rejected_items.append(verify_result["rejection"])

        # Validate grounding (if required by config)
        require_grounding = True
        if config is not None:
            require_grounding = getattr(config, "agent_plan_first_require_grounding", True)
        if require_grounding:
            grounding_result = self._validate_grounding(plan, snapshot)
            if not grounding_result["allowed"]:
                rejected_items.append(grounding_result["rejection"])

        # Check for secrets in the plan
        secret_result = self._check_secrets(plan)
        if secret_result:
            rejected_items.extend(secret_result)

        # Check for external network access
        allow_external = False
        if config is not None:
            allow_external = getattr(config, "agent_plan_first_allow_external_network", False)
        if not allow_external:
            network_result = self._check_network(plan)
            if network_result:
                rejected_items.extend(network_result)

        # Determine overall status
        # Any rejection in environment, run, or verify is critical
        critical_sections = {"environment.install_commands", "run.candidates", "run.candidates[0].cmd", "verify", "grounding", "plan"}
        has_critical_rejection = any(
            (not r.get("section", "").startswith("run.candidates") or not safe_candidates)
            and (r.get("section", "").split("[")[0] in critical_sections
            or r.get("section", "") in critical_sections
            or any(r.get("section", "").startswith(s) for s in critical_sections))
            for r in rejected_items
        )
        if (
            (required_approval_candidates or (not safe_candidates and approval_candidates))
            and "verify" in accepted_sections
        ):
            status = "approval_required"
        elif not rejected_items:
            status = "accepted"
        elif has_critical_rejection or not safe_candidates or "verify" not in accepted_sections:
            status = "rejected"
        else:
            status = "partially_accepted"

        # Build risk summary
        side_effects = []
        if safe_install_commands:
            side_effects.append("filesystem")
        if safe_candidates:
            side_effects.append("process")
        network = "local_only"
        if allow_external:
            network = "external_allowed"

        result = {
            "allowed": status != "rejected",
            "status": status,
            "accepted_sections": accepted_sections,
            "rejected_items": rejected_items,
            "normalized_plan": normalized_plan if status != "rejected" else {},
            "risk_summary": {
                "side_effects": side_effects,
                "network": network,
                "requires_human_input": False,
            },
        }
        if approval_preview_candidates:
            result["approval_preview_candidates"] = approval_preview_candidates
        if status == "approval_required":
            selectable_approvals = list(required_approval_candidates)
            if not safe_candidates:
                selectable_approvals.extend(approval_candidates)
            selected = CommandCandidateSelector().select(
                selectable_approvals,
                command_decisions,
                excluded_candidate_ids,
            )
            selected_candidate = next(
                item for item in selectable_approvals
                if item.candidate_id == selected["candidate_id"]
            )
            evidence_by_id = registry.evidence_by_id()
            result["allowed"] = False
            result["risk_summary"]["requires_human_input"] = True
            result["approval_request"] = build_command_approval_request(
                selected_candidate,
                registry.repository_fingerprint,
                [evidence_by_id[item] for item in selected_candidate.evidence_ids],
                sandbox_policy_fingerprint,
                task_id=snapshot.get("task_id", ""),
                expires_at=(
                    datetime.now(timezone.utc)
                    + timedelta(seconds=int(command_policy_config.get("approval_ttl_seconds", 1800)))
                ).isoformat(),
            )
        return result

    def _validate_command(self, cmd: Any, section: str, index: int) -> Dict:
        """Validate a single command (list of strings)."""
        # Must be a list
        if not isinstance(cmd, list):
            return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "command must be a list of strings"}}

        if len(cmd) == 0:
            return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "command must not be empty"}}

        # All elements must be strings
        for j, arg in enumerate(cmd):
            if not isinstance(arg, str):
                return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "command[%d] must be a string" % j}}

        # Check command root
        root = cmd[0]
        root_basename = root.split("/")[-1] if "/" in root else root

        if root_basename == "uv" and cmd[1:] != ["sync", "--frozen", "--no-dev"]:
            return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "only frozen production uv sync is allowed"}}
        if root_basename == "npm" and not self._safe_npm_build_command(cmd):
            return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "only lockfile-backed repository-local npm builds are allowed"}}

        # Reject dangerous commands
        if root_basename in DANGEROUS_COMMANDS:
            return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "dangerous command: %s" % root_basename}}

        # Check for shell wrapper patterns
        for j in range(len(cmd) - 1):
            pair = (cmd[j], cmd[j + 1])
            if pair in SHELL_WRAPPER_PATTERNS:
                return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "shell wrapper %s %s rejected" % (cmd[j], cmd[j + 1])}}

        # Check for shell metacharacters in any argument
        for j, arg in enumerate(cmd):
            if SHELL_META_PATTERN.search(arg):
                return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "shell metacharacter in command[%d]: %s" % (j, repr(arg[:50]))}}

        # Command root must be in allowlist
        if root not in COMMAND_ROOT_ALLOWLIST and root_basename not in COMMAND_ROOT_ALLOWLIST:
            # Allow if it's a .venv/bin/* path
            if not root.startswith(".venv/bin/"):
                return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "command root '%s' not in allowlist" % root_basename}}

        # Path traversal check in arguments
        for j, arg in enumerate(cmd):
            if PATH_TRAVERSAL_PATTERN.search(arg):
                return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "path traversal in command[%d]" % j}}
            for forbidden in FORBIDDEN_PATH_COMPONENTS:
                if forbidden in arg:
                    return {"allowed": False, "rejection": {"section": section, "item_index": index, "reason": "forbidden path component '%s' in command[%d]" % (forbidden, j)}}

        return {"allowed": True}

    @staticmethod
    def _normalize_declared_console_script(cmd: Any, snapshot: Dict) -> Any:
        """Resolve a repository-declared CLI into the owned virtualenv.

        The model may correctly use the public command documented by the
        project (for example ``qwenpaw app``). Execution must nevertheless be
        pinned to the environment Harness creates. Only exact PEP 621 console
        script names detected from the repository are eligible.
        """
        if not isinstance(cmd, list) or not cmd or not isinstance(cmd[0], str):
            return cmd
        root = cmd[0]
        if "/" in root:
            return cmd
        declared = {
            str(item.get("name"))
            for item in (snapshot.get("detected_signals", {}).get("console_scripts", []) or [])
            if isinstance(item, dict) and item.get("name")
        }
        if root not in declared:
            return cmd
        return [".venv/bin/%s" % root] + list(cmd[1:])

    @staticmethod
    def _safe_npm_build_command(cmd: List[str]) -> bool:
        if len(cmd) not in (4, 5) or cmd[:2] != ["npm", "--prefix"]:
            return False
        prefix = cmd[2]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", prefix):
            return False
        return cmd[3:] in (["ci"], ["run", "build"])

    @staticmethod
    def _prefer_existing_source_artifacts(
        commands: List[List[str]], snapshot: Dict,
    ) -> List[List[str]]:
        """Avoid redundant source builds when the snapshot proved outputs exist."""
        signals = (
            snapshot.get("detected_signals", {})
            if isinstance(snapshot, dict) else {}
        )
        if signals.get("source_frontend_build_required") is not False:
            return commands
        return [
            command
            for command in commands
            if command not in (
                ["npm", "--prefix", "console", "ci"],
                ["npm", "--prefix", "console", "run", "build"],
            )
        ]

    def _validate_candidate(self, cand: Dict, snapshot: Dict, index: int) -> Dict:
        """Validate a run candidate."""
        cmd = cand.get("cmd", [])
        cmd_result = self._validate_command(cmd, "run.candidates[%d].cmd" % index, index)
        if not cmd_result["allowed"]:
            return cmd_result

        return self._validate_candidate_port(cand, index)

    @staticmethod
    def _registry_candidate_for_command(registry: CommandRegistry, cmd: Any):
        """Match a documented command and return its canonical pinned form."""
        if not isinstance(cmd, list):
            return None
        exact = registry.candidate_for_argv(cmd)
        if exact is not None:
            return exact
        if cmd:
            node_installs = [
                item for item in registry.candidates
                if item.source_kind == "node_install"
                and item.argv and item.argv[0] == cmd[0]
                and ("ci" in cmd[1:] or "install" in cmd[1:])
            ]
            if len(node_installs) == 1:
                return node_installs[0]
        for candidate in registry.candidates:
            if candidate.source_kind == "make_target":
                if len(cmd) == 2 and cmd[0] == "make" and candidate.argv[-1] == cmd[1]:
                    return candidate
            elif candidate.source_kind == "package_json_script":
                if len(cmd) >= 2 and cmd[0] == candidate.argv[0]:
                    requested = cmd[2] if len(cmd) >= 3 and cmd[1] == "run" else cmd[1]
                    if candidate.argv[-2:] == ["run", requested]:
                        return candidate
            elif candidate.source_kind == "repository_script" and len(cmd) >= 2:
                if cmd[0] in {"python", "python3", "sh", "bash"}:
                    if candidate.argv[1:] == cmd[1:]:
                        return candidate
                elif cmd[0].startswith("./") and candidate.argv[1:] == [cmd[0][2:]] + cmd[1:]:
                    return candidate
        return None

    @staticmethod
    def _looks_repository_command(cmd: Any) -> bool:
        if not isinstance(cmd, list) or not cmd or not isinstance(cmd[0], str):
            return False
        root = cmd[0]
        basename = Path(root).name
        if root.startswith(".venv/bin/"):
            return basename not in {"python", "python3", "pip"}
        if basename in {"make", "sh", "bash", "zsh", "pnpm", "yarn"}:
            return True
        return basename == "npm" and "run" in cmd[1:]

    @staticmethod
    def _sandbox_fingerprint(config: Any) -> str:
        def value(name, default):
            item = getattr(config, name, default) if config is not None else default
            return item if isinstance(item, type(default)) else default
        return sandbox_policy_fingerprint(
            phase="runtime",
            image=value("docker_image", "python:3.10-slim"),
            network="none",
            gpus=value("docker_gpus", "none"),
            model_cache_dir=value("docker_model_cache_dir", ""),
            security_options={
                "read_only_rootfs": value("docker_read_only_rootfs", False),
                "user": value("docker_user", ""),
                "memory": value("docker_memory", "8g"),
                "cpus": value("docker_cpus", 4.0),
                "pids_limit": value("docker_pids_limit", 512),
                "tmpfs_size": value("docker_tmpfs_size", "1g"),
                "cap_drop_all": value("docker_cap_drop_all", True),
                "no_new_privileges": value("docker_no_new_privileges", True),
                "repo_mount_mode": value("docker_repo_mount_mode", "rw"),
            },
        )

    @staticmethod
    def _validate_candidate_port(cand: Dict, index: int) -> Dict:
        """Validate candidate metadata independently of command provenance."""
        port = cand.get("expected_port", 0)
        if not isinstance(port, (int, float)) or port < 0 or port > 65535:
            return {"allowed": False, "rejection": {"section": "run.candidates", "item_index": index, "reason": "expected_port must be 0-65535"}}

        return {"allowed": True}

    def _validate_verify(self, verify: Dict) -> Dict:
        """Validate the verify section."""
        request = verify.get("request", {})
        if not isinstance(request, dict):
            return {"allowed": False, "rejection": {"section": "verify", "item_index": -1, "reason": "verify.request must be a dict"}}

        # Method must be GET or POST
        method = str(request.get("method", "")).upper()
        if method not in ("GET", "POST"):
            return {"allowed": False, "rejection": {"section": "verify", "item_index": -1, "reason": "verify.request.method must be GET or POST"}}

        # Path must not be an external URL
        path = str(request.get("path", ""))
        if path.startswith("http://") or path.startswith("https://"):
            return {"allowed": False, "rejection": {"section": "verify", "item_index": -1, "reason": "verify.request.path must not be an external URL"}}

        # Must contain {{trace_id}}
        request_str = str(request)
        if "{{trace_id}}" not in request_str:
            return {"allowed": False, "rejection": {"section": "verify", "item_index": -1, "reason": "verify.request must contain {{trace_id}} placeholder"}}

        # Check success_evidence is not too weak
        evidence = str(verify.get("success_evidence", ""))
        if evidence and WEAK_EVIDENCE_PATTERNS.match(evidence.strip()):
            return {"allowed": False, "rejection": {"section": "verify", "item_index": -1, "reason": "verify.success_evidence '%s' is too weak; must reference trace or fresh artifact" % evidence}}

        return {"allowed": True}

    def _validate_grounding(self, plan: Dict, snapshot: Dict) -> Dict:
        """Validate grounding requirements."""
        grounding = plan.get("grounding", [])
        if not isinstance(grounding, list) or len(grounding) == 0:
            # Check if there's a selected candidate that needs grounding
            run = plan.get("run", {})
            selected_id = run.get("selected_candidate_id", "")
            candidates = run.get("candidates", [])
            if selected_id and candidates:
                return {"allowed": False, "rejection": {"section": "grounding", "item_index": -1, "reason": "selected runner candidate requires grounding but none provided"}}

        # Check each grounding entry references files that exist.
        file_tree = set(snapshot.get("file_tree", []))
        layered = snapshot.get("context_mode") == "layered"
        observations = self._grounding_observations(snapshot) if layered else {}
        for i, g in enumerate(grounding):
            if isinstance(g, dict):
                gfile = str(g.get("file", ""))
                if gfile and gfile not in file_tree:
                    return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding file '%s' not found in snapshot file_tree" % gfile}}
                if layered:
                    observation_id = str(g.get("observation_id", ""))
                    observed_candidates = observations.get(observation_id, [])
                    observed = next(
                        (item for item in observed_candidates if item.get("path") == gfile),
                        None,
                    )
                    if not observation_id or not observed:
                        return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding_not_observed"}}
                    if not g.get("sha256") or g.get("sha256") != observed.get("sha256"):
                        return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding_digest_stale"}}
                    start = g.get("line_start")
                    end = g.get("line_end")
                    if not isinstance(start, int) or not isinstance(end, int) or start > end:
                        return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding line range is invalid"}}
                    if start < int(observed.get("line_start", 1)) or end > int(observed.get("line_end", 0)):
                        return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding line range was not observed"}}
                    repo_dir = snapshot.get("repo_dir")
                    if repo_dir:
                        try:
                            current = safe_repo_path(Path(repo_dir), gfile).read_bytes()
                        except (OSError, ValueError):
                            return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding file is unavailable"}}
                        if hashlib.sha256(current).hexdigest() != g.get("sha256"):
                            return {"allowed": False, "rejection": {"section": "grounding", "item_index": i, "reason": "grounding_digest_stale"}}

        return {"allowed": True}

    @staticmethod
    def _grounding_observations(snapshot: Dict) -> Dict[str, List[Dict]]:
        observations: Dict[str, List[Dict]] = {}
        for path, item in (snapshot.get("selected_files") or {}).items():
            if not isinstance(item, dict) or not item.get("observation_id"):
                continue
            observations.setdefault(item["observation_id"], []).append({
                "path": path,
                "line_start": item.get("line_start", 1),
                "line_end": item.get("line_end", 0),
                "sha256": item.get("sha256", ""),
            })
        for record in snapshot.get("observation_ledger", []) or []:
            if not isinstance(record, dict) or record.get("status") != "passed":
                continue
            observation_id = record.get("observation_id")
            evidence = record.get("evidence", {})
            candidates = []
            if isinstance(evidence.get("files"), list):
                candidates.extend(evidence["files"])
            if isinstance(evidence.get("results"), list):
                candidates.extend(evidence["results"])
            for item in candidates:
                if not isinstance(item, dict) or not item.get("path"):
                    continue
                observations.setdefault(observation_id, []).append({
                    "path": item["path"],
                    "line_start": item.get("line_start", item.get("line", 1)),
                    "line_end": item.get("line_end", item.get("line", 1)),
                    "sha256": item.get("sha256", ""),
                })
        return observations

    def _check_secrets(self, plan: Dict) -> List[Dict]:
        """Check for secret patterns in plan string values."""
        sanitizer = AgentInputSanitizer()
        plan_text = str(plan)
        scan = sanitizer.scan_text(plan_text)
        rejections = []
        for redaction in scan.get("redactions", []):
            rejections.append({
                "section": "plan",
                "item_index": -1,
                "reason": "secret pattern detected: %s (count: %d)" % (redaction.get("type", "unknown"), redaction.get("count", 0)),
            })
        return rejections

    def _check_network(self, plan: Dict) -> List[Dict]:
        """Check for external network access patterns."""
        rejections = []
        plan_text = str(plan)
        # Check for URLs that are not localhost
        url_pattern = re.compile(r'https?://([^\s/\'\"]+)')
        for match in url_pattern.finditer(plan_text):
            host = match.group(1)
            if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
                rejections.append({
                    "section": "plan",
                    "item_index": -1,
                    "reason": "external host in plan: %s" % host,
                })
        return rejections
