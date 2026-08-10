"""Plan Policy Gate for LLM plan-first deployment.

Validates LLM-generated deployment plans before they can influence execution.
All LLM plan content is treated as untrusted proposals. The policy gate
ensures commands are safe, paths are bounded, verify includes trace evidence,
and grounding requirements are met.

This is a hard gate: if policy rejects, the plan must NOT be compiled or executed.
"""
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.agent.safety import AgentInputSanitizer
from auto_harness.context.repository import safe_repo_path


# Command root allowlist - only these programs may be proposed
COMMAND_ROOT_ALLOWLIST = frozenset({
    "python", "python3", "pip",
    ".venv/bin/python", ".venv/bin/pip",
    ".venv/bin/streamlit", ".venv/bin/uvicorn",
    "streamlit", "uvicorn", "gradio",
    "conda", "mamba", "micromamba",
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
            result = self._validate_command(cmd, "environment.install_commands", i)
            if result["allowed"]:
                safe_install_commands.append(cmd)
            else:
                rejected_items.append(result["rejection"])
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
            result = self._validate_candidate(cand, snapshot, i)
            if result["allowed"]:
                safe_candidates.append(cand)
            else:
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
            r.get("section", "").split("[")[0] in critical_sections
            or r.get("section", "") in critical_sections
            or any(r.get("section", "").startswith(s) for s in critical_sections)
            for r in rejected_items
        )
        if not rejected_items:
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

        return {
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

    def _validate_candidate(self, cand: Dict, snapshot: Dict, index: int) -> Dict:
        """Validate a run candidate."""
        cmd = cand.get("cmd", [])
        cmd_result = self._validate_command(cmd, "run.candidates[%d].cmd" % index, index)
        if not cmd_result["allowed"]:
            return cmd_result

        # Validate expected_port is reasonable
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
