"""Deployment plan schema and parser for LLM plan-first deployment.

Defines the DeploymentPlan dataclass and the parser that validates
LLM-generated JSON against the required schema.
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auto_harness.providers.json_utils import parse_json_object


# Allowed status values
VALID_STATUSES = frozenset({"ok", "needs_human_input", "no_safe_plan", "invalid"})

# Candidate request phases mirror the command registry phases.
CANDIDATE_REQUEST_PHASES = frozenset({"install", "setup", "run"})


@dataclass
class DeploymentPlan:
    """Structured deployment plan produced by LLM and validated by framework."""
    status: str
    plan_id: str = ""
    summary: str = ""
    grounding: List[Dict] = field(default_factory=list)
    environment: Dict = field(default_factory=dict)
    model_assets: Dict = field(default_factory=dict)
    run: Dict = field(default_factory=dict)
    verify: Dict = field(default_factory=dict)
    risks: List[Dict] = field(default_factory=list)
    fallbacks: List[Dict] = field(default_factory=list)
    selection: Dict = field(default_factory=dict)
    candidate_requests: List[Dict] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "plan_id": self.plan_id,
            "summary": self.summary,
            "grounding": self.grounding,
            "environment": self.environment,
            "model_assets": self.model_assets,
            "run": self.run,
            "verify": self.verify,
            "risks": self.risks,
            "fallbacks": self.fallbacks,
            "selection": self.selection,
            "candidate_requests": self.candidate_requests,
        }


class DeploymentPlanParser:
    """Parses and validates LLM JSON output as a DeploymentPlan.

    All LLM output is treated as untrusted. The parser enforces:
    - Valid JSON structure
    - Required fields when status=ok
    - Commands as list[list[str]], not shell strings
    - Verify request must contain {{trace_id}}
    - selected_candidate_id must match a candidate
    """

    def parse(self, raw_text: str) -> DeploymentPlan:
        """Parse raw LLM text into a validated DeploymentPlan.

        Raises ValueError if the JSON is invalid or required fields are missing.
        """
        # Parse JSON
        try:
            data = parse_json_object(raw_text)
        except Exception as exc:
            raise ValueError("Invalid JSON in deployment plan: %s" % str(exc)) from exc

        if not isinstance(data, dict):
            raise ValueError("Deployment plan must be a JSON object, got %s" % type(data).__name__)

        # Validate status
        status = str(data.get("status", ""))
        if status not in VALID_STATUSES:
            raise ValueError("Invalid status '%s'. Must be one of: %s" % (status, ", ".join(sorted(VALID_STATUSES))))

        # For non-ok statuses, return with minimal validation
        if status != "ok":
            return DeploymentPlan(
                status=status,
                plan_id=str(data.get("plan_id", "")),
                summary=str(data.get("summary", "")),
                raw_response=raw_text,
            )

        # status=ok: validate required fields
        plan_id = str(data.get("plan_id", ""))
        if not plan_id:
            plan_id = "plan_%s" % uuid.uuid4().hex[:8]

        summary = str(data.get("summary", ""))
        if not summary:
            raise ValueError("Deployment plan with status=ok must have 'summary'")

        # Validate grounding
        grounding = data.get("grounding", [])
        if not isinstance(grounding, list) or len(grounding) == 0:
            raise ValueError("Deployment plan with status=ok must have non-empty 'grounding' list")
        for i, g in enumerate(grounding):
            if not isinstance(g, dict):
                raise ValueError("grounding[%d] must be a dict" % i)
            if not g.get("file"):
                raise ValueError("grounding[%d] must have 'file'" % i)
            if not g.get("claim"):
                raise ValueError("grounding[%d] must have 'claim'" % i)
            if not g.get("reason"):
                raise ValueError("grounding[%d] must have 'reason'" % i)
            for field_name in ("line_start", "line_end"):
                if field_name in g and not isinstance(g[field_name], int):
                    raise ValueError("grounding[%d].%s must be an integer" % (i, field_name))
            if "evidence_id" in g and not str(g.get("evidence_id") or "").strip():
                raise ValueError("grounding[%d].evidence_id must be a non-empty string" % i)

        # Grounded selection: reference existing registry candidates by id.
        # This is the Phase B2 alternative to inventing run candidates.
        selection = data.get("selection", {})
        if not isinstance(selection, dict):
            raise ValueError("'selection' must be a dict")
        for key in (
            "selected_environment_candidate_id",
            "selected_run_candidate_id",
            "selected_verify_candidate_id",
        ):
            if key in selection and not str(selection.get(key) or "").strip():
                raise ValueError("selection.%s must be a non-empty string when present" % key)
        selected_run_candidate = str(selection.get("selected_run_candidate_id") or "")

        # Validate environment
        environment = data.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError("'environment' must be a dict")
        install_commands = environment.get("install_commands", [])
        if not isinstance(install_commands, list):
            raise ValueError("environment.install_commands must be a list")
        selected_environment_candidate = str(
            selection.get("selected_environment_candidate_id") or ""
        )
        if not install_commands and not selected_environment_candidate:
            raise ValueError(
                "environment.install_commands must be a non-empty list unless "
                "selection.selected_environment_candidate_id is provided"
            )
        for i, cmd in enumerate(install_commands):
            self._validate_command(cmd, "environment.install_commands[%d]" % i)

        # Validate model_assets (optional, defaults to no models needed)
        model_assets = data.get("model_assets", {})
        if not isinstance(model_assets, dict):
            raise ValueError("'model_assets' must be a dict")

        # Validate run
        run = data.get("run", {})
        if not isinstance(run, dict):
            raise ValueError("'run' must be a dict")
        candidates = run.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("run.candidates must be a list")
        candidate_ids = set()
        for i, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                raise ValueError("run.candidates[%d] must be a dict" % i)
            cand_id = str(cand.get("id", ""))
            if not cand_id:
                raise ValueError("run.candidates[%d] must have 'id'" % i)
            candidate_ids.add(cand_id)
            cmd = cand.get("cmd")
            self._validate_command(cmd, "run.candidates[%d].cmd" % i)
            if not isinstance(cand.get("expected_port"), (int, float)):
                raise ValueError("run.candidates[%d] must have numeric 'expected_port'" % i)
        if not candidates and not selected_run_candidate:
            raise ValueError(
                "run.candidates must be a non-empty list unless "
                "selection.selected_run_candidate_id is provided"
            )

        # Validate selected_candidate_id
        selected_id = str(run.get("selected_candidate_id", ""))
        if not selected_id and not selected_run_candidate:
            raise ValueError("run.selected_candidate_id must be specified")
        if selected_id and candidates and selected_id not in candidate_ids:
            raise ValueError(
                "run.selected_candidate_id '%s' does not match any candidate id: %s"
                % (selected_id, sorted(candidate_ids))
            )

        # Validate verify
        verify = data.get("verify", {})
        if not isinstance(verify, dict):
            raise ValueError("'verify' must be a dict")
        request = verify.get("request", {})
        if not isinstance(request, dict):
            raise ValueError("verify.request must be a dict")
        method = str(request.get("method", "")).upper()
        if method not in ("GET", "POST"):
            raise ValueError("verify.request.method must be GET or POST")
        # Check {{trace_id}} is present somewhere in verify request
        request_str = str(request)
        if "{{trace_id}}" not in request_str:
            raise ValueError("verify.request must contain {{trace_id}} placeholder")
        # Check path is not an external URL
        path = str(request.get("path", ""))
        if path.startswith("http://") or path.startswith("https://"):
            raise ValueError("verify.request.path must not be an external URL")

        # Validate risks and fallbacks (optional)
        risks = data.get("risks", [])
        if not isinstance(risks, list):
            raise ValueError("'risks' must be a list")
        fallbacks = data.get("fallbacks", [])
        if not isinstance(fallbacks, list):
            raise ValueError("'fallbacks' must be a list")

        # Validate candidate requests (Phase B2): the LLM may only *request*
        # new candidates grounded in registry evidence; the framework still
        # normalizes, revalidates, and authorizes them.
        candidate_requests = data.get("candidate_requests", [])
        if not isinstance(candidate_requests, list):
            raise ValueError("'candidate_requests' must be a list")
        for i, request in enumerate(candidate_requests):
            if not isinstance(request, dict):
                raise ValueError("candidate_requests[%d] must be a dict" % i)
            if str(request.get("type", "")) != "candidate_request":
                raise ValueError("candidate_requests[%d].type must be 'candidate_request'" % i)
            if str(request.get("phase", "")) not in CANDIDATE_REQUEST_PHASES:
                raise ValueError("candidate_requests[%d].phase must be one of %s" % (
                    i, sorted(CANDIDATE_REQUEST_PHASES),
                ))
            self._validate_command(request.get("argv"), "candidate_requests[%d].argv" % i)
            grounding_ids = request.get("grounding_evidence_ids", [])
            if not isinstance(grounding_ids, list) or not grounding_ids or not all(
                isinstance(item, str) and item.strip() for item in grounding_ids
            ):
                raise ValueError(
                    "candidate_requests[%d].grounding_evidence_ids must be a non-empty list of strings" % i
                )
            if "expected_port" in request and not isinstance(request.get("expected_port"), (int, float)):
                raise ValueError("candidate_requests[%d].expected_port must be numeric" % i)
            if "cwd" in request and not isinstance(request.get("cwd"), str):
                raise ValueError("candidate_requests[%d].cwd must be a string" % i)

        return DeploymentPlan(
            status=status,
            plan_id=plan_id,
            summary=summary,
            grounding=grounding,
            environment=environment,
            model_assets=model_assets,
            run=run,
            verify=verify,
            risks=risks,
            fallbacks=fallbacks,
            selection=selection,
            candidate_requests=candidate_requests,
            raw_response=raw_text,
        )

    def _validate_command(self, cmd: Any, path: str) -> None:
        """Validate that a command is a list of strings, not a shell string."""
        if isinstance(cmd, str):
            raise ValueError("%s must be a list of strings, not a shell string: %s" % (path, cmd[:100]))
        if not isinstance(cmd, list):
            raise ValueError("%s must be a list, got %s" % (path, type(cmd).__name__))
        if len(cmd) == 0:
            raise ValueError("%s must not be empty" % path)
        for j, arg in enumerate(cmd):
            if not isinstance(arg, str):
                raise ValueError("%s[%d] must be a string, got %s" % (path, j, type(arg).__name__))
