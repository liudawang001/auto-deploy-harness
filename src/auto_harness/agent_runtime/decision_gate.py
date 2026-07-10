"""Unified AgentDecisionGate: observe -> plan -> schema -> critic -> policy -> execute -> artifact.

This is the central architecture for LLM Decision Gates. Each stage gate
(runner, env_solve, model_prepare, repair) uses this same pipeline with
stage-specific planners, schemas, and policy rules.

Key invariants:
- planner mode: LLM generates decisions but NO tools execute
- gated_actor mode: approved tool_calls execute or apply state deltas
- llm_helped is true ONLY when state actually improves
- Every gate writes artifacts to runs/<run_id>/agent_decision_gates/
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.agent_runtime.stage_schemas import (
    GateDecision,
    GateResult,
    RUNNER_TOOLS,
    ENV_TOOLS,
    MODEL_TOOLS,
    REPAIR_TOOLS,
    PIPELINE_STAGES,
    SAFE_RERUN_STAGES,
)
from auto_harness.agent_runtime.stage_planners import (
    RunnerPlanner,
    EnvPlanner,
    ModelPlanner,
    RepairActuatorPlanner,
    PlanPlanner,
    VerifyPlanner,
    parse_gate_decision,
)
from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def format_dependency_constraint(package: str, version_spec: str) -> str:
    """Format a dependency constraint in canonical form: package + version_spec.

    Examples:
        format_dependency_constraint("pydantic", "<2") -> "pydantic<2"
        format_dependency_constraint("numpy", "<2") -> "numpy<2"
        format_dependency_constraint("torch", "==2.3.0") -> "torch==2.3.0"
        format_dependency_constraint("pydantic", "") -> "pydantic"
    """
    return "%s%s" % (package, version_spec) if version_spec else package


# ------------------------------------------------------------------
# Stage-specific policy validators
# ------------------------------------------------------------------

# Allowed command roots for runner candidate commands
_ALLOWED_COMMAND_ROOTS = frozenset({"python", "python3", "streamlit", "gradio", "uvicorn", "gunicorn", "flask"})

# Allowed model sources
_ALLOWED_MODEL_SOURCES = frozenset({"huggingface", "modelscope", "local_cache"})

# Allowed env backends
_ALLOWED_ENV_BACKENDS = frozenset({"venv", "conda", "mamba", "pip"})

# Shell metacharacters that must never appear in command-like inputs
import re
_SHELL_META = re.compile(r'[;&|>`\$]')


class StagePolicyValidator:
    """Stage-specific policy validation for decision gate tool calls.

    Extends the base ToolPolicy with stage-specific rules:
    - runner: candidate_id must exist, cmd no shell metachar, command root allowed
    - env: package name valid, version spec valid, no arbitrary index URL, no source edit
    - model: source in allowlist, repo_id format valid, no path traversal, no token in input
    - repair: action type in allowlist, rerun_from in safe stages, metadata_only not counted as executed
    - plan: stage name in pipeline stages, hints must not change policy
    """

    def validate(self, tool_call: Dict, stage: str, observation: Dict = None) -> Dict:
        """Validate a tool call for a specific stage.

        Returns dict with 'allowed' (bool) and 'reason' (str).
        """
        name = tool_call.get("name", "")
        tool_input = tool_call.get("input") or {}
        obs = observation or {}

        if stage == "runner":
            return self._validate_runner(name, tool_input, obs)
        elif stage == "env_solve":
            return self._validate_env(name, tool_input, obs)
        elif stage == "model_prepare":
            return self._validate_model(name, tool_input, obs)
        elif stage == "repair":
            return self._validate_repair(name, tool_input, obs)
        elif stage == "plan":
            return self._validate_plan(name, tool_input, obs)
        else:
            return {"allowed": True, "reason": "no stage-specific policy for %s" % stage}

    def _validate_runner(self, name: str, tool_input: Dict, obs: Dict) -> Dict:
        """Runner gate policy."""
        if name == "select_runner_candidate":
            candidate_id = str(tool_input.get("candidate_id", ""))
            if not candidate_id:
                return {"allowed": False, "reason": "select_runner_candidate requires candidate_id"}
            # Verify candidate_id exists in observation
            candidates = obs.get("run_candidates", [])
            if not any(c.get("id") == candidate_id for c in candidates):
                return {"allowed": False, "reason": "candidate_id '%s' not found in run_candidates" % candidate_id}
            # Check the candidate's command for shell metacharacters
            for c in candidates:
                if c.get("id") == candidate_id:
                    cmd = c.get("cmd", [])
                    if isinstance(cmd, list):
                        cmd_str = " ".join(str(x) for x in cmd)
                        if _SHELL_META.search(cmd_str):
                            return {"allowed": False, "reason": "candidate command contains shell metacharacters"}
                        # Check command root
                        if cmd:
                            root = str(cmd[0]).split("/")[-1]
                            if root not in _ALLOWED_COMMAND_ROOTS:
                                return {"allowed": False, "reason": "command root '%s' not in allowed list" % root}
                    break
            return {"allowed": True, "reason": "runner candidate selection passes policy"}

        elif name == "add_runner_candidate":
            cmd = tool_input.get("cmd", [])
            if not cmd:
                return {"allowed": False, "reason": "add_runner_candidate requires cmd"}
            cmd_str = " ".join(str(x) for x in cmd) if isinstance(cmd, list) else str(cmd)
            if _SHELL_META.search(cmd_str):
                return {"allowed": False, "reason": "candidate command contains shell metacharacters"}
            if isinstance(cmd, list) and cmd:
                root = str(cmd[0]).split("/")[-1]
                if root not in _ALLOWED_COMMAND_ROOTS:
                    return {"allowed": False, "reason": "command root '%s' not in allowed list" % root}
            return {"allowed": True, "reason": "add_runner_candidate passes policy"}

        elif name == "reject_runner_candidate":
            return {"allowed": True, "reason": "rejection is always allowed"}

        return {"allowed": False, "reason": "unknown runner tool: %s" % name}

    def _validate_env(self, name: str, tool_input: Dict, obs: Dict) -> Dict:
        """Env gate policy."""
        if name in ("inspect_env_log", "propose_dependency_constraint"):
            return {"allowed": True, "reason": "read-only env tool passes policy"}

        if name == "apply_dependency_constraint":
            package = str(tool_input.get("package", ""))
            if not package or not self._valid_package_name(package):
                return {"allowed": False, "reason": "invalid package name: %s" % package}
            version_spec = str(tool_input.get("version_spec", ""))
            if version_spec and not self._valid_version_spec(version_spec):
                return {"allowed": False, "reason": "invalid version spec: %s" % version_spec}
            # No arbitrary index URL
            index_url = tool_input.get("index_url", "")
            if index_url:
                return {"allowed": False, "reason": "arbitrary index URL not allowed"}
            # No source edit
            if tool_input.get("source_edit"):
                return {"allowed": False, "reason": "source edit not allowed in env gate"}
            return {"allowed": True, "reason": "dependency constraint passes policy"}

        if name == "select_environment_backend":
            backend = str(tool_input.get("backend", ""))
            if backend not in _ALLOWED_ENV_BACKENDS:
                return {"allowed": False, "reason": "env backend '%s' not in allowed list" % backend}
            return {"allowed": True, "reason": "environment backend selection passes policy"}

        if name == "select_torch_variant":
            return {"allowed": True, "reason": "torch variant selection passes policy"}

        return {"allowed": False, "reason": "unknown env tool: %s" % name}

    def _validate_model(self, name: str, tool_input: Dict, obs: Dict) -> Dict:
        """Model gate policy."""
        if name in ("inspect_model_config", "inspect_git_lfs_pointers"):
            return {"allowed": True, "reason": "read-only model tool passes policy"}

        if name in ("select_model_source", "select_model_asset_strategy"):
            source = str(tool_input.get("source", ""))
            if source and source not in _ALLOWED_MODEL_SOURCES:
                return {"allowed": False, "reason": "model source '%s' not in allowed list" % source}
            # No path traversal
            target_path = str(tool_input.get("target_path", ""))
            if target_path and ("../" in target_path or "..\\" in target_path):
                return {"allowed": False, "reason": "path traversal in target_path"}
            # No token in input
            for key in tool_input:
                if key.lower() in ("token", "api_key", "password", "secret"):
                    return {"allowed": False, "reason": "secret-like field '%s' not allowed in tool input" % key}
            # No external URL (except HF/ModelScope)
            repo_id = str(tool_input.get("repo_id", ""))
            if repo_id and repo_id.startswith("http") and "huggingface" not in repo_id and "modelscope" not in repo_id:
                return {"allowed": False, "reason": "external URL not allowed as model source"}
            return {"allowed": True, "reason": "model strategy selection passes policy"}

        if name in ("download_model_asset", "link_cached_model_asset"):
            # These are medium-risk, require gated_actor mode (checked elsewhere)
            target_path = str(tool_input.get("target_path", ""))
            if target_path and ("../" in target_path or "..\\" in target_path):
                return {"allowed": False, "reason": "path traversal in target_path"}
            for key in tool_input:
                if key.lower() in ("token", "api_key", "password", "secret"):
                    return {"allowed": False, "reason": "secret-like field '%s' not allowed in tool input" % key}
            return {"allowed": True, "reason": "model asset action passes policy"}

        return {"allowed": False, "reason": "unknown model tool: %s" % name}

    def _validate_repair(self, name: str, tool_input: Dict, obs: Dict) -> Dict:
        """Repair gate policy."""
        if name in ("inspect_log", "classify_failure"):
            return {"allowed": True, "reason": "read-only repair tool passes policy"}

        if name == "apply_dependency_constraint":
            # Reuse env validation
            return self._validate_env(name, tool_input, obs)

        if name == "apply_repair":
            action_type = str(tool_input.get("action_type", ""))
            if action_type == "source_edit":
                return {"allowed": False, "reason": "source edit not allowed in repair gate"}
            return {"allowed": True, "reason": "repair action passes policy"}

        if name == "resume_from_stage":
            stage = str(tool_input.get("stage", ""))
            if stage and stage not in SAFE_RERUN_STAGES:
                return {"allowed": False, "reason": "resume_from_stage '%s' not in safe stages" % stage}
            return {"allowed": True, "reason": "resume stage passes policy"}

        if name == "verify_after_repair":
            return {"allowed": True, "reason": "verify after repair passes policy"}

        return {"allowed": False, "reason": "unknown repair tool: %s" % name}

    def _validate_plan(self, name: str, tool_input: Dict, obs: Dict) -> Dict:
        """Plan gate policy. Plan gate does not execute tools, only validates strategy hints."""
        # Plan gate should never have tool_calls, but validate if present
        stage = str(tool_input.get("stage", ""))
        if stage and stage not in PIPELINE_STAGES:
            return {"allowed": False, "reason": "invalid stage name in plan: %s" % stage}
        # Hints must not change policy
        hints = tool_input.get("hints", {})
        if isinstance(hints, dict):
            for key in hints:
                if key.lower() in ("allow_source_edit", "allow_shell", "bypass_policy", "disable_trace"):
                    return {"allowed": False, "reason": "plan hint '%s' would change policy" % key}
        return {"allowed": True, "reason": "plan hint passes policy"}

    @staticmethod
    def _valid_package_name(name: str) -> bool:
        """Check if a package name is valid (alphanumeric, hyphens, underscores, dots)."""
        import re
        return bool(re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$', name)) and len(name) <= 200

    @staticmethod
    def _valid_version_spec(spec: str) -> bool:
        """Check if a version spec is valid (e.g., '<2', '>=1.0,<2.0', '==1.2.3')."""
        import re
        # Allow common version spec patterns
        parts = spec.split(",")
        for part in parts:
            part = part.strip()
            if not re.match(r'^[<>=!~]+\s*[\d.*]+$', part):
                return False
        return True


# ------------------------------------------------------------------
# Critic for decision gates
# ------------------------------------------------------------------

class GateCritic:
    """Quality gate for decision gate tool calls.

    Checks:
    - Tool is relevant for the stage
    - No secret values in input
    - No hallucinated tool names
    """

    def evaluate(self, tool_call: Dict, stage: str, observation: Dict = None) -> Dict:
        """Evaluate a tool call through the critic gate.

        Returns dict with 'allowed' (bool) and 'reason' (str).
        """
        name = tool_call.get("name", "")
        tool_input = tool_call.get("input") or {}
        input_text = json.dumps(tool_input, ensure_ascii=False).lower()

        # Check for secret values
        secret_field_names = {"api_key", "token", "password", "secret", "credential", "auth_token", "access_token", "private_key", "bearer"}
        for key in tool_input:
            if key.lower() in secret_field_names:
                return {"allowed": False, "reason": "tool input appears to contain secret field '%s'" % key, "issues": ["secret_in_input"]}
        for token in ("api_key=", "token=", "password=", "bearer "):
            if token in input_text:
                return {"allowed": False, "reason": "tool input appears to contain secret value", "issues": ["secret_in_input"]}

        # Stage relevance check
        stage_tools = {
            "runner": RUNNER_TOOLS,
            "env_solve": ENV_TOOLS,
            "model_prepare": MODEL_TOOLS,
            "repair": REPAIR_TOOLS,
        }
        allowed_for_stage = stage_tools.get(stage, ())
        if allowed_for_stage and name not in allowed_for_stage:
            return {"allowed": False, "reason": "tool '%s' not relevant for stage '%s'" % (name, stage), "issues": ["stage_mismatch"]}

        return {"allowed": True, "reason": "tool call is consistent with stage and evidence requirements", "issues": []}


# ------------------------------------------------------------------
# Artifact writer for decision gates
# ------------------------------------------------------------------

class GateArtifactWriter:
    """Writes decision gate artifacts to disk."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.gates_dir = self.run_dir / "agent_decision_gates"
        self.gates_dir.mkdir(parents=True, exist_ok=True)

    def write_gate_result(self, stage: str, result: GateResult) -> Path:
        """Write gate result JSON."""
        path = self.gates_dir / ("%s_gate.json" % stage)
        write_json(path, {
            "stage": result.stage,
            "mode": result.mode,
            "decision_status": result.decision_status,
            "hypothesis": result.hypothesis,
            "tool_call": result.tool_call,
            "critic": result.critic,
            "policy": result.policy,
            "execution": result.execution,
            "state_delta": result.state_delta,
            "llm_helped": result.llm_helped,
            "error": result.error,
            "recorded_at": utc_now_iso(),
        })
        return path

    def write_step(self, stage: str, step_index: int, step: Dict) -> Path:
        """Append a step to the stage steps JSONL."""
        path = self.gates_dir / ("%s_steps.jsonl" % stage)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")
        return path

    def write_summary(self, results: Dict) -> Path:
        """Write the global llm_decision_gates.json summary."""
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / "llm_decision_gates.json"
        write_json(path, results)
        return path


# ------------------------------------------------------------------
# AgentDecisionGate
# ------------------------------------------------------------------

class AgentDecisionGate:
    """Unified decision gate for LLM-driven stage decisions.

    Encapsulates: observe -> LLM planner -> schema validate -> critic -> policy -> execute -> artifact.

    Each stage (runner, env_solve, model_prepare, repair) uses this same pipeline
    with stage-specific planners and policy rules.

    Usage:
        gate = AgentDecisionGate(provider=llm_provider)
        result = gate.decide(
            stage="runner",
            observation={...},
            allowed_tools=["select_runner_candidate"],
            mode="gated_actor",
            run_dir=Path("runs/001"),
        )
    """

    def __init__(self, provider=None, registry=None, critic: GateCritic = None, policy: StagePolicyValidator = None, executor: Callable = None) -> None:
        self.provider = provider
        self.registry = registry
        self.critic = critic or GateCritic()
        self.policy = policy or StagePolicyValidator()
        self.executor = executor  # Optional callable for executing approved tool calls

    def decide(
        self,
        *,
        stage: str,
        observation: Dict,
        allowed_tools: List[str],
        mode: str,
        run_dir: Path,
        max_steps: int = 2,
    ) -> GateResult:
        """Run the decision gate pipeline for a stage.

        Args:
            stage: Stage name (runner, env_solve, model_prepare, repair).
            observation: Stage-specific observation dict.
            allowed_tools: Tools the LLM can choose from.
            mode: Agent mode (planner or gated_actor).
            run_dir: Run directory for artifact writing.
            max_steps: Maximum decision steps.

        Returns:
            GateResult with the full pipeline result.
        """
        run_dir = Path(run_dir)
        artifact_writer = GateArtifactWriter(run_dir)
        planner = self._get_planner(stage)

        result = GateResult(stage=stage, mode=mode)

        for step_index in range(max_steps):
            # 1. Call LLM planner
            decision = planner.plan(observation, provider=self.provider, allowed_tools=allowed_tools)

            # 2. Schema validation (done in parse_gate_decision)
            if decision.status == "invalid":
                result.decision_status = "invalid"
                result.error = "invalid LLM output: %s" % decision.stop_reason
                artifact_writer.write_step(stage, step_index + 1, {
                    "step_index": step_index + 1,
                    "decision": {"status": decision.status, "stop_reason": decision.stop_reason},
                    "reason": "invalid_llm_output",
                })
                break

            # no_action from LLM
            if decision.status == "no_action":
                result.decision_status = "no_action"
                result.hypothesis = decision.hypothesis
                artifact_writer.write_step(stage, step_index + 1, {
                    "step_index": step_index + 1,
                    "decision": {"status": "no_action", "hypothesis": decision.hypothesis, "stop_reason": decision.stop_reason},
                    "reason": "no_action",
                })
                break

            # 3. Decision is ok with a tool_call
            result.decision_status = "ok"
            result.hypothesis = decision.hypothesis
            result.tool_call = decision.tool_call

            # 4. Critic gate
            critic_result = self.critic.evaluate(decision.tool_call, stage, observation)
            result.critic = critic_result
            if not critic_result.get("allowed", False):
                artifact_writer.write_step(stage, step_index + 1, {
                    "step_index": step_index + 1,
                    "decision": {"status": "ok", "tool_call": decision.tool_call, "hypothesis": decision.hypothesis},
                    "critic": critic_result,
                    "reason": "critic_rejected: %s" % critic_result.get("reason", ""),
                })
                # Continue loop so LLM can try a different tool
                continue

            # 5. Policy gate
            policy_result = self.policy.validate(decision.tool_call, stage, observation)
            result.policy = policy_result
            if not policy_result.get("allowed", False):
                artifact_writer.write_step(stage, step_index + 1, {
                    "step_index": step_index + 1,
                    "decision": {"status": "ok", "tool_call": decision.tool_call, "hypothesis": decision.hypothesis},
                    "critic": critic_result,
                    "policy": policy_result,
                    "reason": "policy_rejected: %s" % policy_result.get("reason", ""),
                })
                # Continue loop so LLM can try a different tool
                continue

            # 6. Mode check: planner mode does not execute
            if mode == "planner":
                result.execution = {"executed": False, "status": "planner_mode_would_execute"}
                artifact_writer.write_step(stage, step_index + 1, {
                    "step_index": step_index + 1,
                    "decision": {"status": "ok", "tool_call": decision.tool_call, "hypothesis": decision.hypothesis},
                    "critic": critic_result,
                    "policy": policy_result,
                    "execution": result.execution,
                    "reason": "planner_mode_would_execute",
                })
                break

            # 7. gated_actor mode: execute or apply state delta
            exec_result = self._execute_or_apply(
                decision.tool_call, stage, observation, run_dir, step_index
            )
            result.execution = exec_result
            result.state_delta = exec_result.get("state_delta", {})

            artifact_writer.write_step(stage, step_index + 1, {
                "step_index": step_index + 1,
                "decision": {"status": "ok", "tool_call": decision.tool_call, "hypothesis": decision.hypothesis},
                "critic": critic_result,
                "policy": policy_result,
                "execution": exec_result,
                "state_delta": result.state_delta,
            })

            # If execution succeeded, break
            if exec_result.get("executed") or exec_result.get("applied"):
                break

        # llm_helped: GateResult NEVER self-declares llm_helped=true.
        # A single gate cannot know if state actually improved.
        # llm_helped must be computed by AgentContributionAnalyzer or AgentLoop
        # AFTER observing stage status improvement (before_status -> after_status).
        result.llm_helped = False

        # Write gate result artifact
        artifact_writer.write_gate_result(stage, result)
        return result

    def execute_if_allowed(
        self,
        *,
        decision: Dict,
        observation: Dict,
        mode: str,
        run_dir: Path,
        context: Dict,
    ) -> Dict:
        """Execute a pre-parsed decision if policy allows.

        Used by orchestrator for cases where the decision is already parsed.
        """
        stage = context.get("stage", "")
        tool_call = decision.get("tool_call")
        if not tool_call:
            return {"executed": False, "reason": "no tool_call in decision"}

        # Critic
        critic_result = self.critic.evaluate(tool_call, stage, observation)
        if not critic_result.get("allowed", False):
            return {"executed": False, "reason": "critic_rejected: %s" % critic_result.get("reason", "")}

        # Policy
        policy_result = self.policy.validate(tool_call, stage, observation)
        if not policy_result.get("allowed", False):
            return {"executed": False, "reason": "policy_rejected: %s" % policy_result.get("reason", "")}

        # Mode check
        if mode == "planner":
            return {"executed": False, "reason": "planner_mode_would_execute"}

        # Execute
        return self._execute_or_apply(tool_call, stage, observation, run_dir, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_planner(self, stage: str):
        """Get the stage-specific planner."""
        planners = {
            "runner": RunnerPlanner,
            "env_solve": EnvPlanner,
            "model_prepare": ModelPlanner,
            "repair": RepairActuatorPlanner,
            "plan": PlanPlanner,
            "verify": VerifyPlanner,
        }
        planner_cls = planners.get(stage)
        if planner_cls is None:
            # Return a no-op planner for unsupported stages
            class NoOpPlanner:
                def plan(self, observation, provider=None, allowed_tools=None):
                    return GateDecision(stage=stage, status="no_action", stop_reason="no_provider_for_stage_%s" % stage, raw_response="")
            return NoOpPlanner()
        return planner_cls()

    def _execute_or_apply(
        self,
        tool_call: Dict,
        stage: str,
        observation: Dict,
        run_dir: Path,
        step_index: int,
    ) -> Dict:
        """Execute a tool call or apply a state delta.

        For state-delta tools (select_*, reject_*), applies the delta directly.
        For execution tools (apply_*, download_*, resume_*), delegates to executor.
        """
        name = tool_call.get("name", "")
        tool_input = tool_call.get("input") or {}

        # State-delta tools: apply directly without external execution
        if name in ("select_runner_candidate", "add_runner_candidate", "reject_runner_candidate",
                     "select_environment_backend", "select_torch_variant",
                     "select_model_source", "select_model_asset_strategy",
                     "apply_dependency_constraint", "propose_dependency_constraint",
                     "resume_from_stage", "verify_after_repair"):
            return self._apply_state_delta(name, tool_input, stage, observation, run_dir)

        # Execution tools: delegate to executor if available
        if self.executor:
            try:
                exec_result = self.executor(tool_call, run_dir, step_index)
                if isinstance(exec_result, dict):
                    return exec_result
            except Exception as e:
                return {"executed": False, "status": "error", "error": str(e)}

        # No executor: mark as would_execute
        return {"executed": False, "status": "no_executor", "tool_name": name}

    def _apply_state_delta(
        self,
        tool_name: str,
        tool_input: Dict,
        stage: str,
        observation: Dict,
        run_dir: Path,
    ) -> Dict:
        """Apply a state delta from a state-changing tool call.

        Returns execution result with state_delta describing what changed.
        """
        if tool_name == "select_runner_candidate":
            candidate_id = tool_input.get("candidate_id", "")
            candidates = observation.get("run_candidates", [])
            # Reorder candidates so selected one is first
            selected = [c for c in candidates if c.get("id") == candidate_id]
            others = [c for c in candidates if c.get("id") != candidate_id]
            reordered = selected + others
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    "before": {"candidate_order": [c.get("id") for c in candidates]},
                    "after": {"candidate_order": [c.get("id") for c in reordered]},
                    "reordered_candidates": reordered,
                },
            }

        elif tool_name == "add_runner_candidate":
            new_candidate = {
                "id": tool_input.get("candidate_id", "cand_llm_%d" % hash(frozenset(tool_input.items())) % 10000),
                "cmd": tool_input.get("cmd", []),
                "expected_port": tool_input.get("expected_port"),
                "source": "llm_gate",
                "score": tool_input.get("score", 0.5),
            }
            candidates = observation.get("run_candidates", [])
            updated = [new_candidate] + candidates
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    "before": {"candidate_count": len(candidates)},
                    "after": {"candidate_count": len(updated)},
                    "added_candidate": new_candidate,
                    "updated_candidates": updated,
                },
            }

        elif tool_name == "reject_runner_candidate":
            candidate_id = tool_input.get("candidate_id", "")
            candidates = observation.get("run_candidates", [])
            filtered = [c for c in candidates if c.get("id") != candidate_id]
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    "before": {"candidate_count": len(candidates)},
                    "after": {"candidate_count": len(filtered)},
                    "rejected_id": candidate_id,
                    "remaining_candidates": filtered,
                },
            }

        elif tool_name == "apply_dependency_constraint":
            # Write constraint overlay
            repair_dir = run_dir / "repair_overlay"
            repair_dir.mkdir(parents=True, exist_ok=True)
            constraint = {
                "package": tool_input.get("package", ""),
                "version_spec": tool_input.get("version_spec", ""),
                "scope": tool_input.get("scope", "temporary_overlay"),
                "reason": tool_input.get("reason", ""),
            }
            constraints_path = repair_dir / "constraints.txt"
            # Append constraint
            existing = ""
            if constraints_path.exists():
                existing = constraints_path.read_text(encoding="utf-8")
            line = format_dependency_constraint(constraint["package"], constraint["version_spec"])
            constraints_path.write_text(existing + line + "\n", encoding="utf-8")
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    "constraint": constraint,
                    "overlay_path": str(constraints_path),
                },
            }

        elif tool_name in ("select_environment_backend", "select_torch_variant"):
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    tool_name: tool_input,
                },
            }

        elif tool_name in ("select_model_source", "select_model_asset_strategy"):
            # Write model strategy overlay
            reports_dir = run_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            strategy_path = reports_dir / "model_asset_strategy.json"
            write_json(strategy_path, {
                "source": tool_input.get("source", ""),
                "repo_id": tool_input.get("repo_id", ""),
                "target_path": tool_input.get("target_path", ""),
                "strategy": tool_input.get("strategy", ""),
                "fallback": tool_input.get("fallback", ""),
            })
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    "strategy_path": str(strategy_path),
                },
            }

        elif tool_name == "propose_dependency_constraint":
            # Proposal only, no state change
            return {
                "applied": False,
                "executed": False,
                "status": "proposed",
                "tool_name": tool_name,
                "state_delta": {"changed": False, "proposal": tool_input},
            }

        elif tool_name == "resume_from_stage":
            stage_name = tool_input.get("stage", "")
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {
                    "changed": True,
                    "resume_from_stage": stage_name,
                },
            }

        elif tool_name == "verify_after_repair":
            return {
                "applied": True,
                "executed": True,
                "status": "applied",
                "tool_name": tool_name,
                "state_delta": {"changed": True, "verify_requested": True},
            }

        return {"applied": False, "executed": False, "status": "unknown_tool", "tool_name": tool_name}
