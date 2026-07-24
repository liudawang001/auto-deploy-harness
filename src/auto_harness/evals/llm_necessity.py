"""LLM Necessity Evaluator.

Generates baseline vs agent comparison reports to prove that LLM decision gates
are necessary for complex deployments.

Per design doc:
- llm_required=true only when baseline failed/uncertain AND agent passed
- llm_helped=true only when state actually improved
- Evidence artifacts must exist
- Must actually run baseline and agent pipelines (not read expectations)
- No HarnessOrchestrator references - uses TaskRunner only
- baseline and agent use isolated workspaces
- Exceptions produce infrastructure_error, never completed
"""
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


def _load_run_result(run_dir: Path) -> Dict[str, Any]:
    """Load and normalize run result from a TaskRunner run directory.

    Returns a fixed schema dict regardless of internal artifact format.
    Missing controller_result.json means infrastructure_error.

    Args:
        run_dir: Path to the run directory (e.g., runs/<task_id>).

    Returns:
        Normalized result dict with status, verify_status, trace_id, etc.
    """
    controller_result_path = run_dir / "reports" / "controller_result.json"
    if not controller_result_path.exists():
        return {
            "status": "infrastructure_error",
            "task_id": run_dir.name,
            "controller": "",
            "final_status": "",
            "verify_status": "",
            "trace_id": "",
            "evidence_paths": [],
            "accepted_decisions": [],
            "effective_actions": [],
            "duration_ms": 0,
            "token_usage": {},
            "error": "controller_result.json not found",
        }

    try:
        cr = read_json(controller_result_path)
    except (OSError, ValueError) as exc:
        return {
            "status": "infrastructure_error",
            "task_id": run_dir.name,
            "controller": "",
            "final_status": "",
            "verify_status": "",
            "trace_id": "",
            "evidence_paths": [],
            "accepted_decisions": [],
            "effective_actions": [],
            "duration_ms": 0,
            "token_usage": {},
            "error": "failed to read controller_result: %s" % str(exc)[:500],
        }

    final_status = str(cr.get("status") or cr.get("final_status") or "")
    verify_status = str(cr.get("verify_status") or "")
    verify = cr.get("verify", {})
    if not verify_status and isinstance(verify, dict):
        verify_status = str(verify.get("status") or "")

    # Extract trace_id from the actual verify stage artifact.
    trace_id = ""
    pipeline_path = run_dir / "reports" / "pipeline_results.json"
    if pipeline_path.exists():
        try:
            pipeline = read_json(pipeline_path)
            verify_result = pipeline.get("verify", {})
            verify_data = (
                verify_result.get("data", {})
                if isinstance(verify_result, dict)
                else {}
            )
            trace_id = str(verify_data.get("trace_id") or "")
            if not verify_status and isinstance(verify_result, dict):
                verify_status = str(verify_result.get("status") or "")
        except (OSError, ValueError):
            pass

    # Collect evidence paths
    evidence_paths = []
    evidence_dir = run_dir / "evidence"
    if evidence_dir.exists():
        evidence_paths = [str(p) for p in evidence_dir.glob("*.json")]

    trace_verified = _trace_is_strongly_verified(run_dir, trace_id)

    # Load contribution evidence from either controller implementation.
    contribution_paths = [
        run_dir / "reports" / "llm_contribution_evidence.json",
        run_dir / "reports" / "agent_contribution.json",
    ]
    accepted_decisions = []
    effective_actions = []
    contribution_path = next(
        (path for path in contribution_paths if path.exists()),
        None,
    )
    if contribution_path:
        try:
            contrib = read_json(contribution_path)
            # Extract gate results as accepted decisions
            if isinstance(contrib, dict):
                gate_summary = contrib.get("gate_summary", [])
                for gate in gate_summary:
                    if isinstance(gate, dict) and gate.get("policy_allowed"):
                        accepted_decisions.append(gate)
                        if gate.get("executed") or gate.get("applied"):
                            effective_actions.append(gate)
        except (OSError, ValueError, NameError):
            pass
    repair_apply_path = run_dir / "repairs" / "repair_apply_result.json"
    if repair_apply_path.exists():
        try:
            repair_apply = read_json(repair_apply_path)
            if repair_apply.get("repair_verified"):
                effective_actions.append({
                    "source": "repair_apply",
                    "executed": True,
                    "repair_verified": True,
                })
        except (OSError, ValueError):
            pass

    if not final_status:
        return {
            "status": "infrastructure_error",
            "task_id": cr.get("task_id", run_dir.name),
            "controller": cr.get("controller", ""),
            "final_status": "",
            "verify_status": verify_status,
            "trace_id": trace_id,
            "trace_verified": trace_verified,
            "evidence_paths": evidence_paths,
            "accepted_decisions": accepted_decisions,
            "effective_actions": effective_actions,
            "duration_ms": 0,
            "token_usage": {},
            "error": "controller_result status is missing",
        }

    return {
        "status": "completed",
        "task_id": cr.get("task_id", run_dir.name),
        "controller": cr.get("controller", "langgraph"),
        "final_status": final_status,
        "verify_status": verify_status,
        "trace_id": trace_id,
        "trace_verified": trace_verified,
        "evidence_paths": evidence_paths,
        "accepted_decisions": accepted_decisions,
        "effective_actions": effective_actions,
        "duration_ms": cr.get("duration_ms", 0),
        "token_usage": cr.get("token_usage", {}),
        "error": "",
    }


def _trace_is_strongly_verified(run_dir: Path, trace_id: str) -> bool:
    """Return true only when a passed evidence check contains the current trace."""
    if not trace_id:
        return False
    for path in (run_dir / "evidence").glob("*.json"):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        checks = payload.get("checks", []) if isinstance(payload, dict) else []
        if not checks and isinstance(payload, dict) and isinstance(payload.get("check"), dict):
            checks = [payload["check"]]
        for check in checks:
            if (
                isinstance(check, dict)
                and check.get("status") in ("pass", "passed")
                and trace_id in json.dumps(check, ensure_ascii=False)
            ):
                return True
    return False


class LLMNecessityEvaluator:
    """Evaluates LLM necessity by actually running baseline vs agent pipelines.

    For each case in the manifest:
    1. Run baseline pipeline (agent_mode=off, deterministic planner, no LLM nodes)
    2. Run agent pipeline (agent_mode=gated_actor, LLM planner, all nodes)
    3. Compare results to determine llm_required and llm_helped

    Both baseline and agent use LangGraph controller to avoid controller
    differences polluting the comparison.
    """

    def __init__(
        self,
        output_dir: Path = None,
        config_factory=None,
        runner_factory=None,
    ) -> None:
        self.output_dir = Path(output_dir or "runs/evals/llm_necessity")
        self.config_factory = config_factory
        self.runner_factory = runner_factory

    def evaluate_manifest(self, manifest_path: Path, output_path: Path = None) -> Dict:
        """Evaluate all cases in the manifest and generate a report.

        Args:
            manifest_path: Path to llm_necessity_manifest.json
            output_path: Optional path to write the report JSON

        Returns:
            Report dict with per-case results and summary
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return {"status": "failed", "error": "manifest not found: %s" % manifest_path}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest.get("cases", [])

        results = []
        for case in cases:
            result = self.evaluate_case(case)
            results.append(result)

        summary = self._build_summary(results)
        report = {
            "status": "completed",
            "manifest_version": manifest.get("version", ""),
            "evaluated_at": utc_now_iso(),
            "case_count": len(results),
            "results": results,
            "summary": summary,
        }

        # If any infrastructure errors, report status should reflect it
        infra_errors = [r for r in results if r.get("status") == "infrastructure_error"]
        if infra_errors:
            report["status"] = "failed"
            report["summary"]["infrastructure_error_count"] = len(infra_errors)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, report)

        return report

    def evaluate_case(self, case: Dict) -> Dict:
        """Evaluate a single case by running baseline and agent pipelines.

        Both baseline and agent use LangGraph controller.
        Baseline: deterministic planner, no LLM nodes.
        Agent: LLM planner, all LLM nodes enabled.

        Args:
            case: Case dict from manifest with case_id, fixture_dir, etc.

        Returns:
            Result dict with llm_required, llm_helped, evidence
        """
        case_id = case.get("case_id", "")
        fixture_dir = case.get("fixture_dir", "")

        # Create output directory for this case
        case_output_dir = self.output_dir / case_id
        baseline_dir = case_output_dir / "baseline"
        agent_dir = case_output_dir / "agent"

        # Clean and create directories
        if case_output_dir.exists():
            shutil.rmtree(case_output_dir)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Check fixture exists
        fixture_path = Path(fixture_dir)
        fixture_exists = fixture_path.exists() and any(fixture_path.iterdir())

        if not fixture_exists:
            return self._build_error_result(
                case_id, "infrastructure_error",
                "fixture not found: %s" % fixture_dir,
            )

        # Isolate fixture: copy to separate baseline/agent source dirs
        baseline_source = case_output_dir / "baseline" / "source"
        agent_source = case_output_dir / "agent" / "source"
        shutil.copytree(str(fixture_path), str(baseline_source))
        shutil.copytree(str(fixture_path), str(agent_source))

        # Run baseline pipeline
        baseline_result = self._run_pipeline(
            fixture_dir=baseline_source,
            output_dir=baseline_dir,
            mode="off",
            case=case,
        )

        # Run agent pipeline
        agent_result = self._run_pipeline(
            fixture_dir=agent_source,
            output_dir=agent_dir,
            mode="gated_actor",
            case=case,
            provider="mock",
        )

        # Compare results
        comparison = self._compare_runs(baseline_result, agent_result, case)

        # Write comparison
        write_json(case_output_dir / "comparison.json", comparison)

        return comparison

    def _run_pipeline(
        self,
        *,
        fixture_dir: Path,
        output_dir: Path,
        mode: str,
        case: Dict,
        provider: str = None,
    ) -> Dict:
        """Run a pipeline with the given configuration using TaskRunner.

        Both baseline and agent use LangGraph controller to avoid
        controller differences polluting the comparison.
        """
        try:
            from auto_harness.config import HarnessConfig
            from auto_harness.orchestrator import TaskRunner

            # Build config with only real fields
            config_factory = self.config_factory or HarnessConfig
            config = config_factory()

            # Set common fields
            run_root = Path(output_dir)
            config.runs_dir = str(run_root / "runs")
            config.memory_dir = str(run_root / "memory")
            config.model_cache_dir = str(run_root / "model_cache")
            config.default_controller = "langgraph"

            if mode == "off":
                # Baseline: deterministic planner, no LLM nodes
                config.agent_mode = "off"
                config.langgraph_planner_mode = "deterministic"
                config.langgraph_require_llm = False
                config.langgraph_enable_diagnose = False
                config.langgraph_enable_repair = False
                config.langgraph_enable_agent_verify = False
            else:
                # Agent: LLM planner, all LLM nodes enabled
                config.agent_mode = "gated_actor"
                config.langgraph_planner_mode = "llm"
                config.langgraph_require_llm = True
                config.langgraph_enable_diagnose = True
                config.langgraph_enable_repair = True
                config.langgraph_enable_agent_verify = True
                if provider:
                    config.agent_plan_first_provider = provider
                    config.agent_provider = provider

            # Run permissions from case
            config.allow_dependency_install = case.get("allow_install", False)
            config.allow_service_start = case.get("allow_start", False)

            # Create and run
            runner_factory = self.runner_factory or TaskRunner
            runner = runner_factory(config)

            task_id = runner.deploy(
                repo_url=str(fixture_dir),
                name="%s-%s" % (case.get("case_id", "eval"), mode),
                dry_run=case.get("dry_run", True),
                skip_clone=False,
                allow_install=case.get("allow_install", False),
                allow_start=case.get("allow_start", False),
                controller="langgraph",
            )

            # Load results from the run directory
            run_dir = Path(config.runs_dir) / task_id
            return _load_run_result(run_dir)

        except Exception as exc:
            return {
                "status": "infrastructure_error",
                "task_id": "",
                "controller": "langgraph",
                "final_status": "",
                "verify_status": "",
                "trace_id": "",
                "evidence_paths": [],
                "accepted_decisions": [],
                "effective_actions": [],
                "duration_ms": 0,
                "token_usage": {},
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }

    def _compare_runs(self, baseline: Dict, agent: Dict, case: Dict) -> Dict:
        """Compare baseline and agent runs to determine llm_required and llm_helped."""
        case_id = case.get("case_id", "")

        baseline_status = baseline.get("final_status", baseline.get("status", "unknown"))
        agent_status = agent.get("final_status", agent.get("status", "unknown"))
        baseline_verify = baseline.get("verify_status", "")
        agent_verify = agent.get("verify_status", "")

        # Check for infrastructure errors
        if baseline.get("status") == "infrastructure_error":
            return {
                "case_id": case_id,
                "status": "infrastructure_error",
                "baseline_status": "infrastructure_error",
                "agent_status": agent_status,
                "llm_required": False,
                "llm_helped": False,
                "llm_required_status": "unknown_infrastructure_error",
                "error": baseline.get("error", ""),
            }

        if agent.get("status") == "infrastructure_error":
            return {
                "case_id": case_id,
                "status": "infrastructure_error",
                "baseline_status": baseline_status,
                "agent_status": "infrastructure_error",
                "llm_required": False,
                "llm_helped": False,
                "llm_required_status": "unknown_infrastructure_error",
                "error": agent.get("error", ""),
            }

        # Determine baseline failure
        baseline_failed = (
            baseline.get("status") == "completed"
            and (
                baseline_status in ("failed", "stopped")
                or baseline_verify in ("failed", "fail", "uncertain")
            )
        )

        # Determine agent pass with strong verify
        agent_passed = (
            agent.get("status") == "completed"
            and agent_verify in ("pass", "passed")
            and bool(agent.get("trace_id"))
            and agent.get("trace_verified") is True
        )

        # Effective decisions from agent
        effective = agent.get("effective_actions", [])

        # Determine llm_helped and llm_required
        llm_helped = bool(agent_passed and effective)

        if baseline.get("status") == "infrastructure_error":
            llm_required = False
            llm_required_status = "unknown_infrastructure_error"
        elif baseline_failed and llm_helped:
            llm_required = True
            llm_required_status = "proven_by_baseline_agent_delta"
        else:
            llm_required = False
            llm_required_status = "baseline_did_not_fail"

        # Build causal chain for required cases
        causal_chain = {}
        if llm_required:
            causal_chain = {
                "baseline_failure": {
                    "status": baseline_status,
                    "verify_status": baseline_verify,
                },
                "llm_decision": {
                    "effective_action_count": len(effective),
                },
                "policy_decision": {
                    "accepted_count": len(agent.get("accepted_decisions", [])),
                },
                "effective_action": effective[:3] if effective else [],
                "state_change": {
                    "before_verify": baseline_verify,
                    "after_verify": agent_verify,
                },
                "final_verify": {
                    "status": agent_verify,
                    "trace_id": agent.get("trace_id", ""),
                },
            }

        return {
            "case_id": case_id,
            "target_gate": case.get("target_gate", ""),
            "fixture_exists": True,
            "status": "completed",
            "baseline_status": baseline_status,
            "agent_status": agent_status,
            "baseline_verify_status": baseline_verify,
            "agent_verify_status": agent_verify,
            "llm_required": llm_required,
            "llm_helped": llm_helped,
            "llm_helped_type": "bool",
            "llm_required_status": llm_required_status,
            "effective_action_count": len(effective),
            "causal_chain": causal_chain,
            "evidence_paths": {
                "baseline": baseline.get("evidence_paths", []),
                "agent": agent.get("evidence_paths", []),
            },
        }

    def _build_error_result(self, case_id: str, status: str, error: str) -> Dict:
        """Build an error result for a case."""
        return {
            "case_id": case_id,
            "target_gate": "",
            "fixture_exists": False,
            "status": status,
            "baseline_status": "error",
            "agent_status": "error",
            "llm_required": False,
            "llm_helped": False,
            "llm_helped_type": "bool",
            "llm_required_status": "unknown_infrastructure_error",
            "error": error,
            "evidence_paths": {},
        }

    def _build_summary(self, results: List[Dict]) -> Dict:
        """Build summary from per-case results."""
        total = len(results)
        llm_required_count = sum(1 for r in results if r.get("llm_required"))
        llm_helped_count = sum(1 for r in results if r.get("llm_helped"))
        infra_errors = [r for r in results if r.get("status") == "infrastructure_error"]

        return {
            "total_cases": total,
            "llm_required_count": llm_required_count,
            "llm_helped_count": llm_helped_count,
            "infrastructure_error_count": len(infra_errors),
            "llm_necessity_proven": llm_required_count > 0,
            "all_llm_helped_are_bool": all(
                isinstance(result.get("llm_helped"), bool)
                for result in results
            ),
        }


def generate_report_from_manifest(manifest_path: str, output_path: str = None) -> Dict:
    """Convenience function for CLI integration."""
    evaluator = LLMNecessityEvaluator()
    return evaluator.evaluate_manifest(
        Path(manifest_path),
        Path(output_path) if output_path else None,
    )
