"""LLM Necessity Evaluator.

Generates baseline vs agent comparison reports to prove that LLM decision gates
are necessary for complex deployments.

Per design doc:
- llm_required=true only when baseline failed/uncertain AND agent passed
- llm_helped=true only when state actually improved
- Evidence artifacts must exist
- Must actually run baseline and agent pipelines (not read expectations)
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.time import utc_now_iso


class LLMNecessityEvaluator:
    """Evaluates LLM necessity by actually running baseline vs agent pipelines.

    For each case in the manifest:
    1. Run baseline pipeline (agent_mode=off, no decision gates)
    2. Run agent pipeline (agent_mode=gated_actor, all gates, runtime loop)
    3. Compare results to determine llm_required and llm_helped
    """

    def __init__(self, output_dir: Path = None) -> None:
        self.output_dir = output_dir or Path("runs/evals/llm_necessity")

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

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(output_path, report)

        return report

    def evaluate_case(self, case: Dict) -> Dict:
        """Evaluate a single case by running baseline and agent pipelines.

        Args:
            case: Case dict from manifest

        Returns:
            Result dict with llm_required, llm_helped, evidence
        """
        case_id = case.get("case_id", "")
        target_gate = case.get("target_gate", "")
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
            return self._build_error_result(case_id, target_gate, "fixture_not_found", fixture_dir)

        # Run baseline pipeline
        baseline_result = self._run_pipeline(
            fixture_dir=fixture_path,
            output_dir=baseline_dir,
            mode="off",
            gates=False,
            runtime_loop=False,
            dry_run=True,
        )

        # Run agent pipeline
        agent_result = self._run_pipeline(
            fixture_dir=fixture_path,
            output_dir=agent_dir,
            mode="gated_actor",
            gates=True,
            runtime_loop=True,
            provider="mock",
            dry_run=True,
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
        gates: bool,
        runtime_loop: bool,
        provider: str = None,
        dry_run: bool = True,
    ) -> Dict:
        """Run a pipeline with the given configuration.

        This actually invokes the harness modules, not just reads expectations.
        """
        try:
            # Import here to avoid circular imports
            from auto_harness.config import HarnessConfig
            from auto_harness.orchestrator import HarnessOrchestrator
            from auto_harness.state import StateStore

            # Create config
            config = HarnessConfig(
                agent_mode=mode,
                agent_enable_decision_gates=gates,
                agent_enable_runtime_loop=runtime_loop,
                dry_run=dry_run,
            )

            # Create orchestrator
            store = StateStore(output_dir)
            orchestrator = HarnessOrchestrator(config=config, store=store)

            # Run the pipeline
            task_id = "eval_%s" % output_dir.name
            orchestrator.run_from_dir(task_id, fixture_dir, dry_run=dry_run)

            # Load results
            results_path = output_dir / task_id / "reports" / "pipeline_results.json"
            if results_path.exists():
                return read_json(results_path)

            return {"status": "no_results", "task_id": task_id}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _compare_runs(self, baseline: Dict, agent: Dict, case: Dict) -> Dict:
        """Compare baseline and agent runs to determine llm_required and llm_helped."""
        case_id = case.get("case_id", "")
        target_gate = case.get("target_gate", "")

        # Get status for the target stage
        baseline_status = self._get_stage_status(baseline, target_gate)
        agent_status = self._get_stage_status(agent, target_gate)

        # Check for LLM decisions in agent run
        agent_decisions = agent.get("decisions", [])
        accepted_decision = self._find_accepted_decision(agent_decisions, target_gate)

        # Check for verify evidence
        verify_evidence = agent.get("verify", {})
        final_verify_passed = verify_evidence.get("status") in ("passed", "pass")
        evidence_path_exists = bool(verify_evidence.get("evidence_paths"))

        # Determine llm_required
        baseline_failed = baseline_status in ("failed", "uncertain")
        agent_passed = agent_status in ("passed", "pass")
        has_accepted_decision = accepted_decision is not None
        policy_allowed = accepted_decision.get("policy_allowed", False) if accepted_decision else False
        tool_executed = accepted_decision.get("executed", False) if accepted_decision else False

        llm_required = (
            baseline_failed
            and agent_passed
            and has_accepted_decision
            and policy_allowed
            and (tool_executed or accepted_decision.get("applied", False))
        )

        # For safety cases, llm_required is False
        if "malicious" in case_id:
            llm_required = False

        # Determine llm_helped
        llm_helped = llm_required and agent_passed

        # Build state transition
        state_transition = "%s.%s -> %s.%s" % (
            target_gate, baseline_status,
            target_gate, agent_status,
        )

        return {
            "case_id": case_id,
            "target_gate": target_gate,
            "baseline_status": baseline_status,
            "agent_status": agent_status,
            "agent_decision": accepted_decision.get("tool_name", "") if accepted_decision else "",
            "state_transition": state_transition,
            "llm_required": llm_required,
            "llm_helped": llm_helped,
            "llm_helped_type": "bool",
            "fixture_exists": True,
            "evidence_paths": verify_evidence.get("evidence_paths", []),
            "final_verify_passed": final_verify_passed,
            "evidence_path_exists": evidence_path_exists,
            "has_gate_artifact": self._check_gate_artifact_exists(agent),
        }

    def _get_stage_status(self, results: Dict, stage: str) -> str:
        """Get status for a specific stage from pipeline results."""
        if not results:
            return "no_results"

        # Check stage_status dict
        stage_status = results.get("stage_status", {})
        if stage in stage_status:
            info = stage_status[stage]
            if isinstance(info, dict):
                return info.get("status", "")
            return str(info)

        # Check direct stage result
        stage_result = results.get(stage, {})
        if isinstance(stage_result, dict):
            return stage_result.get("status", "")

        return "unknown"

    def _find_accepted_decision(self, decisions: List[Dict], target_gate: str) -> Optional[Dict]:
        """Find an accepted LLM decision for the target gate."""
        for decision in decisions:
            d = decision.get("decision", {})
            if d.get("stage") == target_gate and d.get("policy_allowed"):
                return d
        return None

    def _check_gate_artifact_exists(self, results: Dict) -> bool:
        """Check if gate artifacts were written."""
        artifacts = results.get("artifacts", {})
        return bool(artifacts.get("agent_decision_gates"))

    def _build_error_result(self, case_id: str, target_gate: str, error: str, fixture_dir: str) -> Dict:
        """Build an error result for a case."""
        return {
            "case_id": case_id,
            "target_gate": target_gate,
            "baseline_status": "error",
            "agent_status": "error",
            "agent_decision": "",
            "state_transition": "",
            "llm_required": False,
            "llm_helped": False,
            "llm_helped_type": "bool",
            "fixture_exists": False,
            "evidence_paths": [],
            "final_verify_passed": False,
            "evidence_path_exists": False,
            "has_gate_artifact": False,
            "error": error,
            "fixture_dir": fixture_dir,
        }

    def _build_summary(self, results: List[Dict]) -> Dict:
        """Build summary from per-case results."""
        total = len(results)
        llm_required_count = sum(1 for r in results if r.get("llm_required"))
        llm_helped_count = sum(1 for r in results if r.get("llm_helped"))
        safety_count = sum(1 for r in results if "malicious" in r.get("case_id", ""))

        gates_covered = set(r.get("target_gate") for r in results)

        # Verify all llm_helped are bool
        all_bool = all(isinstance(r.get("llm_helped"), bool) for r in results)

        return {
            "total_cases": total,
            "llm_required_count": llm_required_count,
            "llm_helped_count": llm_helped_count,
            "safety_cases": safety_count,
            "gates_covered": sorted(gates_covered),
            "llm_necessity_proven": llm_required_count > 0,
            "all_fixtures_exist": all(r.get("fixture_exists") for r in results),
            "all_llm_helped_are_bool": all_bool,
        }


def generate_report_from_manifest(manifest_path: str, output_path: str = None) -> Dict:
    """Convenience function for CLI integration."""
    evaluator = LLMNecessityEvaluator()
    return evaluator.evaluate_manifest(
        Path(manifest_path),
        Path(output_path) if output_path else None,
    )
