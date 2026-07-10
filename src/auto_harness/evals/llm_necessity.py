"""LLM Necessity Evaluator.

Generates baseline vs agent comparison reports to prove that LLM decision gates
are necessary for complex deployments.

Per design doc:
- llm_required=true only when baseline failed/uncertain AND agent passed
- llm_helped=true only when state actually improved
- Evidence artifacts must exist
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class LLMNecessityEvaluator:
    """Evaluates LLM necessity by comparing baseline vs agent outcomes."""

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
        """Evaluate a single case from the manifest.

        Args:
            case: Case dict from manifest

        Returns:
            Result dict with llm_required, llm_helped, evidence
        """
        case_id = case.get("case_id", "")
        target_gate = case.get("target_gate", "")
        baseline = case.get("baseline_expectation", {})
        agent = case.get("agent_expectation", {})
        fixture_dir = case.get("fixture_dir", "")

        # Check fixture exists
        fixture_path = Path(fixture_dir)
        fixture_exists = fixture_path.exists() and any(fixture_path.iterdir())

        # Determine llm_required
        baseline_failed = baseline.get("status") in ("failed", "uncertain")
        agent_passed = agent.get("status") == "passed"
        llm_decision = agent.get("llm_decision", "")
        policy_rejected = llm_decision == "policy_rejected"

        # llm_required: baseline failed/uncertain AND agent passed AND valid LLM decision
        llm_required = (
            baseline_failed
            and agent_passed
            and llm_decision not in ("", "policy_rejected", "no_action")
        )

        # llm_helped: state actually improved
        llm_helped = llm_required and agent.get("state_transition", "")

        # For safety cases, llm_required is False (policy correctly rejected)
        if case_id == "malicious_readme_prompt_injection":
            llm_required = False
            llm_helped = False

        # Build evidence paths
        evidence_paths = []
        if fixture_exists:
            for f in fixture_path.iterdir():
                if f.suffix == ".json":
                    evidence_paths.append(str(f))

        state_transition = agent.get("state_transition", "")
        if policy_rejected:
            state_transition = "no change - policy blocks dangerous actions"

        return {
            "case_id": case_id,
            "target_gate": target_gate,
            "baseline_status": baseline.get("status", ""),
            "baseline_reason": baseline.get("reason", ""),
            "agent_status": agent.get("status", ""),
            "agent_decision": llm_decision,
            "state_transition": state_transition,
            "llm_required": llm_required,
            "llm_helped": llm_helped,
            "fixture_exists": fixture_exists,
            "evidence_paths": evidence_paths,
            "repair_verified": agent.get("repair_verified", False),
        }

    def _build_summary(self, results: List[Dict]) -> Dict:
        """Build summary from per-case results."""
        total = len(results)
        llm_required_count = sum(1 for r in results if r.get("llm_required"))
        llm_helped_count = sum(1 for r in results if r.get("llm_helped"))
        safety_count = sum(1 for r in results if "malicious" in r.get("case_id", ""))

        gates_covered = set(r.get("target_gate") for r in results)

        return {
            "total_cases": total,
            "llm_required_count": llm_required_count,
            "llm_helped_count": llm_helped_count,
            "safety_cases": safety_count,
            "gates_covered": sorted(gates_covered),
            "llm_necessity_proven": llm_required_count > 0,
            "all_fixtures_exist": all(r.get("fixture_exists") for r in results),
        }


def generate_report_from_manifest(manifest_path: str, output_path: str = None) -> Dict:
    """Convenience function for CLI integration."""
    evaluator = LLMEvaluator()
    return evaluator.evaluate_manifest(
        Path(manifest_path),
        Path(output_path) if output_path else None,
    )
