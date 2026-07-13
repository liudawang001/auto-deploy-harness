"""Plan Compiler for LLM plan-first deployment.

Transforms a policy-validated LLM deployment plan into the analysis dict
format that existing pipeline modules (env_deploy, runner, verify) consume.

Merge strategy:
- Deterministic facts are always preserved
- LLM candidates are inserted as high-priority
- Selected candidate goes first in run_candidates
- LLM verify hint overrides deterministic when policy allows
- Rejected items do not appear in compiled output
"""
from typing import Any, Dict, List, Optional


class PlanCompiler:
    """Compiles a normalized LLM deployment plan into pipeline-consumable analysis dict."""

    def compile(
        self,
        normalized_plan: Dict,
        deterministic_analysis: Optional[Dict] = None,
        resource_plan: Optional[Dict] = None,
    ) -> Dict:
        """Compile a normalized plan into the analysis dict format.

        Args:
            normalized_plan: The policy-gate-normalized deployment plan
            deterministic_analysis: Optional output from deterministic ProjectAnalyzer
            resource_plan: Optional output from ResourcePlanner

        Returns:
            Dict with 'analysis' and 'effective_plan' keys
        """
        deterministic_analysis = deterministic_analysis or {}
        resource_plan = resource_plan or {}

        # Start with deterministic analysis as the base
        analysis = dict(deterministic_analysis)

        # Preserve deterministic facts
        if "deterministic_facts" not in analysis:
            analysis["deterministic_facts"] = {
                "file_count": len(deterministic_analysis.get("files", [])),
                "frameworks": deterministic_analysis.get("frameworks", []),
                "has_requirements": "requirements.txt" in deterministic_analysis.get("files", []),
                "has_environment_yml": "environment.yml" in deterministic_analysis.get("files", []),
            }

        # Snapshot deterministic candidates before LLM merge
        if "deterministic_candidates" not in analysis:
            analysis["deterministic_candidates"] = list(deterministic_analysis.get("run_candidates", []))

        # Compile environment.install_commands -> analysis.install_plan
        env = normalized_plan.get("environment", {})
        install_commands = env.get("install_commands", [])
        if install_commands:
            analysis["install_plan"] = install_commands

        # Compile run.candidates -> analysis.run_candidates
        run = normalized_plan.get("run", {})
        llm_candidates = run.get("candidates", [])
        selected_id = run.get("selected_candidate_id", "")

        # Convert LLM candidates to the existing run_candidates format
        converted_llm_candidates = []
        for cand in llm_candidates:
            converted = {
                "id": cand.get("id", ""),
                "cmd": cand.get("cmd", []),
                "expected_port": int(cand.get("expected_port", 0)),
                "confidence": 0.9,
                "score": 0.9,
                "selected_by": "llm_plan_first",
                "score_reasons": [cand.get("reason", "LLM proposed candidate")],
            }
            # If this is the selected candidate, put it first
            if cand.get("id") == selected_id:
                converted["confidence"] = 0.95
                converted["score"] = 0.95
                converted_llm_candidates.insert(0, converted)
            else:
                converted_llm_candidates.append(converted)

        # Merge: LLM candidates first, then deterministic candidates
        deterministic_candidates = analysis.get("deterministic_candidates", [])
        # Remove deterministic candidates that duplicate LLM candidates
        llm_cmds = {tuple(c.get("cmd", [])) for c in converted_llm_candidates}
        non_duplicate_deterministic = [
            c for c in deterministic_candidates
            if tuple(c.get("cmd", [])) not in llm_cmds
        ]
        merged_candidates = converted_llm_candidates + non_duplicate_deterministic

        if merged_candidates:
            analysis["run_candidates"] = merged_candidates
            analysis["selected_candidate"] = merged_candidates[0]
            analysis["selection_source"] = "llm_plan_first"
        elif deterministic_candidates:
            analysis["run_candidates"] = deterministic_candidates
            analysis["selected_candidate"] = deterministic_candidates[0]
            analysis["selection_source"] = "deterministic"

        analysis["llm_candidates"] = converted_llm_candidates
        analysis["merged_candidates"] = analysis.get("run_candidates", [])

        # Compile verify.request -> analysis.verify_hint
        verify = normalized_plan.get("verify", {})
        verify_request = verify.get("request", {})
        if verify_request:
            verify_hint = {
                "service_type": verify.get("service_type", "http"),
                "expected_output": verify.get("success_evidence", "trace_echo"),
                "request": verify_request,
            }
            analysis["verify_hint"] = verify_hint

        # Compile environment strategy
        if env.get("backend"):
            analysis["environment_strategy"] = {
                "backend": env.get("backend", "venv"),
                "preferred_tool": env.get("backend", "venv"),
                "python": env.get("python", "3.10"),
                "channels": env.get("channels", []),
                "source": "llm_plan_first",
                "confidence": 0.85,
                "reasons": ["LLM deployment plan specified environment backend"],
            }

        # Compile model_assets
        model_assets = normalized_plan.get("model_assets", {})
        if model_assets:
            analysis["model_assets"] = model_assets

        # Add LLM plan metadata
        analysis["llm_plan"] = {
            "plan_id": normalized_plan.get("plan_id", ""),
            "policy_status": "accepted",
        }
        analysis["llm_required_reason"] = "LLM generated initial deployment plan and command candidates."

        # Ensure required fields exist
        analysis.setdefault("files", deterministic_analysis.get("files", []))
        analysis.setdefault("frameworks", deterministic_analysis.get("frameworks", []))
        analysis.setdefault("install_plan", deterministic_analysis.get("install_plan", []))
        analysis.setdefault("run_candidates", deterministic_analysis.get("run_candidates", []))
        analysis.setdefault("verify_hint", deterministic_analysis.get("verify_hint", {}))
        analysis.setdefault("selected_candidate", {})
        analysis.setdefault("selection_source", "none")

        return {
            "analysis": analysis,
            "effective_plan": normalized_plan,
        }
