import json
from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import read_json, write_json
from auto_harness.models.result import StageResult
from auto_harness.observability.cost_profile import CostProfileCollector


class ReportGenerator:
    def generate(self, run_dir: Path, task: Dict, results: Dict[str, Dict], execution_audit: Dict = None) -> StageResult:
        report_path = run_dir / "reports" / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# auto-deploy-harness Deployment Report",
            "",
            "## Project",
            "",
            "- Name: `%s`" % task.get("project", {}).get("name", ""),
            "- Repo: `%s`" % task.get("project", {}).get("repo_url", ""),
            "",
            "## Stage Results",
            "",
        ]
        for stage, result in results.items():
            context = result.get("data", {}).get("control_context", {}) if isinstance(result.get("data"), dict) else {}
            skill_names = [item.get("name") for item in context.get("selected_skills", [])]
            memory_ids = [item.get("id") for item in context.get("memory_hits", [])]
            lines.extend([
                "### %s" % stage,
                "",
                "- Status: `%s`" % result.get("status", ""),
                "- Summary: %s" % result.get("summary", ""),
                "- Skills: %s" % (", ".join("`%s`" % name for name in skill_names if name) or "`none`"),
                "- Memory hits: %s" % (", ".join("`%s`" % item for item in memory_ids if item) or "`none`"),
                "",
            ])
        preflight = results.get("host_preflight", {}).get("data", {})
        if isinstance(preflight, dict) and preflight:
            capability = preflight.get("capabilities") or {}
            gpu = capability.get("gpu") or {}
            decision = preflight.get("compatibility_decision") or {}
            policy = preflight.get("policy") or {}
            lines.extend([
                "## Host Preflight",
                "",
                "- GPU probe status: `%s`" % gpu.get("status", ""),
                "- GPU devices: `%s`" % len(gpu.get("devices") or []),
                "- Environment backend: `%s`" % decision.get("backend", ""),
                "- Environment action: `%s`" % decision.get("action", ""),
                "- Target prefix: `%s`" % decision.get("target_prefix", ""),
                "- Compatibility status: `%s`" % decision.get("status", ""),
                "- Mutation authorized: `%s`" % str(bool(policy.get("mutation_authorized"))).lower(),
                "",
            ])
        verify = results.get("verify", {}).get("data", {})
        run_selection = self._run_candidate_selection(results)
        if run_selection:
            lines.extend([
                "## Run Candidate Selection",
                "",
                "- Command: `%s`" % " ".join(str(part) for part in run_selection.get("cmd", [])),
                "- Score: `%.2f`" % float(run_selection.get("score") or 0),
                "- Selected by: `%s`" % run_selection.get("selected_by", ""),
                "- Reasons:",
            ])
            for reason in run_selection.get("score_reasons") or []:
                lines.append("  - %s" % reason)
            lines.append("")
        if verify:
            lines.extend([
                "## Verify",
                "",
                "- Final status: `%s`" % verify.get("status", ""),
                "- Trace ID: `%s`" % verify.get("trace_id", ""),
                "- Next action: `%s`" % verify.get("next_action", ""),
                "",
            ])
        foundation_artifacts = self._write_deployment_foundation_artifacts(
            run_dir, results,
        )
        if foundation_artifacts:
            selection = verify.get("protocol_verify_selection") or {}
            lines.extend([
                "## Deployment Foundation",
                "",
                "- Deployability: `%s`" % (
                    results.get("analyze", {}).get("data", {})
                    .get("deployability", {}).get("status", "")
                ),
                "- Protocol verifier: `%s`" % selection.get("verifier_id", ""),
                "- Protocol selection source: `%s`" % selection.get("source", ""),
                "- Audit artifacts: `%s`" % len(foundation_artifacts),
                "",
            ])
        repair_rerun = self._repair_rerun_summary(run_dir, results)
        if repair_rerun:
            lines.extend([
                "## Repair Rerun Decision",
                "",
                "- Proposed rerun_from: `%s`" % repair_rerun.get("rerun_from_proposed", ""),
                "- Required safe rerun_from: `%s`" % repair_rerun.get("rerun_from_required", ""),
                "- Effective rerun_from: `%s`" % repair_rerun.get("rerun_from_effective", ""),
                "- Source: `%s`" % repair_rerun.get("rerun_from_source", ""),
            ])
            if repair_rerun.get("rerun_reason"):
                lines.append("- Reason: %s" % repair_rerun["rerun_reason"])
            if repair_rerun.get("rerun_from_adjustment_reason"):
                lines.append("- Adjustment: %s" % repair_rerun["rerun_from_adjustment_reason"])
            effectiveness = self._repair_effectiveness(run_dir)
            if effectiveness:
                lines.extend([
                    "- Repair applied: `%s`" % str(bool(effectiveness.get("repair_applied"))).lower(),
                    "- Repair executed: `%s`" % str(bool(effectiveness.get("repair_executed"))).lower(),
                    "- Repair verified: `%s`" % str(bool(effectiveness.get("repair_verified"))).lower(),
                ])
            lines.append("")
        repository_context = self._repository_context_summary(run_dir)
        if repository_context:
            lines.extend([
                "## Repository Context",
                "",
                "- Mode: `%s`" % repository_context.get("mode", ""),
                "- Initial estimated input tokens: `%s`"
                % repository_context.get("initial_input_tokens", 0),
                "- Observation rounds: `%s`"
                % repository_context.get("observation_rounds", 0),
                "- Observed files: `%s`"
                % repository_context.get("observed_files", 0),
                "- Observation tokens: `%s`"
                % repository_context.get("observation_tokens", 0),
                "- Cache hits: `%s`" % repository_context.get("cache_hits", 0),
                "- Rejected reads: `%s`"
                % repository_context.get("rejected_reads", 0),
                "",
            ])
        native_tools = self._native_tool_summary(run_dir)
        if native_tools:
            write_json(
                Path(run_dir) / "reports" / "native_tool_calling_summary.json",
                native_tools,
            )
            lines.extend([
                "## Native Tool Calling",
                "",
                "- Protocol: `%s`" % native_tools.get("provider_protocol", "native_tools"),
                "- Turns: `%s`" % native_tools.get("native_tool_turn_count", 0),
                "- Calls: `%s`" % native_tools.get("native_tool_call_count", 0),
                "- Accepted: `%s`" % native_tools.get("native_tool_call_accepted_count", 0),
                "- Rejected: `%s`" % native_tools.get("native_tool_call_rejected_count", 0),
                "- Reused: `%s`" % native_tools.get("native_tool_call_reused_count", 0),
                "- Conflicts: `%s`" % native_tools.get("native_tool_call_conflict_count", 0),
                "- Truncated results: `%s`" % native_tools.get("native_tool_result_truncated_count", 0),
                "- Stop reason: `%s`" % native_tools.get("stop_reason", ""),
                "",
            ])
        retrieval = self._read_optional(Path(run_dir) / "reports" / "retrieval_summary.json")
        if isinstance(retrieval, dict) and retrieval:
            lines.extend([
                "## Evidence Retrieval",
                "",
                "- Requested/effective mode: `%s` / `%s`" % (
                    retrieval.get("mode_requested", ""), retrieval.get("mode_effective", ""),
                ),
                "- Documents/chunks: `%s` / `%s`" % (
                    retrieval.get("documents", 0), retrieval.get("chunks", 0),
                ),
                "- Queries/hits: `%s` / `%s`" % (
                    retrieval.get("queries", 0), retrieval.get("hits_returned", 0),
                ),
                "- Exact reads/accepted grounding: `%s` / `%s`" % (
                    retrieval.get("hits_exactly_read", 0), retrieval.get("hits_grounding_accepted", 0),
                ),
                "- Retrieval tokens: `%s`" % retrieval.get("retrieval_tokens", 0),
                "- Latency P50/P95 ms: `%s` / `%s`" % (
                    retrieval.get("latency_ms_p50", 0), retrieval.get("latency_ms_p95", 0),
                ),
                "- Degraded: `%s`" % str(bool(retrieval.get("degraded"))).lower(),
                "- RAG helped/required: `%s` / `%s`" % (
                    str(bool(retrieval.get("rag_helped"))).lower(),
                    str(bool(retrieval.get("rag_required"))).lower(),
                ),
                "- Evidence: `retrieval/index_manifest.json`, `retrieval/queries.jsonl`, `reports/retrieval_contribution.json`",
                "",
            ])
        command_authorization = self._command_authorization_summary(run_dir, results)
        if command_authorization:
            write_json(
                Path(run_dir) / "reports" / "command_authorization_summary.json",
                command_authorization,
            )
            lines.extend([
                "## Repository Command Authorization",
                "",
                "- Discovered candidates: `%s`" % command_authorization.get("candidate_count", 0),
                "- Policy decisions: `%s`" % command_authorization.get("decision_count", 0),
                "- Execution authorization attempts: `%s`" % command_authorization.get("attempt_count", 0),
                "- Candidate fallbacks: `%s`" % command_authorization.get("fallback_count", 0),
                "- Approval required: `%s`" % str(bool(command_authorization.get("approval_required"))).lower(),
                "- Final reason: `%s`" % command_authorization.get("final_reason", ""),
                "",
            ])
        agent_metrics = self._read_optional(run_dir / "reports" / "agent_metrics.json")
        metrics = agent_metrics.get("agent_metrics") if isinstance(agent_metrics, dict) else {}
        if metrics:
            lines.extend([
                "## Agent Metrics",
                "",
                "- LLM calls: `%s`" % metrics.get("llm_call_count", 0),
                "- Accepted actions: `%s`" % metrics.get("accepted_action_count", 0),
                "- Rejected actions: `%s`" % metrics.get("rejected_action_count", 0),
                "- Executed actions: `%s`" % metrics.get("executed_action_count", 0),
                "- Repair attempts: `%s`" % metrics.get("repair_attempt_count", 0),
                "- Auto resume count: `%s`" % metrics.get("auto_resume_count", 0),
                "- Verify candidate count: `%s`" % metrics.get("verify_candidate_count", 0),
                "- Agent helped: `%s`" % str(bool(metrics.get("agent_helped"))).lower(),
                "- Help type: %s" % (", ".join("`%s`" % item for item in metrics.get("help_type") or []) or "`none`"),
                "",
            ])
        cost_lines = self._cost_summary_lines(run_dir)
        if cost_lines:
            lines.extend(cost_lines)
        contribution = self._read_optional(run_dir / "reports" / "agent_contribution.json")
        if isinstance(contribution, dict) and contribution:
            lines.extend([
                "## Agent Contribution",
                "",
                "- LLM required: `%s`" % str(bool(contribution.get("llm_required"))).lower(),
                "- LLM helped: `%s`" % str(bool(contribution.get("llm_helped"))).lower(),
                "- Selection source: `%s`" % contribution.get("selection_source", ""),
                "- Reason: %s" % contribution.get("llm_required_reason", ""),
                "- Help type: %s" % (", ".join("`%s`" % item for item in contribution.get("help_type") or []) or "`none`"),
                "- Agent artifacts: `agent_steps.jsonl`, `agent_state.json`, `agent_plan.json`, `agent_plan_revisions.jsonl`",
                "",
            ])
        # Agent Summary section
        agent_summary = self._build_agent_summary(run_dir, results, contribution)
        if agent_summary:
            lines.extend([
                "## Agent Summary",
                "",
                "- Mode: `%s`" % agent_summary.get("mode", "off"),
                "- Runtime loop position: `%s`" % agent_summary.get("runtime_loop_position", "off"),
                "- LLM helped: `%s`" % str(bool(agent_summary.get("llm_helped"))).lower(),
                "- LLM required: `%s`" % str(bool(agent_summary.get("llm_required"))).lower(),
                "- LLM required status: `%s`" % agent_summary.get("llm_required_status", "unknown_without_baseline"),
                "- Helped stages: %s" % (", ".join("`%s`" % s for s in agent_summary.get("helped_stages") or []) or "`none`"),
                "- Accepted tool calls: `%s`" % agent_summary.get("accepted_tool_calls", 0),
                "- Rejected tool calls: `%s`" % agent_summary.get("rejected_tool_calls", 0),
                "- Final verify status: `%s`" % agent_summary.get("final_verify_status", ""),
                "",
            ])
        llm_required_evidence = self._read_optional(run_dir / "reports" / "llm_required_evidence.json")
        if isinstance(llm_required_evidence, dict) and llm_required_evidence:
            lines.extend([
                "## LLM Required Evidence",
                "",
                "- Baseline status: `%s`" % llm_required_evidence.get("baseline_status", ""),
                "- Agent status: `%s`" % llm_required_evidence.get("agent_status", ""),
                "- LLM required: `%s`" % str(bool(llm_required_evidence.get("llm_required"))).lower(),
                "",
            ])
        verified_memory = self._read_optional(run_dir / "reports" / "verified_memory.json")
        if isinstance(verified_memory, dict):
            lines.extend([
                "## Verified Memory",
                "",
                "- Recorded: `%s`" % str(bool(verified_memory.get("recorded"))).lower(),
                "- Memory id: `%s`" % verified_memory.get("memory_id", ""),
                "- Repair action hash: `%s`" % verified_memory.get("repair_action_hash", ""),
                "- Verification trace id: `%s`" % verified_memory.get("verification_trace_id", ""),
                "- Reason: %s" % verified_memory.get("reason", ""),
                "",
            ])
        # Skill Usage / Effects / Outcomes
        skill_sections = self._build_skill_sections(run_dir, results)
        if skill_sections:
            lines.extend(skill_sections)
        execution_audit = execution_audit or self._read_optional(run_dir / "reports" / "execution_audit.json")
        if isinstance(execution_audit, dict) and execution_audit:
            lines.extend([
                "## Execution Audit",
                "",
                "- Requested start stage: `%s`" % execution_audit.get("requested_start_stage", ""),
                "- Effective start stage: `%s`" % execution_audit.get("effective_start_stage", ""),
                "- Fallback applied: `%s`" % str(bool(execution_audit.get("fallback_applied"))).lower(),
                "- Reused stages: %s" % self._format_stage_list(execution_audit.get("reused_stages") or []),
                "- Rerun stages: %s" % self._format_stage_list(execution_audit.get("rerun_stages") or []),
                "",
            ])
        required_env = self._required_env_vars(run_dir, results)
        if required_env:
            lines.extend([
                "## Required Environment Variables",
                "",
                "The following variable names may be required by the deployment. Values are not recorded in reports.",
                "",
            ])
            for name in required_env:
                lines.append("- `%s`: value required from operator or secret manager" % name)
            lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return StageResult("report", "passed", "report generated", {"report_path": str(report_path)}, evidence=[str(report_path)])

    def _write_deployment_foundation_artifacts(
        self, run_dir: Path, results: Dict[str, Dict],
    ) -> List[str]:
        analysis = results.get("analyze", {}).get("data", {})
        verify = results.get("verify", {}).get("data", {})
        if not isinstance(analysis, dict) or not analysis.get("schema_version"):
            return []
        reports = Path(run_dir) / "reports"
        common = {
            "schema_version": 1,
            "repository_fingerprint": str(
                analysis.get("repository_fingerprint") or ""
            ),
            "config_hash": str(
                analysis.get("deployment_foundation_config_hash") or ""
            ),
        }
        artifacts = {
            "project_capabilities.json": analysis.get("capabilities") or {},
            "capability_evidence.json": analysis.get("capability_evidence") or [],
            "deployment_contract.json": analysis.get("deployment_contract") or {},
            "adapter_detections.json": analysis.get("adapter_detections") or [],
            "deployment_candidates.json": analysis.get("deployment_candidates") or [],
            "deployability_assessment.json": analysis.get("deployability") or {},
            "protocol_verify_selection.json": (
                verify.get("protocol_verify_selection") or {
                    "status": "not_run",
                    "reason": "verify stage did not produce protocol selection",
                }
            ),
        }
        paths = []
        for name in sorted(artifacts):
            path = reports / name
            write_json(path, {**common, "data": artifacts[name]})
            paths.append(str(path))
        return paths

    def _native_tool_summary(self, run_dir: Path) -> Dict:
        existing = self._read_optional(
            Path(run_dir) / "reports" / "native_tool_calling_summary.json"
        )
        if isinstance(existing, dict) and existing:
            return existing
        root = Path(run_dir) / "agent_tool_calls"
        calls_path = root / "calls.jsonl"
        if not calls_path.exists():
            return {}
        records = []
        try:
            for line in calls_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(item, dict):
                    records.append(item)
        except OSError:
            return {}
        latest = {}
        for record in records:
            latest[str(record.get("call_id", ""))] = record
        calls = list(latest.values())
        turns_dir = root / "turns"
        return {
            "schema_version": 1,
            "provider_protocol": "native_tools",
            "native_tool_turn_count": len(list(turns_dir.glob("turn_*.json"))) if turns_dir.exists() else 0,
            "native_tool_call_count": len(calls),
            "native_tool_call_accepted_count": sum(
                1 for item in calls if item.get("policy_verdict") == "allowed"
            ),
            "native_tool_call_rejected_count": sum(
                1 for item in calls if item.get("policy_verdict") == "rejected"
            ),
            "native_tool_call_reused_count": 0,
            "native_tool_call_conflict_count": 0,
            "native_tool_result_truncated_count": 0,
            "native_tool_loop_limit_count": 0,
            "stop_reason": "",
        }

    def _repository_context_summary(self, run_dir: Path) -> Dict:
        reports = Path(run_dir) / "reports"
        snapshot = self._read_optional(reports / "project_snapshot.json")
        if not isinstance(snapshot, dict) or snapshot.get("context_mode") != "layered":
            return {}
        records = []
        ledger = reports / "observation_ledger.jsonl"
        if ledger.exists():
            import json
            for line in ledger.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                try:
                    item = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(item, dict):
                    records.append(item)
        observed_paths = set()
        for record in records:
            evidence = record.get("evidence", {})
            candidates = []
            if isinstance(evidence, dict):
                candidates.extend(evidence.get("files", []) or [])
                candidates.extend(evidence.get("results", []) or [])
            for item in candidates:
                if isinstance(item, dict) and item.get("path"):
                    observed_paths.add(item["path"])
        initial_tokens = 0
        turns = sorted((reports / "planner_turns").glob("turn_*.json"))
        if turns:
            first = self._read_optional(turns[0])
            context = first.get("context", {}) if isinstance(first, dict) else {}
            initial_tokens = int(context.get("estimated_input_tokens", 0) or 0)
        return {
            "mode": "layered",
            "initial_input_tokens": initial_tokens,
            "observation_rounds": max(
                [int(item.get("round", 0) or 0) for item in records] or [0]
            ),
            "observed_files": len(observed_paths),
            "observation_tokens": sum(
                int(item.get("content_tokens", 0) or 0) for item in records
            ),
            "cache_hits": sum(bool(item.get("cache_hit")) for item in records),
            "rejected_reads": sum(
                item.get("status") == "rejected" for item in records
            ),
        }

    def _command_authorization_summary(self, run_dir: Path, results: Dict[str, Dict]) -> Dict:
        reports = Path(run_dir) / "reports"
        registry = self._read_optional(reports / "command_registry.json")
        policy = self._read_optional(reports / "command_policy.json")
        attempts = []
        attempts_path = reports / "command_attempts.jsonl"
        if attempts_path.exists():
            for line in attempts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    item = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(item, dict):
                    attempts.append(item)
        fallback_count = 0
        fallbacks_path = reports / "command_fallbacks.jsonl"
        if fallbacks_path.exists():
            fallback_count = sum(
                bool(line.strip()) for line in fallbacks_path.read_text(
                    encoding="utf-8", errors="ignore",
                ).splitlines()
            )
        if not registry and not policy and not attempts and not fallback_count:
            return {}
        final = attempts[-1] if attempts else {}
        return {
            "candidate_count": len(registry.get("candidates", [])) if isinstance(registry, dict) else 0,
            "decision_count": len(policy.get("command_decisions", [])) if isinstance(policy, dict) else 0,
            "attempt_count": len(attempts),
            "fallback_count": fallback_count,
            "approval_required": bool(
                isinstance(policy, dict) and policy.get("approval_request")
            ),
            "final_reason": final.get("reason_code", "") or (
                policy.get("status", "") if isinstance(policy, dict) else ""
            ),
        }

    def _repair_rerun_summary(self, run_dir: Path, results: Dict[str, Dict]) -> Dict:
        plan = self._read_optional(run_dir / "repairs" / "repair_plan.json")
        if isinstance(plan, dict) and (plan.get("rerun_from_proposed") or plan.get("rerun_from_effective") or plan.get("rerun_from")):
            return {
                "rerun_from_proposed": plan.get("rerun_from_proposed", ""),
                "rerun_from_required": plan.get("rerun_from_required", ""),
                "rerun_from_effective": plan.get("rerun_from_effective") or plan.get("rerun_from", ""),
                "rerun_from_source": plan.get("rerun_from_source", ""),
                "rerun_reason": plan.get("rerun_reason", ""),
                "rerun_from_adjustment_reason": plan.get("rerun_from_adjustment_reason", ""),
            }
        for result in results.values():
            data = result.get("data") if isinstance(result, dict) else {}
            loop = data.get("agent_loop") if isinstance(data, dict) else {}
            if isinstance(loop, dict) and loop.get("next_rerun_from"):
                return {
                    "rerun_from_proposed": "",
                    "rerun_from_required": "",
                    "rerun_from_effective": loop.get("next_rerun_from", ""),
                    "rerun_from_source": "agent_loop",
                    "rerun_reason": "",
                    "rerun_from_adjustment_reason": "",
                }
        return {}

    def _build_agent_summary(self, run_dir: Path, results: Dict, contribution: Dict = None) -> Dict:
        """Build agent summary for the report.

        Returns dict with mode, runtime_loop_position, llm_helped, llm_required,
        llm_required_status, helped_stages, accepted_tool_calls, rejected_tool_calls,
        final_verify_status.
        """
        # Get mode from agent_state.json
        agent_state = self._read_optional(run_dir / "agent_state.json")
        mode = "off"
        if isinstance(agent_state, dict):
            mode = agent_state.get("mode", "off")

        # Get runtime_loop_position from config or agent_loop_result
        loop_result = self._read_optional(run_dir / "reports" / "agent_loop_result.json")
        runtime_loop_position = "off"
        if isinstance(loop_result, dict) and loop_result.get("status"):
            runtime_loop_position = "primary" if loop_result.get("mode") == "gated_actor" else "post_pipeline"

        # Get contribution data
        if contribution is None:
            contribution = self._read_optional(run_dir / "reports" / "agent_contribution.json") or {}

        llm_helped = bool(contribution.get("llm_helped"))
        llm_required = bool(contribution.get("llm_required"))
        llm_required_status = contribution.get("llm_required_status", "unknown_without_baseline")

        # Get helped stages from gate results
        gate_results = contribution.get("gate_summary") or []
        helped_stages = [
            g.get("stage") for g in gate_results
            if g.get("policy_allowed") and (g.get("executed") or g.get("applied"))
        ]

        # Get accepted/rejected counts
        accepted_tool_calls = int(contribution.get("accepted_action_count") or 0)
        rejected_tool_calls = int(contribution.get("rejected_action_count") or 0)

        # Get final verify status
        final_verify_status = contribution.get("final_verify_status", "")
        if not final_verify_status:
            verify_result = results.get("verify", {})
            if isinstance(verify_result, dict):
                final_verify_status = verify_result.get("status", "")

        return {
            "mode": mode,
            "runtime_loop_position": runtime_loop_position,
            "llm_helped": llm_helped,
            "llm_required": llm_required,
            "llm_required_status": llm_required_status,
            "helped_stages": helped_stages,
            "accepted_tool_calls": accepted_tool_calls,
            "rejected_tool_calls": rejected_tool_calls,
            "final_verify_status": final_verify_status,
        }

    def _repair_effectiveness(self, run_dir: Path) -> Dict:
        apply_result = self._read_optional(run_dir / "repairs" / "repair_apply_result.json")
        return apply_result if isinstance(apply_result, dict) else {}

    def _run_candidate_selection(self, results: Dict[str, Dict]) -> Dict:
        runner = results.get("runner", {}).get("data", {})
        if isinstance(runner, dict) and isinstance(runner.get("candidate_selection"), dict):
            return runner["candidate_selection"]
        analyze = results.get("analyze", {}).get("data", {})
        candidates = analyze.get("run_candidates") if isinstance(analyze, dict) else []
        if candidates:
            candidate = candidates[0]
            return {
                "cmd": candidate.get("cmd", []),
                "score": float(candidate.get("score") or candidate.get("confidence") or 0),
                "score_reasons": list(candidate.get("score_reasons") or []),
                "selected_by": candidate.get("selected_by") or candidate.get("preferred_by") or "deterministic",
            }
        return {}

    def _format_stage_list(self, stages) -> str:
        names = [stage for stage in stages if isinstance(stage, str) and stage]
        return ", ".join("`%s`" % stage for stage in names) or "`none`"

    def _required_env_vars(self, run_dir: Path, results: Dict[str, Dict]) -> List[str]:
        names = set()
        repair_required = self._read_optional(run_dir / "repairs" / "required_env_vars.json")
        if isinstance(repair_required, dict):
            names.update(self._safe_names(repair_required.get("env_vars") or []))
        repair_plan = self._read_optional(run_dir / "repairs" / "repair_plan.json")
        if isinstance(repair_plan, dict):
            for action in repair_plan.get("actions") or []:
                payload = action.get("payload") if isinstance(action, dict) else {}
                if isinstance(payload, dict):
                    names.update(self._safe_names(payload.get("env_vars") or []))
        for result in results.values():
            data = result.get("data") if isinstance(result, dict) else {}
            if not isinstance(data, dict):
                continue
            diagnosis = data.get("diagnosis")
            if isinstance(diagnosis, dict) and diagnosis.get("category") == "auth_required":
                names.update(self._safe_names(diagnosis.get("required_env_vars") or []))
            names.update(self._safe_names(data.get("external_tokens") or []))
        return sorted(names)

    def _safe_names(self, values) -> List[str]:
        safe = []
        for value in values:
            if not isinstance(value, str):
                continue
            if value and value.upper() == value and all(ch.isalnum() or ch == "_" for ch in value):
                safe.append(value)
        return safe

    def _build_skill_sections(self, run_dir: Path, results: Dict[str, Dict]) -> List[str]:
        """Build Skill Usage, Effects, and Outcomes report sections.

        Returns list of markdown lines, or empty list if no skill data.
        """
        lines: List[str] = []

        # Skill Usage: collect from stage control_context
        skill_usage: Dict[str, List[str]] = {}  # stage -> [skill names]
        for stage, result in results.items():
            context = result.get("data", {}).get("control_context", {}) if isinstance(result.get("data"), dict) else {}
            selected = context.get("selected_skills", [])
            if selected:
                names = []
                for s in selected:
                    name = s.get("name", "")
                    version = s.get("version", "")
                    if name:
                        names.append("%s@%s" % (name, version) if version else name)
                if names:
                    skill_usage[stage] = names

        # Also check project_snapshot.json for plan_first skills
        snapshot = self._read_optional(run_dir / "reports" / "project_snapshot.json")
        if isinstance(snapshot, dict):
            snapshot_skills = snapshot.get("selected_skills", [])
            if snapshot_skills and "plan_first" not in skill_usage:
                names = []
                for s in snapshot_skills:
                    name = s.get("name", "")
                    version = s.get("version", "")
                    if name:
                        names.append("%s@%s" % (name, version) if version else name)
                if names:
                    skill_usage["plan_first"] = names

        if skill_usage:
            lines.extend(["## Skill Usage", ""])
            for stage, names in skill_usage.items():
                lines.append("- %s: %s" % (stage, ", ".join("`%s`" % n for n in names)))
            lines.append("")

        # Skill Effects: from skill_effects.json
        effects = self._read_optional(run_dir / "reports" / "skill_effects.json")
        if isinstance(effects, dict) and effects.get("effects"):
            lines.extend(["## Skill Effects", ""])
            for effect in effects["effects"]:
                name = effect.get("skill_name", "")
                field = effect.get("field_changed", "")
                accepted = effect.get("accepted_by_policy", False)
                effect_type = effect.get("effect_type", "")
                lines.append(
                    "- `%s` influenced `%s`; policy accepted: `%s` (%s)" % (
                        name, field, str(accepted).lower(), effect_type,
                    )
                )
            lines.append("")

        # Skill Outcomes: summary from skill_effects
        if isinstance(effects, dict) and effects.get("effects"):
            total_selected = len(set(e.get("skill_name", "") for e in effects["effects"]))
            influenced = [e for e in effects["effects"] if e.get("field_changed")]
            harmful = [e for e in effects["effects"] if not e.get("accepted_by_policy")]
            verify_result = results.get("verify", {})
            trace_verified = verify_result.get("status", "") in ("passed", "pass") if isinstance(verify_result, dict) else False

            lines.extend(["## Skill Outcomes", ""])
            lines.append("- Selected skills: `%d`" % total_selected)
            lines.append("- Influenced plan: `%d`" % len(influenced))
            lines.append("- Final trace verified: `%s`" % str(trace_verified).lower())
            lines.append("- Harmful skill effects: `%d`" % len(harmful))
            lines.append("")

        return lines

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def _cost_summary_lines(self, run_dir: Path) -> List[str]:
        """Render a Performance & Cost section from a fresh read-only profile.

        Pricing comes from the ambient HarnessConfig when one is loadable from
        the working directory; otherwise tokens and durations still render and
        only the monetary line is skipped. Any failure here must never break
        report generation, so every step is guarded.
        """
        try:
            collector = self._cost_profile_collector()
            profile = collector.collect(Path(run_dir))
        except Exception:
            return []
        tokens = profile.get("tokens") or {}
        coverage = tokens.get("coverage") or {}
        reported = tokens.get("provider_reported") or {}
        estimated = tokens.get("estimated") or {}
        latency = profile.get("llm_latency") or {}
        run_window = profile.get("run") or {}
        stage_rows = [
            stage for stage in profile.get("stages") or []
            if isinstance(stage.get("duration_ms"), int)
        ]
        if not coverage.get("calls_total") and not stage_rows:
            return []
        duration_ms = run_window.get("duration_ms")
        lines = [
            "## Performance & Cost",
            "",
            "- LLM calls: `%s` (with usage telemetry: `%s`)"
            % (coverage.get("calls_total", 0), coverage.get("calls_with_usage", 0)),
            "- Provider reported tokens: input `%s`, output `%s`, total `%s` (cache hit `%s`)"
            % (
                reported.get("input_tokens", 0),
                reported.get("output_tokens", 0),
                reported.get("total_tokens", 0),
                reported.get("cache_hit_tokens", 0),
            ),
            "- Estimated-only tokens (no provider telemetry): total `%s`"
            % estimated.get("total_tokens", 0),
            "- LLM wall time: total `%sms`, avg `%sms`, p95 `%sms`"
            % (latency.get("total_ms", 0), latency.get("avg_ms", 0), latency.get("p95_ms", 0)),
            "- End-to-end duration: %s"
            % ("`%sms`" % duration_ms if isinstance(duration_ms, int) else "`n/a`"),
        ]
        if stage_rows:
            lines.append(
                "- Stage durations: %s"
                % ", ".join("`%s=%sms`" % (stage["stage"], stage["duration_ms"]) for stage in stage_rows)
            )
        cost = profile.get("cost") or {}
        if cost.get("status") == "priced":
            lines.append(
                "- Estimated cost: `%s %s` (pricing as of `%s`, config-provided, not a billing source)"
                % (cost.get("currency"), cost.get("total_cost"), cost.get("pricing_as_of"))
            )
        else:
            lines.append("- Cost: not computed (%s)" % (cost.get("reason") or "unpriced"))
        lines.append("")
        return lines

    @staticmethod
    def _cost_profile_collector() -> CostProfileCollector:
        try:
            from auto_harness.config import HarnessConfig

            return CostProfileCollector(HarnessConfig.load().cost_profile)
        except Exception:
            return CostProfileCollector()
