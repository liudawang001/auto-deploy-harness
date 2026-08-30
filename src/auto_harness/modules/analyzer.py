import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agent import AgentActionPolicy, AgentDecisionEngine, AgentInputSanitizer, AgentObservation
from auto_harness.agents.base import AgentExecutor, AgentRequest
from auto_harness.capabilities import (
    CapabilityDetector,
    DeployabilityAssessor,
    LegacyAnalysisCompiler,
    order_run_candidates,
    score_run_candidates,
)
from auto_harness.agent_runtime.repository_inventory import RepositoryInventoryBuilder
from auto_harness.capabilities.shadow_diff import compute_shadow_diff, enforce_blockers
from auto_harness.command_auth.adapters.entrypoint import ENTRYPOINT_SOURCE_KINDS
from auto_harness.command_auth.discovery import CommandDiscoveryService
from auto_harness.command_auth import CommandAuthorizationEngine
from auto_harness.deployment_contract import (
    DeploymentContractCompiler,
    DeploymentContractParser,
)
from auto_harness.deployment_adapters import (
    CandidateComposer,
    DeploymentAdapterRegistry,
    DetectionContext,
)
from auto_harness.models.result import StageResult
from auto_harness.models.task import RuntimePolicy
from auto_harness.providers.json_utils import parse_json_object


class ProjectAnalyzer:
    def __init__(
        self,
        agent_executor: Optional[AgentExecutor] = None,
        use_agent: bool = False,
        agent_timeout_seconds: int = 900,
        stage_context: Optional[Dict] = None,
        agent_engine: Optional[AgentDecisionEngine] = None,
        agent_mode: str = "off",
        runtime_policy: Optional[RuntimePolicy] = None,
        task_id: str = "",
        agent_max_file_chars: int = 6000,
        deployment_capability_mode: str = "shadow",
        deployment_contract_enabled: bool = True,
        deployment_adapter_registry_enabled: bool = False,
        protocol_verify_registry_enabled: bool = False,
    ) -> None:
        self.agent_executor = agent_executor
        self.use_agent = use_agent
        self.agent_timeout_seconds = agent_timeout_seconds
        self.stage_context = stage_context or {}
        self.agent_engine = agent_engine
        self.agent_mode = agent_mode
        self.runtime_policy = runtime_policy or RuntimePolicy(workspace_root="")
        self.task_id = task_id
        self.agent_max_file_chars = agent_max_file_chars
        self.deployment_capability_mode = deployment_capability_mode
        self.deployment_contract_enabled = deployment_contract_enabled
        self.deployment_adapter_registry_enabled = deployment_adapter_registry_enabled
        self.protocol_verify_registry_enabled = protocol_verify_registry_enabled
        self.agent_policy = AgentActionPolicy()

    def analyze(self, repo_dir: Path) -> StageResult:
        files = self._collect_files(repo_dir)
        capabilities, dependency_manifests = CapabilityDetector().detect(repo_dir, files)
        manifest_paths = [item.path for item in dependency_manifests]
        if "auto-deploy.yaml" in files:
            manifest_paths.append("auto-deploy.yaml")
        repository_inventory = RepositoryInventoryBuilder().build(
            repo_dir,
            files,
            detected_signals={"dependency_files": sorted(set(manifest_paths))},
        )
        legacy_frameworks = self._detect_frameworks(repo_dir, files)
        adapter_context = DetectionContext(
            repo_dir=Path(repo_dir),
            files=tuple(files),
            capabilities=capabilities,
            legacy_frameworks=tuple(legacy_frameworks),
        )
        adapter_registry = DeploymentAdapterRegistry.builtins()
        adapter_detections = adapter_registry.detect_all(adapter_context)
        adapter_proposals = adapter_registry.proposals(
            adapter_context, adapter_detections,
        )
        frameworks = adapter_registry.legacy_frameworks(
            adapter_context, adapter_detections,
        )
        install_plan = self._install_plan(files)
        run_candidates = adapter_registry.legacy_run_candidates(
            adapter_context, adapter_detections,
        )
        self._normalize_candidate_ranking(run_candidates)
        verify_hint = adapter_registry.legacy_verify_hint(
            adapter_context, adapter_detections,
        )
        deployment_candidates = CandidateComposer().compose(
            adapter_proposals["run"],
            adapter_proposals["environment"],
            adapter_proposals["verify"],
        )
        # Phase B1: deterministic repository evidence discovery.  Candidate
        # merge is additive; every command still goes through the unified
        # authorization engine before it can execute.  The discovery registry
        # is only attached when it contributes a new entrypoint candidate so
        # legacy-only projects keep their existing deterministic path.
        # Phase B5 rollout modes: legacy/off keep the baseline chain (no
        # discovery merge, no registry attach, no scoring), shadow computes
        # the auditable diff, enforce fails closed on blockers.
        baseline_run_candidates = list(run_candidates)
        baseline_deployability = DeployabilityAssessor().assess(
            run_candidates=baseline_run_candidates,
            verify_hint=verify_hint,
            deployment_candidates=deployment_candidates,
        ).to_dict()
        baseline_snapshot = {
            "frameworks": frameworks,
            "install_plan": install_plan,
            "run_candidates": baseline_run_candidates,
            "verify_hint": verify_hint,
            "deployability": baseline_deployability,
        }
        legacy_mode = self.deployment_capability_mode in ("legacy", "off")
        entrypoint_registry_attached = False
        if legacy_mode:
            run_candidates = baseline_run_candidates
            entrypoint_discovery = {
                "deterministic_candidates": len(run_candidates),
                "discovery_source_kinds": [],
                "registry_candidates": 0,
                "rejections": [],
                "dockerfile_evidence": [],
                "skipped_reason": "deployment_capability_mode=%s" % (
                    self.deployment_capability_mode
                ),
            }
            discovery_registry = None
        else:
            discovery_registry = CommandDiscoveryService().discover(
                repo_dir,
                files,
                repository_inventory["repository_fingerprint"],
            )
            run_candidates, entrypoint_registry_attached = self._merge_entrypoint_candidates(
                run_candidates, discovery_registry,
            )
            score_run_candidates(
                run_candidates,
                detections=adapter_detections,
                registry=discovery_registry,
            )
            run_candidates = order_run_candidates(run_candidates)
            entrypoint_discovery = self._entrypoint_discovery_summary(
                run_candidates, discovery_registry,
            )
        data: Dict = {
            "files": files[:200],
            "frameworks": frameworks,
            "install_plan": install_plan,
            "run_candidates": run_candidates,
            "verify_hint": verify_hint,
            "environment_strategy": self._environment_strategy(files, frameworks),
            "deterministic_facts": {
                "file_count": len(files),
                "frameworks": frameworks,
                "has_requirements": "requirements.txt" in files,
                "has_environment_yml": "environment.yml" in files or "environment.yaml" in files,
            },
            "deterministic_candidates": [dict(candidate) for candidate in run_candidates],
            "llm_hypotheses": [],
            "llm_candidates": [],
            "merged_candidates": run_candidates,
            "selected_candidate": run_candidates[0] if run_candidates else {},
            "selection_source": "deterministic" if run_candidates else "none",
            "llm_required_reason": "LLM planner disabled or no material contribution.",
            "repository_fingerprint": repository_inventory[
                "repository_fingerprint"
            ],
            "entrypoint_discovery": entrypoint_discovery,
            "adapter_detections": [
                item.to_dict() for item in adapter_detections if item.matched
            ],
            "adapter_proposals": {
                name: [item.to_dict() for item in items]
                for name, items in adapter_proposals.items()
            },
        }
        feature_modes = {
            "deployment_capability_mode": self.deployment_capability_mode,
            "deployment_contract_enabled": self.deployment_contract_enabled,
            "deployment_adapter_registry_enabled": self.deployment_adapter_registry_enabled,
            "protocol_verify_registry_enabled": self.protocol_verify_registry_enabled,
        }
        data["deployment_foundation_config"] = feature_modes
        data["deployment_foundation_config_hash"] = hashlib.sha256(
            json.dumps(feature_modes, sort_keys=True).encode("utf-8")
        ).hexdigest()
        deployability = DeployabilityAssessor().assess(
            run_candidates=run_candidates,
            verify_hint=verify_hint,
            deployment_candidates=deployment_candidates,
        )
        data = LegacyAnalysisCompiler().compile(
            data,
            capabilities=capabilities,
            manifests=dependency_manifests,
            deployability=deployability,
            deployment_candidates=deployment_candidates,
        )
        contract_result = (
            DeploymentContractParser().parse_repo(repo_dir)
            if self.deployment_contract_enabled
            else {
                "found": False,
                "valid": False,
                "path": "auto-deploy.yaml",
                "disabled": True,
            }
        )
        data["deployment_contract"] = self._contract_snapshot(contract_result)
        if contract_result.get("valid"):
            registry, deployment_candidate = (
                DeploymentContractCompiler().compile_registry(
                    repo_dir,
                    contract_result["contract"],
                    discovery_registry,
                )
            )
            data = DeploymentContractCompiler().compile_analysis(
                data,
                contract_result["contract"],
                registry,
                deployment_candidate,
            )
            data["command_registry_scope"] = "contract"
            data["deployability"] = DeployabilityAssessor().assess(
                run_candidates=data["run_candidates"],
                verify_hint=data["verify_hint"],
                deployment_candidates=[deployment_candidate],
            ).to_dict()
        elif contract_result.get("found"):
            data["deployability"] = {
                **data["deployability"],
                "status": "blocked",
                "missing_capabilities": ["deployment_contract.valid"],
                "risk_reasons": [str(contract_result.get("reason_code") or "invalid_contract")],
                "next_resolution": "fix_contract",
            }
        if self.stage_context:
            data["control_context"] = self.stage_context
        if entrypoint_registry_attached:
            data["command_registry"] = discovery_registry.to_dict()
            data["command_registry_scope"] = "discovery"
        if legacy_mode:
            data["rollout_shadow_diff"] = {
                "mode": self.deployment_capability_mode,
                "computed": False,
            }
        else:
            candidate_snapshot = {
                "frameworks": frameworks,
                "install_plan": install_plan,
                "run_candidates": run_candidates,
                "verify_hint": verify_hint,
                "deployability": data.get("deployability") or {},
                "authorization_attempts": self._authorization_preview(
                    run_candidates, discovery_registry, repo_dir,
                ),
            }
            diff = compute_shadow_diff(baseline_snapshot, candidate_snapshot)
            diff["mode"] = self.deployment_capability_mode
            diff["computed"] = True
            data["rollout_shadow_diff"] = diff
            if self.deployment_capability_mode == "enforce":
                blockers = enforce_blockers(diff)
                data["rollout_enforce_blockers"] = blockers
                if blockers:
                    data["deployability"] = {
                        **data.get("deployability", {}),
                        "status": "blocked",
                        "risk_reasons": list(
                            (data.get("deployability") or {}).get("risk_reasons") or []
                        ) + blockers,
                        "next_resolution": "human_input",
                    }
        agent_advice = self._agent_advice(repo_dir, data)
        if agent_advice:
            data["agent_advice"] = agent_advice
        agent_decision = self._agent_planner(repo_dir, data)
        if agent_decision:
            data["agent_decision"] = agent_decision
        self._refresh_agentic_summary(data)
        return StageResult(
            stage="analyze",
            status="passed",
            summary="project analysis completed",
            data=data,
            evidence=[],
        )

    @staticmethod
    def _contract_snapshot(result: Dict) -> Dict:
        snapshot = {
            "found": bool(result.get("found")),
            "valid": bool(result.get("valid")),
            "path": str(result.get("path") or "auto-deploy.yaml"),
        }
        if result.get("reason_code"):
            snapshot["reason_code"] = str(result["reason_code"])
        if result.get("disabled"):
            snapshot["disabled"] = True
        contract = result.get("contract")
        if result.get("valid") and contract is not None:
            snapshot.update(contract.to_dict())
        return snapshot

    @staticmethod
    def _authorization_preview(run_candidates: List[Dict], registry, repo_dir: Path) -> List[Dict]:
        """Pre-execute authorization verdicts used by the shadow diff.

        This is a read-only preview over the discovery registry; it never
        executes commands and never grants more than the runner would.
        """
        if registry is None or not registry.candidates:
            return []
        engine = CommandAuthorizationEngine()
        preview = []
        for item in run_candidates:
            declared = None
            wanted_id = item.get("command_candidate_id")
            if wanted_id:
                declared = next(
                    (c for c in registry.candidates if c.candidate_id == wanted_id),
                    None,
                )
            if declared is None:
                declared = registry.candidate_for_argv(item.get("cmd") or [])
            if declared is None:
                continue
            decision = engine.authorize(
                declared,
                registry,
                repo_dir=Path(repo_dir) if repo_dir else None,
            )
            preview.append({
                "normalized_argv": list(declared.argv),
                "verdict": decision.verdict,
                "reason_code": decision.reason_code,
            })
        return preview

    @staticmethod
    def _entrypoint_discovery_summary(run_candidates: List[Dict], registry) -> Dict:
        source_kinds = sorted({
            str(item.get("source_kind"))
            for item in run_candidates
            if item.get("source_kind")
        })
        return {
            "deterministic_candidates": len(run_candidates),
            "discovery_source_kinds": source_kinds,
            "registry_candidates": len(registry.candidates),
            "rejections": list(registry.discovery_rejections or []),
            "dockerfile_evidence": [
                item.to_dict() for item in registry.evidence
                if item.source_type in {"dockerfile_entrypoint", "dockerfile_expose"}
            ],
        }

    @staticmethod
    def _merge_entrypoint_candidates(run_candidates: List[Dict], registry):
        """Merge Phase B1 discovery candidates into the deterministic list.

        Only evidence-bound entrypoint source kinds join the deterministic
        pipeline; authorization still decides what may execute.  A colliding
        legacy candidate is enriched with the registry binding so it is
        authorized through the same evidence-bound path.  Returns the merged
        list plus whether the discovery registry must be attached: enriching
        a candidate with ``command_candidate_id`` without the registry would
        bypass authorization, so the two always go together.
        """
        merged = list(run_candidates)
        by_argv = {tuple(item.get("cmd") or []): item for item in merged}
        added = 0
        enriched = 0
        for item in registry.candidates:
            if item.phase != "run" or item.source_kind not in ENTRYPOINT_SOURCE_KINDS:
                continue
            binding = {
                "source_kind": item.source_kind,
                "command_candidate_id": item.candidate_id,
                "selected_by": "repository_evidence",
                "cwd": item.cwd,
            }
            existing = by_argv.get(tuple(item.argv))
            if existing is not None:
                if existing.get("command_candidate_id"):
                    continue
                existing.update(binding)
                enriched += 1
                continue
            by_argv[tuple(item.argv)] = {
                "cmd": list(item.argv),
                "expected_port": int(getattr(item, "expected_port", 0) or 0),
                "confidence": float(item.score or 0),
                "score": float(item.score or 0),
                "source": "repository_evidence",
                "score_reasons": list(item.score_reasons or []),
                **binding,
            }
            merged.append(by_argv[tuple(item.argv)])
            added += 1
        return merged, bool(added or enriched)

    def _collect_files(self, repo_dir: Path) -> List[str]:
        result: List[str] = []
        for path in repo_dir.rglob("*"):
            if ".git" in path.parts or path.is_dir():
                continue
            try:
                result.append(str(path.relative_to(repo_dir)))
            except ValueError:
                continue
        return sorted(result)

    def _detect_frameworks(self, repo_dir: Path, files: List[str]) -> List[str]:
        frameworks: List[str] = []
        text = ""
        for name in ("requirements.txt", "pyproject.toml", "README.md", "readme.md", "package.json"):
            path = repo_dir / name
            if path.exists():
                try:
                    text += "\n" + path.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    pass
        for name in files:
            if name.endswith(".py"):
                try:
                    text += "\n" + (repo_dir / name).read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    pass
        for key in ("gradio", "streamlit", "fastapi", "flask", "torch", "transformers", "vllm"):
            if key in text:
                frameworks.append(key)
        if "httpserver" in text or "basehttprequesthandler" in text or "from http.server" in text:
            frameworks.append("http.server")
        if "openai-compatible" in text or "openai compatible" in text or "/v1/chat/completions" in text:
            frameworks.append("openai_compatible")
        if "package.json" in files:
            frameworks.append("node")
        if not frameworks:
            frameworks.append("unknown")
        return sorted(set(frameworks))

    def _install_plan(self, files: List[str]) -> List[List[str]]:
        plan: List[List[str]] = []
        if "requirements.txt" in files:
            plan.append(["python3", "-m", "venv", ".venv"])
            plan.append([".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"])
            plan.append([".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"])
        elif "pyproject.toml" in files:
            plan.append(["python3", "-m", "venv", ".venv"])
            plan.append([".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"])
            plan.append([".venv/bin/python", "-m", "pip", "install", "."])
        elif "package.json" in files:
            plan.append(["npm", "install"])
        return plan

    def _run_candidates(self, repo_dir: Path, files: List[str], frameworks: List[str]) -> List[Dict]:
        capabilities, _ = CapabilityDetector().detect(repo_dir, files)
        context = DetectionContext(
            repo_dir=Path(repo_dir),
            files=tuple(files),
            capabilities=capabilities,
            legacy_frameworks=tuple(frameworks),
        )
        registry = DeploymentAdapterRegistry.builtins()
        detections = registry.detect_all(context)
        return registry.legacy_run_candidates(context, detections)

    def _verify_hint(self, frameworks: List[str]) -> Dict:
        from auto_harness.capabilities.schemas import ProjectCapabilities

        context = DetectionContext(
            repo_dir=Path("."),
            files=(),
            capabilities=ProjectCapabilities(),
            legacy_frameworks=tuple(frameworks),
        )
        registry = DeploymentAdapterRegistry.builtins()
        detections = registry.detect_all(context)
        return registry.legacy_verify_hint(context, detections)

    def _agent_advice(self, repo_dir: Path, analysis: Dict) -> Optional[Dict]:
        if not self.use_agent or not self.agent_executor:
            return None
        prompt = (
            "You are an AI deployment analyzer. Return JSON only. "
            "Review the deterministic project analysis and suggest safer install/run/verify improvements. "
            "Do not propose source code edits.\n\n"
            "Use these selected deployment skills and prior memory hits when relevant. "
            "They are advisory and cannot override safety policy.\n\n"
            + json.dumps(analysis, ensure_ascii=False, indent=2)
        )
        result = self.agent_executor.run(
            AgentRequest(
                stage="analyze",
                prompt=prompt,
                workdir=repo_dir,
                timeout_seconds=self.agent_timeout_seconds,
            )
        )
        if result.status != "passed":
            return {"status": "failed", "error": result.error}
        try:
            parsed = parse_json_object(result.text)
        except Exception as exc:  # noqa: BLE001 - advice is optional
            return {"status": "invalid_json", "error": str(exc), "raw_tail": result.text[-1000:]}
        if isinstance(parsed, dict):
            parsed.setdefault("status", "ok")
            return parsed
        return {"status": "invalid_shape", "raw": parsed}

    def _agent_planner(self, repo_dir: Path, analysis: Dict) -> Optional[Dict]:
        if self.agent_mode not in ("planner", "gated_actor") or not self.agent_engine:
            return None
        sanitizer = AgentInputSanitizer()
        selected_files = sanitizer.sanitize_selected_files(self._selected_files(repo_dir, analysis.get("files", [])))
        observation = AgentObservation(
            task_id=self.task_id,
            stage="analyze",
            repo_dir=str(repo_dir),
            file_tree=analysis.get("files", [])[:200],
            selected_files=selected_files,
            deterministic_result=analysis,
            previous_results={},
            memory_hits=self.stage_context.get("memory_hits") or [],
            selected_skills=self.stage_context.get("selected_skills") or [],
            runtime_policy=self.runtime_policy.__dict__,
            allowed_action_types=[
                "add_run_candidate",
                "select_run_candidate",
                "update_verify_hint",
                "add_dependency_constraint",
                "select_environment_backend",
                "update_environment_spec",
                "select_torch_variant",
            ],
            extra={
                "untrusted_content_risks": sanitizer.risks,
                "redactions": sanitizer.redactions,
            },
        )
        decision = self.agent_engine.decide(observation)
        policy = self.agent_policy.validate(decision, self.runtime_policy, mode=self.agent_mode)
        self.agent_engine.trace_writer.update_policy_result(decision.trace_path, policy)
        merged = self._merge_agent_decision(analysis, decision, policy)
        return {
            "status": decision.status,
            "confidence": decision.confidence,
            "summary": decision.summary,
            "accepted_actions": policy["accepted_actions"],
            "rejected_actions": policy["rejected_actions"],
            "merged": merged,
        }

    def _merge_agent_decision(self, analysis: Dict, decision, policy: Dict) -> Dict:
        merged = {
            "run_candidates_added": 0,
            "verify_hint_updated": False,
            "dependency_constraints_added": 0,
            "preferred_candidate_selected": False,
            "environment_strategy_updated": False,
            "torch_variant_updated": False,
            "candidate_rank_rejections": [],
            "skipped": False,
        }
        if decision.status != "ok" or decision.confidence < 0.5:
            merged["skipped"] = True
            return merged
        for action in policy.get("accepted_actions") or []:
            if float(action.get("confidence") or 0) < 0.5:
                continue
            action_type = action.get("type")
            payload = action.get("payload") or {}
            if action_type == "add_run_candidate":
                candidate = self._candidate_from_payload(payload, action)
                if candidate:
                    analysis.setdefault("run_candidates", []).append(candidate)
                    merged["run_candidates_added"] += 1
            elif action_type == "select_run_candidate":
                selected = self._select_run_candidate(analysis, payload, action)
                if selected:
                    merged["preferred_candidate_selected"] = True
                else:
                    merged["candidate_rank_rejections"].append({
                        "action_type": action_type,
                        "reason": "selected command does not match an existing run candidate",
                        "cmd": payload.get("cmd"),
                    })
            elif action_type == "update_verify_hint":
                verify_hint = payload.get("verify_hint") if isinstance(payload.get("verify_hint"), dict) else payload
                if verify_hint:
                    analysis["verify_hint"] = verify_hint
                    merged["verify_hint_updated"] = True
            elif action_type == "add_dependency_constraint":
                command = self._dependency_constraint_command(payload)
                if command:
                    analysis.setdefault("install_plan", []).append(command)
                    merged["dependency_constraints_added"] += 1
            elif action_type == "select_environment_backend":
                strategy = self._environment_strategy_from_payload(payload, action)
                if strategy:
                    analysis["environment_strategy"] = strategy
                    merged["environment_strategy_updated"] = True
            elif action_type == "update_environment_spec":
                strategy = dict(analysis.get("environment_strategy") or {})
                if payload.get("python"):
                    strategy["python"] = str(payload["python"])
                if payload.get("channels"):
                    strategy["channels"] = [str(item) for item in payload.get("channels") or []]
                if payload.get("conda_dependencies"):
                    strategy["conda_dependencies"] = [str(item) for item in payload.get("conda_dependencies") or []]
                if payload.get("pip_dependencies") or payload.get("packages"):
                    strategy["pip_dependencies"] = [str(item) for item in (payload.get("pip_dependencies") or payload.get("packages") or [])]
                strategy["source"] = "llm_planner"
                strategy.setdefault("reasons", []).append(action.get("reason") or "LLM updated environment spec")
                analysis["environment_strategy"] = strategy
                merged["environment_strategy_updated"] = True
            elif action_type == "select_torch_variant":
                variant = str(payload.get("variant") or payload.get("torch_variant") or "").lower()
                if variant in ("cpu", "cu118", "cu121"):
                    analysis.setdefault("environment_strategy", {})["torch_variant"] = variant
                    analysis["environment_strategy"]["torch_variant_source"] = "llm_planner"
                    merged["torch_variant_updated"] = True
        return merged

    def _refresh_agentic_summary(self, analysis: Dict) -> None:
        decision = analysis.get("agent_decision") if isinstance(analysis.get("agent_decision"), dict) else {}
        accepted = decision.get("accepted_actions") or []
        analysis["llm_hypotheses"] = [
            {
                "action_type": action.get("type"),
                "reason": action.get("reason", ""),
                "confidence": action.get("confidence", 0),
            }
            for action in accepted
            if action.get("type") in ("add_run_candidate", "select_run_candidate", "select_environment_backend", "update_environment_spec", "select_torch_variant")
        ]
        analysis["llm_candidates"] = [
            action.get("payload", {})
            for action in accepted
            if action.get("type") in ("add_run_candidate", "select_run_candidate")
        ]
        analysis["merged_candidates"] = analysis.get("run_candidates") or []
        selected = (analysis.get("run_candidates") or [{}])[0] if analysis.get("run_candidates") else {}
        analysis["selected_candidate"] = selected
        selected_by = selected.get("selected_by") or selected.get("preferred_by") or "deterministic"
        analysis["selection_source"] = "hybrid" if selected_by == "combined" else selected_by
        merged = decision.get("merged") if isinstance(decision.get("merged"), dict) else {}
        if any(merged.get(key) for key in ("run_candidates_added", "preferred_candidate_selected", "verify_hint_updated", "environment_strategy_updated", "torch_variant_updated")):
            analysis["llm_required_reason"] = "LLM materially changed candidate ranking, verify path, or environment strategy."

    def _environment_strategy(self, files: List[str], frameworks: List[str]) -> Dict:
        has_conda = "environment.yml" in files or "environment.yaml" in files
        if has_conda:
            return {
                "backend": "conda",
                "preferred_tool": "mamba",
                "python": "3.10",
                "channels": ["pytorch", "nvidia", "conda-forge"] if "torch" in frameworks else ["conda-forge"],
                "source": "deterministic",
                "confidence": 0.8,
                "reasons": ["environment.yml detected"],
            }
        return {
            "backend": "venv",
            "preferred_tool": "venv",
            "python": "python3",
            "channels": [],
            "source": "deterministic",
            "confidence": 0.6,
            "reasons": ["no conda environment file detected"],
        }

    def _environment_strategy_from_payload(self, payload: Dict, action: Dict) -> Dict:
        backend = str(payload.get("backend") or "").lower()
        if backend not in ("venv", "conda", "mamba", "docker"):
            return {}
        channels = [str(item) for item in payload.get("channels") or []]
        return {
            "backend": backend,
            "preferred_tool": "mamba" if payload.get("prefer_mamba") or backend == "mamba" else backend,
            "python": str(payload.get("python") or "3.10"),
            "channels": channels,
            "source": "llm_planner",
            "confidence": float(action.get("confidence") or payload.get("confidence") or 0),
            "reasons": [str(payload.get("reason") or action.get("reason") or "LLM selected environment backend")],
        }

    def _candidate_from_payload(self, payload: Dict, action: Dict = None) -> Optional[Dict]:
        cmd = payload.get("cmd")
        if not isinstance(cmd, list):
            return None
        action = action or {}
        score = self._candidate_score(payload, action, default=payload.get("confidence") or action.get("confidence") or 0.5)
        reasons = self._score_reasons(payload, action, "LLM added candidate")
        return {
            "cmd": list(cmd),
            "expected_port": int(payload.get("expected_port") or 0),
            "confidence": float(payload.get("confidence") or 0.5),
            "source": "llm_planner",
            "score": score,
            "score_reasons": reasons,
            "selected_by": "llm_planner",
        }

    def _select_run_candidate(self, analysis: Dict, payload: Dict, action: Dict = None) -> bool:
        cmd = payload.get("cmd")
        candidates = analysis.get("run_candidates") or []
        for index, candidate in enumerate(candidates):
            if candidate.get("cmd") == cmd:
                selected = candidates.pop(index)
                selected["preferred_by"] = "llm_planner"
                selected["selected_by"] = "combined" if selected.get("source") != "llm_planner" else "llm_planner"
                if payload.get("expected_port"):
                    selected["expected_port"] = int(payload["expected_port"])
                if payload.get("service_type"):
                    selected["service_type"] = str(payload["service_type"])
                selected["score"] = max(float(selected.get("score") or selected.get("confidence") or 0), self._candidate_score(payload, action or {}, default=action.get("confidence") if action else 0.75))
                selected.setdefault("score_reasons", [])
                selected["score_reasons"].extend(self._score_reasons(payload, action or {}, "LLM selected candidate"))
                candidates.insert(0, selected)
                return True
        return False

    def _normalize_candidate_ranking(self, candidates: List[Dict]) -> None:
        for candidate in candidates:
            confidence = float(candidate.get("confidence") or 0)
            candidate.setdefault("score", confidence)
            candidate.setdefault("score_reasons", self._deterministic_score_reasons(candidate))
            candidate.setdefault("selected_by", "deterministic")

    def _deterministic_score_reasons(self, candidate: Dict) -> List[str]:
        cmd = candidate.get("cmd") or []
        reasons = ["deterministic analyzer confidence %.2f" % float(candidate.get("confidence") or 0)]
        if len(cmd) >= 2:
            reasons.append("matched entrypoint %s" % cmd[-1])
        if candidate.get("expected_port"):
            reasons.append("expected service port %s" % candidate.get("expected_port"))
        return reasons

    def _candidate_score(self, payload: Dict, action: Dict, default=0.5) -> float:
        raw = payload.get("score", action.get("confidence", default))
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return max(0.0, min(1.0, float(default or 0)))

    def _score_reasons(self, payload: Dict, action: Dict, fallback: str) -> List[str]:
        reasons = []
        raw_reasons = payload.get("score_reasons")
        if isinstance(raw_reasons, list):
            reasons.extend(str(item) for item in raw_reasons if item)
        if payload.get("reason"):
            reasons.append(str(payload["reason"]))
        if action.get("reason"):
            reasons.append(str(action["reason"]))
        if not reasons:
            reasons.append(fallback)
        return reasons

    def _dependency_constraint_command(self, payload: Dict) -> Optional[List[str]]:
        package = str(payload.get("package") or "")
        constraint = str(payload.get("constraint") or "")
        if not package:
            return None
        return [".venv/bin/python", "-m", "pip", "install", package + constraint]

    def _selected_files(self, repo_dir: Path, files: List[str]) -> Dict[str, str]:
        selected = {}
        for name in files:
            lowered = name.lower()
            if any(marker in lowered for marker in (".env", "secret", "credential", "token", "key")):
                continue
            if Path(name).name not in ("README.md", "readme.md", "requirements.txt", "pyproject.toml", "app.py", "main.py", "server.py"):
                continue
            path = repo_dir / name
            try:
                if path.is_file():
                    selected[name] = "UNTRUSTED REPO CONTENT:\n" + path.read_text(encoding="utf-8", errors="ignore")[:self.agent_max_file_chars]
            except OSError:
                continue
        return selected
