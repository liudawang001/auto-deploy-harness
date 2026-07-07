import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import write_json
from auto_harness.utils.files import ensure_dir, short_hash
from auto_harness.utils.time import utc_now_iso


class MemoryPromoter:
    """Turns repeated issue memories into human-reviewable skill update proposals."""

    def __init__(self, memory_dir: Path, skills_dir: Path) -> None:
        self.memory_dir = ensure_dir(memory_dir)
        self.skills_dir = Path(skills_dir)
        self.issue_path = self.memory_dir / "deployment_issues.jsonl"
        self.promotion_dir = ensure_dir(self.memory_dir / "promotions")

    def propose(
        self,
        min_count: int = 2,
        stage: Optional[str] = None,
        category: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        output_dir = ensure_dir(output_dir or self.promotion_dir)
        entries = self._read_entries()
        clusters = self._clusters(entries, stage=stage, category=category)
        candidates = [cluster for cluster in clusters if cluster["count"] >= min_count]
        proposals = []
        for cluster in candidates:
            proposal = self._proposal(cluster)
            write_json(output_dir / ("%s.json" % proposal["proposal_id"]), proposal)
            (output_dir / ("%s.md" % proposal["proposal_id"])).write_text(
                self._proposal_markdown(proposal),
                encoding="utf-8",
            )
            proposals.append(proposal)
        return {
            "status": "proposed" if proposals else "no_candidates",
            "min_count": min_count,
            "candidate_count": len(proposals),
            "eligible_memory_count": sum(cluster["count"] for cluster in clusters),
            "output_dir": str(output_dir),
            "proposals": proposals,
        }

    def apply(self, proposal_path: Path, run_regression: bool = True, benchmark_runner=None) -> Dict:
        proposal_path = Path(proposal_path)
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        approval = proposal.get("approval") if isinstance(proposal.get("approval"), dict) else {}
        if approval.get("status") != "approved":
            return {
                "status": "approval_required",
                "proposal_id": proposal.get("proposal_id"),
                "approval": approval or {"required": True, "status": "pending"},
                "error": "memory promotion must be approved before apply",
            }
        target = self.skills_dir / proposal["target_skill"]
        if not target.exists():
            return {
                "status": "failed",
                "proposal_id": proposal.get("proposal_id"),
                "target_skill": str(target),
                "error": "target skill does not exist",
            }
        raw = target.read_text(encoding="utf-8")
        marker = "auto-harness-memory-promotion:%s" % proposal["proposal_id"]
        if marker in raw:
            regression_result = self._run_regression(proposal, proposal_path, benchmark_runner) if run_regression else self._skipped_regression(proposal)
            return {
                "status": "already_applied",
                "proposal_id": proposal["proposal_id"],
                "target_skill": str(target),
                "regression": regression_result,
            }
        block = "\n\n<!-- %s -->\n%s\n<!-- /%s -->\n" % (
            marker,
            proposal["suggested_skill_section"].strip(),
            marker,
        )
        target.write_text(raw.rstrip() + block, encoding="utf-8")
        regression_result = self._run_regression(proposal, proposal_path, benchmark_runner) if run_regression else self._skipped_regression(proposal)
        applied = dict(proposal)
        applied["status"] = "applied"
        applied["applied_at"] = utc_now_iso()
        applied["applied_target"] = str(target)
        applied["regression"] = regression_result
        write_json(proposal_path, applied)
        return {
            "status": "applied",
            "proposal_id": proposal["proposal_id"],
            "target_skill": str(target),
            "regression_binding": proposal.get("regression_binding", {}),
            "regression": regression_result,
        }

    def approve(self, proposal_path: Path, reviewer: str = "operator", note: str = "") -> Dict:
        proposal_path = Path(proposal_path)
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        approval = {
            "required": True,
            "status": "approved",
            "reviewer": reviewer or "operator",
            "approved_at": utc_now_iso(),
            "note": note,
        }
        proposal["approval"] = approval
        proposal["status"] = "approved"
        write_json(proposal_path, proposal)
        return {
            "status": "approved",
            "proposal_id": proposal.get("proposal_id"),
            "approval": approval,
            "regression_binding": proposal.get("regression_binding", {}),
        }

    def _read_entries(self) -> List[Dict]:
        if not self.issue_path.exists():
            return []
        entries = []
        for line in self.issue_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(entry)
        return entries

    def _clusters(self, entries: List[Dict], stage: Optional[str], category: Optional[str]) -> List[Dict]:
        grouped: Dict[str, Dict] = {}
        for entry in entries:
            if stage and entry.get("stage") != stage:
                continue
            if category and entry.get("category") != category:
                continue
            if not self._is_verified_success(entry):
                continue
            frameworks = sorted(str(item) for item in (entry.get("frameworks") or []))
            key_data = {
                "stage": entry.get("stage") or "unknown",
                "category": entry.get("category") or "unknown",
                "frameworks": frameworks,
            }
            key = json.dumps(key_data, ensure_ascii=False, sort_keys=True)
            cluster = grouped.setdefault(
                key,
                {
                    "key": key_data,
                    "count": 0,
                    "entries": [],
                    "memory_ids": [],
                    "symptoms": [],
                    "root_causes": [],
                    "suggested_next_actions": [],
                    "verification_trace_ids": [],
                    "repair_action_hashes": [],
                    "regression_case_ids": [],
                },
            )
            cluster["count"] += 1
            cluster["entries"].append(entry)
            if entry.get("id"):
                cluster["memory_ids"].append(entry["id"])
            if entry.get("symptom"):
                cluster["symptoms"].append(str(entry["symptom"]))
            if entry.get("root_cause"):
                cluster["root_causes"].append(str(entry["root_cause"]))
            if entry.get("suggested_next_action"):
                cluster["suggested_next_actions"].append(str(entry["suggested_next_action"]))
            if entry.get("verification_trace_id"):
                cluster["verification_trace_ids"].append(str(entry["verification_trace_id"]))
            if entry.get("repair_action_hash"):
                cluster["repair_action_hashes"].append(str(entry["repair_action_hash"]))
            for case_id in entry.get("regression_case_ids") or []:
                if case_id:
                    cluster["regression_case_ids"].append(str(case_id))
        return sorted(grouped.values(), key=lambda item: (-item["count"], item["key"]["stage"], item["key"]["category"]))

    def _is_verified_success(self, entry: Dict) -> bool:
        if entry.get("verified_success") is not True:
            return False
        if entry.get("policy_rejected_high_risk") is True or entry.get("rejected_high_risk_action") is True:
            return False
        if not str(entry.get("verification_trace_id") or "").strip():
            return False
        if not str(entry.get("repair_action_hash") or "").strip():
            return False
        regression_case_ids = entry.get("regression_case_ids")
        if not isinstance(regression_case_ids, list) or not regression_case_ids:
            return False
        repair_status = str(entry.get("repair_action_status") or "success").lower()
        if repair_status not in ("success", "succeeded", "passed", "executed"):
            return False
        verify_status = str(entry.get("verify_status") or "passed").lower()
        if verify_status not in ("pass", "passed", "success", "succeeded"):
            return False
        regression_status = str(entry.get("regression_status") or "passed").lower()
        if regression_status not in ("pass", "passed", "success", "succeeded"):
            return False
        return True

    def _proposal(self, cluster: Dict) -> Dict:
        key = cluster["key"]
        proposal_id = "promo_%s" % short_hash(cluster["key"]["stage"] + cluster["key"]["category"] + "".join(cluster["memory_ids"]), 10)
        target_skill = self._target_skill(key["stage"], key["category"])
        section = self._suggested_skill_section(proposal_id, cluster)
        return {
            "proposal_id": proposal_id,
            "created_at": utc_now_iso(),
            "status": "proposed",
            "cluster": {
                "stage": key["stage"],
                "category": key["category"],
                "frameworks": key["frameworks"],
                "count": cluster["count"],
                "memory_ids": cluster["memory_ids"],
                "verified_success_count": cluster["count"],
                "verification_trace_ids": self._unique_head(cluster["verification_trace_ids"], 10),
                "repair_action_hashes": self._unique_head(cluster["repair_action_hashes"], 10),
                "regression_case_ids": self._unique_head(cluster["regression_case_ids"], 12),
            },
            "target_skill": target_skill,
            "review_required": True,
            "approval": {
                "required": True,
                "status": "pending",
                "reviewer": "",
                "approved_at": "",
                "note": "",
            },
            "regression_binding": self._regression_binding(key["stage"], key["category"], key["frameworks"]),
            "apply_command": "PYTHONPATH=src python3 -m auto_harness.cli memory-promote --apply --proposal memory/promotions/%s.json" % proposal_id,
            "suggested_skill_section": section,
        }

    def _regression_binding(self, stage: str, category: str, frameworks: List[str]) -> Dict:
        cases = []
        if stage == "verify" and "gradio" in frameworks:
            cases.extend(["gradio_config_discovery", "gradio_api_shape_variation", "gradio_queue_call_followup"])
        if stage == "verify" and "streamlit" in frameworks:
            cases.extend(["streamlit_error_page", "browser_dom_trace"])
        if stage in ("resource_plan", "model_prepare") or "model" in category:
            cases.extend(["model_download_resume", "cache_hit", "git_lfs_detection", "git_lfs_prepare_execute"])
        if stage == "env_solve" or "dependency" in category:
            cases.extend(["env_solve_legacy_gradio_constraints", "env_solve_torch_cuda_wheel"])
        if stage in ("env_deploy", "runner"):
            cases.extend(["service_exits_after_start", "docker_backend_plan"])
        return {
            "manifest": "tests/fixtures/benchmarks/manifest.json",
            "case_ids": self._unique_head(cases, 8),
            "required_before_apply": True,
            "notes": "Apply 后至少运行绑定 benchmark case；若新增规则可复用 fixture，应补充 manifest。",
        }

    def _target_skill(self, stage: str, category: str) -> str:
        if stage == "verify":
            return "verify-evidence/SKILL.md"
        if stage in ("resource_plan", "model_prepare") or "model" in category:
            return "prepare-model-assets/SKILL.md"
        if stage == "env_solve" or "dependency" in category:
            return "solve-python-cuda-env/SKILL.md"
        if stage in ("env_deploy", "runner"):
            return "deploy-python-webui/SKILL.md"
        return "diagnose-runtime-failure/SKILL.md"

    def _suggested_skill_section(self, proposal_id: str, cluster: Dict) -> str:
        key = cluster["key"]
        frameworks = ", ".join(key["frameworks"]) if key["frameworks"] else "unknown"
        symptoms = self._unique_head(cluster["symptoms"], 3)
        root_causes = self._unique_head(cluster["root_causes"], 3)
        actions = self._unique_head(cluster["suggested_next_actions"], 3)
        lines = [
            "## Memory Promotion: %s / %s" % (key["stage"], key["category"]),
            "",
            "- Proposal: `%s`" % proposal_id,
            "- Frameworks: `%s`" % frameworks,
            "- Observed count: `%s`" % cluster["count"],
            "- Memory ids: `%s`" % "`, `".join(cluster["memory_ids"]),
            "",
            "### 复发症状",
        ]
        lines.extend("- %s" % item for item in symptoms or ["同类阶段多次失败或 uncertain。"])
        lines.append("")
        lines.append("### 可能根因")
        lines.extend("- %s" % item for item in root_causes or ["需要人工 review 后补充稳定根因。"])
        lines.append("")
        lines.append("### 建议规则")
        lines.extend("- %s" % item for item in actions or ["将该复发模式转成阶段内的显式诊断或 verify 规则。"])
        lines.append("- 该规则来自 memory promotion，应用前必须由人确认，不得记录密钥值或一次性路径。")
        return "\n".join(lines)

    def _proposal_markdown(self, proposal: Dict) -> str:
        cluster = proposal["cluster"]
        return "\n".join([
            "# Memory Promotion Proposal",
            "",
            "- Proposal ID: `%s`" % proposal["proposal_id"],
            "- Target skill: `%s`" % proposal["target_skill"],
            "- Stage: `%s`" % cluster["stage"],
            "- Category: `%s`" % cluster["category"],
            "- Frameworks: `%s`" % "`, `".join(cluster["frameworks"]),
            "- Count: `%s`" % cluster["count"],
            "- Review required: `%s`" % proposal["review_required"],
            "- Approval status: `%s`" % proposal["approval"]["status"],
            "- Regression cases: `%s`" % "`, `".join(proposal["regression_binding"]["case_ids"]),
            "",
            "## Suggested Skill Section",
            "",
            proposal["suggested_skill_section"],
            "",
        ])

    def _unique_head(self, values: List[str], limit: int) -> List[str]:
        result = []
        seen = set()
        for value in values:
            compact = " ".join(value.split())[:500]
            if not compact or compact in seen:
                continue
            seen.add(compact)
            result.append(compact)
            if len(result) >= limit:
                break
        return result

    def _run_regression(self, proposal: Dict, proposal_path: Path, benchmark_runner=None) -> Dict:
        binding = proposal.get("regression_binding") if isinstance(proposal.get("regression_binding"), dict) else {}
        manifest = binding.get("manifest")
        case_ids = binding.get("case_ids") or []
        if not manifest or not case_ids:
            return {
                "status": "skipped",
                "reason": "no regression binding case ids",
                "binding": binding,
            }
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        output_path = proposal_path.with_suffix(".regression.json")
        if benchmark_runner is None:
            from auto_harness.benchmarks import BenchmarkRunner
            benchmark_runner = BenchmarkRunner()
        report = benchmark_runner.run(manifest_path, output_path=output_path, case_ids=case_ids)
        return {
            "status": report.get("status"),
            "manifest": str(manifest_path),
            "case_ids": case_ids,
            "output_path": str(output_path),
            "case_count": len(report.get("cases") or []),
            "failed_case_ids": [case.get("id") for case in report.get("cases", []) if case.get("status") != "passed"],
        }

    def _skipped_regression(self, proposal: Dict) -> Dict:
        return {
            "status": "skipped",
            "reason": "regression execution disabled by caller",
            "binding": proposal.get("regression_binding", {}),
        }
