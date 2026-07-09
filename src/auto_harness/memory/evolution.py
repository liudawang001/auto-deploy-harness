"""Memory Evolution Manager: propose, regress, shadow, promote, reject skill candidates.

This is the core orchestrator for the evidence-gated memory-to-skill evolution loop.

Flow:
  verified memory → MemoryQualityGate → cluster → MemoryCurator →
  candidate draft → SkillPatchValidator → write candidate json/md →
  regression gate → shadow eval → promote (with rollback) → outcome tracking

The manager never modifies official skill files directly — that's done by
SkillPatchApplier under strict gating.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.memory.quality import MemoryQualityGate
from auto_harness.memory.curator import MemoryCurator
from auto_harness.skills.patch import SkillPatchValidator, SkillPatchApplier
from auto_harness.models.base import write_json
from auto_harness.utils.files import ensure_dir, short_hash
from auto_harness.utils.time import utc_now_iso


class MemoryEvolutionManager:
    """Manage the full memory-to-skill evolution lifecycle.

    Methods:
      propose(): Generate a candidate from verified memories
      run_regression(): Run regression for a candidate
      promote(): Promote a candidate to active skill (after gates)
      reject(): Mark a candidate as rejected
    """

    def __init__(self, memory_dir: Path, skills_dir: Path, provider=None):
        self.memory_dir = ensure_dir(Path(memory_dir))
        self.skills_dir = Path(skills_dir)
        self.provider = provider
        self.quality_gate = MemoryQualityGate()
        self.curator = MemoryCurator(provider=provider)
        self.patch_validator = SkillPatchValidator()
        self.patch_applier = SkillPatchApplier()
        self.issue_path = self.memory_dir / "deployment_issues.jsonl"
        self.candidate_dir = ensure_dir(self.memory_dir / "skill_candidates")

    def propose(
        self,
        min_verified_count: int = 3,
        stage: str = None,
        category: str = None,
        output_dir: Path = None,
    ) -> Dict:
        """Generate a skill patch candidate from verified memories.

        Steps:
        1. Read deployment_issues.jsonl
        2. Filter with MemoryQualityGate
        3. Cluster by stage/category/frameworks
        4. Require count >= min_verified_count
        5. Load target skill content
        6. Call MemoryCurator
        7. Validate patch with SkillPatchValidator
        8. Write candidate json + markdown

        Returns:
            Dict with status, candidate_count, candidates list.
        """
        output_dir = ensure_dir(output_dir or self.candidate_dir)

        # 1. Read entries
        entries = self._read_entries()

        # 2. Filter with quality gate
        verified = self.quality_gate.filter_verified(entries)

        # 3. Cluster
        clusters = self._clusters(verified, stage=stage, category=category)

        # 4. Filter by min count
        eligible_clusters = [c for c in clusters if c["count"] >= min_verified_count]

        if not eligible_clusters:
            return {
                "status": "no_candidates",
                "min_verified_count": min_verified_count,
                "total_verified": len(verified),
                "cluster_count": len(clusters),
                "eligible_cluster_count": 0,
                "candidates": [],
            }

        # 5-8. Generate candidates from each cluster
        candidates = []
        for cluster in eligible_clusters:
            candidate = self._generate_candidate(cluster, output_dir)
            if candidate:
                candidates.append(candidate)

        return {
            "status": "proposed" if candidates else "no_candidates",
            "min_verified_count": min_verified_count,
            "total_verified": len(verified),
            "cluster_count": len(clusters),
            "eligible_cluster_count": len(eligible_clusters),
            "candidates": candidates,
            "output_dir": str(output_dir),
        }

    def run_regression(self, candidate_path: Path, benchmark_runner=None) -> Dict:
        """Run regression for a candidate.

        Args:
            candidate_path: Path to candidate_<id>.json
            benchmark_runner: Optional BenchmarkRunner instance.

        Returns:
            Dict with regression status.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")

        # Read regression binding
        binding = candidate.get("regression_binding", {})
        manifest = binding.get("manifest", "")
        case_ids = binding.get("case_ids", [])

        # Check case_ids is non-empty
        if not case_ids:
            result = {
                "status": "regression_failed",
                "candidate_id": candidate_id,
                "error": "no regression case_ids bound",
            }
            # Write regression artifact
            write_json(candidate_path.with_suffix(".regression.json"), result)
            # Update candidate status
            candidate["regression"] = {"status": "failed", "error": "no case_ids"}
            candidate["status"] = "regression_failed"
            write_json(candidate_path, candidate)
            return result

        # Run benchmark
        if benchmark_runner is None:
            try:
                from auto_harness.benchmarks import BenchmarkRunner
                benchmark_runner = BenchmarkRunner()
            except ImportError:
                result = {
                    "status": "regression_failed",
                    "candidate_id": candidate_id,
                    "error": "BenchmarkRunner not available",
                }
                write_json(candidate_path.with_suffix(".regression.json"), result)
                return result

        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path

        output_path = candidate_path.with_suffix(".regression.json")
        try:
            report = benchmark_runner.run(manifest_path, output_path=output_path, case_ids=case_ids)
        except Exception as exc:
            result = {
                "status": "regression_failed",
                "candidate_id": candidate_id,
                "error": "benchmark runner error: %s" % str(exc)[:200],
            }
            write_json(output_path, result)
            candidate["regression"] = result
            candidate["status"] = "regression_failed"
            write_json(candidate_path, candidate)
            return result

        failed_case_ids = [
            case.get("id") for case in report.get("cases", [])
            if case.get("status") != "passed"
        ]

        regression_status = "passed" if not failed_case_ids else "failed"
        result = {
            "status": regression_status,
            "candidate_id": candidate_id,
            "manifest": str(manifest_path),
            "case_ids": case_ids,
            "output_path": str(output_path),
            "failed_case_ids": failed_case_ids,
        }

        # Write regression artifact
        write_json(output_path, result)

        # Update candidate
        candidate["regression"] = result
        if regression_status == "passed":
            candidate["status"] = "regression_passed"
        else:
            candidate["status"] = "regression_failed"
        write_json(candidate_path, candidate)

        return result

    def promote(self, candidate_path: Path, require_shadow: bool = True) -> Dict:
        """Promote a candidate to active skill after all gates pass.

        Gates checked:
        - candidate.status in candidate|shadow_passed|regression_passed
        - quality_gate.passed == true
        - regression.status == passed
        - base_skill_sha256 == current target skill sha
        - patch validator passed
        - if require_shadow: shadow.helped_count >= 2 and shadow.harmful_count == 0

        Args:
            candidate_path: Path to candidate_<id>.json
            require_shadow: Whether to require shadow evaluation.

        Returns:
            Dict with promotion status.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")

        # Gate 1: status check
        allowed_statuses = {"candidate", "shadow_passed", "regression_passed"}
        if candidate.get("status") not in allowed_statuses:
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "candidate status '%s' not in %s" % (candidate.get("status"), allowed_statuses),
            }

        # Gate 2: quality gate
        quality = candidate.get("quality_gate", {})
        if not quality.get("passed"):
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "quality gate not passed",
            }

        # Gate 3: regression
        regression = candidate.get("regression", {})
        if regression.get("status") != "passed":
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "regression not passed (status: %s)" % regression.get("status"),
            }

        # Gate 4: base sha match (checked by SkillPatchApplier)
        # Gate 5: patch validator
        patch_markdown = candidate.get("patch", {}).get("markdown", "")
        validation = self.patch_validator.validate(patch_markdown)
        if not validation["valid"]:
            return {
                "status": "failed",
                "candidate_id": candidate_id,
                "error": "patch validation failed: %s" % ", ".join(validation["reject_reasons"]),
            }

        # Gate 6: shadow evaluation (if required)
        if require_shadow:
            shadow = candidate.get("shadow", {})
            helped_count = shadow.get("helped_count", 0)
            harmful_count = shadow.get("harmful_count", 0)
            if helped_count < 2:
                return {
                    "status": "failed",
                    "candidate_id": candidate_id,
                    "error": "shadow helped_count %d < 2" % helped_count,
                }
            if harmful_count > 0:
                return {
                    "status": "failed",
                    "candidate_id": candidate_id,
                    "error": "shadow harmful_count %d > 0" % harmful_count,
                }

        # All gates passed — apply patch
        apply_result = self.patch_applier.apply_candidate(candidate, self.skills_dir)
        if apply_result["status"] not in ("applied", "already_applied"):
            return {
                "status": apply_result["status"],
                "candidate_id": candidate_id,
                "error": apply_result.get("error", "apply failed"),
            }

        # Update candidate with promotion info
        candidate["promotion"] = {
            "status": "promoted",
            "promoted_at": utc_now_iso(),
            "previous_sha256": apply_result.get("previous_sha256", ""),
            "new_sha256": apply_result.get("new_sha256", ""),
            "rollback_path": apply_result.get("rollback_path", ""),
        }
        candidate["status"] = "active"
        write_json(candidate_path, candidate)

        return {
            "status": "promoted",
            "candidate_id": candidate_id,
            "target_skill": apply_result.get("target_skill", ""),
            "previous_sha256": apply_result.get("previous_sha256", ""),
            "new_sha256": apply_result.get("new_sha256", ""),
            "rollback_path": apply_result.get("rollback_path", ""),
        }

    def reject(self, candidate_path: Path, reason: str) -> Dict:
        """Mark a candidate as rejected.

        Args:
            candidate_path: Path to candidate_<id>.json
            reason: Human-readable rejection reason.

        Returns:
            Dict with rejection status.
        """
        candidate_path = Path(candidate_path)
        if not candidate_path.exists():
            return {"status": "failed", "error": "candidate file not found"}

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_id = candidate.get("candidate_id", "unknown")
        candidate["status"] = "rejected"
        candidate["rejection"] = {
            "reason": reason,
            "rejected_at": utc_now_iso(),
        }
        write_json(candidate_path, candidate)

        return {
            "status": "rejected",
            "candidate_id": candidate_id,
            "reason": reason,
        }

    def _read_entries(self) -> List[Dict]:
        """Read all entries from deployment_issues.jsonl."""
        if not self.issue_path.exists():
            return []
        entries = []
        for line in self.issue_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _clusters(self, entries: List[Dict], stage: str = None, category: str = None) -> List[Dict]:
        """Cluster verified entries by stage/category/frameworks."""
        grouped: Dict[str, Dict] = {}
        for entry in entries:
            if stage and entry.get("stage") != stage:
                continue
            if category and entry.get("category") != category:
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
                    "repair_actions": [],
                    "verification_trace_ids": [],
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
            if entry.get("repair_actions"):
                for action in entry["repair_actions"]:
                    if action:
                        cluster["repair_actions"].append(str(action))
            if entry.get("verification_trace_id"):
                cluster["verification_trace_ids"].append(str(entry["verification_trace_id"]))
            for case_id in entry.get("regression_case_ids") or []:
                if case_id:
                    cluster["regression_case_ids"].append(str(case_id))

        return sorted(grouped.values(), key=lambda item: (-item["count"], item["key"]["stage"], item["key"]["category"]))

    def _generate_candidate(self, cluster: Dict, output_dir: Path) -> Optional[Dict]:
        """Generate a single candidate from a cluster."""
        key = cluster["key"]

        # Determine target skill
        target_skill = self._target_skill(key["stage"], key["category"])

        # Load target skill content
        target_skill_path = self.skills_dir / target_skill
        target_content = ""
        if target_skill_path.exists():
            target_content = target_skill_path.read_text(encoding="utf-8")

        # Compute base skill sha
        base_skill_sha = _sha256(target_content) if target_content else ""

        # Call curator
        curation = self.curator.curate(cluster, target_skill_content=target_content[:2000])
        if curation.get("status") != "ok":
            return None

        draft = curation["candidate_draft"]
        patch_markdown = draft.get("skill_patch", {}).get("markdown", "")
        section_title = draft.get("skill_patch", {}).get("section_title", "Memory Evolution Patch")

        # Validate patch
        validation = self.patch_validator.validate(patch_markdown)
        if not validation["valid"]:
            return None

        # Build candidate
        candidate_id = "skillcand_%s" % short_hash(
            key["stage"] + key["category"] + "".join(cluster["memory_ids"][:5]),
            12,
        )

        # Determine regression binding
        regression_binding = self._regression_binding(key["stage"], key["category"], key["frameworks"])

        candidate = {
            "candidate_id": candidate_id,
            "created_at": utc_now_iso(),
            "status": "candidate",
            "source_memory_ids": cluster["memory_ids"],
            "target_skill": target_skill,
            "base_skill_sha256": base_skill_sha,
            "curator": {
                "provider": "mock" if self.provider is None else getattr(self.provider, "__class__", type).__name__,
                "raw_response_hash": curation.get("raw_response_hash", ""),
            },
            "pattern": draft.get("pattern", {}),
            "reusable_rule": draft.get("reusable_rule", {}),
            "patch": {
                "section_title": section_title,
                "markdown": patch_markdown,
            },
            "quality_gate": {
                "passed": True,
                "reasons": validation.get("reasons", []),
                "reject_reasons": validation.get("reject_reasons", []),
            },
            "regression_binding": regression_binding,
            "shadow": {
                "enabled": False,
                "helped_count": 0,
                "harmful_count": 0,
            },
            "promotion": {
                "status": "not_promoted",
                "promoted_at": "",
                "previous_sha256": "",
                "new_sha256": "",
                "rollback_path": "",
            },
        }

        # Write candidate files
        candidate_json_path = output_dir / ("candidate_%s.json" % candidate_id)
        candidate_md_path = output_dir / ("candidate_%s.md" % candidate_id)
        write_json(candidate_json_path, candidate)
        candidate_md_path.write_text(
            "# Skill Candidate: %s\n\n%s" % (candidate_id, patch_markdown),
            encoding="utf-8",
        )

        return candidate

    def _target_skill(self, stage: str, category: str) -> str:
        """Map stage/category to target skill path."""
        if stage == "verify":
            return "verify-evidence/SKILL.md"
        if stage in ("resource_plan", "model_prepare") or "model" in category:
            return "prepare-model-assets/SKILL.md"
        if stage == "env_solve" or "dependency" in category:
            return "solve-python-cuda-env/SKILL.md"
        if stage in ("env_deploy", "runner"):
            return "deploy-python-webui/SKILL.md"
        return "diagnose-runtime-failure/SKILL.md"

    def _regression_binding(self, stage: str, category: str, frameworks: List[str]) -> Dict:
        """Determine regression binding for a candidate."""
        cases = []
        if stage == "verify" and "gradio" in frameworks:
            cases.extend(["gradio_config_discovery", "gradio_api_shape_variation"])
        if stage == "verify" and "streamlit" in frameworks:
            cases.extend(["streamlit_error_page"])
        if stage in ("resource_plan", "model_prepare") or "model" in category:
            cases.extend(["model_download_resume", "cache_hit"])
        if stage == "env_solve" or "dependency" in category:
            cases.extend(["env_solve_legacy_gradio_constraints"])
        return {
            "manifest": "tests/fixtures/benchmarks/manifest.json",
            "case_ids": cases[:8],
            "required_before_promote": True,
        }


def _sha256(text: str) -> str:
    """Compute sha256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
