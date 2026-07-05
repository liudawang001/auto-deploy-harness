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
            "output_dir": str(output_dir),
            "proposals": proposals,
        }

    def apply(self, proposal_path: Path) -> Dict:
        proposal_path = Path(proposal_path)
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
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
            return {
                "status": "already_applied",
                "proposal_id": proposal["proposal_id"],
                "target_skill": str(target),
            }
        block = "\n\n<!-- %s -->\n%s\n<!-- /%s -->\n" % (
            marker,
            proposal["suggested_skill_section"].strip(),
            marker,
        )
        target.write_text(raw.rstrip() + block, encoding="utf-8")
        applied = dict(proposal)
        applied["status"] = "applied"
        applied["applied_at"] = utc_now_iso()
        applied["applied_target"] = str(target)
        write_json(proposal_path, applied)
        return {
            "status": "applied",
            "proposal_id": proposal["proposal_id"],
            "target_skill": str(target),
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
        return sorted(grouped.values(), key=lambda item: (-item["count"], item["key"]["stage"], item["key"]["category"]))

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
            },
            "target_skill": target_skill,
            "review_required": True,
            "apply_command": "PYTHONPATH=src python3 -m auto_harness.cli memory-promote --apply --proposal memory/promotions/%s.json" % proposal_id,
            "suggested_skill_section": section,
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
