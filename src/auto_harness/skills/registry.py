import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SkillDoc:
    name: str
    description: str
    path: str
    content: str
    sha256: str
    score: int = 0

    def to_context(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "sha256": self.sha256,
            "score": self.score,
            "content": self.content,
        }


class SkillRegistry:
    """Loads repo-local deployment skills from Markdown control documents."""

    def __init__(self, skills_dir: Path, max_chars: int = 6000) -> None:
        self.skills_dir = skills_dir
        self.max_chars = max_chars

    def select_for_stage(self, stage: str, analysis: Optional[Dict] = None, limit: int = 3) -> List[SkillDoc]:
        analysis = analysis or {}
        scored: List[SkillDoc] = []
        for skill in self._load_all():
            score = self._score(stage, analysis, skill)
            if score > 0:
                skill.score = score
                scored.append(skill)
        return sorted(scored, key=lambda item: (-item.score, item.name))[:limit]

    def _load_all(self) -> List[SkillDoc]:
        if not self.skills_dir.exists():
            return []
        skills: List[SkillDoc] = []
        for path in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = self._parse_frontmatter(raw)
            name = meta.get("name") or path.parent.name
            description = meta.get("description") or ""
            content = body.strip()
            if self.max_chars and len(content) > self.max_chars:
                content = content[: self.max_chars] + "\n\n[truncated]"
            sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            skills.append(
                SkillDoc(
                    name=name,
                    description=description,
                    path=str(path),
                    content=content,
                    sha256=sha,
                )
            )
        return skills

    def _parse_frontmatter(self, raw: str) -> Tuple[Dict[str, str], str]:
        if not raw.startswith("---\n"):
            return {}, raw
        end = raw.find("\n---", 4)
        if end == -1:
            return {}, raw
        meta: Dict[str, str] = {}
        for line in raw[4:end].splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta, raw[end + 4 :]

    def _score(self, stage: str, analysis: Dict, skill: SkillDoc) -> int:
        haystack = "%s\n%s\n%s" % (skill.name, skill.description, skill.content[:1000])
        haystack = haystack.lower()
        stage_aliases = {
            "analyze": ["analyze", "analysis", "classify", "project"],
            "resource_plan": ["resource", "plan", "gpu", "cuda", "disk", "model asset"],
            "env_deploy": ["env", "deploy", "install", "dependency"],
            "model_prepare": ["model", "asset", "download", "cache", "huggingface", "modelscope"],
            "runner": ["runner", "run", "startup", "service"],
            "verify": ["verify", "evidence", "trace", "gradio api"],
        }.get(stage, [stage])
        score = 0
        for alias in stage_aliases:
            if alias in haystack:
                score += 5
                break

        frameworks = analysis.get("frameworks") or []
        for framework in frameworks:
            token = str(framework).lower()
            if token and token in haystack:
                score += 2

        verify_hint = analysis.get("verify_hint", {})
        service_type = verify_hint.get("service_type") if isinstance(verify_hint, dict) else ""
        if service_type and str(service_type).lower() in haystack:
            score += 1
        return score
