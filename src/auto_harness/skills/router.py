"""Skill Router: select the most relevant skills for a deployment stage.

Uses structured frontmatter fields (stages, frameworks, failure_categories,
allowed_tools) instead of simple text matching. Scores skills based on
stage match, framework match, failure category match, tool overlap,
and history-based bonuses/penalties.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.skills.schema import SkillSchemaParser, SkillSpec

# Scoring constants
SCORE_STAGE_MATCH = 8
SCORE_FRAMEWORK_MATCH = 5
SCORE_FAILURE_CATEGORY_MATCH = 5
SCORE_TOOL_OVERLAP = 3
SCORE_MODEL_SOURCE_MATCH = 2
SCORE_ENV_BACKEND_MATCH = 2
SCORE_RECENT_VERIFIED_SUCCESS = 3
SCORE_RECENT_POLICY_ACCEPTED = 2
PENALTY_DEPRECATED = -20
PENALTY_RECENT_HARMFUL = -10
PENALTY_REGRESSION_FAILED = -20
PENALTY_SIDE_EFFECT_IN_PLANNER = -5


@dataclass
class SkillRouteRequest:
    """Input to the SkillRouter."""
    stage: str
    analysis: Dict = field(default_factory=dict)
    failure_category: str = ""
    frameworks: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    mode: str = "planner"  # planner | gated_actor
    history: Dict = field(default_factory=dict)


@dataclass
class RoutedSkill:
    """A skill selected by the router with score and reasons."""
    spec: SkillSpec
    score: int = 0
    match_reasons: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)

    def to_context(self) -> Dict:
        """Convert to a context dict."""
        return {
            "name": self.spec.name,
            "version": self.spec.version,
            "type": self.spec.type,
            "sha256": self.spec.sha256,
            "score": self.score,
            "match_reasons": self.match_reasons,
            "penalties": self.penalties,
            "stages": self.spec.stages,
            "frameworks": self.spec.frameworks,
            "allowed_tools": self.spec.allowed_tools,
        }


class SkillRouter:
    """Routes the most relevant skills for a deployment stage.

    Uses structured frontmatter fields for scoring instead of
    simple text matching. Applies history-based bonuses and penalties.
    """

    def __init__(
        self,
        skills_dir: Path,
        metrics_store: Optional[Dict] = None,
        max_chars: int = 6000,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        self.metrics_store = metrics_store or {}
        self.max_chars = max_chars
        self.parser = SkillSchemaParser()

    def route(self, request: SkillRouteRequest, limit: int = 3) -> List[RoutedSkill]:
        """Select the most relevant skills for the given request.

        Args:
            request: The routing request with stage, analysis, etc.
            limit: Maximum number of skills to return.

        Returns:
            List of RoutedSkill sorted by score (highest first).
        """
        all_specs = self._load_all()
        routed: List[RoutedSkill] = []

        for spec in all_specs:
            score, reasons, penalties = self._score_skill(spec, request)
            if score > 0 or reasons:  # Include skills with any match, even if penalties bring score down
                routed.append(RoutedSkill(
                    spec=spec,
                    score=score,
                    match_reasons=reasons,
                    penalties=penalties,
                ))

        # Sort by score descending, then by name for stability
        routed.sort(key=lambda r: (-r.score, r.spec.name))
        return routed[:limit]

    def _load_all(self) -> List[SkillSpec]:
        """Load all valid skills from the skills directory."""
        if not self.skills_dir.exists():
            return []

        specs: List[SkillSpec] = []
        for path in sorted(self.skills_dir.glob("*/SKILL.md")):
            spec = self.parser.parse_file(path)
            # Only include skills with valid name and type
            if spec.name and spec.type:
                # Truncate content if needed
                if self.max_chars and len(spec.content) > self.max_chars:
                    spec.content = spec.content[:self.max_chars] + "\n\n[truncated]"
                specs.append(spec)
        return specs

    def _score_skill(
        self,
        spec: SkillSpec,
        request: SkillRouteRequest,
    ) -> tuple:
        """Score a single skill against the request.

        Returns (score, match_reasons, penalties).
        """
        score = 0
        reasons: List[str] = []
        penalties: List[str] = []

        # 1. Stage match: +8
        if request.stage in spec.stages:
            score += SCORE_STAGE_MATCH
            reasons.append("stage=%s" % request.stage)

        # 2. Framework match: +5
        request_frameworks = request.frameworks or request.analysis.get("frameworks", [])
        matching_frameworks = set(request_frameworks) & set(spec.frameworks)
        if matching_frameworks:
            score += SCORE_FRAMEWORK_MATCH
            reasons.append("framework=%s" % ",".join(sorted(matching_frameworks)))

        # 3. Failure category match: +5
        if request.failure_category and request.failure_category in spec.failure_categories:
            score += SCORE_FAILURE_CATEGORY_MATCH
            reasons.append("failure_category=%s" % request.failure_category)

        # 4. Allowed tool overlap: +3
        if request.allowed_tools and spec.allowed_tools:
            overlap = set(request.allowed_tools) & set(spec.allowed_tools)
            if overlap:
                score += SCORE_TOOL_OVERLAP
                reasons.append("tool_overlap=%d" % len(overlap))

        # 5. Model source match: +2
        analysis = request.analysis or {}
        model_sources = []
        for asset in analysis.get("model_assets", []):
            if isinstance(asset, dict) and asset.get("source"):
                model_sources.append(asset["source"])
        if model_sources and spec.model_sources:
            matching_sources = set(model_sources) & set(spec.model_sources)
            if matching_sources:
                score += SCORE_MODEL_SOURCE_MATCH
                reasons.append("model_source=%s" % ",".join(sorted(matching_sources)))

        # 6. Env backend match: +2
        env_backend = analysis.get("env_solution", {}).get("backend", "")
        if env_backend and spec.env_backends and env_backend in spec.env_backends:
            score += SCORE_ENV_BACKEND_MATCH
            reasons.append("env_backend=%s" % env_backend)

        # 7. History-based scoring
        # Look up by full key (name@sha8), name alone, or from metrics_store
        request_history = request.history or {}
        skill_key = "%s@%s" % (spec.name, spec.sha256[:8])
        history = request_history.get(skill_key) or request_history.get(spec.name) or self.metrics_store.get(skill_key) or self.metrics_store.get(spec.name) or {}

        # Recent verified success: +3
        if history.get("recent_verified_success"):
            score += SCORE_RECENT_VERIFIED_SUCCESS
            reasons.append("recent_verified_success")

        # Recent policy accepted: +2
        if history.get("recent_policy_accepted"):
            score += SCORE_RECENT_POLICY_ACCEPTED
            reasons.append("recent_policy_accepted")

        # 8. Penalties

        # Deprecated: -20
        if spec.deprecated:
            score += PENALTY_DEPRECATED
            penalties.append("deprecated")

        # Recent harmful outcome: -10
        if history.get("recent_harmful"):
            score += PENALTY_RECENT_HARMFUL
            penalties.append("recent_harmful")

        # Regression failed: -20
        if history.get("regression_failed"):
            score += PENALTY_REGRESSION_FAILED
            penalties.append("regression_failed")

        # Side-effect skill in planner mode: -5
        if spec.side_effects and request.mode == "planner":
            score += PENALTY_SIDE_EFFECT_IN_PLANNER
            penalties.append("side_effect_in_planner_mode")

        return score, reasons, penalties
