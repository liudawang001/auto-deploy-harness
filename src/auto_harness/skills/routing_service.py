"""SkillRoutingService: unified skill routing for all pipeline stages.

Provides a single interface for routing skills in plan, replan, verify,
and repair stages. Replaces the ad-hoc _route_skills() in
PlanFirstDeploymentLoop with a reusable service.

Memory filtering:
- verified_success == True: included at trust_level "verified"
- fix_status == unresolved: included at trust_level "unresolved" as warning
- unresolved suggested actions are NOT treated as verified rules.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from auto_harness.skills.router import SkillRouter, SkillRouteRequest
from auto_harness.skills.context import SkillContextBuilder


class SkillRoutingService:
    """Unified skill routing for all pipeline stages.

    Usage:
        service = SkillRoutingService(router, context_builder, memory_store)
        result = service.route(stage="plan", analysis=analysis)
        # result has: memory_hits, selected_skills, skill_context, request, artifact
    """

    def __init__(
        self,
        router: SkillRouter,
        context_builder: SkillContextBuilder,
        memory_store: Any = None,
        route_limit: int = 3,
        memory_limit: int = 5,
    ):
        self.router = router
        self.context_builder = context_builder
        self.memory_store = memory_store
        self.route_limit = route_limit
        self.memory_limit = memory_limit

    def route(
        self,
        *,
        stage: str,
        analysis: Dict[str, Any],
        failure_category: str = "",
        allowed_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Route skills for a pipeline stage.

        Args:
            stage: Pipeline stage (plan, replan, verify, repair).
            analysis: Project analysis or compiled analysis dict.
            failure_category: Failure category for repair routing.
            allowed_tools: Tools allowed at this stage.

        Returns:
            Dict with:
            - memory_hits: Relevant memory entries with trust_level
            - selected_skills: List of skill context dicts
            - skill_context: Compact context for LLM
            - request: The SkillRouteRequest used (for artifact)
            - artifact: Safe summary for route artifact (no full content)
        """
        # Route skills
        frameworks = list(analysis.get("frameworks", []))
        request = SkillRouteRequest(
            stage=stage,
            analysis=analysis,
            failure_category=failure_category,
            frameworks=frameworks,
            allowed_tools=allowed_tools or [],
        )
        routed = self.router.route(request, limit=self.route_limit)

        # Build skill context
        skill_context = self.context_builder.build(routed, stage=stage)

        # Build selected_skills list
        selected_skills = [r.to_context() for r in routed]

        # Query memory
        memory_hits = []
        if self.memory_store:
            raw_hits = self.memory_store.query(
                stage, analysis, limit=self.memory_limit
            )
            for hit in raw_hits:
                enriched = dict(hit)
                # Classify trust level
                if enriched.get("verified_success"):
                    enriched["trust_level"] = "verified"
                elif enriched.get("fix_status") == "unresolved":
                    enriched["trust_level"] = "unresolved"
                else:
                    enriched["trust_level"] = "unknown"
                memory_hits.append(enriched)

        # Build artifact (safe summary, no full skill content)
        artifact = {
            "stage": stage,
            "failure_category": failure_category,
            "frameworks": frameworks,
            "selected_skills": [
                {
                    "name": s.get("name", ""),
                    "version": s.get("version", ""),
                    "sha256": s.get("sha256", ""),
                    "score": s.get("score", 0),
                    "match_reasons": s.get("match_reasons", []),
                }
                for s in selected_skills
            ],
            "memory_hit_count": len(memory_hits),
            "memory_trust_levels": {
                lvl: sum(1 for h in memory_hits if h.get("trust_level") == lvl)
                for lvl in ("verified", "unresolved", "unknown")
            },
        }

        return {
            "memory_hits": memory_hits,
            "selected_skills": selected_skills,
            "skill_context": skill_context,
            "request": {
                "stage": stage,
                "failure_category": failure_category,
                "frameworks": frameworks,
                "allowed_tools": allowed_tools or [],
            },
            "artifact": artifact,
        }
