"""Compile schema-v2 analysis additions without changing legacy behaviour."""

from typing import Dict, List


class LegacyAnalysisCompiler:
    def compile(
        self,
        analysis: Dict,
        *,
        capabilities,
        manifests: List,
        deployability,
        deployment_candidates=None,
    ) -> Dict:
        result = dict(analysis)
        result["schema_version"] = 2
        result["capabilities"] = capabilities.to_dict()
        result["capability_evidence"] = [item.to_dict() for item in capabilities.evidence]
        result["dependency_manifests"] = [item.to_dict() for item in manifests]
        result["deployment_candidates"] = [item.to_dict() for item in (deployment_candidates or [])]
        result["deployability"] = deployability.to_dict()
        result["legacy_compatibility"] = {
            "compiled": True,
            "warnings": [],
            "preserved_fields": [
                "frameworks", "install_plan", "run_candidates", "verify_hint",
            ],
        }
        return result
