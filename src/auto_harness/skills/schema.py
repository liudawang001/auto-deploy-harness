"""Skill Schema Parser: parse and validate SKILL.md frontmatter.

Each SKILL.md must use a unified YAML-like frontmatter with required fields:
name, version, type, stages, risk_level, side_effects, allowed_tools,
success_signals, regression_cases.

The parser extracts structured SkillSpec from the raw Markdown file,
validates required fields and value domains, and returns a list of
validation errors for invalid specs.
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Pipeline stages that skill stages must belong to
VALID_STAGES = frozenset({
    "analyze", "resource_plan", "host_preflight", "env_solve", "env_deploy",
    "model_prepare", "runner", "verify", "report",
    "plan_first", "replan", "repair",
})

# Allowed skill types
VALID_TYPES = frozenset({
    "analysis_skill", "execution_skill", "verification_skill",
    "repair_skill", "security_skill",
})

# Allowed risk levels
VALID_RISK_LEVELS = frozenset({"low", "medium", "high"})

# Semver-like pattern: X.Y.Z
SEMVER_PATTERN = re.compile(r'^\d+\.\d+\.\d+$')


@dataclass
class SkillSpec:
    """Structured specification parsed from a SKILL.md file."""
    name: str
    version: str
    type: str
    stages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    failure_categories: List[str] = field(default_factory=list)
    risk_level: str = "low"
    side_effects: bool = False
    allowed_tools: List[str] = field(default_factory=list)
    success_signals: List[str] = field(default_factory=list)
    regression_cases: List[str] = field(default_factory=list)
    path: str = ""
    sha256: str = ""
    content: str = ""
    deprecated: bool = False
    replacement: str = ""
    model_sources: List[str] = field(default_factory=list)
    env_backends: List[str] = field(default_factory=list)
    owners: List[str] = field(default_factory=list)

    def to_context(self) -> Dict:
        """Convert to a context dict for LLM consumption."""
        return {
            "name": self.name,
            "version": self.version,
            "type": self.type,
            "stages": self.stages,
            "frameworks": self.frameworks,
            "failure_categories": self.failure_categories,
            "risk_level": self.risk_level,
            "side_effects": self.side_effects,
            "allowed_tools": self.allowed_tools,
            "success_signals": self.success_signals,
            "regression_cases": self.regression_cases,
            "sha256": self.sha256,
            "deprecated": self.deprecated,
            "replacement": self.replacement,
            "path": self.path,
        }


class SkillSchemaParser:
    """Parses and validates SKILL.md files into SkillSpec objects."""

    def parse_file(self, path: Path) -> SkillSpec:
        """Parse a SKILL.md file and return a SkillSpec.

        Args:
            path: Path to the SKILL.md file.

        Returns:
            SkillSpec with parsed frontmatter and body content.
        """
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return SkillSpec(name="", version="", type="", path=str(path))

        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        spec = self.parse_text(raw, path=str(path))
        spec.sha256 = sha
        return spec

    def parse_text(self, raw: str, path: str = "") -> SkillSpec:
        """Parse SKILL.md text content into a SkillSpec.

        Args:
            raw: The full SKILL.md text content.
            path: Optional file path for reference.

        Returns:
            SkillSpec with parsed frontmatter and body content.
        """
        meta, body = self._parse_frontmatter(raw)

        # Parse list fields from comma-separated or YAML-like values
        stages = self._parse_list_field(meta.get("stages", ""))
        frameworks = self._parse_list_field(meta.get("frameworks", ""))
        failure_categories = self._parse_list_field(meta.get("failure_categories", ""))
        allowed_tools = self._parse_list_field(meta.get("allowed_tools", ""))
        success_signals = self._parse_list_field(meta.get("success_signals", ""))
        regression_cases = self._parse_list_field(meta.get("regression_cases", ""))
        model_sources = self._parse_list_field(meta.get("model_sources", ""))
        env_backends = self._parse_list_field(meta.get("env_backends", ""))
        owners = self._parse_list_field(meta.get("owners", ""))

        # Parse boolean fields
        side_effects = self._parse_bool(meta.get("side_effects", "false"))
        deprecated = self._parse_bool(meta.get("deprecated", "false"))

        return SkillSpec(
            name=meta.get("name", ""),
            version=meta.get("version", ""),
            type=meta.get("type", ""),
            stages=stages,
            frameworks=frameworks,
            failure_categories=failure_categories,
            risk_level=meta.get("risk_level", "low"),
            side_effects=side_effects,
            allowed_tools=allowed_tools,
            success_signals=success_signals,
            regression_cases=regression_cases,
            path=path,
            content=body.strip(),
            deprecated=deprecated,
            replacement=meta.get("replacement", ""),
            model_sources=model_sources,
            env_backends=env_backends,
            owners=owners,
        )

    def validate(self, spec: SkillSpec) -> List[str]:
        """Validate a SkillSpec and return a list of error messages.

        Args:
            spec: The SkillSpec to validate.

        Returns:
            List of validation error strings. Empty list means valid.
        """
        errors: List[str] = []

        # name must be non-empty
        if not spec.name:
            errors.append("name is required and must be non-empty")

        # version must be semver-like
        if not spec.version:
            errors.append("version is required")
        elif not SEMVER_PATTERN.match(spec.version):
            errors.append("version must be semver-like (e.g. 1.0.0), got: %s" % spec.version)

        # type must be in allowlist
        if not spec.type:
            errors.append("type is required")
        elif spec.type not in VALID_TYPES:
            errors.append("type must be one of %s, got: %s" % (sorted(VALID_TYPES), spec.type))

        # stages must be non-empty and valid
        if not spec.stages:
            errors.append("stages is required and must be non-empty")
        else:
            invalid_stages = [s for s in spec.stages if s not in VALID_STAGES]
            if invalid_stages:
                errors.append("invalid stages: %s (must be in %s)" % (invalid_stages, sorted(VALID_STAGES)))

        # risk_level must be valid
        if spec.risk_level not in VALID_RISK_LEVELS:
            errors.append("risk_level must be one of %s, got: %s" % (sorted(VALID_RISK_LEVELS), spec.risk_level))

        # side_effects=True skill should not enter planner-only execute context
        # (This is a warning, not a hard error - the router will penalize it)

        # deprecated=True must have replacement
        if spec.deprecated and not spec.replacement:
            errors.append("deprecated skill must have a replacement field")

        return errors

    def _parse_frontmatter(self, raw: str) -> Tuple[Dict[str, str], str]:
        """Parse YAML-like frontmatter from SKILL.md.

        Supports both simple key: value and key: [item1, item2] syntax,
        as well as multi-line YAML lists (key: followed by - item lines).

        Returns (meta_dict, body_text).
        """
        if not raw.startswith("---"):
            return {}, raw

        # Find the closing ---
        end_match = re.search(r'\n---', raw[3:])
        if not end_match:
            return {}, raw

        frontmatter = raw[3:3 + end_match.start()]
        body = raw[3 + end_match.start() + 4:]

        meta: Dict[str, str] = {}
        lines = frontmatter.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            # Check if this line starts a multi-line list (key: followed by - item on next lines)
            if ":" in line and not line.lstrip().startswith("-"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Check if value is empty and next lines start with -
                if not value:
                    # Look ahead for multi-line list items
                    list_items = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith("- "):
                        item = lines[j].strip()[2:].strip().strip("'\"")
                        if item:
                            list_items.append(item)
                        j += 1
                    if list_items:
                        meta[key] = "[" + ", ".join(list_items) + "]"
                        i = j
                        continue

                # Remove surrounding quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                meta[key] = value

            i += 1

        return meta, body

    def _parse_list_field(self, value: str) -> List[str]:
        """Parse a list field from frontmatter.

        Supports: [item1, item2] or item1, item2 or single value.
        """
        value = value.strip()
        if not value:
            return []

        # Remove surrounding brackets
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]

        # Split by comma and clean
        items = []
        for item in value.split(","):
            item = item.strip().strip("'\"")
            if item:
                items.append(item)
        return items

    def _parse_bool(self, value: str) -> bool:
        """Parse a boolean field from frontmatter."""
        return value.strip().lower() in ("true", "yes", "1")
