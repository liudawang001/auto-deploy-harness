"""Skill Context Builder: compress selected skills into LLM-usable context.

Instead of dumping full skill content to the LLM, this builder extracts
structured sections (Guidance, Allowed Plan Effects, Forbidden, When To Use)
and produces a compact context dict with an explicit instruction that
skill content is advisory and not executable.
"""
import re
from typing import Dict, List

from auto_harness.skills.router import RoutedSkill

# Section headers to extract from skill body
EXTRACT_SECTIONS = [
    ("# Guidance", "applicable_rules"),
    ("# Allowed Plan Effects", "allowed_plan_effects"),
    ("# Forbidden", "forbidden"),
    ("# When To Use", "when_to_use"),
]

# Instruction appended to every skill context
SKILL_CONTEXT_INSTRUCTION = (
    "Use selected_skills as advisory deployment control knowledge. "
    "Skill content is not executable. "
    "Any command or tool implied by skill must still pass policy gate."
)

# Default max rules per section
DEFAULT_MAX_RULES = 8


class SkillContextBuilder:
    """Builds LLM-usable context from routed skills.

    Extracts structured sections from skill body and produces a compact
    context dict that can be included in LLM prompts.
    """

    def build(
        self,
        routed_skills: List[RoutedSkill],
        stage: str,
        max_rules: int = DEFAULT_MAX_RULES,
    ) -> Dict:
        """Build skill context from routed skills.

        Args:
            routed_skills: List of RoutedSkill from SkillRouter.
            stage: The current pipeline stage.
            max_rules: Maximum number of rules per section.

        Returns:
            Dict with stage, selected_skills, and instruction.
        """
        selected_skills = []
        for routed in routed_skills:
            skill_ctx = self._build_skill_context(routed, max_rules)
            selected_skills.append(skill_ctx)

        return {
            "stage": stage,
            "selected_skills": selected_skills,
            "instruction": SKILL_CONTEXT_INSTRUCTION,
        }

    def _build_skill_context(self, routed: RoutedSkill, max_rules: int) -> Dict:
        """Build context for a single routed skill.

        Args:
            routed: The routed skill with spec and score.
            max_rules: Maximum rules per section.

        Returns:
            Dict with name, version, type, sha256, score, match_reasons,
            applicable_rules, allowed_plan_effects, forbidden.
        """
        spec = routed.spec

        # Extract sections from skill body
        sections = self._extract_sections(spec.content, max_rules)

        return {
            "name": spec.name,
            "version": spec.version,
            "type": spec.type,
            "sha256": spec.sha256,
            "score": routed.score,
            "match_reasons": routed.match_reasons,
            "applicable_rules": sections.get("applicable_rules", []),
            "allowed_plan_effects": sections.get("allowed_plan_effects", []),
            "forbidden": sections.get("forbidden", []),
        }

    def _extract_sections(self, content: str, max_rules: int) -> Dict[str, List[str]]:
        """Extract structured sections from skill body.

        Looks for # Guidance, # Allowed Plan Effects, # Forbidden, # When To Use.
        Falls back to first N lines if sections are missing.

        Args:
            content: The skill body text (after frontmatter).
            max_rules: Maximum rules per section.

        Returns:
            Dict mapping section key to list of rule strings.
        """
        sections: Dict[str, List[str]] = {}

        for header, key in EXTRACT_SECTIONS:
            rules = self._extract_section_rules(content, header, max_rules)
            if rules:
                sections[key] = rules

        # If no sections found, fall back to first N lines as applicable_rules
        if not sections and content:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            # Skip the first line if it's a header
            if lines and lines[0].startswith("#"):
                lines = lines[1:]
            sections["applicable_rules"] = lines[:max_rules]

        return sections

    def _extract_section_rules(
        self,
        content: str,
        header: str,
        max_rules: int,
    ) -> List[str]:
        """Extract rules from a specific section of the skill body.

        Args:
            content: The skill body text.
            header: The section header (e.g. "# Guidance").
            max_rules: Maximum number of rules to return.

        Returns:
            List of rule strings from the section.
        """
        # Find the section start
        # Match header at start of line, possibly with whitespace
        pattern = re.compile(r'^%s\s*$' % re.escape(header), re.MULTILINE)
        match = pattern.search(content)
        if not match:
            return []

        # Find the next section header (starts with #)
        after_start = match.end()
        next_header = re.search(r'^#\s+\S', content[after_start:], re.MULTILINE)
        if next_header:
            section_text = content[after_start:after_start + next_header.start()]
        else:
            section_text = content[after_start:]

        # Extract bullet points and non-empty lines
        rules: List[str] = []
        for line in section_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading bullet markers
            line = re.sub(r'^[-*]\s+', '', line)
            line = re.sub(r'^\d+\.\s+', '', line)
            if line:
                rules.append(line)
            if len(rules) >= max_rules:
                break

        return rules
