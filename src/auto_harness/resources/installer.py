"""Install packaged configuration and skills into an operator workspace."""
from importlib import resources
from pathlib import Path
from typing import Dict


def initialize_workspace(root: Path, force: bool = False) -> Dict[str, object]:
    root = Path(root)
    created = []
    preserved = []
    package_root = resources.files("auto_harness.resources")
    targets = [(package_root.joinpath("default.json"), root / "configs" / "default.json")]
    skills_root = package_root.joinpath("skills")
    for skill_dir in skills_root.iterdir():
        if skill_dir.is_dir():
            targets.append((skill_dir.joinpath("SKILL.md"), root / "skills" / skill_dir.name / "SKILL.md"))
    for source, target in targets:
        if target.exists() and not force:
            preserved.append(str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        created.append(str(target))
    (root / "runs").mkdir(parents=True, exist_ok=True)
    return {"status": "initialized", "created": created, "preserved": preserved}
