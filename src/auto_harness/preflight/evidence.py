"""Preflight artifact writer."""
from pathlib import Path
from typing import Dict

from auto_harness.models.base import write_json


class PreflightEvidenceWriter:
    def __init__(self, run_dir: Path) -> None:
        self.root = Path(run_dir) / "preflight"

    def write(self, name: str, payload: Dict) -> str:
        path = self.root / ("%s.json" % name)
        write_json(path, payload)
        return str(path)
