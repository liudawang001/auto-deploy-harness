import json
from pathlib import Path
from typing import Any, Dict

from auto_harness.utils.time import utc_now_iso


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, stage: str, event_type: str, data: Dict[str, Any] = None) -> None:
        event = {
            "ts": utc_now_iso(),
            "stage": stage,
            "type": event_type,
            "data": data or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

