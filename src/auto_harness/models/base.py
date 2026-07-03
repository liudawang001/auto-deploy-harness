import dataclasses
import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_plain(data), f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

