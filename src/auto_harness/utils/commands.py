from pathlib import Path
from typing import Iterable, List


def command_name(cmd: List[str]) -> str:
    if not cmd:
        return ""
    return Path(cmd[0]).name


def is_allowed_command(cmd: List[str], allowed_commands: Iterable[str]) -> bool:
    return command_name(cmd) in set(allowed_commands)

