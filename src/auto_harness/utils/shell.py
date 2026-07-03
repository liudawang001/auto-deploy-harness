import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CommandResult:
    cmd: List[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_command(
    cmd: List[str],
    cwd: Path,
    timeout_seconds: int = 900,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return CommandResult(cmd, str(cwd), proc.returncode, proc.stdout, proc.stderr, False)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            cmd,
            str(cwd),
            124,
            exc.stdout or "",
            exc.stderr or "",
            True,
        )

