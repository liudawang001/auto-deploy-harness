"""Process reconciler: detects external state of local processes.

Reconciles local process operations against the OS:
- PID exists with matching start time, command, and cwd → reuse
- PID exists but identity doesn't match → conflict (PID reuse)
- PID doesn't exist → retry
- Port not ready but process matches → manual (can't auto-reuse)
- Missing identity fields → manual (can't verify)

Key principle: bool(pid) ≠ process alive. We must verify PID + start
time + command hash + cwd to safely reuse a process.
"""
import hashlib
import subprocess
from pathlib import Path
from typing import Callable, Dict

from auto_harness.recovery.download import reconcile_result


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_command(cmd: str) -> str:
    """Normalize command string for comparison.

    Collapses consecutive whitespace but preserves argument differences.
    Does NOT ignore argument changes — that would be unsafe.
    """
    return " ".join(cmd.split())


class ProcessProbe:
    """Probe a local process by PID.

    Uses /proc on Linux, ps + lsof on macOS/other Unix.
    Returns a dict with process identity information.
    """

    def _cwd(self, pid):
        """Get the current working directory of a process."""
        proc_cwd = Path("/proc") / str(pid) / "cwd"
        if proc_cwd.exists():
            try:
                return str(proc_cwd.resolve())
            except OSError:
                return ""
        # macOS / other Unix: use lsof
        try:
            completed = subprocess.run(
                ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        for line in completed.stdout.splitlines():
            if line.startswith("n"):
                return str(Path(line[1:]).resolve())
        return ""

    def observe(self, pid):
        """Observe a process by PID.

        Returns a dict with:
        - exists: bool
        - identity_complete: bool (all fields available)
        - start_time: str (ps lstart format)
        - command: str (actual command from ps)
        - cwd: str (resolved working directory)
        """
        if int(pid or 0) <= 0:
            return {"exists": False}
        try:
            completed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {"exists": False}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {"exists": False}
        fields = completed.stdout.strip().split(None, 5)
        if len(fields) < 6:
            return {"exists": True, "identity_complete": False}
        return {
            "exists": True,
            "identity_complete": True,
            "start_time": " ".join(fields[:5]),
            "command": fields[5],
            "cwd": self._cwd(pid),
        }


class ProcessReconciler:
    """Reconciler for local process operations.

    Checks if a previously started process is still running and
    whether its identity matches what was recorded.
    """

    resource_type = "local_process"

    def __init__(self, probe: ProcessProbe, port_probe: Callable[[str, int], bool]) -> None:
        self.probe = probe
        self.port_probe = port_probe

    def reconcile(self, operation):
        """Reconcile a local process operation.

        Decision logic:
        1. PID doesn't exist → retry
        2. PID exists but identity incomplete → manual
        3. PID start time doesn't match → conflict (PID reuse)
        4. Command hash doesn't match → conflict
        5. Missing expected cwd → manual
        6. Cwd doesn't match → conflict
        7. Port not ready → manual (process exists but not serving)
        8. All match → reuse
        """
        identity = operation["resource_identity"]
        recorded = operation.get("observed_resource", {})
        observed = self.probe.observe(int(recorded.get("pid") or 0))

        # 1. Process no longer exists
        if not observed.get("exists"):
            return reconcile_result("retry", "recorded process no longer exists")

        # 2. Can't verify identity
        if not observed.get("identity_complete"):
            return reconcile_result("manual", "process identity cannot be verified")

        # 3. PID reuse: start time changed
        if observed["start_time"] != recorded.get("process_start_time"):
            return reconcile_result(
                "conflict", "pid was reused", observed=observed,
            )

        # 4. Command changed
        if sha256_text(normalize_command(observed["command"])) != identity.get("command_hash"):
            return reconcile_result(
                "conflict", "process command changed", observed=observed,
            )

        # 5. Missing expected cwd
        if not identity.get("repo_path"):
            return reconcile_result(
                "manual", "expected process cwd is missing", observed=observed,
            )

        # 6. Cwd doesn't match
        expected_cwd = str(Path(identity["repo_path"]).resolve())
        if not observed.get("cwd"):
            return reconcile_result(
                "manual", "process cwd cannot be verified", observed=observed,
            )
        if observed["cwd"] != expected_cwd:
            return reconcile_result(
                "conflict", "process cwd changed", observed=observed,
            )

        # 7. Port not ready
        port = int(identity.get("expected_port") or 0)
        if port and not self.port_probe("127.0.0.1", port):
            return reconcile_result(
                "manual",
                "matching process exists but port is not ready",
                observed=observed,
            )

        # 8. All match
        return reconcile_result(
            "reuse", "same process is still running", observed=observed,
        )
