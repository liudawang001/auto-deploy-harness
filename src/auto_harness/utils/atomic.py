"""Atomic file write and file locking utilities.

Provides:
- atomic_write_text(): Write text to a file atomically using tempfile + os.replace
- FileLock: Context manager for file-based locking (fcntl on Unix, no-op fallback)

These are used by SkillPatchApplier and SkillRollbackManager to prevent
concurrent modification or corruption of skill files during promotion/rollback.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to a file atomically.

    Writes to a temporary file first, then atomically replaces the target.
    This prevents partial writes from corrupting the file if the process
    is interrupted mid-write.

    Args:
        path: Target file path.
        text: Content to write.
        encoding: Text encoding (default utf-8).
    """
    path = Path(path)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same directory (same filesystem for os.replace)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".atomic_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replace
        os.replace(tmp_path, str(path))
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class FileLock:
    """File-based lock using fcntl (Unix) or no-op fallback.

    Usage:
        with FileLock(path) as lock:
            # exclusive access to file at path
            ...

    On Unix (macOS/Linux), uses fcntl.flock with LOCK_EX/LOCK_UN.
    On other platforms, falls back to a no-op (still provides atomic_write safety).

    The lock file is <path>.lock — a separate file used only for locking.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = Path(str(self.path) + ".lock")
        self._fd = None

    def __enter__(self):
        """Acquire exclusive lock."""
        # Ensure lock file parent exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        self._fd = self.lock_path.open("w")
        try:
            import fcntl
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        except (ImportError, AttributeError):
            # No fcntl (Windows) — fall back to no-op
            # atomic_write_text still provides write safety
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock and close lock file."""
        try:
            import fcntl
            if self._fd is not None:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        except (ImportError, AttributeError):
            pass
        if self._fd is not None:
            try:
                self._fd.close()
            except OSError:
                pass
        # Don't delete lock file — it may be reused
        return False
