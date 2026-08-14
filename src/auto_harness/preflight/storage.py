"""Host storage (RAM + cache filesystem) preflight probe.

Uses the stdlib for filesystem usage and os.sysconf for RAM, so it never
depends on psutil or external tools. An optional command runner can inject
``/proc/meminfo``-style RAM facts for Linux hosts.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Optional


class StorageProbe:
    """Collect RAM total/available and cache filesystem total/free."""

    def __init__(self, command_runner=None) -> None:
        self.command_runner = command_runner

    def probe(self, cache_root: Path) -> Dict:
        cache_root = Path(cache_root)
        if not cache_root.exists():
            cache_root.mkdir(parents=True, exist_ok=True)
        disk = shutil.disk_usage(str(cache_root))
        ram_total = self._ram_total()
        ram_available = self._ram_available(ram_total)
        return {
            "schema_version": 1,
            "cache_root": str(cache_root),
            "ram_total_bytes": ram_total,
            "ram_available_bytes": ram_available,
            "disk_total_bytes": int(disk.total),
            "disk_used_bytes": int(disk.used),
            "disk_free_bytes": int(disk.free),
        }

    def _ram_total(self) -> int:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages and page_size:
                return int(pages) * int(page_size)
        except (AttributeError, OSError, ValueError):
            pass
        return 0

    def _ram_available(self, ram_total: int) -> int:
        # Prefer MemAvailable from /proc/meminfo (Linux) when readable.
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) * 1024
        except (OSError, ValueError):
            pass
        # Conservative fallback: treat all RAM as available (solver applies its own safety ratio).
        return ram_total
