"""Static repository command evidence adapters."""

from auto_harness.command_auth.adapters.make import discover_make
from auto_harness.command_auth.adapters.node import discover_node
from auto_harness.command_auth.adapters.python_cli import discover_python_cli
from auto_harness.command_auth.adapters.repository_script import discover_repository_scripts

__all__ = [
    "discover_make",
    "discover_node",
    "discover_python_cli",
    "discover_repository_scripts",
]
