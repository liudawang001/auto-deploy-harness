import re
from typing import Dict

SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+([<>=!~]=?[A-Za-z0-9_.+*~-]+)?$")


def safe_package_spec(package: str) -> bool:
    if not package or not SAFE_PACKAGE_RE.match(package):
        return False
    lowered = package.lower()
    forbidden = ("--extra-index-url", "--trusted-host", " -e ", "git+", "http://", "https://", "/", "\\")
    return not any(token in lowered for token in forbidden)


def install_package_command(package: str) -> Dict:
    if not safe_package_spec(package):
        return {"status": "rejected", "reason": "unsafe package spec", "cmd": []}
    return {
        "status": "ready",
        "reason": "",
        "cmd": [".venv/bin/python", "-m", "pip", "install", package],
    }
