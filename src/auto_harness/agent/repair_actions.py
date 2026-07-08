import re
from typing import Dict

SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.-]+([<>=!~]=?[A-Za-z0-9_.+*,-]+)?$")


def safe_package_spec(package: str) -> bool:
    if not package or not SAFE_PACKAGE_RE.match(package):
        return False
    lowered = package.lower()
    forbidden = ("--extra-index-url", "--trusted-host", " -e ", "git+", "http://", "https://", "/", "\\")
    return not any(token in lowered for token in forbidden)


def install_package_command(package: str, env_context: Dict = None) -> Dict:
    if not safe_package_spec(package):
        return {"status": "rejected", "reason": "unsafe package spec", "cmd": []}
    env_context = env_context or {}
    backend = str(env_context.get("backend") or env_context.get("environment_backend") or "venv")
    prefix = str(env_context.get("conda_prefix") or env_context.get("environment_prefix") or "")
    if backend in ("conda", "mamba") and prefix:
        tool = "mamba" if backend == "mamba" else "conda"
        return {
            "status": "ready",
            "reason": "",
            "cmd": [tool, "run", "-p", prefix, "python", "-m", "pip", "install", package],
        }
    python = str(env_context.get("python_executable") or ".venv/bin/python")
    return {
        "status": "ready",
        "reason": "",
        "cmd": [python, "-m", "pip", "install", package],
    }
