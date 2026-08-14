"""Minimal child-process environments for untrusted deployment code.

Provider credentials belong to the harness process.  Installers, target
services, and verification probes are untrusted and must never inherit the
parent environment implicitly.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Optional


_SAFE_EXACT_NAMES = frozenset({
    "PATH",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
})

_SECRET_NAME_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|"
    r"PRIVATE_?KEY|ACCESS_?KEY)(?:$|_)",
    re.IGNORECASE,
)

_FORBIDDEN_PREFIXES = (
    "DEEPSEEK_",
    "XUNFEI_",
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GITHUB_",
    "GH_",
    "ANTHROPIC_",
    "OPENAI_",
)

_FORBIDDEN_EXACT = frozenset({
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "MODELSCOPE_API_TOKEN",
    "AUTO_HARNESS_LLM_API_KEY",
    "SSH_AUTH_SOCK",
    "GOOGLE_APPLICATION_CREDENTIALS",
})


def is_secret_environment_name(name: str) -> bool:
    """Return True when an environment name may carry credentials."""
    normalized = str(name or "").strip().upper()
    if not normalized:
        return True
    return (
        normalized in _FORBIDDEN_EXACT
        or normalized.startswith(_FORBIDDEN_PREFIXES)
        or bool(_SECRET_NAME_RE.search(normalized))
    )


def local_docker_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Resolve a local Docker socket without exposing the user's Docker HOME."""
    source = dict(os.environ if environ is None else environ)
    host = str(source.get("DOCKER_HOST") or "").strip()
    if not host:
        try:
            context_env = {
                name: source[name]
                for name in (
                    "PATH", "HOME", "USERPROFILE", "DOCKER_CONFIG", "DOCKER_CONTEXT",
                )
                if source.get(name)
            }
            completed = subprocess.run(
                [
                    "docker", "context", "inspect", "--format",
                    "{{.Endpoints.docker.Host}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env=context_env,
            )
            if completed.returncode == 0:
                host = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            host = ""
    if host.startswith(("unix://", "npipe://")):
        return {"DOCKER_HOST": host}
    return {}


class ChildEnvironmentPolicy:
    """Build fail-closed environments for target-controlled subprocesses."""

    def __init__(self, environ: Optional[Mapping[str, str]] = None) -> None:
        self.environ = dict(os.environ if environ is None else environ)

    def build_for_install(
        self,
        *,
        home_dir: Optional[Path] = None,
        extra: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        return self._build("install", home_dir=home_dir, extra=extra)

    def build_for_service(
        self,
        *,
        home_dir: Optional[Path] = None,
        extra: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        return self._build("service", home_dir=home_dir, extra=extra)

    def build_for_verify(
        self,
        *,
        home_dir: Optional[Path] = None,
        extra: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, str]:
        return self._build("verify", home_dir=home_dir, extra=extra)

    def _build(
        self,
        phase: str,
        *,
        home_dir: Optional[Path],
        extra: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        child: Dict[str, str] = {}
        for name, value in self.environ.items():
            upper = str(name).upper()
            if is_secret_environment_name(upper):
                continue
            if upper in _SAFE_EXACT_NAMES or upper.startswith("LC_"):
                child[str(name)] = str(value)

        child.setdefault("PATH", os.defpath)
        child["PYTHONDONTWRITEBYTECODE"] = "1"
        child["AUTO_HARNESS_CHILD_PHASE"] = phase

        if home_dir is not None:
            safe_home = Path(home_dir)
            safe_home.mkdir(parents=True, exist_ok=True)
            child["HOME"] = str(safe_home)
            if os.name == "nt":
                child["USERPROFILE"] = str(safe_home)

        for name, value in (extra or {}).items():
            if is_secret_environment_name(name):
                raise ValueError(
                    "secret-like environment variable is not allowed in child process: %s"
                    % name
                )
            child[str(name)] = str(value)
        return child


__all__ = [
    "ChildEnvironmentPolicy",
    "is_secret_environment_name",
    "local_docker_environment",
]
