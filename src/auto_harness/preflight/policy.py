"""Fine-grained policy for environment probes and mutations."""
import re
from pathlib import Path
from typing import Dict, List


UNSAFE_PACKAGE_TOKENS = (
    "http://", "https://", "git+", "file:", "--extra-index",
    "--trusted-host", "|", "&&", "$(",
)
ALLOWED_PIP_INDEX_URLS = {
    "https://download.pytorch.org/whl/cpu",
    "https://download.pytorch.org/whl/cu118",
    "https://download.pytorch.org/whl/cu121",
}
CONDA_SPEC_RE = re.compile(
    r"^[A-Za-z0-9_.-]+(?:[<>=!~]{1,2}[A-Za-z0-9*_.+!,<>=-]+)?$"
)


class EnvironmentPreflightPolicy:
    MUTATION_SUBCOMMANDS = {"create", "install"}

    def evaluate(self, decision: Dict, repo_dir: Path, config, allow_mutation: bool = False) -> Dict:
        reasons: List[str] = []
        decision_status = decision.get("status")
        allowed = decision_status in ("allowed", "uncertain") and decision.get("action") != "block"
        backend = decision.get("backend", "venv")
        prefix = decision.get("target_prefix", "")
        if backend in ("conda", "mamba", "micromamba"):
            if not decision.get("tool"):
                allowed = False
                reasons.append("selected environment tool is unavailable")
            if not self._prefix_allowed(prefix, repo_dir, config):
                allowed = False
                reasons.append("target prefix is outside configured conda_envs_dir")
        return {
            "schema_version": 1,
            "status": decision_status if allowed else "blocked",
            "allowed": allowed,
            "mutation_authorized": bool(
                allowed and decision_status == "allowed" and allow_mutation
            ),
            "reasons": reasons or list(decision.get("reasons") or []),
        }

    def validate_mutation_command(
        self,
        command: List[str],
        decision: Dict,
        repo_dir: Path,
        config,
    ) -> Dict:
        if not command:
            return {"allowed": False, "reason": "empty command"}
        tool = str(Path(command[0]).resolve()) if Path(command[0]).is_absolute() else command[0]
        expected_tool = decision.get("tool") or decision.get("backend")
        if tool != expected_tool:
            return {"allowed": False, "reason": "tool path does not match preflight decision"}
        subcommand = command[1] if len(command) > 1 else ""
        if subcommand == "run":
            return self._validate_run(command, decision, repo_dir)
        if subcommand not in self.MUTATION_SUBCOMMANDS:
            return {"allowed": False, "reason": "unsupported environment mutation subcommand"}
        parsed = self._parse_conda_mutation(command)
        if not parsed.get("allowed"):
            return parsed
        prefix = parsed["prefix"]
        if prefix != decision.get("target_prefix") or not self._prefix_allowed(prefix, repo_dir, config):
            return {"allowed": False, "reason": "environment prefix policy failed"}
        channels = parsed["channels"]
        allowed_channels = set(getattr(config, "conda_allowed_channels", []) or [])
        if any(channel not in allowed_channels for channel in channels):
            return {"allowed": False, "reason": "channel allowlist failed"}
        for package in parsed["packages"]:
            package_result = self._validate_conda_spec(package, allowed_channels)
            if not package_result.get("allowed"):
                return package_result
        return {"allowed": True, "reason": "typed environment command allowed"}

    def _validate_run(self, command, decision, repo_dir):
        prefix = self._argument(command, "-p")
        expected = [command[0], "run", "-p", prefix, "python", "-m", "pip", "install"]
        allowed = (
            command.count("-p") == 1
            and prefix == decision.get("target_prefix")
            and command[:len(expected)] == expected
            and len(command) > len(expected)
        )
        if not allowed:
            return {"allowed": False, "reason": "invalid environment run command"}
        arguments = command[len(expected):]
        index = 0
        package_count = 0
        while index < len(arguments):
            token = str(arguments[index])
            if token in ("-r", "--requirement"):
                if index + 1 >= len(arguments):
                    return {"allowed": False, "reason": "requirement path is missing"}
                requirement_result = self._validate_requirements_file(
                    arguments[index + 1], repo_dir,
                )
                if not requirement_result.get("allowed"):
                    return requirement_result
                package_count += 1
                index += 2
                continue
            if token == "--index-url":
                if index + 1 >= len(arguments) or arguments[index + 1] not in ALLOWED_PIP_INDEX_URLS:
                    return {"allowed": False, "reason": "pip index URL is not allowed"}
                index += 2
                continue
            if token.startswith("-") or not self._valid_pip_requirement(token):
                return {"allowed": False, "reason": "invalid pip requirement"}
            package_count += 1
            index += 1
        return {
            "allowed": package_count > 0,
            "reason": "fixed-prefix pip install" if package_count > 0 else "pip requirement is missing",
        }

    def _parse_conda_mutation(self, command):
        prefixes = []
        channels = []
        packages = []
        index = 2
        while index < len(command):
            token = str(command[index])
            if token in ("-y", "--yes"):
                index += 1
                continue
            if token in ("-p", "--prefix"):
                if index + 1 >= len(command):
                    return {"allowed": False, "reason": "environment prefix is missing"}
                prefixes.append(str(command[index + 1]))
                index += 2
                continue
            if token in ("-c", "--channel"):
                if index + 1 >= len(command):
                    return {"allowed": False, "reason": "Conda channel is missing"}
                channels.append(str(command[index + 1]))
                index += 2
                continue
            if token.startswith("-"):
                return {"allowed": False, "reason": "unsupported Conda option"}
            packages.append(token)
            index += 1
        if len(prefixes) != 1:
            return {"allowed": False, "reason": "exactly one environment prefix is required"}
        if not packages:
            return {"allowed": False, "reason": "Conda package specification is missing"}
        return {
            "allowed": True,
            "prefix": prefixes[0],
            "channels": channels,
            "packages": packages,
        }

    def _validate_conda_spec(self, raw_spec, allowed_channels):
        spec = str(raw_spec).strip()
        lowered = spec.lower()
        if (
            not spec
            or any(item in lowered for item in UNSAFE_PACKAGE_TOKENS)
            or "/" in spec
            or "\\" in spec
            or "@" in spec
            or any(char.isspace() for char in spec)
        ):
            return {"allowed": False, "reason": "unsafe Conda package specification"}
        if "::" in spec:
            channel, spec = spec.split("::", 1)
            if channel not in allowed_channels:
                return {"allowed": False, "reason": "channel-qualified package is not allowed"}
        if not CONDA_SPEC_RE.fullmatch(spec):
            return {"allowed": False, "reason": "invalid Conda package specification"}
        return {"allowed": True, "reason": "Conda package specification allowed"}

    def _validate_requirements_file(self, raw_path, repo_dir):
        repo = Path(repo_dir).resolve()
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = repo / path
        path = path.resolve()
        try:
            path.relative_to(repo)
        except ValueError:
            return {"allowed": False, "reason": "requirements file is outside repository"}
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            return {"allowed": False, "reason": "requirements file is unavailable or too large"}
        try:
            lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
        except (OSError, UnicodeError):
            return {"allowed": False, "reason": "requirements file is not readable"}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-") or not self._valid_pip_requirement(line):
                return {"allowed": False, "reason": "requirements file contains an unsafe entry"}
        return {"allowed": True, "reason": "repository requirements file allowed"}

    @staticmethod
    def _valid_pip_requirement(raw_spec):
        try:
            from packaging.requirements import Requirement
            requirement = Requirement(str(raw_spec))
        except (ImportError, ValueError):
            return False
        return requirement.url is None

    def _prefix_allowed(self, prefix, repo_dir, config):
        if not prefix:
            return False
        repo = Path(repo_dir).resolve()
        root = Path(getattr(config, "conda_envs_dir", ".conda/envs"))
        if not root.is_absolute():
            root = repo / root
        root = root.resolve()
        target = Path(prefix).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return False
        return target != root and target.name not in ("base", "root")

    @staticmethod
    def _argument(command, name):
        if name not in command:
            return ""
        index = command.index(name)
        return command[index + 1] if index + 1 < len(command) else ""

    @staticmethod
    def _arguments(command, name):
        return [
            command[index + 1]
            for index, item in enumerate(command[:-1])
            if item == name
        ]
