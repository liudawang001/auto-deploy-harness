import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

from auto_harness.env.schemas import EnvironmentSpec
from auto_harness.utils.files import safe_name


ALLOWED_CONDA_CHANNELS = {"defaults", "conda-forge", "pytorch", "nvidia", "fastai"}


class CondaProbe:
    def probe(self) -> Dict:
        conda = shutil.which("conda") or ""
        mamba = shutil.which("mamba") or ""
        micromamba = shutil.which("micromamba") or ""
        return {
            "conda": conda,
            "mamba": mamba,
            "micromamba": micromamba,
            "available": bool(conda or mamba or micromamba),
        }


class CondaEnvironmentParser:
    def parse_repo(self, repo_dir: Path, default_python: str = "3.10") -> Dict:
        for name in ("environment.yml", "environment.yaml"):
            path = Path(repo_dir) / name
            if path.exists():
                return self.parse(path, default_python=default_python)
        return {"found": False}

    def parse(self, path: Path, default_python: str = "3.10") -> Dict:
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return {"found": False, "error": "environment file is not readable"}
        data = self._load_yaml(raw)
        parser = data.pop("_parser", "pyyaml")
        name = safe_name(str(data.get("name") or path.parent.name or "auto-harness"))
        raw_channels = [str(item).strip() for item in (data.get("channels") or []) if str(item).strip()]
        channels = self._safe_channels(raw_channels)
        conda_deps, pip_deps, python = self._dependencies(data.get("dependencies") or [], default_python)
        torch = self._torch_from_deps(conda_deps + pip_deps)
        return {
            "found": True,
            "parser": parser,
            "path": str(path),
            "name": name,
            "channels": channels,
            "rejected_channels": [item for item in raw_channels if item not in channels],
            "python": python,
            "conda_dependencies": conda_deps,
            "pip_dependencies": pip_deps,
            "torch": torch,
        }

    def _load_yaml(self, raw: str) -> Dict:
        try:
            import yaml
            data = yaml.safe_load(raw) or {}
            if isinstance(data, dict):
                data["_parser"] = "pyyaml"
                return data
        except Exception:
            pass
        return self._fallback_parse(raw)

    def _fallback_parse(self, raw: str) -> Dict:
        data: Dict = {"channels": [], "dependencies": [], "_parser": "fallback"}
        section = ""
        in_pip = False
        pip_items: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*:", stripped):
                key, value = stripped.split(":", 1)
                section = key.strip()
                in_pip = False
                if key.strip() == "name" and value.strip():
                    data["name"] = value.strip()
                continue
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item == "pip:":
                    in_pip = True
                    pip_items = []
                    data["dependencies"].append({"pip": pip_items})
                    continue
                if section == "channels":
                    data["channels"].append(item)
                elif section == "dependencies" and in_pip:
                    pip_items.append(item)
                elif section == "dependencies":
                    data["dependencies"].append(item)
        return data

    def _dependencies(self, values: List, default_python: str) -> tuple:
        conda_deps: List[str] = []
        pip_deps: List[str] = []
        python = default_python
        for item in values:
            if isinstance(item, str):
                conda_deps.append(item)
                if item.startswith("python="):
                    python = item.split("=", 1)[1]
            elif isinstance(item, dict) and isinstance(item.get("pip"), list):
                pip_deps.extend(str(dep) for dep in item["pip"] if str(dep).strip())
        return conda_deps, pip_deps, python

    def _safe_channels(self, channels: List) -> List[str]:
        safe = []
        for channel in channels:
            item = str(channel).strip()
            if item in ALLOWED_CONDA_CHANNELS and item not in safe:
                safe.append(item)
        return safe

    def _torch_from_deps(self, deps: List[str]) -> Dict:
        packages = [self._package_name(dep) for dep in deps]
        cuda = ""
        for dep in deps:
            if str(dep).startswith("pytorch-cuda="):
                cuda = str(dep).split("=", 1)[1]
        return {
            "packages": [pkg for pkg in ("pytorch", "torch", "torchvision", "torchaudio") if pkg in packages],
            "conda_cuda": cuda,
            "requires_torch": any(pkg in packages for pkg in ("pytorch", "torch", "torchvision", "torchaudio")),
        }

    def _package_name(self, spec: str) -> str:
        return re.split(r"[<>=~!;\[]", str(spec).strip(), maxsplit=1)[0].strip().lower().replace("_", "-")


class CondaBackend:
    def __init__(
        self,
        backend: str = "conda",
        envs_dir: str = ".conda/envs",
        prefer_mamba: bool = True,
        allowed_channels: List[str] = None,
        default_python: str = "3.10",
        probe: CondaProbe = None,
    ) -> None:
        self.backend = backend
        self.envs_dir = envs_dir
        self.prefer_mamba = prefer_mamba
        self.allowed_channels = set(allowed_channels or sorted(ALLOWED_CONDA_CHANNELS))
        self.default_python = default_python
        self.probe = probe or CondaProbe()

    def build_spec(self, repo_dir: Path, env_solution: Dict, conda_file: Dict = None) -> EnvironmentSpec:
        conda_file = conda_file or {}
        strategy = env_solution.get("environment_strategy") or {}
        decision = env_solution.get("compatibility_decision") or {}
        name = safe_name(str(conda_file.get("name") or strategy.get("name") or Path(repo_dir).name or "auto-harness"))
        python = str(strategy.get("python") or conda_file.get("python") or env_solution.get("python") or self.default_python)
        backend = str(decision.get("backend") or env_solution.get("backend") or strategy.get("backend") or self.backend)
        if backend == "auto":
            backend = "conda"
        channels = self._channels(strategy.get("channels") or conda_file.get("channels") or [])
        prefix = str(decision.get("target_prefix") or (Path(self.envs_dir) / name))
        return EnvironmentSpec(
            backend=backend,
            name=name,
            prefix=prefix,
            python=python,
            channels=channels,
            conda_dependencies=list(conda_file.get("conda_dependencies") or []),
            pip_dependencies=list(conda_file.get("pip_dependencies") or []),
            torch=dict(env_solution.get("torch_solution") or {}),
            source_files=[conda_file.get("path")] if conda_file.get("path") else [],
            tool_path=str(decision.get("tool") or ""),
            action=str(decision.get("action") or "create"),
            spec_hash=str(decision.get("spec_hash") or ""),
            project_id=str(decision.get("project_id") or ""),
            repo_fingerprint=str(decision.get("repo_fingerprint") or ""),
        )

    def command_plan(self, spec: EnvironmentSpec, pip_plan: List[List[str]] = None) -> Dict:
        tool = spec.tool_path or self._tool(spec.backend)
        if spec.action == "reuse":
            return {
                "environment_backend": spec.backend,
                "environment_prefix": spec.prefix,
                "environment_python": str(Path(spec.prefix) / "bin" / "python"),
                "tool": tool,
                "action": "reuse",
                "commands": [],
                "spec": self._spec_dict(spec),
            }
        channels = self._channel_args(spec.channels)
        create = [tool, "create", "-y", "-p", spec.prefix] + channels + [
            "python=%s" % spec.python,
            "pip",
        ]
        conda_install = self._conda_install(tool, spec, channels)
        pip_commands = self._pip_commands(tool, spec, pip_plan or [])
        commands = [create]
        if conda_install:
            commands.append(conda_install)
        commands.extend(pip_commands)
        return {
            "environment_backend": spec.backend,
            "environment_prefix": spec.prefix,
            "environment_python": str(Path(spec.prefix) / "bin" / "python"),
            "tool": tool,
            "action": spec.action,
            "commands": commands,
            "spec": self._spec_dict(spec),
        }

    def run_cmd(self, spec: EnvironmentSpec, cmd: List[str]) -> List[str]:
        tool = spec.tool_path or self._tool(spec.backend)
        raw = list(cmd)
        if raw and raw[0].endswith("/python"):
            raw = ["python"] + raw[1:]
        elif raw and raw[0].startswith(".venv/bin/"):
            raw = [Path(raw[0]).name] + raw[1:]
        elif raw and (raw[0] in ("python", "python3") or raw[0].startswith("python")):
            raw = ["python"] + raw[1:]
        return [tool, "run", "-p", spec.prefix] + raw

    def pip_install_cmd(self, env_context: Dict, package: str) -> List[str]:
        backend = str(env_context.get("backend") or "venv")
        prefix = str(env_context.get("conda_prefix") or env_context.get("environment_prefix") or "")
        tool = "mamba" if backend == "mamba" else "conda"
        if backend in ("conda", "mamba") and prefix:
            return [tool, "run", "-p", prefix, "python", "-m", "pip", "install", package]
        python = str(env_context.get("python_executable") or ".venv/bin/python")
        return [python, "-m", "pip", "install", package]

    def _tool(self, backend: str) -> str:
        if backend == "mamba":
            return "mamba"
        if backend == "micromamba":
            return "micromamba"
        return "conda"

    def _channels(self, channels: List[str]) -> List[str]:
        result = []
        for channel in channels:
            item = str(channel).strip()
            if item in self.allowed_channels and item not in result:
                result.append(item)
        return result

    def _channel_args(self, channels: List[str]) -> List[str]:
        args: List[str] = []
        if channels:
            args.append("--override-channels")
        for channel in channels:
            args.extend(["-c", channel])
        return args

    def _conda_install(self, tool: str, spec: EnvironmentSpec, channels: List[str]) -> List[str]:
        deps = [dep for dep in spec.conda_dependencies if not str(dep).startswith("python=") and dep != "pip"]
        torch = spec.torch.get("selected") if isinstance(spec.torch.get("selected"), dict) else {}
        variant = str(torch.get("variant") or "")
        packages = ["pytorch" if pkg == "torch" else pkg for pkg in torch.get("packages") or []]
        if variant.startswith("cu"):
            cuda = "12.1" if variant == "cu121" else "11.8" if variant == "cu118" else ""
            if cuda and "pytorch-cuda=%s" % cuda not in deps:
                deps.append("pytorch-cuda=%s" % cuda)
        elif variant == "cpu" and packages and "cpuonly" not in deps:
            deps.append("cpuonly")
        for package in packages:
            if package not in deps:
                deps.append(package)
        if not deps:
            return []
        return [tool, "install", "-y", "-p", spec.prefix] + channels + deps

    def _pip_commands(self, tool: str, spec: EnvironmentSpec, pip_plan: List[List[str]]) -> List[List[str]]:
        commands = []
        selected_torch = bool((spec.torch.get("selected") or {}).get("packages"))
        pip_dependencies = [
            dep for dep in spec.pip_dependencies
            if not (selected_torch and self._is_torch_package(dep))
        ]
        if pip_dependencies:
            commands.append(
                [tool, "run", "-p", spec.prefix, "python", "-m", "pip", "install"]
                + pip_dependencies
            )
        for cmd in pip_plan:
            if "-m" in cmd and "pip" in cmd:
                index = cmd.index("pip")
                pip_args = cmd[index + 1:]
                if selected_torch and any(
                    self._is_torch_package(item) for item in pip_args
                    if not str(item).startswith("-")
                ):
                    continue
                commands.append(
                    [tool, "run", "-p", spec.prefix, "python", "-m", "pip"]
                    + pip_args
                )
            elif cmd and Path(cmd[0]).name == "uv" and cmd[1:] == ["sync", "--frozen", "--no-dev"]:
                commands.append(
                    [tool, "run", "-p", spec.prefix, "uv"] + cmd[1:]
                )
            elif self._safe_npm_build_command(cmd):
                # Frontend tools belong to the host project checkout, not the
                # isolated Python environment.  Keep the exact argv command.
                commands.append(list(cmd))
        return commands

    @staticmethod
    def _safe_npm_build_command(cmd: List[str]) -> bool:
        if len(cmd) not in (4, 5) or cmd[:2] != ["npm", "--prefix"]:
            return False
        prefix = str(cmd[2])
        return bool(re.fullmatch(
            r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", prefix,
        )) and cmd[3:] in (["ci"], ["run", "build"])

    @staticmethod
    def _is_torch_package(spec: str) -> bool:
        name = re.split(
            r"[<>=~!;\[]", str(spec).strip(), maxsplit=1,
        )[0].lower().replace("_", "-")
        return name in {"torch", "pytorch", "torchvision", "torchaudio"}

    def _spec_dict(self, spec: EnvironmentSpec) -> Dict:
        return json.loads(json.dumps(spec.__dict__, ensure_ascii=False))
