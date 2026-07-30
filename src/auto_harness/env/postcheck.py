"""Deterministic post-check for created Conda environments."""
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from auto_harness.preflight.conda import _completed
from auto_harness.preflight.schemas import EnvironmentPostcheckEvidence


PROBE_SCRIPT = (
    "import importlib.metadata as m,json,sys;"
    "names=%r;"
    "safe=lambda n:(m.version(n) if n else '');"
    "\ndef version(n):\n"
    " try:return safe(n)\n"
    " except m.PackageNotFoundError:return ''\n"
    "pkgs={n:version(n) for n in names};"
    "gpu={'required':%r,'framework':'torch','available':False,'device_count':0,'device_names':[],'torch_cuda_version':''};"
    "\ntry:\n import torch\n gpu.update(available=bool(torch.cuda.is_available()),device_count=int(torch.cuda.device_count()),"
    "device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
    "torch_cuda_version=str(torch.version.cuda or ''))\nexcept Exception: pass\n"
    "print(json.dumps({'executable':sys.executable,'version':'.'.join(map(str,sys.version_info[:3])),'packages':pkgs,'gpu_runtime':gpu}))"
)


class EnvironmentPostchecker:
    def __init__(self, command_runner=None, timeout_seconds: int = 30) -> None:
        self.command_runner = command_runner or subprocess.run
        self.timeout_seconds = timeout_seconds

    def check(
        self,
        tool: str,
        prefix: Path,
        python_constraint: str,
        package_specs: List[str],
        gpu_required: bool,
        spec_hash: str,
    ) -> Dict:
        prefix = Path(prefix).resolve()
        names = [self._probe_name(spec) for spec in package_specs if self._probe_name(spec)]
        script = PROBE_SCRIPT % (names, bool(gpu_required))
        command = [tool, "run", "-p", str(prefix), "python", "-c", script]
        try:
            result = self.command_runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return EnvironmentPostcheckEvidence(
                prefix=str(prefix), spec_hash=spec_hash, errors=["postcheck_timeout"]
            ).to_dict()
        except (OSError, subprocess.SubprocessError) as exc:
            return EnvironmentPostcheckEvidence(
                prefix=str(prefix), spec_hash=spec_hash, errors=[str(exc)[:500]]
            ).to_dict()
        code, stdout, stderr = _completed(result)
        if code != 0:
            return EnvironmentPostcheckEvidence(
                prefix=str(prefix), spec_hash=spec_hash, errors=[stderr[-1000:] or "postcheck_failed"]
            ).to_dict()
        try:
            payload = json.loads((stdout or "")[-20000:])
        except ValueError:
            return EnvironmentPostcheckEvidence(
                prefix=str(prefix), spec_hash=spec_hash, errors=["invalid_postcheck_json"]
            ).to_dict()
        executable = str(payload.get("executable", ""))
        version = str(payload.get("version", ""))
        python_ok = self._python_satisfied(version, python_constraint) and self._inside(executable, prefix)
        installed = {
            self._canonical_name(k): str(v)
            for k, v in (payload.get("packages") or {}).items()
        }
        mismatches = [spec for spec in package_specs if not self._package_satisfied(spec, installed)]
        gpu = payload.get("gpu_runtime") or {}
        gpu_ok = not gpu_required or bool(gpu.get("available"))
        status = "passed" if python_ok and not mismatches and gpu_ok else "failed"
        return EnvironmentPostcheckEvidence(
            status=status,
            prefix=str(prefix),
            python={
                "executable": executable,
                "version": version,
                "constraint_satisfied": python_ok,
            },
            packages={
                "checked": names,
                "all_satisfied": not mismatches,
                "mismatches": mismatches,
            },
            gpu_runtime={"required": gpu_required, **gpu},
            spec_hash=spec_hash,
            errors=[] if status == "passed" else ["environment_postcheck_failed"],
        ).to_dict()

    def _package_satisfied(self, spec, installed):
        name = self._probe_name(spec)
        if not name or name in ("python", "pip", "pytorch-cuda", "cpuonly"):
            return True
        version = installed.get(self._canonical_name(name))
        if not version:
            return False
        raw = str(spec).strip()
        try:
            from packaging.requirements import Requirement
            from packaging.specifiers import SpecifierSet
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                return True
            return not requirement.specifier or version in requirement.specifier
        except ImportError:
            return False
        except ValueError:
            constraint = self._conda_constraint(raw, name)
            if constraint is None:
                return False
            if not constraint:
                return True
            try:
                return version in SpecifierSet(constraint)
            except ValueError:
                return False

    @classmethod
    def _probe_name(cls, spec):
        name = cls._package_name(spec)
        return "torch" if name == "pytorch" else name

    @staticmethod
    def _package_name(spec):
        raw = str(spec).strip()
        if "::" in raw:
            raw = raw.split("::", 1)[1]
        try:
            from packaging.requirements import Requirement
            return EnvironmentPostchecker._canonical_name(Requirement(raw).name)
        except (ImportError, ValueError):
            name = re.split(r"[<>=~!;\[]", raw, maxsplit=1)[0]
            return EnvironmentPostchecker._canonical_name(name)

    @staticmethod
    def _python_satisfied(actual, expected):
        if not expected:
            return True
        raw = str(expected).strip()
        if re.fullmatch(r"\d+\.\d+(?:\.\*)?", raw):
            raw = "==%s.*" % raw.removesuffix(".*")
        elif re.fullmatch(r"\d+\.\d+\.\d+", raw):
            raw = "==%s" % raw
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
            return Version(str(actual)) in SpecifierSet(raw)
        except (ImportError, ValueError):
            return False

    @classmethod
    def _conda_constraint(cls, raw, name):
        value = str(raw).strip()
        if "::" in value:
            value = value.split("::", 1)[1]
        prefix = re.split(r"[<>=~!]", value, maxsplit=1)[0].strip()
        constraint = value[len(prefix):].strip()
        if not constraint:
            return ""
        if constraint.startswith("=") and not constraint.startswith(("==", ">=", "<=", "!=")):
            version_and_build = constraint[1:].split("=", 1)
            version = version_and_build[0].strip()
            if not re.fullmatch(r"[A-Za-z0-9*_.+-]+", version):
                return None
            return "==%s" % (version if "*" in version else version + ".*")
        return constraint

    @staticmethod
    def _canonical_name(name):
        try:
            from packaging.utils import canonicalize_name
            return str(canonicalize_name(str(name)))
        except ImportError:
            return str(name).strip().lower().replace("_", "-")

    @staticmethod
    def _inside(executable, prefix):
        try:
            Path(executable).resolve().relative_to(prefix)
            return True
        except ValueError:
            return False
