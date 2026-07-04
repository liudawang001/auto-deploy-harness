from pathlib import Path
from typing import Dict, List, Tuple

from auto_harness.models.result import StageResult


class EnvSolveModule:
    """Builds a safer dependency installation plan without executing commands."""

    def solve(self, repo_dir: Path, analysis: Dict, resource_plan: Dict) -> StageResult:
        requirements = self._read_requirements(repo_dir)
        frameworks = set(analysis.get("frameworks") or [])
        base_plan = [list(cmd) for cmd in analysis.get("install_plan") or []]
        constraints, reasons = self._constraints(requirements, frameworks)
        install_plan = self._apply_constraints(base_plan, constraints)
        risk_reasons = self._risk_reasons(requirements, frameworks, resource_plan)
        solved_analysis = dict(analysis)
        solved_analysis["install_plan"] = install_plan
        solved_analysis["env_solution"] = {
            "backend": "local_venv" if install_plan else "unknown",
            "python": self._python_choice(resource_plan),
            "python_range": resource_plan.get("python_range", "unknown"),
            "constraints": constraints,
            "constraint_reasons": reasons,
            "torch_variant": resource_plan.get("torch_variant", ""),
            "gpu_required": bool(resource_plan.get("gpu_required")),
            "risk_reasons": risk_reasons,
        }
        status = "passed" if install_plan else "uncertain"
        summary = "environment solution generated" if status == "passed" else "no install plan to solve"
        return StageResult(
            "env_solve",
            status,
            summary,
            {
                "backend": solved_analysis["env_solution"]["backend"],
                "python": solved_analysis["env_solution"]["python"],
                "install_plan": install_plan,
                "constraints": constraints,
                "constraint_reasons": reasons,
                "risk_reasons": risk_reasons,
                "analysis": solved_analysis,
            },
        )

    def _read_requirements(self, repo_dir: Path) -> List[str]:
        path = repo_dir / "requirements.txt"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
        return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    def _constraints(self, requirements: List[str], frameworks: set) -> Tuple[List[str], List[str]]:
        constraints = []
        reasons = []
        req_text = "\n".join(requirements).lower()
        old_gradio = "gradio" in frameworks and not ("gradio>=" in req_text and "gradio>=4" in req_text)
        if old_gradio:
            if not self._has_package_constraint(requirements, "numpy", "<2"):
                constraints.append("numpy<2")
                reasons.append("old or unpinned gradio project detected; numpy 2.x can break legacy dependencies")
            if not self._has_package_constraint(requirements, "pydantic", "<2"):
                constraints.append("pydantic<2")
                reasons.append("old or unpinned gradio project detected; pydantic v2 can break legacy gradio stacks")
        if "opencv-python" in req_text and "opencv-python-headless" not in req_text:
            constraints.append("opencv-python-headless")
            reasons.append("headless deployment should prefer opencv-python-headless")
        return self._dedupe(constraints), self._dedupe(reasons)

    def _has_package_constraint(self, requirements: List[str], package: str, token: str) -> bool:
        prefix = package.lower()
        for requirement in requirements:
            normalized = requirement.lower().replace(" ", "")
            if normalized.startswith(prefix) and token in normalized:
                return True
        return False

    def _apply_constraints(self, base_plan: List[List[str]], constraints: List[str]) -> List[List[str]]:
        if not constraints:
            return base_plan
        plan = [list(cmd) for cmd in base_plan]
        for index, cmd in enumerate(plan):
            if len(cmd) >= 4 and cmd[-2:] == ["-r", "requirements.txt"] and "pip" in cmd:
                plan[index] = cmd + constraints
                return plan
        if plan:
            python_cmd = ".venv/bin/python" if any(".venv" in " ".join(cmd) for cmd in plan) else "python3"
            plan.append([python_cmd, "-m", "pip", "install"] + constraints)
        return plan

    def _risk_reasons(self, requirements: List[str], frameworks: set, resource_plan: Dict) -> List[str]:
        reasons = []
        req_text = "\n".join(requirements).lower()
        if resource_plan.get("gpu_required"):
            reasons.append("GPU/CUDA signals detected; torch wheel variant must match local CUDA runtime")
        if "torch" in frameworks and "torch" not in req_text:
            reasons.append("torch framework detected but requirements do not pin torch")
        if "flash-attn" in req_text:
            reasons.append("flash-attn may require CUDA toolkit and build isolation tuning")
        if "bitsandbytes" in req_text:
            reasons.append("bitsandbytes requires compatible CUDA and platform support")
        return reasons

    def _python_choice(self, resource_plan: Dict) -> str:
        python_range = str(resource_plan.get("python_range") or "")
        if "3.10" in python_range:
            return "3.10"
        if "3.11" in python_range:
            return "3.11"
        return "python3"

    def _dedupe(self, values: List[str]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
