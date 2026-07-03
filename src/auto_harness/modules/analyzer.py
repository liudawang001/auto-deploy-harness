import json
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.agents.base import AgentExecutor, AgentRequest
from auto_harness.models.result import StageResult
from auto_harness.providers.json_utils import parse_json_object


class ProjectAnalyzer:
    def __init__(self, agent_executor: Optional[AgentExecutor] = None, use_agent: bool = False, agent_timeout_seconds: int = 900) -> None:
        self.agent_executor = agent_executor
        self.use_agent = use_agent
        self.agent_timeout_seconds = agent_timeout_seconds

    def analyze(self, repo_dir: Path) -> StageResult:
        files = self._collect_files(repo_dir)
        frameworks = self._detect_frameworks(repo_dir, files)
        install_plan = self._install_plan(files)
        run_candidates = self._run_candidates(repo_dir, files, frameworks)
        data: Dict = {
            "files": files[:200],
            "frameworks": frameworks,
            "install_plan": install_plan,
            "run_candidates": run_candidates,
            "verify_hint": self._verify_hint(frameworks),
        }
        agent_advice = self._agent_advice(repo_dir, data)
        if agent_advice:
            data["agent_advice"] = agent_advice
        return StageResult(
            stage="analyze",
            status="passed",
            summary="project analysis completed",
            data=data,
            evidence=[],
        )

    def _collect_files(self, repo_dir: Path) -> List[str]:
        result: List[str] = []
        for path in repo_dir.rglob("*"):
            if ".git" in path.parts or path.is_dir():
                continue
            try:
                result.append(str(path.relative_to(repo_dir)))
            except ValueError:
                continue
        return sorted(result)

    def _detect_frameworks(self, repo_dir: Path, files: List[str]) -> List[str]:
        frameworks: List[str] = []
        text = ""
        for name in ("requirements.txt", "pyproject.toml", "README.md", "readme.md", "package.json"):
            path = repo_dir / name
            if path.exists():
                try:
                    text += "\n" + path.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    pass
        for key in ("gradio", "streamlit", "fastapi", "flask", "torch", "transformers"):
            if key in text:
                frameworks.append(key)
        if "package.json" in files:
            frameworks.append("node")
        if not frameworks:
            frameworks.append("unknown")
        return sorted(set(frameworks))

    def _install_plan(self, files: List[str]) -> List[List[str]]:
        plan: List[List[str]] = []
        if "requirements.txt" in files:
            plan.append(["python3", "-m", "venv", ".venv"])
            plan.append([".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"])
        elif "pyproject.toml" in files:
            plan.append(["python3", "-m", "venv", ".venv"])
            plan.append([".venv/bin/python", "-m", "pip", "install", "."])
        elif "package.json" in files:
            plan.append(["npm", "install"])
        return plan

    def _run_candidates(self, repo_dir: Path, files: List[str], frameworks: List[str]) -> List[Dict]:
        candidates: List[Dict] = []
        for entry in ("app.py", "main.py", "server.py", "webui.py", "demo.py"):
            if entry in files:
                candidates.append({"cmd": [".venv/bin/python", entry], "expected_port": 7860, "confidence": 0.7})
        if "streamlit" in frameworks:
            for entry in ("app.py", "main.py", "demo.py"):
                if entry in files:
                    candidates.append({"cmd": [".venv/bin/streamlit", "run", entry], "expected_port": 8501, "confidence": 0.8})
        return candidates

    def _verify_hint(self, frameworks: List[str]) -> Dict:
        if "gradio" in frameworks:
            return {
                "service_type": "webui",
                "expected_output": "web_result",
                "request": {
                    "method": "POST",
                    "path": "/api/predict",
                    "json": {"data": ["{{trace_id}}"]},
                },
            }
        if "fastapi" in frameworks or "flask" in frameworks:
            return {
                "service_type": "api",
                "expected_output": "json_or_text",
                "request": {"method": "GET"},
            }
        return {"service_type": "unknown", "expected_output": "unknown"}

    def _agent_advice(self, repo_dir: Path, analysis: Dict) -> Optional[Dict]:
        if not self.use_agent or not self.agent_executor:
            return None
        prompt = (
            "You are an AI deployment analyzer. Return JSON only. "
            "Review the deterministic project analysis and suggest safer install/run/verify improvements. "
            "Do not propose source code edits.\n\n"
            + json.dumps(analysis, ensure_ascii=False, indent=2)
        )
        result = self.agent_executor.run(
            AgentRequest(
                stage="analyze",
                prompt=prompt,
                workdir=repo_dir,
                timeout_seconds=self.agent_timeout_seconds,
            )
        )
        if result.status != "passed":
            return {"status": "failed", "error": result.error}
        try:
            parsed = parse_json_object(result.text)
        except Exception as exc:  # noqa: BLE001 - advice is optional
            return {"status": "invalid_json", "error": str(exc), "raw_tail": result.text[-1000:]}
        if isinstance(parsed, dict):
            parsed.setdefault("status", "ok")
            return parsed
        return {"status": "invalid_shape", "raw": parsed}
