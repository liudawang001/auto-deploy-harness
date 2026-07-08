import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

from auto_harness.config import HarnessConfig
from auto_harness.models.base import read_json, write_json
from auto_harness.orchestrator import TaskRunner
from auto_harness.utils.time import compact_timestamp


class LiveAgentSmokeRunner:
    """Runs optional live agent smoke and writes a redacted manifest."""

    def run(
        self,
        repo: Path,
        provider: str,
        execute: bool,
        output_dir: Path,
        config: HarnessConfig = None,
        analyze_planner: bool = True,
        resume_attempts: int = 1,
    ) -> Dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        missing = self._missing_provider_env(provider)
        if missing:
            manifest_path = output_dir / "live-agent-smoke-manifest.json"
            manifest = self._skipped_manifest(provider, missing, manifest_path)
            return {
                "status": "skipped",
                "reason": "provider environment is not configured",
                "missing_env": missing,
                "manifest_path": str(manifest_path),
                "manifest": manifest,
            }
        config = config or HarnessConfig.load()
        config.runs_dir = str(output_dir / "runs")
        config.agent_mode = "gated_actor"
        config.agent_provider = provider
        config.agent_enable_analyze_planner = bool(analyze_planner)
        config.agent_enable_log_diagnosis = True
        config.agent_enable_verify_planner = True
        config.agent_enable_repair_actions = True
        config.agent_auto_resume_after_repair = True
        runner = TaskRunner(config)
        task_id = runner.deploy(
            str(repo),
            "live-agent-smoke",
            dry_run=not execute,
            skip_clone=False,
            allow_install=execute,
            allow_start=execute,
        )
        resumed = 0
        for _ in range(max(0, int(resume_attempts or 0))):
            run_dir = Path(config.runs_dir) / task_id
            if not self._should_resume(run_dir):
                break
            runner.resume(task_id, dry_run=not execute)
            resumed += 1
        run_dir = Path(config.runs_dir) / task_id
        manifest_path = output_dir / "live-agent-smoke-manifest.json"
        manifest = self.build_manifest(run_dir, provider_name=provider, output_path=manifest_path)
        manifest["resume_attempt_count"] = resumed
        write_json(manifest_path, manifest)
        return {"status": "completed", "task_id": task_id, "run_dir": str(run_dir), "manifest_path": str(manifest_path), "manifest": manifest}

    def _missing_provider_env(self, provider: str) -> List[str]:
        if provider != "xunfei":
            return []
        missing = []
        if not (os.environ.get("XUNFEI_API_URL") or os.environ.get("XUNFEI_API_BASE")):
            missing.append("XUNFEI_API_URL or XUNFEI_API_BASE")
        for name in ("XUNFEI_API_KEY", "XUNFEI_MODEL"):
            if not os.environ.get(name):
                missing.append(name)
        return missing

    def _skipped_manifest(self, provider_name: str, missing_env: List[str], output_path: Path) -> Dict:
        manifest = {
            "task_id": "",
            "provider_name": provider_name,
            "model_name": "",
            "stage_summary": {},
            "agent_action_count": 0,
            "rejected_action_count": 0,
            "repair_executed_count": 0,
            "resume_attempt_count": 0,
            "final_verify_status": "skipped",
            "artifact_paths": [],
            "sha256": {},
            "external_gate": {
                "status": "external_required",
                "reason": "provider environment is not configured",
                "missing_env": missing_env,
            },
        }
        write_json(Path(output_path), manifest)
        return manifest

    def build_manifest(self, run_dir: Path, provider_name: str, output_path: Path = None) -> Dict:
        run_dir = Path(run_dir)
        pipeline = self._read_optional(run_dir / "reports" / "pipeline_results.json") or {}
        repair_apply = self._read_optional(run_dir / "repairs" / "repair_apply_result.json") or {}
        agent_counts = self._agent_counts(run_dir)
        manifest = {
            "task_id": run_dir.name,
            "provider_name": provider_name,
            "model_name": self._model_name(run_dir),
            "stage_summary": {stage: {"status": result.get("status"), "summary": result.get("summary")} for stage, result in pipeline.items() if isinstance(result, dict)},
            "agent_action_count": agent_counts["action_count"],
            "rejected_action_count": agent_counts["rejected_action_count"],
            "repair_executed_count": self._repair_executed_count(run_dir, repair_apply),
            "resume_attempt_count": self._resume_attempt_count(run_dir),
            "final_verify_status": pipeline.get("verify", {}).get("status", ""),
            "artifact_paths": self._artifact_paths(run_dir),
        }
        manifest["sha256"] = {item: self._sha256(run_dir / item) for item in manifest["artifact_paths"]}
        if output_path:
            write_json(Path(output_path), manifest)
        return manifest

    def _repair_executed_count(self, run_dir: Path, latest_apply: Dict) -> int:
        count = int(latest_apply.get("executed_action_count") or 0)
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            return count
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if '"memory_recorded"' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            repair_apply = data.get("repair_apply") if isinstance(data.get("repair_apply"), dict) else {}
            count += int(repair_apply.get("executed_action_count") or 0)
        return count

    def _should_resume(self, run_dir: Path) -> bool:
        pipeline = self._read_optional(run_dir / "reports" / "pipeline_results.json") or {}
        verify = pipeline.get("verify") if isinstance(pipeline.get("verify"), dict) else {}
        if verify.get("status") in ("pass", "passed"):
            return False
        for result in pipeline.values():
            if not isinstance(result, dict):
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            loop = data.get("agent_loop") if isinstance(data.get("agent_loop"), dict) else {}
            if loop.get("should_auto_resume"):
                return True
        return False

    def _resume_attempt_count(self, run_dir: Path) -> int:
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            return 0
        count = 0
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if '"resume_requested"' in line:
                count += 1
        return count

    def _agent_counts(self, run_dir: Path) -> Dict:
        action_count = 0
        rejected = 0
        for path in (run_dir / "logs" / "agent_calls").glob("*.json"):
            data = self._read_optional(path) or {}
            decision = data.get("parsed_decision") if isinstance(data.get("parsed_decision"), dict) else {}
            action_count += len(decision.get("actions") or [])
            policy = data.get("policy_result") if isinstance(data.get("policy_result"), dict) else {}
            rejected += len(policy.get("rejected_actions") or [])
        return {"action_count": action_count, "rejected_action_count": rejected}

    def _model_name(self, run_dir: Path) -> str:
        for path in (run_dir / "logs" / "agent_calls").glob("*.json"):
            data = self._read_optional(path) or {}
            if data.get("model"):
                return str(data["model"])
        return ""

    def _artifact_paths(self, run_dir: Path) -> List[str]:
        candidates = [
            "task.json",
            "state.json",
            "events.jsonl",
            "reports/pipeline_results.json",
            "repairs/repair_plan.json",
            "repairs/repair_apply_result.json",
        ]
        candidates.extend(str(path.relative_to(run_dir)) for path in sorted((run_dir / "evidence").glob("*verify*.json")))
        candidates.extend(str(path.relative_to(run_dir)) for path in sorted((run_dir / "logs" / "agent_calls").glob("*.json")))
        return [item for item in candidates if (run_dir / item).exists()]

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _read_optional(self, path: Path):
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
