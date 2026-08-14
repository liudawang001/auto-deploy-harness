from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.assets import GitLFSProgressParser, HuggingFaceDownloader, ModelCache, ModelScopeDownloader
from auto_harness.assets.manifest import AssetManifest, ModelAsset
from auto_harness.diagnostics import LogClassifier
from auto_harness.models.base import write_json
from auto_harness.models.result import StageResult
from auto_harness.runtime import ChildEnvironmentPolicy
from auto_harness.utils.commands import is_allowed_command
from auto_harness.utils.shell import run_command


class ModelPrepareModule:
    def __init__(
        self,
        cache: ModelCache,
        huggingface_downloader: HuggingFaceDownloader = None,
        modelscope_downloader: ModelScopeDownloader = None,
        command_runner=None,
        log_classifier: LogClassifier = None,
        git_lfs_progress_parser: GitLFSProgressParser = None,
    ) -> None:
        self.cache = cache
        self.huggingface_downloader = huggingface_downloader or HuggingFaceDownloader()
        self.modelscope_downloader = modelscope_downloader or ModelScopeDownloader()
        self.command_runner = command_runner or run_command
        self._uses_default_command_runner = command_runner is None
        self.child_environment_policy = ChildEnvironmentPolicy()
        self.log_classifier = log_classifier or LogClassifier()
        self.git_lfs_progress_parser = git_lfs_progress_parser or GitLFSProgressParser()

    def prepare(
        self,
        run_dir: Path,
        resource_plan: Dict,
        execute: bool = False,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        repo_dir: Optional[Path] = None,
        allowed_commands: Optional[List[str]] = None,
        timeout_seconds: int = 900,
    ) -> StageResult:
        raw_assets = resource_plan.get("model_assets") or []
        git_lfs_plan = resource_plan.get("git_lfs") if isinstance(resource_plan.get("git_lfs"), dict) else {}
        git_submodule_plan = resource_plan.get("git_submodules") if isinstance(resource_plan.get("git_submodules"), dict) else {}
        assets = []
        progress = {
            "status": "planned" if raw_assets else "empty",
            "downloaded_bytes": 0,
            "total_bytes": self._resource_total(raw_assets),
            "current_file": "",
        }

        def update_progress(update: Dict) -> None:
            progress.update({key: value for key, value in update.items() if value is not None})
            if progress_callback:
                progress_callback(dict(progress))

        git_lfs_result = self._prepare_git_lfs(
            git_lfs_plan,
            execute=execute,
            repo_dir=repo_dir or run_dir / "workspace" / "repo",
            allowed_commands=allowed_commands or [],
            timeout_seconds=timeout_seconds,
            progress_callback=update_progress,
        )
        git_submodule_result = self._prepare_git_submodules(
            git_submodule_plan,
            execute=execute,
            repo_dir=repo_dir or run_dir / "workspace" / "repo",
            allowed_commands=allowed_commands or [],
            timeout_seconds=timeout_seconds,
            progress_callback=update_progress,
        )

        for raw in raw_assets:
            asset = ModelAsset(**{key: value for key, value in raw.items() if key in ModelAsset.__dataclass_fields__})
            self.cache.reserve(asset)
            if execute:
                if asset.source == "huggingface":
                    update_progress({"status": "downloading", "asset_id": asset.asset_id})
                    asset = self.huggingface_downloader.download(asset, update_progress)
                elif asset.source == "modelscope":
                    update_progress({"status": "downloading", "asset_id": asset.asset_id})
                    asset = self.modelscope_downloader.download(asset, update_progress)
                else:
                    asset.status = "unsupported"
                    asset.last_error = "model source is not supported for download yet"
            assets.append(asset)
            progress["downloaded_bytes"] = sum(asset.downloaded_bytes for asset in assets)
            if progress_callback:
                progress_callback(dict(progress))

        manifest = AssetManifest(
            assets=assets,
            total_expected_size_bytes=self._total_size(assets),
            cache_root=str(self.cache.root),
            status=self._manifest_status(assets),
        )
        manifest_path = run_dir / "reports" / "model_assets_manifest.json"
        write_json(manifest_path, manifest)
        if not assets:
            status = self._combined_status(git_lfs_result, git_submodule_result, assets, execute)
            if self._git_repo_assets_required(git_lfs_result, git_submodule_result):
                summary = self._git_repo_assets_summary(git_lfs_result, git_submodule_result, execute and status == "passed")
            else:
                summary = "no external model assets detected"
            return StageResult(
                "model_prepare",
                status,
                summary,
                {
                    "manifest_path": str(manifest_path),
                    "assets": [],
                    "cache": self.cache.summary(),
                    "progress": progress,
                    "git_lfs": git_lfs_result,
                    "git_submodules": git_submodule_result,
                    "executed": execute,
                },
                evidence=[str(manifest_path)],
                error=self._first_git_error(git_lfs_result, git_submodule_result) if status == "failed" else None,
            )
        status = self._combined_status(git_lfs_result, git_submodule_result, assets, execute)
        summary = "model assets prepared" if execute and status == "passed" else "model assets planned; download not executed"
        if execute and status != "passed":
            summary = "model asset preparation incomplete"
        return StageResult(
            "model_prepare",
            status,
            summary,
            {
                "manifest_path": str(manifest_path),
                "assets": [asset.__dict__ for asset in assets],
                "cache": self.cache.summary(),
                "executed": execute,
                "progress": progress,
                "git_lfs": git_lfs_result,
                "git_submodules": git_submodule_result,
            },
            evidence=[str(manifest_path)],
            error=self._first_git_error(git_lfs_result, git_submodule_result) or (self._first_error(assets) if status != "passed" else None),
        )

    def _prepare_git_lfs(
        self,
        plan: Dict,
        execute: bool,
        repo_dir: Path,
        allowed_commands: List[str],
        timeout_seconds: int,
        progress_callback,
    ) -> Dict:
        if not plan or not plan.get("required"):
            return {"required": False, "executed": False, "commands": []}
        commands = plan.get("prepare_commands") or [["git", "lfs", "install"], ["git", "lfs", "pull"]]
        result = {
            "required": True,
            "available": plan.get("available"),
            "executed": False,
            "commands": [],
            "status": "planned",
            "pointer_count": plan.get("pointer_count", 0),
            "total_pointer_size_bytes": plan.get("total_pointer_size_bytes"),
            "progress": {},
        }
        if not execute:
            return result
        result["executed"] = True
        for cmd in commands:
            progress_callback({"status": "git_lfs_running", "current_file": " ".join(cmd)})
            if not is_allowed_command(cmd, allowed_commands):
                result.update({
                    "status": "failed",
                    "error": "disallowed command: %s" % (cmd[0] if cmd else ""),
                    "diagnosis": {
                        "category": "command_rejected",
                        "signal": cmd[0] if cmd else "",
                        "suggested_fix": "allow git command before executing git lfs preparation",
                        "confidence": 0.9,
                    },
                })
                return result
            command_result = self._run_repo_command(cmd, repo_dir, timeout_seconds)
            parsed_progress = self.git_lfs_progress_parser.parse(command_result.stdout + "\n" + command_result.stderr)
            if parsed_progress:
                result["progress"] = parsed_progress
                progress_callback(parsed_progress)
            record = {
                "cmd": command_result.cmd,
                "exit_code": command_result.exit_code,
                "stdout_tail": command_result.stdout[-4000:],
                "stderr_tail": command_result.stderr[-4000:],
                "timed_out": command_result.timed_out,
            }
            if parsed_progress:
                record["progress"] = parsed_progress
            result["commands"].append(record)
            if command_result.exit_code != 0:
                diagnosis = self.log_classifier.classify(command_result.stderr + "\n" + command_result.stdout)
                result.update({
                    "status": "failed",
                    "error": command_result.stderr[-2000:] or command_result.stdout[-2000:],
                    "diagnosis": diagnosis,
                })
                return result
        result["status"] = "ready"
        final_progress = {"status": "git_lfs_ready", "current_file": "", "percent": 100}
        if result.get("progress"):
            final_progress.update({key: value for key, value in result["progress"].items() if key not in ("status", "percent")})
        result["progress"] = final_progress
        progress_callback(final_progress)
        return result

    def _prepare_git_submodules(
        self,
        plan: Dict,
        execute: bool,
        repo_dir: Path,
        allowed_commands: List[str],
        timeout_seconds: int,
        progress_callback,
    ) -> Dict:
        if not plan or not plan.get("required"):
            return {"required": False, "executed": False, "commands": []}
        commands = plan.get("prepare_commands") or [
            ["git", "submodule", "sync", "--recursive"],
            ["git", "submodule", "update", "--init", "--recursive"],
        ]
        result = {
            "required": True,
            "available": plan.get("available"),
            "executed": False,
            "commands": [],
            "status": "planned",
            "submodule_count": plan.get("submodule_count", 0),
            "submodules": plan.get("submodules") or [],
            "progress": {},
        }
        if not execute:
            return result
        result["executed"] = True
        for cmd in commands:
            progress_callback({"status": "git_submodule_running", "current_file": " ".join(cmd)})
            if not is_allowed_command(cmd, allowed_commands):
                result.update({
                    "status": "failed",
                    "error": "disallowed command: %s" % (cmd[0] if cmd else ""),
                    "diagnosis": {
                        "category": "command_rejected",
                        "signal": cmd[0] if cmd else "",
                        "suggested_fix": "allow git command before executing git submodule preparation",
                        "confidence": 0.9,
                    },
                })
                return result
            command_result = self._run_repo_command(cmd, repo_dir, timeout_seconds)
            record = {
                "cmd": command_result.cmd,
                "exit_code": command_result.exit_code,
                "stdout_tail": command_result.stdout[-4000:],
                "stderr_tail": command_result.stderr[-4000:],
                "timed_out": command_result.timed_out,
            }
            result["commands"].append(record)
            if command_result.exit_code != 0:
                diagnosis = self.log_classifier.classify(command_result.stderr + "\n" + command_result.stdout)
                result.update({
                    "status": "failed",
                    "error": command_result.stderr[-2000:] or command_result.stdout[-2000:],
                    "diagnosis": diagnosis,
                })
                return result
        result["status"] = "ready"
        final_progress = {
            "status": "git_submodule_ready",
            "current_file": "",
            "submodule_count": result.get("submodule_count", 0),
        }
        result["progress"] = final_progress
        progress_callback(final_progress)
        return result

    def _run_repo_command(self, cmd: List[str], repo_dir: Path, timeout_seconds: int):
        if not self._uses_default_command_runner:
            return self.command_runner(cmd, repo_dir, timeout_seconds=timeout_seconds)
        return self.command_runner(
            cmd,
            repo_dir,
            timeout_seconds=timeout_seconds,
            env=self.child_environment_policy.build_for_install(
                home_dir=repo_dir.parent / "model_prepare_home",
            ),
        )

    def _total_size(self, assets):
        sizes = [asset.expected_size_bytes for asset in assets if asset.expected_size_bytes]
        if not sizes:
            return None
        return sum(sizes)

    def _resource_total(self, raw_assets):
        sizes = [raw.get("expected_size_bytes") for raw in raw_assets if raw.get("expected_size_bytes")]
        if not sizes:
            return None
        return sum(sizes)

    def _manifest_status(self, assets):
        if not assets:
            return "empty"
        if any(asset.status == "failed" for asset in assets):
            return "failed"
        if any(asset.status == "unsupported" for asset in assets):
            return "partial"
        if all(asset.status in ("downloaded", "cached") for asset in assets):
            return "ready"
        return "planned"

    def _first_error(self, assets):
        for asset in assets:
            if asset.last_error:
                return asset.last_error
        return None

    def _combined_status(self, git_lfs_result: Dict, git_submodule_result: Dict, assets, execute: bool) -> str:
        if git_lfs_result.get("status") == "failed" or git_submodule_result.get("status") == "failed":
            return "failed"
        if execute and any(asset.status == "failed" for asset in assets):
            return "failed"
        if execute and any(asset.status == "unsupported" for asset in assets):
            return "uncertain"
        return "passed"

    def _git_repo_assets_required(self, git_lfs_result: Dict, git_submodule_result: Dict) -> bool:
        return bool(git_lfs_result.get("required") or git_submodule_result.get("required"))

    def _git_repo_assets_summary(self, git_lfs_result: Dict, git_submodule_result: Dict, prepared: bool) -> str:
        if git_lfs_result.get("required") and not git_submodule_result.get("required"):
            return "git lfs model assets prepared" if prepared else "git lfs model assets planned; pull not executed"
        if git_submodule_result.get("required") and not git_lfs_result.get("required"):
            return "git submodule assets prepared" if prepared else "git submodule assets planned; preparation not executed"
        return "git repository assets prepared" if prepared else "git repository assets planned; preparation not executed"

    def _first_git_error(self, git_lfs_result: Dict, git_submodule_result: Dict):
        return git_lfs_result.get("error") or git_submodule_result.get("error")

    def prepare_frozen(
        self,
        run_dir: Path,
        plan,
        decision,
        cache_dir: Optional[Path] = None,
        disk_safety_ratio: float = 1.2,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> StageResult:
        """Consume a frozen ModelFilePlan + allowed ResourceDecision (Document A).

        Downloads exactly plan.files, re-verifies required files, and writes the
        atomic complete marker. Does NOT re-resolve the model or re-scan the repo.
        """
        if getattr(decision, "status", None) != "allowed":
            return StageResult(
                "model_prepare",
                "blocked",
                "resource decision is not allowed: %s" % getattr(decision, "status", ""),
                {"status": getattr(decision, "status", "blocked")},
            )
        model_identity = getattr(plan, "model_identity", "") or ""
        if ":" not in model_identity:
            return StageResult("model_prepare", "failed", "invalid model identity", {})
        source = model_identity.split(":", 1)[0]
        rest = model_identity.split(":", 1)[1]
        repo_id, revision = rest.rsplit("@", 1) if "@" in rest else (rest, "")
        downloader = (
            self.huggingface_downloader if source == "huggingface" else self.modelscope_downloader
        )
        cache_path = Path(cache_dir) if cache_dir else Path(plan.model_identity.replace(":", "_").replace("/", "_"))
        result = downloader.download_plan(
            repo_id, revision, plan, cache_path,
            progress_callback=progress_callback,
            disk_safety_ratio=disk_safety_ratio,
        )
        if result.get("status") == "complete":
            return StageResult(
                "model_prepare",
                "passed",
                "frozen model file plan downloaded and verified",
                {
                    "status": "complete",
                    "complete_marker_path": result.get("complete_marker_path", ""),
                    "cache_dir": str(cache_path),
                },
                evidence=[result.get("complete_marker_path", "")] if result.get("complete_marker_path") else [],
            )
        return StageResult(
            "model_prepare",
            "failed",
            result.get("status", "failed"),
            result,
            error=result.get("error") or result.get("status", "failed"),
        )
