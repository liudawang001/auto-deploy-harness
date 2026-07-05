from pathlib import Path
from typing import Callable, Dict, List, Optional

from auto_harness.assets import GitLFSProgressParser, HuggingFaceDownloader, ModelCache, ModelScopeDownloader
from auto_harness.assets.manifest import AssetManifest, ModelAsset
from auto_harness.diagnostics import LogClassifier
from auto_harness.models.base import write_json
from auto_harness.models.result import StageResult
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
            status = "failed" if git_lfs_result.get("status") == "failed" else "passed"
            if git_lfs_result.get("required") and execute and status == "passed":
                summary = "git lfs model assets prepared"
            elif git_lfs_result.get("required"):
                summary = "git lfs model assets planned; pull not executed"
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
                    "executed": execute,
                },
                evidence=[str(manifest_path)],
                error=git_lfs_result.get("error") if status == "failed" else None,
            )
        if git_lfs_result.get("status") == "failed":
            status = "failed"
        elif execute and any(asset.status == "failed" for asset in assets):
            status = "failed"
        elif execute and any(asset.status == "unsupported" for asset in assets):
            status = "uncertain"
        else:
            status = "passed"
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
            },
            evidence=[str(manifest_path)],
            error=git_lfs_result.get("error") or (self._first_error(assets) if status != "passed" else None),
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
            command_result = self.command_runner(cmd, repo_dir, timeout_seconds=timeout_seconds)
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
