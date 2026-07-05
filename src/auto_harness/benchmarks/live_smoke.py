from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LiveSmokeTarget:
    id: str
    source: str
    repo: str
    name: str
    purpose: str
    expected_signals: List[str]
    required_env_vars: List[str]
    estimated_minutes: int
    command: List[str]
    optional: bool = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "repo": self.repo,
            "name": self.name,
            "purpose": self.purpose,
            "expected_signals": self.expected_signals,
            "required_env_vars": self.required_env_vars,
            "estimated_minutes": self.estimated_minutes,
            "command": self.command,
            "optional": self.optional,
        }


class LiveSmokePlanner:
    """Builds an explicit optional live E2E smoke matrix without running network tasks."""

    def __init__(self, default_execute: bool = False) -> None:
        self.default_execute = default_execute

    def plan(self, include_long_running: bool = False, execution_backend: str = "local") -> Dict:
        targets = self._targets(include_long_running=include_long_running, execution_backend=execution_backend)
        return {
            "status": "planned",
            "kind": "optional_live_e2e_smoke",
            "network_required": True,
            "runs_commands": False,
            "execution_backend": execution_backend,
            "target_count": len(targets),
            "total_estimated_minutes": sum(target.estimated_minutes for target in targets),
            "required_env_vars": sorted({env for target in targets for env in target.required_env_vars}),
            "targets": [target.to_dict() for target in targets],
            "notes": [
                "This plan is intentionally not part of the default benchmark manifest.",
                "Run targets manually only when network, tokens, disk and time budget are available.",
                "Secrets must be injected through environment variables and must not be written to reports or memory.",
            ],
        }

    def _targets(self, include_long_running: bool, execution_backend: str) -> List[LiveSmokeTarget]:
        base = [
            self._target(
                target_id="hf_tiny_gradio_space",
                source="huggingface",
                repo="hf-internal-testing/tiny-random-gpt2",
                name="hf-tiny-gradio",
                purpose="Validate Hugging Face model asset discovery/download/cache with a tiny model repo.",
                expected_signals=[
                    "resource_plan.model_assets includes huggingface source",
                    "model_prepare manifest is written",
                    "verify remains evidence-based and does not pass without trace",
                ],
                required_env_vars=[],
                estimated_minutes=5,
                execution_backend=execution_backend,
            ),
            self._target(
                target_id="modelscope_tiny_model",
                source="modelscope",
                repo="damo/nlp_structbert_backbone_base_std",
                name="modelscope-tiny",
                purpose="Validate ModelScope file listing/download/cache plumbing against a small public model.",
                expected_signals=[
                    "resource_plan.model_assets includes modelscope source",
                    "model_prepare progress is refreshed during download",
                    "manifest records cache path and file metadata",
                ],
                required_env_vars=[],
                estimated_minutes=10,
                execution_backend=execution_backend,
            ),
            self._target(
                target_id="git_lfs_small_weight_repo",
                source="github",
                repo="https://github.com/git-lfs/git-lfs-test-server.git",
                name="git-lfs-smoke",
                purpose="Validate Git LFS detection and controlled prepare command behavior on a real Git repository.",
                expected_signals=[
                    "resource_plan.git_lfs is populated when LFS patterns/pointers exist",
                    "git lfs commands remain guarded by allowed_commands",
                    "model_prepare progress captures git lfs output when executed",
                ],
                required_env_vars=[],
                estimated_minutes=10,
                execution_backend=execution_backend,
            ),
        ]
        if include_long_running:
            base.append(
                self._target(
                    target_id="hf_medium_transformers_demo",
                    source="huggingface",
                    repo="Qwen/Qwen2.5-0.5B-Instruct",
                    name="hf-medium-transformers",
                    purpose="Validate longer model download, cache reuse and first inference verify progress.",
                    expected_signals=[
                        "download progress persists to state.json",
                        "cache hit works on second run",
                        "verify progress includes first_inference_probe_started and verify_completed",
                    ],
                    required_env_vars=["HF_TOKEN"],
                    estimated_minutes=45,
                    execution_backend=execution_backend,
                )
            )
        return base

    def _target(
        self,
        target_id: str,
        source: str,
        repo: str,
        name: str,
        purpose: str,
        expected_signals: List[str],
        required_env_vars: List[str],
        estimated_minutes: int,
        execution_backend: str,
    ) -> LiveSmokeTarget:
        command = [
            "PYTHONPATH=src",
            "python3",
            "-m",
            "auto_harness.cli",
            "deploy",
            "--repo",
            repo,
            "--name",
            name,
        ]
        if self.default_execute:
            command.extend(["--execute", "--allow-install", "--allow-start"])
        else:
            command.append("--dry-run")
        if execution_backend != "local":
            command.extend(["--execution-backend", execution_backend])
        return LiveSmokeTarget(
            id=target_id,
            source=source,
            repo=repo,
            name=name,
            purpose=purpose,
            expected_signals=expected_signals,
            required_env_vars=required_env_vars,
            estimated_minutes=estimated_minutes,
            command=command,
        )
