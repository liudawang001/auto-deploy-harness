from pathlib import Path
from typing import Dict, List

from auto_harness.assets import ModelAssetDetector
from auto_harness.models.result import StageResult


class ResourcePlanner:
    def __init__(self, detector: ModelAssetDetector = None) -> None:
        self.detector = detector or ModelAssetDetector()

    def plan(self, repo_dir: Path, analysis: Dict) -> StageResult:
        assets = self.detector.detect(repo_dir, analysis)
        frameworks = set(analysis.get("frameworks") or [])
        gpu_required = self._gpu_required(repo_dir, frameworks)
        estimated_disk = self._estimate_disk(assets, frameworks)
        data = {
            "python_range": self._python_range(repo_dir),
            "gpu_required": gpu_required,
            "cuda_required": "unknown" if gpu_required else "",
            "torch_variant": "cuda_or_cpu" if "torch" in frameworks else "",
            "estimated_vram_gb": 16 if gpu_required else 0,
            "estimated_disk_gb": estimated_disk,
            "external_tokens": self._external_tokens(repo_dir, assets),
            "risk_level": self._risk_level(gpu_required, estimated_disk, assets),
            "risk_reasons": self._risk_reasons(gpu_required, assets),
            "model_assets": [asset.__dict__ for asset in assets],
        }
        return StageResult(
            "resource_plan",
            "passed",
            "resource plan generated",
            data,
            evidence=[],
        )

    def _python_range(self, repo_dir: Path) -> str:
        pyproject = repo_dir / "pyproject.toml"
        if pyproject.exists():
            text = pyproject.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if "requires-python" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return "unknown"

    def _gpu_required(self, repo_dir: Path, frameworks: set) -> bool:
        if {"torch", "transformers"}.intersection(frameworks):
            text = self._read_key_text(repo_dir).lower()
            gpu_tokens = ("cuda", "gpu", "flash-attn", "xformers", "bitsandbytes", "vllm")
            return any(token in text for token in gpu_tokens)
        return False

    def _estimate_disk(self, assets: List, frameworks: set) -> int:
        hinted = [asset.expected_size_bytes for asset in assets if asset.expected_size_bytes]
        if hinted:
            return max(1, int(sum(hinted) / (1024 ** 3)) + 5)
        if assets:
            return 20
        if {"torch", "transformers"}.intersection(frameworks):
            return 8
        return 2

    def _external_tokens(self, repo_dir: Path, assets: List) -> List[str]:
        tokens = []
        text = self._read_key_text(repo_dir)
        if any(asset.source == "huggingface" for asset in assets) or "HF_TOKEN" in text or "HUGGINGFACE" in text:
            tokens.append("HF_TOKEN")
        if any(asset.source == "modelscope" for asset in assets):
            tokens.append("MODELSCOPE_TOKEN")
        return sorted(set(tokens))

    def _risk_level(self, gpu_required: bool, estimated_disk: int, assets: List) -> str:
        if gpu_required or estimated_disk >= 20 or assets:
            return "high"
        if estimated_disk >= 8:
            return "medium"
        return "low"

    def _risk_reasons(self, gpu_required: bool, assets: List) -> List[str]:
        reasons = []
        if gpu_required:
            reasons.append("GPU/CUDA signals detected")
        if assets:
            reasons.append("external model assets detected")
        return reasons

    def _read_key_text(self, repo_dir: Path) -> str:
        text = ""
        for name in ("README.md", "readme.md", "requirements.txt", "pyproject.toml", "app.py", "main.py"):
            path = repo_dir / name
            if path.exists():
                text += "\n" + path.read_text(encoding="utf-8", errors="ignore")
        return text
