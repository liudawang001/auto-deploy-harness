import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

from auto_harness.assets.manifest import ModelAsset


class ModelAssetDetector:
    HF_REPO = re.compile(r"(?:huggingface\.co/|hf\.co/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
    FROM_PRETRAINED = re.compile(r"from_pretrained\(\s*['\"]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]")
    HF_SNAPSHOT = re.compile(r"snapshot_download\(\s*repo_id\s*=\s*['\"]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]")
    MODELSCOPE = re.compile(r"(?:modelscope\.cn/models/|model_id\s*=\s*['\"])([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
    SIZE_HINT = re.compile(r"(\d+(?:\.\d+)?)\s*(gb|gib|mb|mib)", re.IGNORECASE)

    def detect(self, repo_dir: Path, analysis: Dict = None) -> List[ModelAsset]:
        seen: Set[str] = set()
        assets: List[ModelAsset] = []
        for rel_path, text in self._candidate_texts(repo_dir):
            for repo_id in self._find_huggingface_ids(text):
                key = "huggingface:%s" % repo_id
                if key in seen:
                    continue
                seen.add(key)
                assets.append(
                    ModelAsset(
                        asset_id=key,
                        source="huggingface",
                        repo_id=repo_id,
                        origin=rel_path,
                        expected_size_bytes=self._size_hint(text),
                    )
                )
            for repo_id in self._find_modelscope_ids(text):
                key = "modelscope:%s" % repo_id
                if key in seen:
                    continue
                seen.add(key)
                assets.append(
                    ModelAsset(
                        asset_id=key,
                        source="modelscope",
                        repo_id=repo_id,
                        origin=rel_path,
                        expected_size_bytes=self._size_hint(text),
                    )
                )
        return assets

    def _candidate_texts(self, repo_dir: Path) -> Iterable:
        names = {"README.md", "readme.md", "app.py", "main.py", "server.py", "webui.py", "demo.py", "config.json"}
        for path in sorted(repo_dir.rglob("*")):
            if path.is_dir() or ".git" in path.parts:
                continue
            if path.name not in names and path.suffix not in (".py", ".md", ".json", ".yaml", ".yml"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            yield str(path.relative_to(repo_dir)), text

    def _find_huggingface_ids(self, text: str) -> List[str]:
        found = []
        for pattern in (self.HF_REPO, self.FROM_PRETRAINED, self.HF_SNAPSHOT):
            found.extend(match.group(1).strip("/") for match in pattern.finditer(text))
        return self._dedupe(found)

    def _find_modelscope_ids(self, text: str) -> List[str]:
        return self._dedupe(match.group(1).strip("/") for match in self.MODELSCOPE.finditer(text))

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        result = []
        seen = set()
        for value in values:
            value = value.strip().strip(".,);]")
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _size_hint(self, text: str):
        match = self.SIZE_HINT.search(text)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = 1024 * 1024 * 1024 if unit in ("gb", "gib") else 1024 * 1024
        return int(value * multiplier)
