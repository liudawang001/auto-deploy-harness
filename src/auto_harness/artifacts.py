import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List

from auto_harness.models.base import write_json
from auto_harness.utils.time import utc_now_iso


class DeploymentPackageExporter:
    """Exports a compact, auditable deployment package for one run."""

    DEFAULT_DIRS = ("reports", "evidence", "repairs")
    DEFAULT_FILES = ("task.json", "state.json", "events.jsonl")
    EXCLUDED_ROOTS = ("workspace",)

    def export(self, run_dir: Path, output_path: Path, include_logs: bool = False) -> Dict:
        run_dir = Path(run_dir)
        output_path = Path(output_path)
        if not run_dir.exists():
            return {"status": "failed", "error": "run directory does not exist", "run_dir": str(run_dir)}
        task_id = run_dir.name
        files = self._collect_files(run_dir, include_logs=include_logs)
        manifest = {
            "status": "generated",
            "task_id": task_id,
            "generated_at": utc_now_iso(),
            "source_run_dir": str(run_dir),
            "output_path": str(output_path),
            "include_logs": include_logs,
            "excluded_roots": list(self.EXCLUDED_ROOTS) + ([] if include_logs else ["logs"]),
            "files": [self._file_record(run_dir, path) for path in files],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz") as tar:
            for path in files:
                tar.add(path, arcname=str(Path(task_id) / path.relative_to(run_dir)))
            payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
            info = tarfile.TarInfo(str(Path(task_id) / "deployment_package_manifest.json"))
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        sidecar = self._sidecar_path(output_path)
        write_json(sidecar, manifest)
        manifest["package_sha256"] = self._sha256(output_path)
        manifest["package_size"] = output_path.stat().st_size
        write_json(sidecar, manifest)
        return {
            "status": "generated",
            "task_id": task_id,
            "output_path": str(output_path),
            "manifest_path": str(sidecar),
            "file_count": len(files),
            "package_sha256": manifest["package_sha256"],
            "package_size": manifest["package_size"],
        }

    def _collect_files(self, run_dir: Path, include_logs: bool) -> List[Path]:
        files: List[Path] = []
        for name in self.DEFAULT_FILES:
            path = run_dir / name
            if path.is_file():
                files.append(path)
        dirs = list(self.DEFAULT_DIRS)
        if include_logs:
            dirs.append("logs")
        for dirname in dirs:
            root = run_dir / dirname
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and not self._is_generated_package(path):
                    files.append(path)
        return sorted(set(files), key=lambda path: str(path.relative_to(run_dir)))

    def _file_record(self, run_dir: Path, path: Path) -> Dict:
        return {
            "path": str(path.relative_to(run_dir)),
            "size": path.stat().st_size,
            "sha256": self._sha256(path),
        }

    def _sidecar_path(self, output_path: Path) -> Path:
        name = output_path.name
        if name.endswith(".tar.gz"):
            return output_path.with_name(name[:-7] + ".manifest.json")
        return output_path.with_name(name + ".manifest.json")

    def _is_generated_package(self, path: Path) -> bool:
        return path.suffix in (".gz", ".tgz") or path.name.endswith(".manifest.json")

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
