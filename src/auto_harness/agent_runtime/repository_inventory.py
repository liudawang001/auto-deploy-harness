"""Bounded repository inventory for layered planner context."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


class RepositoryInventoryBuilder:
    """Build metadata-only repository facts without exposing file contents."""

    def build(
        self,
        repo_dir: Path,
        file_tree: Iterable[str],
        *,
        detected_signals: Dict = None,
        total_file_count: int = 0,
        tree_truncated: bool = False,
        excluded: Dict = None,
    ) -> Dict:
        root = Path(repo_dir).resolve()
        entries: List[Dict] = []
        types = Counter()
        fingerprint = hashlib.sha256()
        for raw in file_tree:
            rel = str(raw).replace("\\", "/")
            path = root / rel
            suffix = Path(rel).suffix.lower() or "[no_extension]"
            types[suffix] += 1
            try:
                stat = path.stat()
                size = int(stat.st_size)
                mtime_ns = int(stat.st_mtime_ns)
            except OSError:
                size = -1
                mtime_ns = -1
            entries.append({"path": rel, "size": size, "type": suffix})
            fingerprint.update(
                ("%s\0%s\0%s\n" % (rel, size, mtime_ns)).encode("utf-8")
            )
        fingerprint.update(
            json.dumps(detected_signals or {}, sort_keys=True).encode("utf-8")
        )
        manifest_hashes = {}
        for rel in (detected_signals or {}).get("dependency_files", []):
            try:
                manifest_hashes[rel] = hashlib.sha256(
                    (root / rel).read_bytes()
                ).hexdigest()
            except OSError:
                continue
        fingerprint.update(
            json.dumps(manifest_hashes, sort_keys=True).encode("utf-8")
        )
        return {
            "schema_version": 1,
            "repository_fingerprint": fingerprint.hexdigest(),
            "tree": {
                "entries": entries,
                "total_file_count": int(total_file_count or len(entries)),
                "truncated": bool(tree_truncated),
            },
            "file_type_counts": dict(sorted(types.items())),
            "detected_signals": detected_signals or {},
            "manifest_sha256": manifest_hashes,
            "excluded": {
                "sensitive_files": int((excluded or {}).get("sensitive_files", 0)),
                "binary_files": int((excluded or {}).get("binary_files", 0)),
                "oversized_files": int((excluded or {}).get("oversized_files", 0)),
            },
        }


def rebuild_repository_inventory(
    repo_dir: Path, snapshot: Dict, *, max_tree_entries: int = 5000
) -> Dict:
    """Rescan repository metadata so additions and deletions invalidate plans."""
    from auto_harness.agent_runtime.project_snapshot import ProjectSnapshotBuilder

    collector = ProjectSnapshotBuilder(max_tree_entries=max_tree_entries)
    tree = collector.collect_file_tree(Path(repo_dir))
    return RepositoryInventoryBuilder().build(
        Path(repo_dir),
        tree,
        detected_signals=snapshot.get("detected_signals", {}),
        total_file_count=collector._last_total_file_count,
        tree_truncated=collector._last_total_file_count > len(tree),
        excluded=collector._last_excluded,
    )
