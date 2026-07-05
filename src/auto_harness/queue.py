from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.orchestrator import TaskRunner
from auto_harness.utils.files import ensure_dir, safe_name, short_hash
from auto_harness.utils.time import compact_timestamp, utc_now_iso


class DeploymentQueue:
    """Persistent local deployment queue used by explicit CLI-driven workers."""

    TERMINAL_STATUSES = ("completed", "failed", "cancelled")

    def __init__(self, queue_dir: Path, runner: TaskRunner) -> None:
        self.queue_dir = ensure_dir(Path(queue_dir))
        self.items_dir = ensure_dir(self.queue_dir / "items")
        self.runner = runner

    def submit(
        self,
        repo_url: str,
        name: str = "",
        dry_run: bool = True,
        skip_clone: bool = False,
        allow_install: bool = False,
        allow_start: bool = False,
        require_gpu: bool = False,
        priority: int = 100,
    ) -> Dict:
        base = safe_name(name or repo_url.rstrip("/").rsplit("/", 1)[-1].replace(".git", "") or "task")
        stamp = compact_timestamp()
        job_id = "job_%s_%s_%s" % (base, stamp, short_hash(repo_url + name + stamp + str(priority), 6))
        now = utc_now_iso()
        item = {
            "job_id": job_id,
            "status": "queued",
            "repo_url": repo_url,
            "name": name,
            "dry_run": dry_run,
            "skip_clone": skip_clone,
            "allow_install": allow_install,
            "allow_start": allow_start,
            "require_gpu": require_gpu,
            "priority": priority,
            "attempts": 0,
            "task_id": "",
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        self._write(item)
        return item

    def list(self, status: Optional[str] = None) -> Dict:
        items = [item for item in self._items() if not status or item.get("status") == status]
        return {
            "status": "listed",
            "queue_dir": str(self.queue_dir),
            "count": len(items),
            "items": items,
            "status_counts": self._status_counts(self._items()),
        }

    def run_next(self, max_jobs: int = 1, gpu_slots: int = 0) -> Dict:
        selected: List[Dict] = []
        skipped: List[Dict] = []
        used_gpu = 0
        for item in self._queued_items():
            if item.get("require_gpu") and used_gpu >= gpu_slots:
                skipped.append({"job_id": item["job_id"], "reason": "gpu slot unavailable"})
                continue
            selected.append(item)
            if item.get("require_gpu"):
                used_gpu += 1
            if len(selected) >= max(1, max_jobs):
                break
        results = self._run_items(selected, max_workers=max(1, max_jobs))
        return {
            "status": "completed" if results else "idle",
            "requested_max_jobs": max_jobs,
            "worker_count": min(len(selected), max(1, max_jobs)),
            "gpu_slots": gpu_slots,
            "started": len(results),
            "results": results,
            "skipped": skipped,
        }

    def _run_items(self, items: List[Dict], max_workers: int) -> List[Dict]:
        if not items:
            return []
        if len(items) == 1 or max_workers <= 1:
            return [self._run_item(item) for item in items]
        indexed_results: Dict[int, Dict] = {}
        with ThreadPoolExecutor(max_workers=min(len(items), max_workers)) as executor:
            futures = {executor.submit(self._run_item, item): index for index, item in enumerate(items)}
            for future in as_completed(futures):
                indexed_results[futures[future]] = future.result()
        return [indexed_results[index] for index in range(len(items))]

    def _run_item(self, item: Dict) -> Dict:
        item["status"] = "running"
        item["attempts"] = int(item.get("attempts") or 0) + 1
        item["started_at"] = utc_now_iso()
        item["updated_at"] = item["started_at"]
        self._write(item)
        try:
            task_id = self.runner.deploy(
                item["repo_url"],
                item.get("name") or "",
                dry_run=bool(item.get("dry_run", True)),
                skip_clone=bool(item.get("skip_clone", False)),
                allow_install=bool(item.get("allow_install", False)),
                allow_start=bool(item.get("allow_start", False)),
            )
            item["status"] = "completed"
            item["task_id"] = task_id
            item["completed_at"] = utc_now_iso()
            item["updated_at"] = item["completed_at"]
            item["error"] = ""
        except Exception as exc:  # noqa: BLE001 - queue must persist failure and continue
            item["status"] = "failed"
            item["error"] = str(exc)
            item["failed_at"] = utc_now_iso()
            item["updated_at"] = item["failed_at"]
        self._write(item)
        return {
            "job_id": item["job_id"],
            "status": item["status"],
            "task_id": item.get("task_id", ""),
            "attempts": item.get("attempts", 0),
            "error": item.get("error", ""),
        }

    def _queued_items(self) -> List[Dict]:
        return [item for item in self._items() if item.get("status") == "queued"]

    def _items(self) -> List[Dict]:
        items = []
        for path in sorted(self.items_dir.glob("*.json")):
            try:
                item = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(item, dict):
                items.append(item)
        return sorted(items, key=lambda item: (int(item.get("priority") or 100), item.get("created_at") or ""))

    def _write(self, item: Dict) -> None:
        write_json(self.items_dir / ("%s.json" % item["job_id"]), item)

    def _status_counts(self, items: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            status = item.get("status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return counts
