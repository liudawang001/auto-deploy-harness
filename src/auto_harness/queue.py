from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_harness.models.base import read_json, write_json
from auto_harness.orchestrator import TaskRunner
from auto_harness.runtime import GpuResourceProbe
from auto_harness.utils.files import ensure_dir, safe_name, short_hash
from auto_harness.utils.time import compact_timestamp, utc_now_iso

# Whitelist for LLM snapshot keys persisted in queue items
_ALLOWED_LLM_SNAPSHOT_KEYS = frozenset({
    "agent_provider",
    "plan_first_provider",
    "model",
    "context_window_tokens",
    "max_output_tokens",
    "agent_repo_context_mode",
    "agent_repo_inventory_budget_tokens",
    "agent_repo_core_budget_tokens",
    "agent_repo_observation_budget_tokens",
    "agent_repo_max_observation_rounds",
    "agent_repo_max_requests_per_round",
    "agent_repo_max_observed_files",
    "agent_repo_max_chars_per_read",
    "agent_repo_max_lines_per_read",
    "agent_repo_search_max_results",
    "agent_repo_search_max_files",
    "agent_repo_search_max_bytes",
    "agent_repo_tree_max_entries",
})

_FORBIDDEN_LLM_SNAPSHOT_KEYS = frozenset({
    "api_key",
    "token",
    "secret",
    "password",
    "authorization",
})


class DeploymentQueue:
    """Persistent local deployment queue used by explicit CLI-driven workers."""

    TERMINAL_STATUSES = ("completed", "failed", "cancelled")

    def __init__(self, queue_dir: Path, runner: TaskRunner, gpu_probe: GpuResourceProbe = None, claim_ttl_seconds: int = 3600) -> None:
        self.queue_dir = ensure_dir(Path(queue_dir))
        self.items_dir = ensure_dir(self.queue_dir / "items")
        self.locks_dir = ensure_dir(self.queue_dir / "locks")
        self.runner = runner
        self.gpu_probe = gpu_probe or GpuResourceProbe()
        self.claim_ttl_seconds = max(0, int(claim_ttl_seconds))

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
        llm: Optional[Dict] = None,
    ) -> Dict:
        # Validate and whitelist LLM snapshot
        llm_snapshot = _sanitize_llm_snapshot(llm)
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
        if llm_snapshot:
            item["llm"] = llm_snapshot
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

    def run_next(self, max_jobs: int = 1, gpu_slots: int = None) -> Dict:
        gpu_probe = self.gpu_probe.probe() if gpu_slots is None else {
            "status": "manual",
            "source": "manual",
            "available_slots": max(0, gpu_slots),
            "gpus": [],
        }
        effective_gpu_slots = int(gpu_probe.get("available_slots") or 0)
        selected: List[Dict] = []
        skipped: List[Dict] = []
        recovered_locks: List[Dict] = []
        used_gpu = 0
        for item in self._queued_items():
            if item.get("require_gpu") and used_gpu >= effective_gpu_slots:
                skipped.append({"job_id": item["job_id"], "reason": "gpu slot unavailable"})
                continue
            claimed = self._claim_item(item)
            if not claimed:
                skipped.append({"job_id": item["job_id"], "reason": "job already claimed"})
                continue
            if claimed.get("_stale_lock_recovered"):
                recovered_locks.append({
                    "job_id": item["job_id"],
                    "lock_age_seconds": claimed.get("_stale_lock_age_seconds"),
                })
            selected.append(claimed)
            if item.get("require_gpu"):
                used_gpu += 1
            if len(selected) >= max(1, max_jobs):
                break
        results = self._run_items(selected, max_workers=max(1, max_jobs))
        return {
            "status": "completed" if results else "idle",
            "requested_max_jobs": max_jobs,
            "worker_count": min(len(selected), max(1, max_jobs)),
            "gpu_slots": effective_gpu_slots,
            "gpu_probe": gpu_probe,
            "started": len(results),
            "results": results,
            "skipped": skipped,
            "recovered_locks": recovered_locks,
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
        try:
            # Use task-level config when the runner has real config/provider_registry
            runner_config = getattr(self.runner, "config", None)
            runner_registry = getattr(self.runner, "provider_registry", None)
            if runner_config is not None and runner_registry is not None:
                job_config = copy.deepcopy(runner_config)
                llm_snapshot = item.get("llm")
                if isinstance(llm_snapshot, dict):
                    _apply_queue_llm_snapshot(job_config, llm_snapshot)
                job_runner = TaskRunner(
                    job_config,
                    provider_registry=runner_registry,
                )
                _validate_job_providers(job_runner, job_config, item)
                effective_runner = job_runner
            else:
                # Test double path — delegate directly
                effective_runner = self.runner

            task_id = effective_runner.deploy(
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
        finally:
            self._write(item)
            self._release_lock(item)
        return {
            "job_id": item["job_id"],
            "status": item["status"],
            "task_id": item.get("task_id", ""),
            "attempts": item.get("attempts", 0),
            "error": item.get("error", ""),
        }

    def _claim_item(self, item: Dict) -> Optional[Dict]:
        lock_path = self._lock_path(item["job_id"])
        recovered = self._recover_stale_lock(lock_path)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        try:
            os.write(fd, ("pid=%s\nclaimed_at=%s\n" % (os.getpid(), utc_now_iso())).encode("utf-8"))
        finally:
            os.close(fd)
        current = self._read_item(item["job_id"])
        if not current or current.get("status") != "queued":
            self._release_lock({"job_id": item["job_id"]})
            return None
        now = utc_now_iso()
        current["status"] = "running"
        current["attempts"] = int(current.get("attempts") or 0) + 1
        current["started_at"] = now
        current["updated_at"] = now
        current["claim"] = {"pid": os.getpid(), "claimed_at": now, "lock_path": str(lock_path)}
        current["_lock_path"] = str(lock_path)
        if recovered:
            current["claim"]["stale_lock_recovered"] = True
            current["claim"]["stale_lock_age_seconds"] = recovered.get("age_seconds")
            current["_stale_lock_recovered"] = True
            current["_stale_lock_age_seconds"] = recovered.get("age_seconds")
        self._write(current)
        return current

    def _recover_stale_lock(self, lock_path: Path) -> Optional[Dict]:
        if not lock_path.exists() or self.claim_ttl_seconds <= 0:
            return None
        try:
            age = max(0.0, os.path.getmtime(str(lock_path)))
            age_seconds = max(0, int(time.time() - age))
        except OSError:
            return None
        if age_seconds < self.claim_ttl_seconds:
            return None
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        return {"age_seconds": age_seconds}

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
        public_item = {key: value for key, value in item.items() if not key.startswith("_")}
        write_json(self._item_path(item["job_id"]), public_item)

    def _read_item(self, job_id: str) -> Optional[Dict]:
        path = self._item_path(job_id)
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def _item_path(self, job_id: str) -> Path:
        return self.items_dir / ("%s.json" % job_id)

    def _lock_path(self, job_id: str) -> Path:
        return self.locks_dir / ("%s.lock" % job_id)

    def _release_lock(self, item: Dict) -> None:
        lock_path = Path(item.get("_lock_path") or self._lock_path(item["job_id"]))
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _status_counts(self, items: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            status = item.get("status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# LLM snapshot helpers
# ---------------------------------------------------------------------------

def _sanitize_llm_snapshot(llm: Optional[Dict]) -> Optional[Dict]:
    """Whitelist and validate an LLM snapshot for queue persistence."""
    if llm is None:
        return None
    if not isinstance(llm, dict):
        return None
    forbidden = _FORBIDDEN_LLM_SNAPSHOT_KEYS.intersection(
        str(k).lower() for k in llm
    )
    if forbidden:
        raise ValueError(
            "LLM snapshot must not contain secret keys: %s"
            % ", ".join(sorted(forbidden))
        )
    snapshot = {}
    for key in _ALLOWED_LLM_SNAPSHOT_KEYS:
        if key in llm and llm[key] is not None:
            snapshot[key] = llm[key]
    return snapshot if snapshot else None


def _apply_queue_llm_snapshot(config: Any, llm_snapshot: Dict) -> None:
    """Restore provider selection, runtime limits, and governance budgets."""
    from auto_harness.providers.settings import (
        normalize_provider_name,
        set_runtime_overrides,
    )

    repository_fields = (
        "agent_repo_context_mode",
        "agent_repo_inventory_budget_tokens",
        "agent_repo_core_budget_tokens",
        "agent_repo_observation_budget_tokens",
        "agent_repo_max_observation_rounds",
        "agent_repo_max_requests_per_round",
        "agent_repo_max_observed_files",
        "agent_repo_max_chars_per_read",
        "agent_repo_max_lines_per_read",
        "agent_repo_search_max_results",
        "agent_repo_search_max_files",
        "agent_repo_search_max_bytes",
        "agent_repo_tree_max_entries",
    )
    for field in repository_fields:
        if field not in llm_snapshot:
            continue
        value = llm_snapshot[field]
        setattr(config, field, value if field == "agent_repo_context_mode" else int(value))

    agent_provider = llm_snapshot.get("agent_provider", "")
    plan_provider = llm_snapshot.get("plan_first_provider", "") or agent_provider
    if not agent_provider:
        return
    config.agent_provider = normalize_provider_name(agent_provider)
    config.agent_plan_first_provider = normalize_provider_name(plan_provider)

    provider_names = {
        config.agent_provider,
        config.agent_plan_first_provider,
    }
    model = llm_snapshot.get("model")
    context_window = llm_snapshot.get("context_window_tokens")
    max_output = llm_snapshot.get("max_output_tokens")
    if len(provider_names) > 1 and any(
        value is not None for value in (model, context_window, max_output)
    ):
        raise ValueError(
            "uniform queue LLM overrides require a single effective provider"
        )
    set_runtime_overrides(
        config,
        provider_names,
        model=model,
        context_window_tokens=context_window,
        max_output_tokens=max_output,
    )
    if context_window is not None:
        config.agent_context_window_tokens = int(context_window)
    if max_output is not None:
        config.agent_context_reserved_output_tokens = int(max_output)


def _validate_job_providers(job_runner, config, item: Dict) -> None:
    """Pre-check agent and plan-first providers before deploy.

    LangGraph ``planner_mode=auto`` may select the real plan-first provider
    even when the legacy ``agent_plan_first`` flag is false, so both paths
    must be ready before a queued deployment starts.
    """
    from auto_harness.providers.errors import ErrorCategory, ProviderError

    agent_provider = getattr(config, "agent_provider", "deepseek")
    plan_provider = getattr(config, "agent_plan_first_provider", "deepseek")
    # LangGraph planner_mode=auto selects the real plan-first provider even
    # when the legacy agent_plan_first flag is false, so validate both paths.
    providers_to_check = [
        (agent_provider, "agent"),
        (plan_provider, "plan_first"),
    ]

    for provider_name, purpose in providers_to_check:
        try:
            provider = job_runner.provider_registry.create(
                provider_name,
                config=config,
                purpose=purpose,
            )
        except Exception as exc:
            raise ProviderError(
                "queue job provider pre-check failed for %s: %s" % (provider_name, exc),
                provider_name=str(provider_name),
                category=ErrorCategory.CONFIGURATION_ERROR,
            ) from exc
        missing_checker = getattr(provider, "missing_configuration", None)
        missing = list(missing_checker()) if callable(missing_checker) else []
        if missing:
            raise ProviderError(
                "%s provider configuration is incomplete" % provider_name,
                provider_name=str(provider_name),
                category=ErrorCategory.CONFIGURATION_ERROR,
                safe_detail="missing: %s" % ", ".join(missing),
            )
