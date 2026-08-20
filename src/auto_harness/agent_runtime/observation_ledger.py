"""Append-only, redacted repository observation ledger and budgets."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from auto_harness.agent_runtime.schemas import ToolCall
from auto_harness.tools.repository_executor import RepositoryToolExecutor
from auto_harness.utils.time import utc_now_iso


def estimate_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, (len(text) + 3) // 4)


class ObservationLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def append(self, record: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def find_cache(self, cache_key: str) -> Dict:
        for item in reversed(self.load()):
            if item.get("cache_key") == cache_key and item.get("status") == "passed":
                return item
        return {}

    @staticmethod
    def cache_key(tool: str, tool_input: Dict, repository_fingerprint: str) -> str:
        payload = json.dumps(
            {"tool": tool, "input": tool_input, "repository_fingerprint": repository_fingerprint},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RepositoryObservationService:
    """Execute one bounded observation round and persist redacted results."""

    def __init__(self, config: Any = None, executor=None):
        self.config = config
        self.executor = executor or RepositoryToolExecutor(config=config)
        from auto_harness.tools.registry import ToolRegistry
        from auto_harness.tools.retrieval_executor import RetrievalToolExecutor
        self.registry = ToolRegistry(config=config)
        self.retrieval_executor = RetrievalToolExecutor(
            config=config, registry=self.registry,
        )

    def execute_round(
        self,
        requests: Iterable,
        *,
        repo_dir: Path,
        ledger_path: Path,
        repository_fingerprint: str,
        round_number: int,
        budget: Dict,
        stage: str = "plan",
        task_id: str = "",
        run_dir: Any = None,
    ) -> Dict:
        maximum_rounds = self._cfg("agent_repo_max_observation_rounds", 4)
        if int(round_number) > maximum_rounds:
            return {"status": "rejected", "stop_reason": "observation_round_limit_exceeded", "results": [], "budget": budget}
        ledger = ObservationLedger(ledger_path)
        existing_records = [
            item for item in ledger.load()
            if item.get("repository_fingerprint") == repository_fingerprint
            and item.get("status") == "passed"
        ]
        accounted_tokens = sum(
            int(item.get("content_tokens", 0) or 0)
            for item in existing_records
        )
        accounted_paths = set()
        for item in existing_records:
            evidence = item.get("evidence", {})
            candidates = []
            if isinstance(evidence, dict):
                candidates.extend(evidence.get("files", []) or [])
                candidates.extend(evidence.get("results", []) or [])
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("path"):
                    accounted_paths.add(candidate["path"])
        results = []
        remaining_tokens = min(
            int(budget.get("remaining_tokens", self._cfg("agent_repo_observation_budget_tokens", 24000))),
            max(
                0,
                self._cfg("agent_repo_observation_budget_tokens", 24000)
                - accounted_tokens,
            ),
        )
        remaining_files = min(
            int(budget.get("remaining_files", self._cfg("agent_repo_max_observed_files", 20))),
            max(
                0,
                self._cfg("agent_repo_max_observed_files", 20)
                - len(accounted_paths),
            ),
        )
        for request in requests:
            item = request.to_dict() if hasattr(request, "to_dict") else dict(request)
            request_id = str(item.get("request_id", ""))
            tool = str(item.get("tool", ""))
            tool_input = item.get("input", {})
            if tool == "retrieve_deployment_context":
                decision = self._validate_retrieval_input(tool_input, stage)
            else:
                decision = self.executor.policy.validate_and_normalize(
                    tool, tool_input, Path(repo_dir)
                )
            if not decision.get("allowed"):
                rejected = self._record(
                    request_id=request_id,
                    tool=tool,
                    status="rejected",
                    error=decision.get("reason", "repository_observation_rejected"),
                    evidence={},
                    content_tokens=0,
                    round_number=round_number,
                    repository_fingerprint=repository_fingerprint,
                    cache_key="",
                    normalized_input=None,
                )
                ledger.append(rejected)
                results.append(rejected)
                continue
            normalized_input = decision["normalized_input"]
            key = ledger.cache_key(tool, normalized_input, repository_fingerprint)
            cached = ledger.find_cache(key)
            if cached:
                cached_result = self._record(
                    request_id=request_id,
                    tool=tool,
                    status="cache_hit",
                    error="",
                    evidence={},
                    content_tokens=0,
                    round_number=round_number,
                    repository_fingerprint=repository_fingerprint,
                    cache_key=key,
                    observation_id=cached.get("observation_id", ""),
                    normalized_input=normalized_input,
                )
                cached_result["source_observation_id"] = cached.get(
                    "observation_id", ""
                )
                ledger.append(cached_result)
                results.append(cached_result)
                continue
            if remaining_tokens <= 0 or remaining_files <= 0:
                rejected = self._record(
                    request_id=request_id,
                    tool=tool,
                    status="rejected",
                    error="observation_budget_exhausted",
                    evidence={},
                    content_tokens=0,
                    round_number=round_number,
                    repository_fingerprint=repository_fingerprint,
                    cache_key=key,
                    normalized_input=normalized_input,
                )
                ledger.append(rejected)
                results.append(rejected)
                continue
            execution_context = {
                "repo_dir": str(repo_dir), "round": round_number,
                "stage": stage, "task_id": task_id,
                "repository_fingerprint": repository_fingerprint,
                "run_dir": str(run_dir or Path(ledger_path).parent.parent),
                "requested_by": "json_action",
            }
            selected_executor = (
                self.retrieval_executor
                if tool == "retrieve_deployment_context" else self.executor
            )
            tool_result = selected_executor.execute(
                ToolCall(name=tool, input=normalized_input), execution_context,
            )
            evidence = tool_result.evidence or {}
            cost = estimate_tokens(evidence)
            file_count = _evidence_file_count(evidence)
            if cost > remaining_tokens or file_count > remaining_files:
                result = self._record(
                    request_id=request_id,
                    tool=tool,
                    status="rejected",
                    error="observation_budget_exhausted",
                    evidence={},
                    content_tokens=0,
                    round_number=round_number,
                    repository_fingerprint=repository_fingerprint,
                    cache_key=key,
                    normalized_input=normalized_input,
                )
                ledger.append(result)
                results.append(result)
                continue
            remaining_tokens -= cost
            remaining_files -= file_count
            observation_id = "obs_%s" % key[:12]
            result = self._record(
                request_id=request_id,
                tool=tool,
                status=tool_result.status,
                error=tool_result.error or "",
                evidence=evidence,
                content_tokens=cost,
                round_number=round_number,
                repository_fingerprint=repository_fingerprint,
                cache_key=key,
                observation_id=observation_id,
                normalized_input=normalized_input,
            )
            if normalized_input.get("retrieved_from_query_id"):
                result["retrieved_from_query_id"] = normalized_input["retrieved_from_query_id"]
                result["retrieval_chunk_ids"] = list(normalized_input.get("retrieval_chunk_ids") or [])
            ledger.append(result)
            results.append(result)
            if normalized_input.get("retrieved_from_query_id") and run_dir:
                from auto_harness.retrieval.artifacts import RetrievalArtifacts
                settings = getattr(self.config, "retrieval", {}) if self.config else {}
                RetrievalArtifacts(Path(run_dir)).finalize(
                    requested_mode=str(settings.get("mode", "lexical")),
                )
        next_budget = {
            "remaining_rounds": max(0, maximum_rounds - int(round_number)),
            "remaining_tokens": remaining_tokens,
            "remaining_files": remaining_files,
        }
        return {"status": "passed", "kind": "observation_result", "protocol_version": 1, "round": int(round_number), "results": results, "budget": next_budget}

    def _validate_retrieval_input(self, value: Any, stage: str) -> Dict:
        schema = self.registry.tools.get("retrieve_deployment_context")
        visible = {
            item["name"] for item in self.registry.executable_for_stage(
                stage, agent_mode="planner",
            )
        }
        if schema is None or schema.name not in visible:
            return {"allowed": False, "reason": "retrieval is disabled"}
        try:
            from auto_harness.providers.protocols import validate_tool_arguments
            validate_tool_arguments(schema.input_schema, value)
        except (TypeError, ValueError) as exc:
            return {"allowed": False, "reason": str(exc)[:300]}
        return {
            "allowed": True,
            "reason": "retrieval request allowed",
            "normalized_input": dict(value),
        }

    @staticmethod
    def _record(
        *, request_id, tool, status, error, evidence, content_tokens,
        round_number, repository_fingerprint, cache_key, observation_id="",
        normalized_input=None,
    ) -> Dict:
        return {
            "schema_version": 1,
            "observation_id": observation_id,
            "request_id": request_id,
            "tool": tool,
            "normalized_input": normalized_input,
            "status": status,
            "error": error,
            "evidence": evidence,
            "content_tokens": int(content_tokens),
            "cache_hit": status == "cache_hit",
            "round": int(round_number),
            "repository_fingerprint": repository_fingerprint,
            "cache_key": cache_key,
            "created_at": utc_now_iso(),
        }

    def _cfg(self, name: str, default: int) -> int:
        if isinstance(self.config, dict):
            return int(self.config.get(name, default))
        return int(getattr(self.config, name, default))


def _evidence_file_count(evidence: Dict) -> int:
    if isinstance(evidence.get("files"), list):
        return len(evidence["files"])
    paths = set()
    for item in evidence.get("results", []) if isinstance(evidence.get("results"), list) else []:
        if isinstance(item, dict) and item.get("path"):
            paths.add(item["path"])
    return len(paths)


def enrich_plan_grounding(plan: Dict, snapshot: Dict, records: List[Dict]) -> Dict:
    """Attach proof fields only when a cited file was actually observed."""
    result = dict(plan or {})
    observed_by_path: Dict[str, List[Dict]] = {}
    for path, item in (snapshot.get("selected_files") or {}).items():
        if isinstance(item, dict) and item.get("observation_id"):
            observed_by_path.setdefault(path, []).append(item)
    for record in records or []:
        if not isinstance(record, dict) or record.get("status") != "passed":
            continue
        evidence = record.get("evidence", {})
        candidates = []
        if isinstance(evidence.get("files"), list):
            candidates.extend(evidence["files"])
        if isinstance(evidence.get("results"), list):
            candidates.extend(evidence["results"])
        for item in candidates:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            normalized = dict(item)
            normalized["observation_id"] = record.get("observation_id", "")
            normalized.setdefault("line_start", normalized.get("line", 1))
            normalized.setdefault("line_end", normalized.get("line", 1))
            observed_by_path.setdefault(item["path"], []).append(normalized)

    grounding = []
    for raw in result.get("grounding", []) or []:
        item = dict(raw) if isinstance(raw, dict) else raw
        if isinstance(item, dict):
            candidates = observed_by_path.get(str(item.get("file", "")), [])
            if candidates:
                observed = max(
                    candidates,
                    key=lambda value: int(value.get("line_end", 0)) - int(value.get("line_start", 1)),
                )
                # Proof metadata is authoritative framework data. Replace
                # missing, aliased, or stale model-supplied values only after
                # an exact observed repository path match.
                item["observation_id"] = observed.get("observation_id", "")
                item["sha256"] = observed.get("sha256", "")
                observed_start = int(observed.get("line_start", 1))
                observed_end = int(observed.get("line_end", observed_start))
                requested_start = item.get("line_start")
                requested_end = item.get("line_end")
                if (
                    not isinstance(requested_start, int)
                    or not isinstance(requested_end, int)
                    or requested_start < observed_start
                    or requested_end > observed_end
                    or requested_start > requested_end
                ):
                    item["line_start"] = observed_start
                    item["line_end"] = observed_end
        grounding.append(item)
    result["grounding"] = grounding
    return result
