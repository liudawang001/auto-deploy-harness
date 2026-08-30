import json
from dataclasses import replace
from typing import Any, Dict, List, TYPE_CHECKING

from auto_harness.context.logs import LogCompactor
from auto_harness.context.memory import compact_memory_hits
from auto_harness.context.repository import RepoEvidenceSelector

if TYPE_CHECKING:
    from auto_harness.agent.schemas import AgentObservation

_STAGE_RESULT_FIELDS = {
    "stage",
    "status",
    "summary",
    "error_type",
    "exit_code",
    "operation_id",
    "failure_signature",
    "stop_reason",
}


def compact_agent_observation(
    observation: "AgentObservation",
    *,
    profile: str = "default",
    aggressive: bool = False,
    skill_budget_tokens: int = 2000,
    memory_budget_tokens: int = 2000,
) -> "AgentObservation":
    diagnose = profile in {"diagnose", "repair"}
    max_files = 1 if aggressive else 3
    max_file_chars = 400 if aggressive else (1200 if diagnose else 1000)
    selector = RepoEvidenceSelector()
    selected_files = selector.select(
        observation.selected_files,
        observation.deterministic_result,
        max_files=max_files,
        max_chars=max_file_chars,
    )
    file_tree_limit = 10 if aggressive else (0 if diagnose else 50)
    file_tree = list(observation.file_tree or [])[:file_tree_limit]
    if len(observation.file_tree or []) > file_tree_limit:
        file_tree.append(
            "... omitted %s paths"
            % (len(observation.file_tree or []) - file_tree_limit)
        )
    deterministic = compact_failure_context(
        observation.deterministic_result,
        max_chars=600 if aggressive else 2000,
    )
    compacted_memories = compact_memory_hits(
        observation.memory_hits,
        limit=1 if aggressive else 3,
        max_text_chars=300 if aggressive else 1000,
    )
    compacted_skills = [
        compact_value(
            item,
            max_text_chars=400 if aggressive else 1200,
            max_items=8,
        )
        for item in (observation.selected_skills or [])[: (1 if aggressive else 2)]
    ]
    return replace(
        observation,
        file_tree=file_tree,
        selected_files=selected_files,
        deterministic_result=deterministic,
        previous_results=summarize_stage_results(observation.previous_results),
        memory_hits=fit_items_to_budget(
            compacted_memories,
            max(1, memory_budget_tokens // (2 if aggressive else 1)),
        ),
        selected_skills=fit_items_to_budget(
            compacted_skills,
            max(1, skill_budget_tokens // (2 if aggressive else 1)),
        ),
        runtime_policy=compact_value(
            observation.runtime_policy,
            max_text_chars=400 if aggressive else 800,
            max_items=12 if aggressive else 20,
        ),
        extra=compact_value(
            observation.extra,
            max_text_chars=300 if aggressive else 1000,
            max_items=6 if aggressive else 12,
        ),
    )


def compact_project_snapshot(
    snapshot: Dict[str, Any],
    *,
    aggressive: bool = False,
    skill_budget_tokens: int = 2000,
    memory_budget_tokens: int = 2000,
) -> Dict[str, Any]:
    text_limit = 400 if aggressive else 1000
    item_limit = 6 if aggressive else 20
    result = {
        key: compact_value(
            value,
            max_text_chars=text_limit,
            max_items=item_limit,
        )
        for key, value in snapshot.items()
        if key
        not in {
            "file_tree",
            "selected_files",
            "memory_hits",
            "selected_skills",
            "skill_context",
        }
    }
    if aggressive and isinstance(snapshot.get("command_registry"), dict):
        registry = snapshot["command_registry"]
        registry_candidates = list(registry.get("candidates") or [])

        def candidate_priority(item):
            argv = list(item.get("argv") or []) if isinstance(item, dict) else []
            if not isinstance(item, dict) or item.get("phase") != "run":
                return (10,)
            lowered = [str(token).lower() for token in argv]
            cwd = str(item.get("cwd") or ".").lower()
            if "--serve-only" in argv:
                return (0,)
            if "run_cli" in lowered or lowered[-2:] == ["langflow", "run"]:
                return (1,)
            if any(
                token in {"serve", "server", "start", "run", "backend", "webui"}
                or token.startswith(("serve_", "run_", "start_"))
                for token in lowered[1:]
            ) and cwd not in {"docs", "doc", "documentation"}:
                return (2,)
            if "--webui-only" in argv:
                return (3,)
            if "--webui" in argv:
                return (4,)
            if cwd in {"docs", "doc", "documentation"}:
                return (9,)
            return (5,)

        registry_candidates.sort(key=candidate_priority)
        result["command_registry"] = {
            "schema_version": registry.get("schema_version", 1),
            "repository_fingerprint": registry.get("repository_fingerprint", ""),
            "candidates": [
                compact_value(item, max_text_chars=240, max_items=8)
                for item in registry_candidates[:8]
            ],
            "evidence_count": len(registry.get("evidence") or []),
            "candidate_count": len(registry.get("candidates") or []),
        }
    tree_limit = 80 if aggressive else 200
    file_tree = list(snapshot.get("file_tree") or [])
    result["file_tree"] = file_tree[:tree_limit]
    result["file_tree_summary"] = {
        "total_file_count": len(file_tree),
        "omitted_file_count": max(0, len(file_tree) - tree_limit),
    }
    result["selected_files"] = RepoEvidenceSelector().select(
        snapshot.get("selected_files") or {},
        snapshot.get("detected_signals") or {},
        max_files=2 if aggressive else 4,
        max_chars=700 if aggressive else 1200,
    )
    result["memory_hits"] = fit_items_to_budget(
        compact_memory_hits(
            snapshot.get("memory_hits") or [],
            limit=1 if aggressive else 3,
            max_text_chars=400 if aggressive else 800,
        ),
        max(1, memory_budget_tokens // (2 if aggressive else 1)),
    )
    return result


def compact_failure_context(value: Any, max_chars: int = 4000) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {"error": compact_text(str(value or ""), max_chars)}
    result = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in {"stdout", "stderr", "log", "logs", "output"}:
            result[key] = LogCompactor().compact(item, max_chars=max_chars)
        else:
            result[key] = compact_value(
                item, max_text_chars=max_chars, max_items=30
            )
    return result


def summarize_stage_results(results: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(results, dict):
        return {}
    summaries = {}
    for stage, value in list(results.items())[:12]:
        if not isinstance(value, dict):
            summaries[str(stage)] = {"status": "", "summary": compact_text(str(value), 500)}
            continue
        summary = {
            key: compact_value(value[key], max_text_chars=1000, max_items=10)
            for key in _STAGE_RESULT_FIELDS
            if key in value
        }
        evidence = value.get("evidence_paths")
        if isinstance(evidence, list):
            summary["evidence_paths"] = [
                str(path) for path in evidence[:5] if not str(path).startswith("/")
            ]
            summary["evidence_count"] = len(evidence)
        actions = value.get("attempted_actions") or value.get("actions")
        if isinstance(actions, list):
            summary["attempted_actions"] = [
                compact_value(action, max_text_chars=500, max_items=8)
                for action in actions[:5]
            ]
        summaries[str(stage)] = summary
    return summaries


def compact_value(
    value: Any,
    *,
    max_text_chars: int = 2000,
    max_items: int = 20,
    depth: int = 0,
):
    if depth >= 5:
        return "[nested content omitted]"
    if isinstance(value, str):
        return compact_text(value, max_text_chars)
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            str(key): compact_value(
                item,
                max_text_chars=max_text_chars,
                max_items=max_items,
                depth=depth + 1,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            result["_omitted_item_count"] = len(items) - max_items
        return result
    if isinstance(value, (list, tuple)):
        result = [
            compact_value(
                item,
                max_text_chars=max_text_chars,
                max_items=max_items,
                depth=depth + 1,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            result.append({"_omitted_item_count": len(value) - max_items})
        return result
    return value


def compact_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    head = max_chars * 2 // 3
    tail = max_chars - head
    return (
        value[:head]
        + "\n... [context omitted %s chars] ...\n" % (len(value) - max_chars)
        + value[-tail:]
    )


def fit_items_to_budget(
    items: List[Any],
    max_tokens: int,
) -> List[Any]:
    """Fit structured items to a conservative UTF-8 byte/token budget."""
    budget = max(0, int(max_tokens))
    result = []
    used = 2  # JSON list brackets
    for item in items or []:
        fitted = _fit_value_to_budget(item, max(0, budget - used))
        encoded = json.dumps(
            fitted,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        cost = len(encoded.encode("utf-8")) + 1
        if fitted in ({}, [], "") or used + cost > budget:
            break
        result.append(fitted)
        used += cost
    return result


def fit_value_to_budget(value: Any, max_tokens: int):
    return _fit_value_to_budget(value, max_tokens)


def _fit_value_to_budget(value: Any, max_tokens: int):
    budget = max(0, int(max_tokens))
    if budget <= 2:
        return {}
    candidate = value
    for text_cap, item_cap in (
        (1000, 20),
        (500, 12),
        (240, 8),
        (100, 5),
        (40, 3),
    ):
        candidate = compact_value(
            value,
            max_text_chars=text_cap,
            max_items=item_cap,
        )
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if len(encoded.encode("utf-8")) <= budget:
            return candidate
    return {}
