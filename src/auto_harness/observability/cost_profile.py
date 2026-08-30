"""Performance and cost profile aggregation over persisted run artifacts.

Pure read-side collector. Token usage and per-call LLM latency already land
on disk through the LLMCallExecutor telemetry (``logs/agent_calls/*.json``,
``reports/planner_turns/turn_*.json``) and stage wall-clock time comes from
``events.jsonl`` stage_update timestamps. This module aggregates those into a
per-run profile and a cross-run portfolio profile.

It never mutates run artifacts and never invents numbers: usage without a
pricing entry is reported as unpriced tokens, ``estimated`` token counts are
kept strictly separate from ``provider_reported`` counts, and runs without
telemetry are surfaced as data coverage instead of being silently dropped.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from auto_harness.controllers.outcomes import SUCCESS_STATUSES
from auto_harness.models.base import read_json, write_json
from auto_harness.utils.atomic import atomic_write_text
from auto_harness.utils.time import utc_now_iso

SCHEMA_VERSION = 1

EMPTY_COST_PROFILE = {
    "currency": "USD",
    "pricing_as_of": "",
    "pricing": {},
}

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
)

_KNOWN_USAGE_SOURCES = ("provider_reported", "estimated")


def _empty_bucket() -> Dict[str, int]:
    bucket = {key: 0 for key in _USAGE_KEYS}
    bucket["call_count"] = 0
    return bucket


def _percentile(values: List[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return int(ordered[index])


def _latency_summary(values: List[int]) -> Dict[str, int]:
    if not values:
        return {
            "count": 0,
            "total_ms": 0,
            "avg_ms": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "max_ms": 0,
        }
    total = int(sum(values))
    count = len(values)
    return {
        "count": count,
        "total_ms": total,
        "avg_ms": int(round(total / count)),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": int(max(values)),
    }


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ms_between(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds() * 1000))


def _read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _usage_from_record(record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (usage, source) for one persisted LLM call artifact.

    The persisted estimated-source records only carry ``input_tokens`` (the
    conservative estimator never sees the completion), so ``total_tokens`` is
    backfilled as input + output on a copy to keep bucket sums coherent.
    """
    context = record.get("context")
    if not isinstance(context, dict):
        return None, ""
    usage = context.get("usage")
    if not isinstance(usage, dict):
        return None, ""
    source = str(usage.get("source") or "")
    has_numbers = any(
        isinstance(usage.get(key), (int, float)) and not isinstance(usage.get(key), bool)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    )
    if not has_numbers:
        return None, ""
    if source not in _KNOWN_USAGE_SOURCES:
        source = "unattributed"
    normalized = dict(usage)
    input_tokens = normalized.get("input_tokens")
    output_tokens = normalized.get("output_tokens")
    total_tokens = normalized.get("total_tokens")
    total_is_number = isinstance(total_tokens, (int, float)) and not isinstance(total_tokens, bool)
    if not total_is_number or int(total_tokens) == 0:
        if isinstance(input_tokens, (int, float)) and not isinstance(input_tokens, bool):
            output = output_tokens if isinstance(output_tokens, (int, float)) and not isinstance(output_tokens, bool) else 0
            normalized["total_tokens"] = int(input_tokens) + int(output)
    return normalized, source


def _model_from_record(record: Dict[str, Any]) -> str:
    context = record.get("context")
    model = record.get("model")
    if not model and isinstance(context, dict):
        model = context.get("model")
    return str(model or "unknown")


def _latency_from_record(record: Dict[str, Any]) -> Optional[int]:
    value = record.get("latency_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    context = record.get("context")
    if isinstance(context, dict):
        response = context.get("provider_response")
        if isinstance(response, dict):
            value = response.get("latency_ms")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
    return None


def _add_usage(bucket: Dict[str, int], usage: Dict[str, Any]) -> None:
    for key in _USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            bucket[key] += int(value)


class CostProfileCollector:
    """Aggregate token/cost/latency/duration metrics from run artifacts."""

    def __init__(self, cost_profile_config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(EMPTY_COST_PROFILE)
        if isinstance(cost_profile_config, dict):
            config.update(cost_profile_config)
        self._currency = str(config.get("currency") or "USD")
        self._pricing_as_of = str(config.get("pricing_as_of") or "")
        pricing = config.get("pricing")
        self._pricing = pricing if isinstance(pricing, dict) else {}

    # ------------------------------------------------------------------ #
    # per-run profile
    # ------------------------------------------------------------------ #

    def collect(self, run_dir: Path) -> Dict[str, Any]:
        profile, _, _ = self._profile(run_dir)
        return profile

    def _profile(self, run_dir: Path) -> Tuple[Dict[str, Any], List[int], Dict[str, List[int]]]:
        run_dir = Path(run_dir)
        call_records = self._call_records(run_dir)

        buckets = {source: _empty_bucket() for source in _KNOWN_USAGE_SOURCES}
        buckets["unattributed"] = _empty_bucket()
        # Only provider-reported usage feeds per-model rows and pricing;
        # estimated counts must never be turned into money.
        model_buckets: Dict[str, Dict[str, int]] = {}
        calls_total = 0
        calls_with_usage = 0
        latencies: List[int] = []
        llm_wall_ms = 0

        for record in call_records:
            calls_total += 1
            latency = _latency_from_record(record)
            if latency is not None:
                latencies.append(latency)
                llm_wall_ms += latency
            usage, source = _usage_from_record(record)
            if usage is None:
                continue
            calls_with_usage += 1
            _add_usage(buckets[source], usage)
            buckets[source]["call_count"] += 1
            if source == "provider_reported":
                model_bucket = model_buckets.setdefault(_model_from_record(record), _empty_bucket())
                _add_usage(model_bucket, usage)
                model_bucket["call_count"] += 1

        stages = self._stage_segments(run_dir / "events.jsonl")
        run_window = self._run_window(run_dir / "events.jsonl")
        success = self._success(run_dir, run_window)

        by_model = []
        for model in sorted(model_buckets):
            bucket = model_buckets[model]
            row = dict(bucket)
            row["model"] = model
            row["total_tokens"] = bucket["input_tokens"] + bucket["output_tokens"]
            by_model.append(row)

        profile: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "task_id": run_dir.name,
            "generated_at": utc_now_iso(),
            "tokens": {
                "provider_reported": buckets["provider_reported"],
                "estimated": buckets["estimated"],
                "unattributed": buckets["unattributed"],
                "by_model": by_model,
                "coverage": {
                    "calls_total": calls_total,
                    "calls_with_usage": calls_with_usage,
                    "calls_without_usage": calls_total - calls_with_usage,
                },
            },
            "llm_latency": _latency_summary(latencies),
            "stages": stages,
            "run": run_window,
            "success": success,
            "cost": self._cost_section(model_buckets),
        }
        stage_durations: Dict[str, List[int]] = {}
        for stage in stages:
            duration = stage.get("duration_ms")
            if isinstance(duration, int):
                stage_durations.setdefault(str(stage["stage"]), []).append(duration)
        return profile, latencies, stage_durations

    def _call_records(self, run_dir: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for pattern in ("logs/agent_calls/*.json", "reports/planner_turns/turn_*.json"):
            for path in sorted(run_dir.glob(pattern)):
                data = _read_optional_json(path)
                if data is not None:
                    records.append(data)
        if not list(run_dir.glob("reports/planner_turns/turn_*.json")):
            # plan-first runs persist their final plan call as a raw artifact;
            # only count it when no planner turn artifacts exist, so the same
            # provider call is never summed twice.
            raw_plan = _read_optional_json(run_dir / "reports" / "llm_deployment_plan.raw.json")
            if raw_plan is not None:
                usage, _ = _usage_from_record(raw_plan)
                if usage is not None:
                    records.append(raw_plan)
        return records

    def _stage_segments(self, events_path: Path) -> List[Dict[str, Any]]:
        if not events_path.exists():
            return []
        running_since: Dict[str, datetime] = {}
        segments: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for line in self._event_lines(events_path):
            event = self._parse_event(line)
            if event is None or event.get("type") != "stage_update":
                continue
            stage = str(event.get("stage") or "")
            if not stage:
                continue
            ts = _parse_ts(event.get("ts"))
            data = event.get("data")
            data = data if isinstance(data, dict) else {}
            status = str(data.get("status") or "")
            if status == "running":
                if ts is not None:
                    running_since[stage] = ts
                continue
            if stage not in segments:
                segments[stage] = {
                    "stage": stage,
                    "status": status,
                    "duration_ms": None,
                    "attempts": 0,
                }
                order.append(stage)
            entry = segments[stage]
            entry["attempts"] += 1
            entry["status"] = status
            started = running_since.pop(stage, None)
            if started is not None and ts is not None:
                # keep the most recent attempt duration for rerun stages
                entry["duration_ms"] = _ms_between(started, ts)
        return [segments[stage] for stage in order]

    def _run_window(self, events_path: Path) -> Dict[str, Any]:
        window: Dict[str, Any] = {
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
        }
        if not events_path.exists():
            return window
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None
        terminal_ts: Optional[datetime] = None
        for line in self._event_lines(events_path):
            event = self._parse_event(line)
            if event is None:
                continue
            ts = _parse_ts(event.get("ts"))
            if ts is None:
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts
            if event.get("type") == "controller_terminal":
                terminal_ts = ts
        if first_ts is None:
            return window
        finished = terminal_ts or last_ts
        window["started_at"] = first_ts.isoformat()
        if finished is not None:
            window["finished_at"] = finished.isoformat()
            window["duration_ms"] = _ms_between(first_ts, finished)
        return window

    def _success(self, run_dir: Path, run_window: Dict[str, Any]) -> Dict[str, Any]:
        controller = _read_optional_json(run_dir / "reports" / "controller_result.json") or {}
        final_status = str(controller.get("status") or "")
        verify_status = str(controller.get("verify_status") or "")
        if not final_status or not verify_status:
            terminal = self._terminal_event(run_dir / "events.jsonl")
            final_status = final_status or str(terminal.get("status") or "")
            verify_status = verify_status or str(terminal.get("verify_status") or "")
        return {
            "final_status": final_status,
            "verify_status": verify_status,
            "verify_passed": verify_status.lower() in {"pass", "passed"},
            "success": final_status.lower() in SUCCESS_STATUSES,
        }

    def _terminal_event(self, events_path: Path) -> Dict[str, Any]:
        if not events_path.exists():
            return {}
        for line in reversed(self._event_lines(events_path)):
            event = self._parse_event(line)
            if event is not None and event.get("type") == "controller_terminal":
                data = event.get("data")
                return data if isinstance(data, dict) else {}
        return {}

    @staticmethod
    def _event_lines(events_path: Path) -> List[str]:
        try:
            with events_path.open("r", encoding="utf-8") as handle:
                return handle.readlines()
        except OSError:
            return []

    @staticmethod
    def _parse_event(line: str) -> Optional[Dict[str, Any]]:
        try:
            event = json.loads(line)
        except ValueError:
            return None
        return event if isinstance(event, dict) else None

    # ------------------------------------------------------------------ #
    # cost
    # ------------------------------------------------------------------ #

    def _cost_section(self, model_buckets: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        section: Dict[str, Any] = {
            "status": "unpriced",
            "reason": "",
            "currency": self._currency,
            "pricing_as_of": self._pricing_as_of,
            "models": [],
            "unpriced_models": [],
            "total_cost": 0.0,
        }
        if not self._pricing:
            section["reason"] = "no pricing configured"
            return section
        if not model_buckets:
            section["reason"] = "no provider reported usage"
            return section
        total_cost = 0.0
        for model in sorted(model_buckets):
            bucket = model_buckets[model]
            entry = self._pricing.get(model)
            if not isinstance(entry, dict):
                section["unpriced_models"].append(
                    {"model": model, "total_tokens": int(bucket["total_tokens"])}
                )
                continue
            input_tokens = int(bucket["input_tokens"])
            output_tokens = int(bucket["output_tokens"])
            cache_hit = min(int(bucket["cache_hit_tokens"]), input_tokens)
            cache_miss = input_tokens - cache_hit
            cost = (
                cache_hit * float(entry.get("cache_hit_input_per_million") or 0.0) / 1_000_000
                + cache_miss * float(entry.get("input_per_million") or 0.0) / 1_000_000
                + output_tokens * float(entry.get("output_per_million") or 0.0) / 1_000_000
            )
            total_cost += cost
            section["models"].append(
                {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_hit_tokens": cache_hit,
                    "cost": round(cost, 6),
                }
            )
        if section["models"]:
            section["status"] = "priced"
        if section["unpriced_models"] and section["models"]:
            section["reason"] = "some models have no pricing entry"
        elif section["unpriced_models"]:
            section["reason"] = "no pricing entry for observed models"
        section["total_cost"] = round(total_cost, 4)
        return section

    # ------------------------------------------------------------------ #
    # portfolio across runs
    # ------------------------------------------------------------------ #

    def collect_many(
        self,
        runs_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        runs_dir = Path(runs_dir)
        profiles: List[Dict[str, Any]] = []
        latencies: List[int] = []
        stage_durations: Dict[str, List[int]] = {}
        errors: List[Dict[str, str]] = []
        for run_dir in self._discover_run_dirs(runs_dir):
            try:
                profile, run_latencies, run_stages = self._profile(run_dir)
            except OSError as exc:
                errors.append({"task_id": run_dir.name, "error": str(exc)})
                continue
            profiles.append(profile)
            latencies.extend(run_latencies)
            for stage, values in run_stages.items():
                stage_durations.setdefault(stage, []).extend(values)
        report = self._portfolio(runs_dir, profiles, latencies, stage_durations, errors)
        if output_path is not None:
            output_path = Path(output_path)
            write_json(output_path, report)
            atomic_write_text(output_path.with_suffix(".md"), self.markdown(report))
        return report

    def _discover_run_dirs(self, runs_dir: Path) -> List[Path]:
        if not runs_dir.exists():
            return []
        discovered = []
        for path in sorted(runs_dir.iterdir()):
            if not path.is_dir():
                continue
            markers = ("task.json", "state.json", "events.jsonl")
            if any((path / marker).exists() for marker in markers):
                discovered.append(path)
        return discovered

    def _portfolio(
        self,
        runs_dir: Path,
        profiles: List[Dict[str, Any]],
        latencies: List[int],
        stage_durations: Dict[str, List[int]],
        errors: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {}
        buckets = {source: _empty_bucket() for source in _KNOWN_USAGE_SOURCES}
        model_buckets: Dict[str, Dict[str, int]] = {}
        runs_with_reported = 0
        runs_with_any_usage = 0
        run_durations: List[int] = []
        run_rows: List[Dict[str, Any]] = []
        for profile in profiles:
            tokens = profile["tokens"]
            final_status = profile["success"]["final_status"] or "unknown"
            status_counts[final_status] = status_counts.get(final_status, 0) + 1
            if tokens["provider_reported"]["call_count"] > 0:
                runs_with_reported += 1
            if tokens["coverage"]["calls_with_usage"] > 0:
                runs_with_any_usage += 1
            for source in _KNOWN_USAGE_SOURCES:
                _add_usage(buckets[source], tokens[source])
                buckets[source]["call_count"] += int(tokens[source].get("call_count") or 0)
            for row in tokens.get("by_model") or []:
                model_bucket = model_buckets.setdefault(str(row["model"]), _empty_bucket())
                for key in _USAGE_KEYS:
                    model_bucket[key] += int(row.get(key) or 0)
                model_bucket["call_count"] += int(row.get("call_count") or 0)
            duration = profile["run"]["duration_ms"]
            if isinstance(duration, int):
                run_durations.append(duration)
            run_rows.append(
                {
                    "task_id": profile["task_id"],
                    "final_status": profile["success"]["final_status"],
                    "verify_status": profile["success"]["verify_status"],
                    "run_duration_ms": profile["run"]["duration_ms"],
                    "llm_call_count": tokens["coverage"]["calls_total"],
                    "provider_reported_tokens": tokens["provider_reported"]["total_tokens"],
                    "estimated_tokens": tokens["estimated"]["total_tokens"],
                    "llm_wall_ms": profile["llm_latency"]["total_ms"],
                    "cost": (profile["cost"] or {}).get("total_cost"),
                }
            )
        run_count = len(profiles)
        success_count = sum(
            1 for row in run_rows if str(row["final_status"] or "").lower() in SUCCESS_STATUSES
        )
        known_status_rows = [
            row for row in run_rows if str(row["final_status"] or "").strip()
        ]
        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "runs_dir": str(runs_dir),
            "run_count": run_count,
            "status_counts": status_counts,
            "success_rate": round(success_count / run_count, 4) if run_count else 0.0,
            "runs_with_known_status": len(known_status_rows),
            "success_rate_known_status": (
                round(
                    sum(
                        1
                        for row in known_status_rows
                        if str(row["final_status"] or "").lower() in SUCCESS_STATUSES
                    )
                    / len(known_status_rows),
                    4,
                )
                if known_status_rows
                else 0.0
            ),
            "data_coverage": {
                "runs_total": run_count,
                "runs_with_provider_reported_usage": runs_with_reported,
                "runs_with_any_usage": runs_with_any_usage,
                "runs_without_usage": run_count - runs_with_any_usage,
                "collect_errors": errors,
            },
            "tokens": {
                "provider_reported": buckets["provider_reported"],
                "estimated": buckets["estimated"],
            },
            "cost": self._cost_section(model_buckets),
            "llm_latency": _latency_summary(latencies),
            "run_durations": _latency_summary(run_durations),
            "stage_durations": {
                stage: _latency_summary(values)
                for stage, values in sorted(stage_durations.items())
            },
            "runs": run_rows,
        }
        return report

    # ------------------------------------------------------------------ #
    # markdown rendering
    # ------------------------------------------------------------------ #

    def markdown(self, report: Dict[str, Any]) -> str:
        lines: List[str] = [
            "# Performance & Cost Profile",
            "",
            "- Generated at: `%s`" % report.get("generated_at", ""),
            "- Runs dir: `%s`" % report.get("runs_dir", ""),
            "- Run count: `%s`" % report.get("run_count", 0),
            "",
        ]
        lines.extend(self._outcome_markdown(report))
        lines.extend(self._tokens_markdown(report))
        lines.extend(self._latency_markdown(report))
        lines.extend(self._stage_markdown(report))
        lines.extend(self._runs_markdown(report))
        lines.extend(self._coverage_markdown(report))
        return "\n".join(lines) + "\n"

    def _outcome_markdown(self, report: Dict[str, Any]) -> List[str]:
        lines = ["## Outcome", ""]
        status_counts = report.get("status_counts") or {}
        if status_counts:
            lines.append("| final status | runs |")
            lines.append("|---|---|")
            for status in sorted(status_counts):
                lines.append("| %s | %s |" % (status, status_counts[status]))
        lines.extend(
            [
                "",
                "- Success rate (all runs): `%s`" % report.get("success_rate", 0.0),
                "- Success rate (runs with a recorded terminal status): `%s` over `%s` runs"
                % (report.get("success_rate_known_status", 0.0), report.get("runs_with_known_status", 0)),
                "",
            ]
        )
        return lines

    def _tokens_markdown(self, report: Dict[str, Any]) -> List[str]:
        lines = ["## Tokens & Cost", ""]
        tokens = report.get("tokens") or {}
        reported = tokens.get("provider_reported") or {}
        estimated = tokens.get("estimated") or {}
        lines.extend(
            [
                "- Provider reported tokens: input `%s`, output `%s`, total `%s` (cache hit `%s`)"
                % (
                    reported.get("input_tokens", 0),
                    reported.get("output_tokens", 0),
                    reported.get("total_tokens", 0),
                    reported.get("cache_hit_tokens", 0),
                ),
                "- Estimated-only tokens (no provider telemetry): total `%s`"
                % estimated.get("total_tokens", 0),
                "",
            ]
        )
        cost = report.get("cost") or {}
        if cost.get("status") == "priced":
            lines.append("| model | input | output | cache hit | cost (%s) |" % cost.get("currency", ""))
            lines.append("|---|---|---|---|---|")
            for row in cost.get("models") or []:
                lines.append(
                    "| %s | %s | %s | %s | %s |"
                    % (
                        row.get("model"),
                        row.get("input_tokens"),
                        row.get("output_tokens"),
                        row.get("cache_hit_tokens"),
                        row.get("cost"),
                    )
                )
            lines.append("")
            lines.append("- Total cost: `%s %s`" % (cost.get("currency"), cost.get("total_cost")))
        else:
            lines.append("- Cost: not computed (%s)" % (cost.get("reason") or "unpriced"))
        if cost.get("unpriced_models"):
            names = ", ".join(
                "`%s`" % row.get("model") for row in cost["unpriced_models"]
            )
            lines.append("- Unpriced models (tokens counted, no price configured): %s" % names)
        if cost.get("pricing_as_of"):
            lines.append("- Pricing as of: `%s` (config-provided, not a billing source)" % cost["pricing_as_of"])
        lines.append("")
        return lines

    def _latency_markdown(self, report: Dict[str, Any]) -> List[str]:
        latency = report.get("llm_latency") or {}
        run_durations = report.get("run_durations") or {}
        return [
            "## LLM Latency & Run Duration",
            "",
            "- LLM calls: `%s`, wall time total `%sms`, avg `%sms`, p50 `%sms`, p95 `%sms`, max `%sms`"
            % (
                latency.get("count", 0),
                latency.get("total_ms", 0),
                latency.get("avg_ms", 0),
                latency.get("p50_ms", 0),
                latency.get("p95_ms", 0),
                latency.get("max_ms", 0),
            ),
            "- End-to-end run duration: count `%s`, avg `%sms`, p50 `%sms`, p95 `%sms`, max `%sms`"
            % (
                run_durations.get("count", 0),
                run_durations.get("avg_ms", 0),
                run_durations.get("p50_ms", 0),
                run_durations.get("p95_ms", 0),
                run_durations.get("max_ms", 0),
            ),
            "",
        ]

    def _stage_markdown(self, report: Dict[str, Any]) -> List[str]:
        stage_durations = report.get("stage_durations") or {}
        lines = ["## Stage Durations (across runs)", ""]
        if not stage_durations:
            lines.extend(["- No stage timing data found.", ""])
            return lines
        lines.append("| stage | runs | avg ms | p50 ms | p95 ms | max ms |")
        lines.append("|---|---|---|---|---|---|")
        for stage, stats in stage_durations.items():
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    stage,
                    stats.get("count", 0),
                    stats.get("avg_ms", 0),
                    stats.get("p50_ms", 0),
                    stats.get("p95_ms", 0),
                    stats.get("max_ms", 0),
                )
            )
        lines.append("")
        return lines

    def _runs_markdown(self, report: Dict[str, Any]) -> List[str]:
        runs = report.get("runs") or []
        lines = ["## Runs", ""]
        if not runs:
            lines.extend(["- No runs found.", ""])
            return lines
        lines.append("| task | status | duration s | llm calls | reported tokens | cost |")
        lines.append("|---|---|---|---|---|---|")
        for row in runs:
            duration = row.get("run_duration_ms")
            duration_s = round(duration / 1000, 1) if isinstance(duration, int) else "n/a"
            cost = row.get("cost")
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    row.get("task_id"),
                    row.get("final_status") or "unknown",
                    duration_s,
                    row.get("llm_call_count", 0),
                    row.get("provider_reported_tokens", 0),
                    cost if cost is not None else "n/a",
                )
            )
        lines.append("")
        return lines

    def _coverage_markdown(self, report: Dict[str, Any]) -> List[str]:
        coverage = report.get("data_coverage") or {}
        lines = [
            "## Data Coverage",
            "",
            "- Runs with provider-reported usage: `%s` / `%s`"
            % (coverage.get("runs_with_provider_reported_usage", 0), coverage.get("runs_total", 0)),
            "- Runs with any usage telemetry: `%s`" % coverage.get("runs_with_any_usage", 0),
            "- Runs without usage telemetry: `%s` (older trace format or mock provider)"
            % coverage.get("runs_without_usage", 0),
            "",
        ]
        return lines
