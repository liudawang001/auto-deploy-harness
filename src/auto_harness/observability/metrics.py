"""Evidence-derived metrics shared by Agent, repair, recovery, verify, and skills."""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from auto_harness.models.base import read_json, write_json
from auto_harness.utils.atomic import FileLock, atomic_write_text
from auto_harness.utils.time import utc_now_iso


@dataclass(frozen=True)
class AgentMetricEvent:
    schema_version: int
    event_id: str
    task_id: str
    category: str
    name: str
    stage: str
    outcome: str
    value: float
    source_artifact: str
    occurred_at: str
    dimensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricEventWriter:
    """Durably writes sanitized metric events under a run report directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "reports" / "agent_metric_events.jsonl"

    def replace(self, events: Iterable[AgentMetricEvent]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True)
            for event in events
        ]
        text = "\n".join(lines)
        if text:
            text += "\n"
        with FileLock(self.path):
            atomic_write_text(self.path, text)
        return self.path

    def append(self, event: AgentMetricEvent) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
        with FileLock(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return self.path


class UnifiedMetricsCollector:
    """Build normalized events and aggregates from persisted run artifacts."""

    def collect(self, run_dir: Path, output_path: Path = None) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        task_id = run_dir.name
        events: List[AgentMetricEvent] = []
        events.extend(self._llm_and_policy_events(run_dir, task_id))
        events.extend(self._repair_events(run_dir, task_id))
        events.extend(self._recovery_events(run_dir, task_id))
        events.extend(self._verify_events(run_dir, task_id))
        events.extend(self._skill_events(run_dir, task_id))
        events.extend(self._deployment_capability_events(run_dir, task_id))
        events.sort(key=lambda item: (item.category, item.name, item.source_artifact, item.event_id))

        events_path = MetricEventWriter(run_dir).replace(events)
        payload = {
            "schema_version": 1,
            "task_id": task_id,
            "generated_at": utc_now_iso(),
            "event_count": len(events),
            "events_path": str(events_path),
            "summary": self._aggregate(events),
            "provenance": sorted({
                event.source_artifact for event in events if event.source_artifact
            }),
        }
        output = Path(output_path or run_dir / "reports" / "unified_metrics.json")
        write_json(output, payload)
        return payload

    def _event(
        self,
        *,
        task_id: str,
        category: str,
        name: str,
        stage: str = "",
        outcome: str = "",
        value: float = 1,
        source_artifact: str,
        dimensions: Dict[str, Any] = None,
        discriminator: str = "",
    ) -> AgentMetricEvent:
        identity = "|".join([
            task_id, category, name, stage, outcome, source_artifact, discriminator,
        ])
        return AgentMetricEvent(
            schema_version=1,
            event_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            task_id=task_id,
            category=category,
            name=name,
            stage=stage,
            outcome=outcome,
            value=float(value),
            source_artifact=source_artifact,
            occurred_at=utc_now_iso(),
            dimensions=dimensions or {},
        )

    def _llm_and_policy_events(self, run_dir: Path, task_id: str) -> List[AgentMetricEvent]:
        events = []
        trace_dir = run_dir / "logs" / "agent_calls"
        for path in sorted(trace_dir.glob("*.json")):
            data = self._read(path)
            if not data:
                continue
            source = self._source(run_dir, path)
            stage = str(data.get("stage") or path.stem.split("_", 1)[0])
            parsed = data.get("parsed_decision")
            outcome = "parsed" if parsed not in (None, {}, "") else "invalid"
            events.append(self._event(
                task_id=task_id,
                category="llm",
                name="call",
                stage=stage,
                outcome=outcome,
                source_artifact=source,
                dimensions={
                    "provider": str(data.get("provider", ""))[:80],
                    "model": str(data.get("model", ""))[:120],
                    "latency_ms": int(data.get("latency_ms") or 0),
                },
            ))
            policy = data.get("policy_result") if isinstance(data.get("policy_result"), dict) else {}
            for outcome_name, key in (
                ("accepted", "accepted_actions"),
                ("rejected", "rejected_actions"),
            ):
                for index, action in enumerate(policy.get(key) or []):
                    action = action if isinstance(action, dict) else {}
                    events.append(self._event(
                        task_id=task_id,
                        category="policy",
                        name="action_evaluated",
                        stage=stage,
                        outcome=outcome_name,
                        source_artifact=source,
                        dimensions={"action_type": str(action.get("type", ""))[:80]},
                        discriminator="%s:%s" % (key, index),
                    ))
        return events

    def _repair_events(self, run_dir: Path, task_id: str) -> List[AgentMetricEvent]:
        events = []
        apply_path = run_dir / "repairs" / "repair_apply_result.json"
        apply_result = self._read(apply_path)
        if apply_result:
            source = self._source(run_dir, apply_path)
            action_results = apply_result.get("action_results") or []
            if action_results:
                for index, action in enumerate(action_results):
                    action = action if isinstance(action, dict) else {}
                    executed = bool(action.get("executed"))
                    events.append(self._event(
                        task_id=task_id,
                        category="repair",
                        name="action_apply",
                        stage="repair",
                        outcome="executed" if executed else "not_executed",
                        source_artifact=source,
                        dimensions={
                            "action_type": str(
                                action.get("action_type") or action.get("type") or ""
                            )[:80],
                            "exit_code": action.get("exit_code"),
                        },
                        discriminator=str(index),
                    ))
            else:
                executed_count = int(apply_result.get("executed_action_count") or 0)
                if executed_count:
                    events.append(self._event(
                        task_id=task_id,
                        category="repair",
                        name="actions_executed",
                        stage="repair",
                        outcome="executed",
                        value=executed_count,
                        source_artifact=source,
                    ))

        loop_path = run_dir / "repairs" / "repair_loop_state.json"
        loop_state = self._read(loop_path)
        for index, attempt in enumerate((loop_state or {}).get("history") or []):
            attempt = attempt if isinstance(attempt, dict) else {}
            events.append(self._event(
                task_id=task_id,
                category="repair",
                name="attempt",
                stage=str(attempt.get("stage") or "repair"),
                outcome=str(attempt.get("status") or "recorded"),
                source_artifact=self._source(run_dir, loop_path),
                discriminator=str(index),
            ))
        return events

    def _recovery_events(self, run_dir: Path, task_id: str) -> List[AgentMetricEvent]:
        events = []
        operations_dir = run_dir / "operations"
        for path in sorted(operations_dir.glob("*.json")):
            if path.name.endswith("_result.json"):
                continue
            operation = self._read(path)
            if not operation or not operation.get("operation_id"):
                continue
            source = self._source(run_dir, path)
            operation_id = str(operation["operation_id"])
            events.append(self._event(
                task_id=task_id,
                category="recovery",
                name="operation",
                stage=str(operation.get("stage", "")),
                outcome=str(operation.get("status", "")),
                source_artifact=source,
                dimensions={
                    "operation_id": operation_id,
                    "idempotency_key": str(
                        operation.get("idempotency_key") or operation_id
                    ),
                    "attempt": int(operation.get("attempt") or 0),
                    "resource_type": str(operation.get("resource_type", ""))[:80],
                },
            ))
            reconcile = operation.get("reconcile_result")
            if isinstance(reconcile, dict) and reconcile.get("decision") == "reuse":
                events.append(self._event(
                    task_id=task_id,
                    category="recovery",
                    name="duplicate_execution_prevented",
                    stage=str(operation.get("stage", "")),
                    outcome="reuse",
                    source_artifact=source,
                    dimensions={"operation_id": operation_id},
                ))

        faults_path = operations_dir / "fault_injections.jsonl"
        for index, fault in enumerate(self._read_jsonl(faults_path)):
            events.append(self._event(
                task_id=task_id,
                category="recovery",
                name="fault_injected",
                stage=str(fault.get("point", "")).split(":", 1)[0],
                outcome=str(fault.get("point", "")),
                source_artifact=self._source(run_dir, faults_path),
                dimensions={"operation_id": str(fault.get("operation_id", ""))},
                discriminator=str(index),
            ))
        return events

    def _verify_events(self, run_dir: Path, task_id: str) -> List[AgentMetricEvent]:
        pipeline_path = run_dir / "reports" / "pipeline_results.json"
        pipeline = self._read(pipeline_path)
        verify = (pipeline or {}).get("verify")
        if not isinstance(verify, dict):
            return []
        status = str(verify.get("status", ""))
        return [self._event(
            task_id=task_id,
            category="verify",
            name="final_result",
            stage="verify",
            outcome=status,
            source_artifact=self._source(run_dir, pipeline_path),
        )]

    def _skill_events(self, run_dir: Path, task_id: str) -> List[AgentMetricEvent]:
        effects_path = run_dir / "reports" / "skill_effects.json"
        effects = self._read(effects_path)
        events = []
        verify_status = ""
        pipeline = self._read(run_dir / "reports" / "pipeline_results.json") or {}
        if isinstance(pipeline.get("verify"), dict):
            verify_status = str(pipeline["verify"].get("status", ""))
        for index, effect in enumerate((effects or {}).get("effects") or []):
            effect = effect if isinstance(effect, dict) else {}
            influenced = bool(effect.get("field_changed"))
            accepted = bool(effect.get("accepted_by_policy"))
            harmful = influenced and accepted and verify_status not in ("pass", "passed")
            events.append(self._event(
                task_id=task_id,
                category="skill",
                name="effect",
                stage=str(effect.get("stage", "")),
                outcome="harmful" if harmful else ("influenced" if influenced else "selected"),
                source_artifact=self._source(run_dir, effects_path),
                dimensions={
                    "skill_name": str(effect.get("skill_name", ""))[:120],
                    "accepted_by_policy": accepted,
                },
                discriminator=str(index),
            ))
        return events

    def _aggregate(self, events: List[AgentMetricEvent]) -> Dict[str, Any]:
        counters = {
            "llm_calls": 0,
            "policy_accepted": 0,
            "policy_rejected": 0,
            "repair_actions_executed": 0,
            "repair_attempts": 0,
            "recovery_operations": 0,
            "duplicate_execution_prevented": 0,
            "faults_injected": 0,
            "verify_passes": 0,
            "verify_failures": 0,
            "skill_influences": 0,
            "skill_harms": 0,
        }
        for event in events:
            amount = int(event.value)
            if event.category == "llm" and event.name == "call":
                counters["llm_calls"] += amount
            elif event.category == "policy" and event.outcome == "accepted":
                counters["policy_accepted"] += amount
            elif event.category == "policy" and event.outcome == "rejected":
                counters["policy_rejected"] += amount
            elif event.category == "repair" and event.name in ("action_apply", "actions_executed") and event.outcome == "executed":
                counters["repair_actions_executed"] += amount
            elif event.category == "repair" and event.name == "attempt":
                counters["repair_attempts"] += amount
            elif event.category == "recovery" and event.name == "operation":
                counters["recovery_operations"] += amount
            elif event.category == "recovery" and event.name == "duplicate_execution_prevented":
                counters["duplicate_execution_prevented"] += amount
            elif event.category == "recovery" and event.name == "fault_injected":
                counters["faults_injected"] += amount
            elif event.category == "verify" and event.outcome in ("pass", "passed"):
                counters["verify_passes"] += amount
            elif event.category == "verify":
                counters["verify_failures"] += amount
            elif event.category == "skill" and event.outcome in ("influenced", "harmful"):
                counters["skill_influences"] += amount
                counters["skill_harms"] += amount if event.outcome == "harmful" else 0

        policy_total = counters["policy_accepted"] + counters["policy_rejected"]
        skill_total = counters["skill_influences"]
        return {
            "counters": counters,
            "rates": {
                "policy_accept_rate": (
                    counters["policy_accepted"] / policy_total if policy_total else 0.0
                ),
                "skill_harm_rate": (
                    counters["skill_harms"] / skill_total if skill_total else 0.0
                ),
            },
        }

    def _deployment_capability_events(
        self, run_dir: Path, task_id: str,
    ) -> List[AgentMetricEvent]:
        """Phase B4: emit the deployment capability metrics as events."""
        metrics = compute_deployment_metrics(run_dir)
        events = []
        source = self._source(run_dir, run_dir / "reports" / "deployment_metrics.json")
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value:
                    events.append(self._event(
                        task_id=task_id,
                        category="deployment",
                        name=name,
                        stage="deploy",
                        outcome="recorded",
                        value=float(value),
                        source_artifact=source,
                    ))
            elif isinstance(value, dict):
                labels = {
                    key: count for key, count in value.items()
                    if isinstance(count, (int, float)) and count
                }
                for label, count in labels.items():
                    events.append(self._event(
                        task_id=task_id,
                        category="deployment",
                        name=name,
                        stage="deploy",
                        outcome="recorded",
                        value=float(count),
                        source_artifact=source,
                        dimensions={"label": str(label)[:120]},
                        discriminator="%s:%s" % (name, label),
                    ))
        return events

    @staticmethod
    def _source(run_dir: Path, path: Path) -> str:
        try:
            return str(Path(path).relative_to(run_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _read(path: Path) -> Dict[str, Any]:
        if not Path(path).exists():
            return {}
        try:
            value = read_json(Path(path))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not Path(path).exists():
            return []
        items = []
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if isinstance(value, dict):
                    items.append(value)
        except (OSError, ValueError):
            return []
        return items


def _foundation_artifact(reports_dir: Path, name: str) -> Any:
    """Read a foundation audit artifact written in the wrapped envelope."""
    path = Path(reports_dir) / name
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def compute_deployment_metrics(run_dir: Path) -> Dict[str, Any]:
    """Phase B4: aggregate deployment-capability metrics from run artifacts.

    Metric names follow the universal deployment expansion plan.  Every
    counter is derived from persisted, secret-redacted artifacts only.
    """
    run_dir = Path(run_dir)
    reports = run_dir / "reports"
    capabilities = _foundation_artifact(reports, "project_capabilities.json") or {}
    candidates = _foundation_artifact(reports, "deployment_candidates.json") or []
    deployability = _foundation_artifact(reports, "deployability_assessment.json") or {}
    verify_selection = _foundation_artifact(
        reports, "protocol_verify_selection.json",
    ) or {}
    llm_resolution = _foundation_artifact(reports, "llm_resolution.json") or {}
    candidate_attempts = _read_jsonl_public(reports / "candidate_attempts.jsonl")
    authorization_attempts = _read_jsonl_public(reports / "command_attempts.jsonl")
    fallbacks = _read_jsonl_public(reports / "command_fallbacks.jsonl")

    frameworks = capabilities.get("service_frameworks") or []
    ui = capabilities.get("ui_frameworks") or []
    runtimes = capabilities.get("inference_runtimes") or []
    ml = capabilities.get("ml_libraries") or []
    languages = capabilities.get("languages") or ["unknown"]
    frameworks_known = bool(frameworks or ui or runtimes or ml)
    unknown = not frameworks_known

    def _labeled(counter: Dict[str, int], key: str) -> Dict[str, int]:
        counter = dict(counter)
        counter["total"] = sum(counter.values())
        return counter

    by_source: Dict[str, int] = {}
    by_adapter: Dict[str, int] = {}
    for candidate in candidates if isinstance(candidates, list) else []:
        source = str(candidate.get("source") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        for adapter in candidate.get("adapter_ids") or []:
            by_adapter[adapter] = by_adapter.get(adapter, 0) + 1
    selected_id = str(deployability.get("selected_candidate_id") or "")
    selected = next(
        (
            candidate for candidate in candidates if isinstance(candidates, list)
            and candidate.get("candidate_id") == selected_id
        ),
        None,
    ) if isinstance(candidates, list) else None

    rejected_by_reason: Dict[str, int] = {}
    for attempt in authorization_attempts:
        if str(attempt.get("verdict")) in ("candidate_rejected", "hard_denied"):
            reason = str(attempt.get("reason_code") or "unknown")
            rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
    fallback_by_outcome: Dict[str, int] = {}
    for record in fallbacks:
        outcome = str(record.get("reason") or "unknown")
        fallback_by_outcome[outcome] = fallback_by_outcome.get(outcome, 0) + 1
    for attempt in candidate_attempts:
        outcome = str(attempt.get("outcome") or "unknown")
        fallback_by_outcome[outcome] = fallback_by_outcome.get(outcome, 0) + 1

    verify_status = str(verify_selection.get("status") or "")
    verifier_id = str(verify_selection.get("verifier_id") or "unknown")
    strong = bool(
        (verify_selection.get("shadow_decision") or {}).get("strong_evidence")
        or verify_selection.get("strong_evidence")
    )
    verify_pass_total = 1 if verify_status == "passed" and strong else 0
    verify_attempt_total = 1 if verify_status else 0

    # A false success is a passing verify without strong current-trace
    # evidence; unsafe execution is a command that ran without an
    # auto-allowed or approved authorization decision.
    false_success_total = 1 if (verify_status == "passed" and not strong) else 0
    safe_decisions = {
        str(attempt.get("candidate_id"))
        for attempt in authorization_attempts
        if str(attempt.get("verdict")) in ("auto_allowed", "approval_required")
    }
    unsafe_command_execution_total = sum(
        1 for attempt in candidate_attempts
        if str(attempt.get("outcome")) == "authorized"
        and str(attempt.get("candidate_id"))
        and str(attempt.get("candidate_id")) not in safe_decisions
    )

    llm_outcome = str(llm_resolution.get("contribution") or "deterministic_no_llm")
    return {
        "project_framework_unknown_total": 1 if unknown else 0,
        "project_deployability_ready_total": _labeled(
            {
                str(deployability.get("status") or "unknown"): (
                    1 if deployability.get("status") == "ready" else 0
                )
            },
            "status",
        ),
        "deployment_candidate_total": {
            "total": len(candidates) if isinstance(candidates, list) else 0,
            "by_source": _labeled(by_source, "source"),
            "by_adapter": _labeled(by_adapter, "adapter"),
        },
        "deployment_candidate_selected_total": {
            "total": 1 if selected else 0,
            "by_source": _labeled(
                {str((selected or {}).get("source") or "unknown"): 1 if selected else 0},
                "source",
            ),
        },
        "deployment_candidate_rejected_total": _labeled(rejected_by_reason, "reason_code"),
        "deployment_candidate_fallback_total": _labeled(fallback_by_outcome, "outcome"),
        "unknown_framework_verified_total": _labeled(
            {
                str(languages[0] if languages else "unknown"): (
                    1 if unknown and verify_status == "passed" and strong else 0
                )
            },
            "runtime_family",
        ),
        "llm_unknown_fallback_total": _labeled(
            {llm_outcome: 1 if llm_resolution else 0},
            "outcome",
        ),
        "protocol_verify_attempt_total": _labeled(
            {verifier_id: verify_attempt_total}, "verifier",
        ),
        "protocol_verify_pass_total": _labeled(
            {verifier_id: verify_pass_total}, "verifier",
        ),
        "deployment_false_success_total": false_success_total,
        "unsafe_command_execution_total": unsafe_command_execution_total,
    }


def _read_jsonl_public(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    items = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                items.append(value)
    except OSError:
        return []
    return items
