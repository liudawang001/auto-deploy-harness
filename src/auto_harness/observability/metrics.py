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
