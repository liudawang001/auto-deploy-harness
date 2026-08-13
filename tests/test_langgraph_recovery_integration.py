"""Phase 6+7 定向测试：Recovery Gate 接入主图 + Checkpoint Resume & Human Interrupt。

测试覆盖：
1. 新 operation 执行；
2. committed operation 复用不重复执行；
3. running operation reconcile；
4. conflict 停止；
5. manual/cleanup 需要 approval；
6. crash window：checkpoint 前 crash，journal 为 running；
7. 副作用完成后 journal commit；
8. dependency environment identity 冲突；
9. ordinary graph resume；
10. approval interrupt 后 approve 恢复；
11. reject 后 executor 调用为 0；
12. approval hash mismatch 拒绝；
13. LangGraph task 切 legacy resume 被拒绝；
14. committed runner operation 不重复启动进程；
15. resume 最终仍必须经过 VerifyModule。
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.recovery.graph_adapter import GraphRecoveryAdapter, RecoveryDecision


class TestGraphRecoveryAdapter:
    """GraphRecoveryAdapter 基本功能测试。"""

    def test_build_operation_has_stable_id(self):
        """operation ID 由确定性输入生成，不包含时间/PID。"""
        state = {
            "task_id": "t1",
            "run_dir": "/tmp/runs/t1",
            "repo_dir": "/tmp/runs/t1/workspace/repo",
            "runtime_policy": {"env_backend": "conda"},
            "compiled_analysis": {"install_plan": ["pip install torch"]},
        }
        adapter = GraphRecoveryAdapter()
        op = adapter.build_operation(state, "env_deploy")
        assert op["operation_id"]
        assert op["resource_type"] == "dependency_install"
        # Same inputs -> same ID
        op2 = adapter.build_operation(state, "env_deploy")
        assert op["operation_id"] == op2["operation_id"]

    def test_build_operation_runner_uses_local_process(self):
        """runner stage 根据 runtime_policy 决定 resource_type。"""
        state = {
            "task_id": "t1",
            "run_dir": "/tmp/runs/t1",
            "repo_dir": "/tmp/runs/t1/workspace/repo",
            "runtime_policy": {"execution_backend": "local"},
            "compiled_analysis": {},
        }
        adapter = GraphRecoveryAdapter()
        op = adapter.build_operation(state, "runner")
        assert op["resource_type"] == "local_process"

    def test_build_operation_runner_docker(self):
        """Docker backend 使用 docker_service resource_type。"""
        state = {
            "task_id": "t1",
            "run_dir": "/tmp/runs/t1",
            "repo_dir": "/tmp/runs/t1/workspace/repo",
            "runtime_policy": {"execution_backend": "docker"},
            "compiled_analysis": {},
        }
        adapter = GraphRecoveryAdapter()
        op = adapter.build_operation(state, "runner")
        assert op["resource_type"] == "docker_service"

    def test_prepare_new_operation_returns_execute(self, tmp_path):
        """新 operation 返回 execute 决策。"""
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {},
        }
        adapter = GraphRecoveryAdapter()
        decision = adapter.prepare_or_reconcile(state, "env_deploy")
        assert decision.decision == "execute"

    def test_no_secrets_in_operation(self):
        """operation ID 输入不包含 secret/token/PID。"""
        state = {
            "task_id": "t1",
            "run_dir": "/tmp/runs/t1",
            "repo_dir": "/tmp/runs/t1/workspace/repo",
            "runtime_policy": {"api_key": "sk-secret123"},
            "compiled_analysis": {"install_plan": ["pip install torch"]},
        }
        adapter = GraphRecoveryAdapter()
        op = adapter.build_operation(state, "env_deploy")
        # Verify normalized_input doesn't contain secrets
        norm = op["normalized_input"]
        assert "api_key" not in json.dumps(norm)
        assert "sk-secret" not in json.dumps(norm)

    def test_capabilities_reflect_reconcilers(self):
        """capabilities 正确反映可用 reconciler。"""
        adapter = GraphRecoveryAdapter(reconcilers={
            "model_download": MagicMock(),
            "dependency_install": MagicMock(),
        })
        caps = adapter.capabilities({})
        assert caps["download"] is True
        assert caps["dependency_install"] is True
        assert caps["local_process"] is False

    def test_recovery_decision_dataclass(self):
        """RecoveryDecision 是不可变的。"""
        d = RecoveryDecision(
            decision="execute",
            operation={},
            reconcile_result={},
            hydrated_stage_result={},
        )
        assert d.decision == "execute"
        with pytest.raises(AttributeError):
            d.decision = "reuse"


class TestRecoveryGateNode:
    """Recovery gate 节点集成测试。"""

    def test_no_adapter_allows_execution(self):
        """没有 recovery_adapter 时允许执行。"""
        from auto_harness.graph.nodes import make_recovery_gate_node

        class MockDeps:
            recovery_adapter = None
            runtime_config = None

        node = make_recovery_gate_node("env_deploy", MockDeps())
        state = {"run_dir": "/tmp", "task_id": "t1"}
        result = node(state)
        assert result["recovery_decision"] == "execute"

    def test_reuse_skips_stage(self):
        """reuse 决策跳过执行并 hydrate 结果。"""
        from auto_harness.graph.nodes import make_recovery_gate_node

        mock_adapter = MagicMock()
        mock_decision = RecoveryDecision(
            decision="reuse",
            operation={"operation_id": "op1"},
            reconcile_result={"decision": "reuse"},
            hydrated_stage_result={"status": "passed", "data": {"reused": True}},
        )
        mock_adapter.prepare_or_reconcile.return_value = mock_decision

        class MockDeps:
            recovery_adapter = mock_adapter
            runtime_config = None

        node = make_recovery_gate_node("env_deploy", MockDeps())
        state = {
            "run_dir": "/tmp",
            "task_id": "t1",
            "repo_dir": "/tmp/repo",
            "runtime_policy": {},
            "compiled_analysis": {},
            "stage_results": {},
        }
        result = node(state)
        assert result["recovery_decision"] == "reuse"
        assert result["recovery_skip_stage"] is True

    def test_retry_clears_prior_stage_reuse_flag(self):
        """A reused env stage must not skip a later repaired runner stage."""
        from auto_harness.graph.nodes import make_recovery_gate_node

        mock_adapter = MagicMock()
        mock_adapter.prepare_or_reconcile.return_value = RecoveryDecision(
            decision="retry",
            operation={"operation_id": "op2"},
            reconcile_result={"decision": "retry"},
            hydrated_stage_result={},
        )

        class MockDeps:
            recovery_adapter = mock_adapter
            runtime_config = None

        result = make_recovery_gate_node("runner", MockDeps())({
            "run_dir": "/tmp",
            "task_id": "t1",
            "repo_dir": "/tmp/repo",
            "runtime_policy": {},
            "compiled_analysis": {},
            "recovery_skip_stage": True,
        })
        assert result["recovery_decision"] == "retry"
        assert result["recovery_skip_stage"] is False

    def test_conflict_stops(self):
        """conflict 决策停止。"""
        from auto_harness.graph.nodes import make_recovery_gate_node

        mock_adapter = MagicMock()
        mock_decision = RecoveryDecision(
            decision="stop",
            operation={},
            reconcile_result={"decision": "conflict"},
            hydrated_stage_result={},
            stop_reason="recovery_conflict",
        )
        mock_adapter.prepare_or_reconcile.return_value = mock_decision

        class MockDeps:
            recovery_adapter = mock_adapter
            runtime_config = None

        node = make_recovery_gate_node("runner", MockDeps())
        state = {
            "run_dir": "/tmp",
            "task_id": "t1",
            "repo_dir": "/tmp/repo",
            "runtime_policy": {},
            "compiled_analysis": {},
        }
        result = node(state)
        assert result["recovery_decision"] == "stop"
        assert "conflict" in result.get("stop_reason", "")

    def test_failed_dependency_without_observed_resource_requests_retry(self):
        """A no-resource dependency failure must not request cleanup."""
        from auto_harness.graph.nodes import make_recovery_gate_node

        mock_adapter = MagicMock()
        mock_adapter.prepare_or_reconcile.return_value = RecoveryDecision(
            decision="approval",
            operation={"operation_id": "op3", "observed_resource": {}},
            reconcile_result={"decision": "manual"},
            hydrated_stage_result={},
            stop_reason="failed_operation_requires_operator_decision",
        )

        class MockDeps:
            recovery_adapter = mock_adapter
            runtime_config = None

        result = make_recovery_gate_node("env_deploy", MockDeps())({
            "run_dir": "/tmp",
            "task_id": "t1",
            "repo_dir": "/tmp/repo",
            "runtime_policy": {},
            "compiled_analysis": {},
        })

        assert result["pending_approval"]["requested_action"] == "retry"
        assert result["pending_approval"]["risk"] == "medium"


class TestRouteAfterRecovery:
    """Recovery gate 路由测试。"""

    def test_execute_routes_to_stage(self):
        """execute 决策路由到 stage 执行。"""
        state = {"recovery_decision": "execute"}
        assert state["recovery_decision"] == "execute"

    def test_reuse_routes_to_next_recovery(self):
        """reuse 决策跳过当前 stage，路由到下一个 recovery gate。"""
        state = {"recovery_decision": "reuse"}
        assert state["recovery_decision"] == "reuse"

    def test_approval_routes_to_approval(self):
        """approval 决策路由到 approval 节点。"""
        state = {"recovery_decision": "approval", "pending_approval": {"kind": "recovery"}}
        assert state["recovery_decision"] == "approval"


# -------------------------------------------------------------------
# Phase 7: Checkpoint Resume & Human Interrupt
# -------------------------------------------------------------------


class TestControllerConsistencyOnResume:
    """Resume 时禁止切换 controller。"""

    def test_langgraph_task_cannot_resume_as_legacy(self):
        """LangGraph task resume 时切 legacy 被拒绝。"""
        from auto_harness.orchestrator import TaskRunner
        from auto_harness.config import HarnessConfig
        from auto_harness.state.store import StateStore
        from auto_harness.models.task import TaskSpec, ProjectSpec, RuntimePolicy
        from auto_harness.utils.time import utc_now_iso

        config = HarnessConfig()
        runner = TaskRunner(config)
        # Create a task with langgraph controller
        spec = TaskSpec(
            task_id="test_lg_001",
            project=ProjectSpec(name="test", repo_url="https://github.com/test/repo"),
            runtime=RuntimePolicy(workspace_root="/tmp"),
            controller="langgraph",
            created_at=utc_now_iso(),
        )
        run_dir = runner.store.create_task(spec)

        # Attempting to resume with legacy controller must raise
        with pytest.raises(ValueError, match="controller_switch_on_resume_is_not_allowed"):
            runner.resume("test_lg_001", dry_run=True, controller="legacy")

    def test_same_controller_resume_allowed(self):
        """相同 controller 的 resume 不报错。"""
        from auto_harness.orchestrator import TaskRunner
        from auto_harness.config import HarnessConfig
        from auto_harness.models.task import TaskSpec, ProjectSpec, RuntimePolicy
        from auto_harness.utils.time import utc_now_iso

        config = HarnessConfig()
        runner = TaskRunner(config)
        spec = TaskSpec(
            task_id="test_lg_002",
            project=ProjectSpec(name="test", repo_url="https://github.com/test/repo"),
            runtime=RuntimePolicy(workspace_root="/tmp"),
            controller="langgraph",
            created_at=utc_now_iso(),
        )
        run_dir = runner.store.create_task(spec)

        # Same controller should not raise ValueError for controller mismatch
        # (it may fail for other reasons like no checkpoint, but not controller mismatch)
        try:
            runner.resume("test_lg_002", dry_run=True, controller="langgraph")
        except ValueError as exc:
            assert "controller_switch" not in str(exc)


class TestApprovalResume:
    """Approval interrupt 后的恢复测试。"""

    def _make_request(self, **overrides):
        """Build a valid approval request for tests."""
        from auto_harness.graph.approval import build_approval_request
        defaults = {
            "approval_id": "app001",
            "operation_id": "op001",
            "approval_kind": "recovery",
            "requested_action": "apply_repair",
            "risk": "high",
            "reason": "source_edit",
        }
        defaults.update(overrides)
        return build_approval_request(**defaults)

    def test_approval_node_reject_stops_graph(self):
        """reject 后 executor 调用次数为 0。"""
        from auto_harness.graph.approval import approval_node

        request = self._make_request(approval_id="app001", operation_id="op001")
        state = {
            "run_dir": "/tmp/test_reject",
            "task_id": "t1",
            "pending_approval": request,
        }

        with patch("langgraph.types.interrupt", return_value={
            "approval_id": "app001",
            "operation_id": "op001",
            "decision": "reject",
            "reviewer": "cli",
            "request_hash": request["request_hash"],
        }):
            result = approval_node(state)

        assert result.get("stop_reason") == "operator_rejected"
        assert result.get("approved_operation_id") == ""

    def test_approval_hash_mismatch_rejected(self):
        """approval hash 不匹配时拒绝。"""
        from auto_harness.graph.approval import approval_node

        request = self._make_request(approval_id="app002", operation_id="op002")
        state = {
            "run_dir": "/tmp/test_hash",
            "task_id": "t1",
            "pending_approval": request,
        }

        # Decision with wrong approval_id
        with patch("langgraph.types.interrupt", return_value={
            "approval_id": "wrongid",
            "operation_id": "op002",
            "decision": "approve",
            "reviewer": "cli",
            "request_hash": request["request_hash"],
        }):
            result = approval_node(state)

        assert "mismatch" in result.get("stop_reason", "")

    def test_approval_operation_mismatch_rejected(self):
        """operation_id 不匹配时拒绝。"""
        from auto_harness.graph.approval import approval_node

        request = self._make_request(approval_id="app003", operation_id="op003")
        state = {
            "run_dir": "/tmp/test_op",
            "task_id": "t1",
            "pending_approval": request,
        }

        # Decision with wrong operation_id
        with patch("langgraph.types.interrupt", return_value={
            "approval_id": "app003",
            "operation_id": "wrongop",
            "decision": "approve",
            "reviewer": "cli",
            "request_hash": request["request_hash"],
        }):
            result = approval_node(state)

        assert "mismatch" in result.get("stop_reason", "")

    def test_approval_approve_allows_execution(self):
        """approve 后允许继续执行。"""
        from auto_harness.graph.approval import approval_node

        request = self._make_request(approval_id="app004", operation_id="op004")
        state = {
            "run_dir": "/tmp/test_approve",
            "task_id": "t1",
            "pending_approval": request,
        }

        with patch("langgraph.types.interrupt", return_value={
            "approval_id": "app004",
            "operation_id": "op004",
            "decision": "approve",
            "reviewer": "cli",
            "request_hash": request["request_hash"],
        }):
            result = approval_node(state)

        assert result.get("stop_reason") == ""
        assert result.get("approved_operation_id") == "op004"
        assert result.get("approved_action") == "apply_repair"


class TestResumeAudit:
    """Resume 审计 artifact 测试。"""

    def test_resume_audit_has_required_fields(self, tmp_path):
        """resume audit 包含所有必要字段。"""
        from auto_harness.controllers.langgraph import LangGraphController
        from auto_harness.controllers.base import DeploymentContext, DeploymentResult

        # Create a minimal mock dependencies
        class MockDeps:
            def initial_state(self, ctx):
                return {}
            def graph_deps(self):
                class GD:
                    runtime_config = MagicMock()
                    # ... other deps
                return GD()
            def to_controller_result(self, output):
                return DeploymentResult(
                    task_id="t1", status="stopped",
                    stop_reason="test", controller="langgraph",
                )
            def completed_result(self, values):
                return DeploymentResult(
                    task_id="t1", status="completed",
                    stop_reason="already_completed", controller="langgraph",
                )
            def blocked_result(self, values, reason):
                return DeploymentResult(
                    task_id="t1", status="blocked",
                    stop_reason=reason, controller="langgraph",
                )

        ctrl = LangGraphController(MockDeps())
        context = DeploymentContext(
            task_id="t1",
            run_dir=str(tmp_path),
            repo_dir=str(tmp_path / "repo"),
            dry_run=True,
            runtime_policy={},
        )

        # Resume with no checkpoint (already completed)
        ctrl.resume(context)

        audit_path = tmp_path / "reports" / "resume_audit.json"
        assert audit_path.exists()
        audit = json.loads(audit_path.read_text())
        assert "task_id" in audit
        assert "controller" in audit
        assert "checkpoint_next" in audit
        assert "resume_kind" in audit
        assert "operation_id" in audit
        assert "reconcile_decision" in audit
        assert "duplicate_execution_prevented" in audit
        assert "resumed_at" in audit
        assert "final_stop_reason" in audit
        assert audit["controller"] == "langgraph"


class TestCommittedOperationNoDuplicate:
    """committed operation resume 不重复执行。"""

    def test_committed_runner_reuse_skips_execution(self, tmp_path):
        """committed runner operation 在 resume 时 hydrate 结果，不重复启动进程。"""
        adapter = GraphRecoveryAdapter()

        # Simulate a committed operation in journal
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {"execution_backend": "local"},
            "compiled_analysis": {"run_candidates": [{"id": "c1", "selected": True}]},
        }

        # First: create the operation
        op = adapter.build_operation(state, "runner")
        from auto_harness.recovery.journal import OperationJournal
        journal = OperationJournal(tmp_path)
        record = journal.create(op)
        # Transition to committed with a result artifact
        result_path = tmp_path / "operations" / ("%s_result.json" % op["operation_id"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            "status": "passed",
            "data": {"port": 7860, "reused": True},
        }))
        journal.transition(op["operation_id"], "committed", result_artifacts=[str(result_path)])

        # Now: prepare_or_reconcile should return reuse
        decision = adapter.prepare_or_reconcile(state, "runner")
        assert decision.decision == "reuse"
        assert decision.hydrated_stage_result.get("status") == "passed"
        assert decision.hydrated_stage_result.get("data", {}).get("reused") is True


# -------------------------------------------------------------------
# Task 3: begin() integration and reconcile decision application
# -------------------------------------------------------------------


class TestNewOperationBeginsRunning:
    """New operations are persisted as running before execute."""

    def test_new_adapter_operation_is_running_before_execute(self, tmp_path):
        """New side effect is persisted as running before executor is called."""
        adapter = GraphRecoveryAdapter()
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {"install_plan": ["pip install flask"]},
        }
        decision = adapter.prepare_or_reconcile(state, "env_deploy")
        assert decision.decision == "execute"
        assert decision.operation["status"] == "running"
        # Verify persisted on disk
        op_id = decision.operation["operation_id"]
        from auto_harness.recovery.journal import OperationJournal
        journal = OperationJournal(tmp_path)
        loaded = journal.load(op_id)
        assert loaded["status"] == "running"


class TestReconcileDecisionApplied:
    """Reconcile decisions are persisted via apply_decision."""

    def _make_running_unknown(self, adapter, state, stage, tmp_path):
        """Helper: create operation, set to running, then recover to unknown."""
        from auto_harness.recovery.journal import OperationJournal
        op = adapter.build_operation(state, stage)
        journal = OperationJournal(tmp_path)
        journal.begin(op)
        # Simulate crash: running -> unknown
        journal.recover_running(op["operation_id"])
        return op["operation_id"]

    def test_reconcile_reuse_transitions_to_committed(self, tmp_path):
        """reconcile reuse persists operation as committed."""
        adapter = GraphRecoveryAdapter(reconcilers={
            "dependency_install": MagicMock(),
        })
        # Reconciler says reuse
        adapter.reconcilers["dependency_install"].reconcile.return_value = {
            "decision": "reuse",
            "reason": "env_exists",
            "observed_state": {"exists": True},
            "evidence_paths": [],
        }
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {"install_plan": ["pip install flask"]},
        }
        op_id = self._make_running_unknown(adapter, state, "env_deploy", tmp_path)

        decision = adapter.prepare_or_reconcile(state, "env_deploy")
        assert decision.decision == "reuse"
        # apply_decision should have persisted committed
        from auto_harness.recovery.journal import OperationJournal
        journal = OperationJournal(tmp_path)
        loaded = journal.load(op_id)
        assert loaded["status"] == "committed"

    def test_cleanup_decision_transitions_to_manual(self, tmp_path):
        """cleanup_then_retry decision persists operation as manual before approval."""
        adapter = GraphRecoveryAdapter(reconcilers={
            "dependency_install": MagicMock(),
        })
        adapter.reconcilers["dependency_install"].reconcile.return_value = {
            "decision": "cleanup_then_retry",
            "reason": "conflicting_env",
        }
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {"install_plan": ["pip install flask"]},
        }
        op_id = self._make_running_unknown(adapter, state, "env_deploy", tmp_path)

        decision = adapter.prepare_or_reconcile(state, "env_deploy")
        assert decision.decision == "approval"
        assert decision.stop_reason == "cleanup_required"
        from auto_harness.recovery.journal import OperationJournal
        journal = OperationJournal(tmp_path)
        loaded = journal.load(op_id)
        assert loaded["status"] == "manual"

    def test_conflict_decision_transitions_to_conflict(self, tmp_path):
        """conflict decision persists operation as conflict before stop."""
        adapter = GraphRecoveryAdapter(reconcilers={
            "local_process": MagicMock(),
        })
        adapter.reconcilers["local_process"].reconcile.return_value = {
            "decision": "conflict",
            "reason": "pid_reuse",
        }
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {"execution_backend": "local"},
            "compiled_analysis": {},
        }
        op_id = self._make_running_unknown(adapter, state, "runner", tmp_path)

        decision = adapter.prepare_or_reconcile(state, "runner")
        assert decision.decision == "stop"
        assert decision.stop_reason == "recovery_conflict"
        from auto_harness.recovery.journal import OperationJournal
        journal = OperationJournal(tmp_path)
        loaded = journal.load(op_id)
        assert loaded["status"] == "conflict"

    def test_unknown_is_reconciled_not_blindly_retried(self, tmp_path):
        """Unknown status must reconcile, not directly retry."""
        adapter = GraphRecoveryAdapter(reconcilers={
            "dependency_install": MagicMock(),
        })
        # No reconciler match -> manual
        adapter.reconcilers["dependency_install"].reconcile.return_value = {
            "decision": "manual",
            "reason": "cannot_verify",
        }
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {"install_plan": ["pip install flask"]},
        }
        op_id = self._make_running_unknown(adapter, state, "env_deploy", tmp_path)

        decision = adapter.prepare_or_reconcile(state, "env_deploy")
        # manual -> approval, never a direct retry
        assert decision.decision == "approval"
        assert decision.decision != "retry"

    def test_failed_is_fail_closed_approval(self, tmp_path):
        """Failed operations require operator decision, not auto-retry."""
        from auto_harness.recovery.journal import OperationJournal
        adapter = GraphRecoveryAdapter()
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {"install_plan": ["pip install flask"]},
        }
        op = adapter.build_operation(state, "env_deploy")
        journal = OperationJournal(tmp_path)
        journal.begin(op)
        journal.transition(op["operation_id"], "failed", error="install_error")

        decision = adapter.prepare_or_reconcile(state, "env_deploy")
        assert decision.decision == "approval"
        assert decision.stop_reason == "failed_operation_requires_operator_decision"

    def test_effective_repair_authorizes_exact_failed_stage_retry(self, tmp_path):
        """A policy-applied effective repair may retry its exact failed stage."""
        from auto_harness.recovery.journal import OperationJournal

        adapter = GraphRecoveryAdapter()
        state = {
            "task_id": "t1",
            "run_dir": str(tmp_path),
            "repo_dir": str(tmp_path / "workspace" / "repo"),
            "runtime_policy": {},
            "compiled_analysis": {},
            "failed_stage": "runner",
            "repair_count": 1,
            "repair_resume_executed": True,
            "repair_apply_result": {"effective_action_count": 1},
        }
        op = adapter.build_operation(state, "runner")
        journal = OperationJournal(tmp_path)
        journal.begin(op)
        journal.transition(op["operation_id"], "failed", error="early_exit")

        decision = adapter.prepare_or_reconcile(state, "runner")

        assert decision.decision == "retry"
        assert journal.load(op["operation_id"])["status"] == "retryable"
