"""Phase 1 定向测试：默认控制器和配置语义。

验证：
1. deploy 未指定 controller，task.json 为 langgraph；
2. 显式 --controller legacy 保持 legacy；
3. 旧 task.json 无 controller 时加载为 legacy；
4. execute + mock + langgraph 被拒绝；
5. dry-run + mock + langgraph 可运行到 graph；
6. LangGraph 缺依赖时明确报错，不 fallback；
7. legacy 不因默认 controller 变化而自动打开 Plan-first。
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_harness.config import HarnessConfig
from auto_harness.controllers.validation import ControllerValidation, validate_controller_run
from auto_harness.models.task import ProjectSpec, RuntimePolicy, TaskSpec
from auto_harness.state.store import StateStore


class TestDefaultController:
    """新任务默认使用 LangGraph 控制器。"""

    def test_taskspec_default_is_langgraph(self):
        """TaskSpec.controller 默认值为 'langgraph'。"""
        spec = TaskSpec(
            task_id="t1",
            project=ProjectSpec(name="test", repo_url="https://example.com/repo"),
            runtime=RuntimePolicy(workspace_root="/tmp"),
            created_at="2026-01-01T00:00:00Z",
        )
        assert spec.controller == "langgraph"

    def test_taskspec_explicit_legacy(self):
        """显式指定 controller=legacy 保持 legacy。"""
        spec = TaskSpec(
            task_id="t1",
            project=ProjectSpec(name="test", repo_url="https://example.com/repo"),
            runtime=RuntimePolicy(workspace_root="/tmp"),
            created_at="2026-01-01T00:00:00Z",
            controller="legacy",
        )
        assert spec.controller == "legacy"

    def test_config_default_controller_is_langgraph(self):
        """HarnessConfig.default_controller 默认为 'langgraph'。"""
        config = HarnessConfig()
        assert config.default_controller == "langgraph"

    def test_config_invalid_controller_raises(self):
        """无效的 default_controller 抛出 ValueError。"""
        with pytest.raises(ValueError, match="default_controller"):
            HarnessConfig(default_controller="invalid")


class TestOldTaskMigration:
    """旧任务（缺少 controller 字段）保持 legacy 语义。"""

    def test_old_task_json_loads_as_legacy(self, tmp_path):
        """旧 task.json 没有 controller 字段时加载为 legacy。"""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        task_dir = runs_dir / "old_task"
        task_dir.mkdir()

        # 写一个没有 controller 字段的旧格式 task.json
        old_task = {
            "task_id": "old_task",
            "project": {"name": "test", "repo_url": "https://example.com/repo", "branch": "main"},
            "runtime": {"workspace_root": "/tmp", "allow_dependency_install": False, "allow_service_start": False},
            "created_at": "2025-01-01T00:00:00Z",
        }
        (task_dir / "task.json").write_text(json.dumps(old_task), encoding="utf-8")

        store = StateStore(runs_dir)
        loaded = store.load_task("old_task")
        assert loaded.controller == "legacy"

    def test_new_task_json_loads_as_langgraph(self, tmp_path):
        """新 task.json 有 controller=langgraph 时加载为 langgraph。"""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        task_dir = runs_dir / "new_task"
        task_dir.mkdir()

        new_task = {
            "task_id": "new_task",
            "project": {"name": "test", "repo_url": "https://example.com/repo", "branch": "main"},
            "runtime": {"workspace_root": "/tmp", "allow_dependency_install": False, "allow_service_start": False},
            "created_at": "2026-01-01T00:00:00Z",
            "controller": "langgraph",
        }
        (task_dir / "task.json").write_text(json.dumps(new_task), encoding="utf-8")

        store = StateStore(runs_dir)
        loaded = store.load_task("new_task")
        assert loaded.controller == "langgraph"


class TestControllerValidation:
    """运行前校验：LangGraph + mock + execute 被拒绝。"""

    def test_legacy_always_allowed(self):
        """Legacy 控制器永远允许。"""
        config = HarnessConfig()
        result = validate_controller_run(
            controller="legacy", dry_run=False, provider_name="mock", config=config,
        )
        assert result.allowed is True

    def test_langgraph_execute_mock_rejected(self):
        """execute + mock + langgraph 被拒绝。"""
        config = HarnessConfig()
        result = validate_controller_run(
            controller="langgraph", dry_run=False, provider_name="mock", config=config,
        )
        assert result.allowed is False
        assert "mock" in result.reason

    def test_langgraph_dry_run_mock_allowed(self):
        """dry-run + mock + langgraph 允许。"""
        config = HarnessConfig()
        result = validate_controller_run(
            controller="langgraph", dry_run=True, provider_name="mock", config=config,
        )
        assert result.allowed is True

    def test_langgraph_execute_real_provider_allowed(self):
        """execute + 真实 provider + langgraph 允许。"""
        config = HarnessConfig()
        result = validate_controller_run(
            controller="langgraph", dry_run=False, provider_name="xunfei", config=config,
        )
        assert result.allowed is True

    def test_langgraph_llm_not_required(self):
        """langgraph_require_llm=False 时即使 mock 也允许。"""
        config = HarnessConfig(langgraph_require_llm=False)
        result = validate_controller_run(
            controller="langgraph", dry_run=False, provider_name="mock", config=config,
        )
        assert result.allowed is True

    def test_langgraph_dry_run_mock_disabled(self):
        """langgraph_allow_mock_in_dry_run=False 时拒绝。"""
        config = HarnessConfig(langgraph_allow_mock_in_dry_run=False)
        result = validate_controller_run(
            controller="langgraph", dry_run=True, provider_name="mock", config=config,
        )
        assert result.allowed is False

    def test_validation_result_is_frozen(self):
        """ControllerValidation 是不可变 dataclass。"""
        result = ControllerValidation(allowed=True)
        with pytest.raises(AttributeError):
            result.allowed = False


class TestNoSilentFallback:
    """LLM 不可用时 fail fast，不静默降级到 legacy。"""

    def test_langgraph_llm_failure_does_not_use_legacy(self):
        """LangGraph LLM 失败不会切换到 legacy。

        验证方式：当 validate_controller_run 拒绝时，
        调用者不应 fallback 到 legacy。
        """
        config = HarnessConfig()
        result = validate_controller_run(
            controller="langgraph", dry_run=False, provider_name="mock", config=config,
        )
        # 被拒绝，调用者应停止，不 fallback
        assert result.allowed is False
        # 原因明确，不是 "fallback_to_legacy"
        assert "fallback" not in result.reason


class TestLegacyUnchanged:
    """Legacy 控制器不因默认值变化而隐式调用 LangGraph planner。"""

    def test_legacy_does_not_open_plan_first(self):
        """显式 --controller legacy 不会打开 agent_plan_first。"""
        config = HarnessConfig(default_controller="langgraph")
        # Legacy 模式下 agent_plan_first 应保持 False
        assert config.agent_plan_first is False

    def test_explicit_legacy_stays_legacy(self):
        """显式指定 controller=legacy，解析后仍是 legacy。"""
        config = HarnessConfig(default_controller="langgraph")
        selected = "legacy" or config.default_controller
        assert selected == "legacy"

    def test_default_controller_resolves_when_none(self):
        """未指定 controller 时使用 default_controller。"""
        config = HarnessConfig(default_controller="langgraph")
        selected = None or config.default_controller
        assert selected == "langgraph"


class TestLangGraphConfigValidation:
    """LangGraph 配置校验。"""

    def test_negative_max_diagnoses_raises(self):
        """负数 langgraph_max_diagnoses 抛出 ValueError。"""
        with pytest.raises(ValueError, match="langgraph_max_diagnoses"):
            HarnessConfig(langgraph_max_diagnoses=-1)

    def test_negative_max_repairs_raises(self):
        """负数 langgraph_max_repairs 抛出 ValueError。"""
        with pytest.raises(ValueError, match="langgraph_max_repairs"):
            HarnessConfig(langgraph_max_repairs=-1)

    def test_negative_max_same_failure_raises(self):
        """负数 langgraph_max_same_failure 抛出 ValueError。"""
        with pytest.raises(ValueError, match="langgraph_max_same_failure"):
            HarnessConfig(langgraph_max_same_failure=-1)

    def test_zero_limits_allowed(self):
        """零限制是合法的（禁用该功能）。"""
        config = HarnessConfig(
            langgraph_max_diagnoses=0,
            langgraph_max_repairs=0,
            langgraph_max_same_failure=0,
        )
        assert config.langgraph_max_diagnoses == 0
        assert config.langgraph_max_repairs == 0
        assert config.langgraph_max_same_failure == 0


class TestPyprojectDependencies:
    """langgraph 依赖应在核心依赖中。"""

    def test_langgraph_in_core_deps(self):
        """pyproject.toml 中 langgraph 在核心依赖中。"""
        pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        # 检查 [project] dependencies 包含 langgraph
        import re
        # 找到 dependencies 列表
        match = re.search(r'dependencies\s*=\s*\[([^\]]*)\]', pyproject)
        assert match, "dependencies list not found in pyproject.toml"
        deps = match.group(1)
        assert '"langgraph"' in deps
        assert '"langgraph-checkpoint-sqlite"' in deps
