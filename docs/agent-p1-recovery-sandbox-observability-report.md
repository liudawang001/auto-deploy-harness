# Agent P1 恢复、沙箱与可观测性实施报告

## 1. 范围

本阶段完成 `docs/agent-p0-hardening-optimization-plan.md` 中的三项 P1：

1. LangGraph checkpoint 故障注入和跨进程恢复证据。
2. Docker install 与 runtime/verify 权限分层。
3. 统一采集 LLM、Policy、Repair、Recovery、Verify、Skill 指标。

本阶段不调用真实 LLM API，不执行 Docker daemon，不访问外网，也不声称完成真实 GPU 部署验证。

## 2. LangGraph 故障注入与幂等恢复

### 2.1 新增故障窗口

`src/auto_harness/recovery/faults.py` 提供持久化、一次性 `FaultInjector`：

```text
<stage>:before_side_effect
<stage>:after_side_effect_before_commit
<stage>:after_commit_before_checkpoint
```

故障点通过 `GraphNodeDependencies` 注入 `make_stage_node`。默认配置为空，不影响正常任务：

```json
{
  "langgraph_fault_injection_points": []
}
```

也可以临时使用环境变量：

```bash
AUTO_HARNESS_LANGGRAPH_FAULT_INJECTION=env_deploy:after_side_effect_before_commit
```

每个 `(operation_id, point)` 在 run 目录中最多触发一次。触发前先持久化 marker 和
`operations/fault_injections.jsonl`，重启后不会在相同位置无限崩溃。

### 2.2 幂等键和 durable result

- `GraphRecoveryAdapter.build_operation()` 写入 `idempotency_key`，其值等于稳定 `operation_id`。
- 成功的 stage result 在 journal commit 前写入 `<operation_id>_result.json`。
- 如果副作用已完成但进程在 commit 前退出，reconciler 判定 `reuse` 后可以读取该结果。
- 如果 journal 已 committed 但 LangGraph checkpoint 未推进，恢复时 hydrate 结果并跳过 stage。
- 修复了 `make_stage_node` 中 `reuse` 先被 `recovery_execution_not_allowed` 截断的问题。

这套逻辑不允许未知副作用被直接重试。没有 reconciler 或无法证明外部状态时仍进入
manual/approval，而不是为了自动化率重复执行。

### 2.3 证据

`tests/test_p1_recovery_fault_windows.py` 覆盖：

- 副作用前退出：executor 调用次数为 0，journal 为 running。
- 副作用后 commit 前退出：结果已持久化，新实例 reconcile 后复用，调用次数保持 1。
- commit 后 checkpoint 前退出：committed 结果 hydrate，调用次数保持 1。
- 新 `FaultInjector` 实例读取旧 marker，不重复触发。
- 独立 Python 子进程以退出码 73 模拟崩溃，父进程恢复后副作用计数仍为 1。

## 3. Docker 分阶段安全策略

`DockerSandboxBackend.for_phase()` 提供三个固定 profile：

| 策略 | install | runtime | verify |
|---|---|---|---|
| repo mount | `rw` | `ro` | `ro` |
| root filesystem | writable | read-only | read-only |
| user | 可配置 | 默认 `65532:65532` | 默认 `65532:65532` |
| model cache | `rw` | `ro` | `ro` |
| capabilities | drop ALL | drop ALL | drop ALL |
| no-new-privileges | 开启 | 开启 | 开启 |
| GPU | 按配置 | 按配置 | 强制 `none` |
| HOME | 镜像默认 | `/tmp` | `/tmp` |

安全不变量由 phase profile 强制设置，不会被全局 `repo_mount_mode=rw`、
`read_only_rootfs=false` 或关闭 capability policy 的配置覆盖。内存、CPU、PID 和 tmpfs
额度仍允许配置。

当前接线：

- `EnvDeployModule` 使用 `install`。
- `RunnerModule` 使用 `runtime`。
- `verify` profile 已提供并进入测试，当前 VerifyModule 仍主要从宿主侧验证服务。

边界：本阶段验证生成的 Docker 命令和安全元数据，不代表已在 Docker daemon 中执行。
现有 install 容器的依赖持久化/镜像构建仍需真实 Docker 方案单独验证。

## 4. 统一指标

`src/auto_harness/observability/metrics.py` 新增：

- `AgentMetricEvent`
- `MetricEventWriter`
- `UnifiedMetricsCollector`

事件统一字段包括：

```text
schema_version, event_id, task_id, category, name, stage,
outcome, value, source_artifact, occurred_at, dimensions
```

数据源只读取已持久化工件：

| 类别 | 来源 |
|---|---|
| LLM / Policy | `logs/agent_calls/*.json` |
| Repair | `repairs/repair_apply_result.json`、`repair_loop_state.json` |
| Recovery | `operations/*.json`、`fault_injections.jsonl` |
| Verify | `reports/pipeline_results.json` |
| Skill | `reports/skill_effects.json` |

输出：

```text
reports/agent_metric_events.jsonl
reports/unified_metrics.json
reports/agent_metrics.json
```

`agent_metric_events.jsonl` 每次重新从源工件生成，稳定 `event_id` 防止重复采集。
`unified_metrics.json` 保存 counters、rates 和 provenance。当前核心指标包括：

- LLM 调用数；
- Policy accept/reject；
- Repair action 执行数和 attempt 数；
- Recovery operation 和 duplicate execution prevented；
- 故障注入次数；
- Verify pass/failure；
- Skill influence/harm。

这些指标能证明工件中记录了什么，不能单独证明 LLM 决策质量或线上收益。LLM 必要性仍需
真实 baseline/agent paired run。

## 5. Readiness 和 Benchmark

新增 benchmark：

1. `langgraph_fault_injection_idempotency`
2. `docker_phase_security_profiles`
3. `unified_metrics_consistency`

新增 capability：

- `fault_window_idempotency`
- `docker_phase_profiles`
- `unified_agent_observability`

本阶段本地结果：

```text
P1 相关测试：46 passed
P1 benchmark：3 passed
readiness：ready_for_external_smoke
benchmark manifest：73 cases
完整测试：1222 passed, 18 failed
完整 benchmark：71 passed, 2 not_run
```

18 项完整测试失败均为当前执行沙箱禁止绑定本地 socket 的
`PermissionError: [Errno 1] Operation not permitted`。2 个 benchmark `not_run` 为
`dashboard_http_server` 和 `langgraph_self_repair_controller_e2e`，原因同样是 socket bind
受限。本报告不把环境阻断标记为通过，也不把本地确定性测试标记为真实联网、真实 Docker
或真实 GPU 验证。

## 6. 面试可陈述范围

可以陈述：

- 为 LangGraph 副作用节点设计稳定幂等键、operation journal、三窗口故障注入和跨进程恢复测试。
- 将 Docker 安装与运行/验证命令策略分层，运行阶段强制只读和非 root。
- 从真实运行工件生成统一 Agent 指标事件，并保留 provenance。

不能陈述：

- 已证明任意副作用都能自动恢复。
- 已完成生产级分布式 exactly-once。
- 已在真实 Docker/GPU 环境验证全部 profile。
- 统一指标已经证明 LLM 显著提升成功率。
