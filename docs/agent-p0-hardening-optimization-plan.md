# Agent P0 主链路加固执行计划

## 1. 目标

本轮不增加 LLM 任意命令、源码修改或成功判定权限，优先修复会导致“假 Agent 闭环”的确定性缺陷：

1. Repair Planner、Policy、Applier 使用同一动作契约。
2. 未实现动作必须在执行前失败关闭，禁止静默忽略和部分执行。
3. Skill evolution 只保留一条可修改 Skill 的主链路。
4. Skill candidate 必须经过审批、回归、Shadow、晋升和回滚状态机。
5. 当前 Provider 协议如实标记为 `json_action`，不包装成原生 Tool Calling。

## 2. 已完成改动

### 2.1 Repair Action Contract

涉及文件：

- `src/auto_harness/repair/actions.py`
- `src/auto_harness/repair/planner.py`
- `src/auto_harness/repair/policy.py`
- `src/auto_harness/repair/apply.py`

实现：

- 新增 `RepairActionRegistry`，集中声明支持的动作、动作类型和必需 payload。
- Planner 在 plan 中写入 `action_contract`；不支持或 payload 不完整时，状态改为 `needs_manual_review`。
- Policy 对 Registry 未注册动作返回 `unsupported repair action type`。
- Applier 在执行前对全部动作做预检；任一动作无效时整份 plan 拒绝，不执行合法子集。
- `pin_dependency` 复用受控 pip 安装器。
- `request_env_var_name_only` 作为旧名称兼容输入，规范化为 `set_env_var_name_only`。
- Prompt、LangGraph observation 和 legacy controller observation 只暴露 Applier 可消费动作。

### 2.2 Skill Candidate Lifecycle

涉及文件：

- `src/auto_harness/memory/lifecycle.py`
- `src/auto_harness/memory/evolution.py`
- `src/auto_harness/skills/shadow.py`
- `src/auto_harness/skills/rollback.py`
- `src/auto_harness/cli.py`

主状态：

```text
proposed
  -> approved
  -> regression_passed
  -> shadow_passed
  -> active
  -> rolled_back
```

失败状态：

```text
regression_failed
shadow_failed
rejected
```

约束：

- 未审批 candidate 不能运行 regression。
- 未通过 regression 不能进入 shadow。
- 默认未达到 `shadow_passed` 不能 promote。
- `--no-require-shadow` 只允许从 `regression_passed` promote，必须显式指定。
- 每次迁移写入 `.lifecycle.jsonl`，事件包含 `previous_event_hash` 和 `event_hash`。
- 相同 `run_id` 的 Shadow 结果只能计数一次，禁止重复回放抬高 helped_count。
- 回滚通过同一 lifecycle 写入 `rolled_back`。

### 2.3 单一 Skill 写入口

涉及文件：

- `src/auto_harness/memory/promotion.py`
- `src/auto_harness/benchmarks/runner.py`
- `README.md`

实现：

- `memory-promote` 仅保留旧 proposal 读取和生成能力。
- `memory-promote --apply` 固定失败，不再修改 `skills/*/SKILL.md`。
- 统一使用 `memory-evolve --propose/--approve/--regression/--shadow/--promote`。
- 原验证旧 apply 的 benchmark 已切换到 `MemoryEvolutionManager` 主链路。

### 2.4 Provider 协议真实性

当前结论：

- Mock Provider：`json_action`
- Xunfei Provider：`json_action`
- 当前没有 provider-native function/tool calling adapter

README 和运行证据必须使用“Schema 化 JSON Action + Python Policy/Executor”，不能写成“原生 Typed Tool Calling”。

## 3. 测试与验收

新增测试：

- `tests/test_repair_action_contract.py`
- `tests/test_memory_lifecycle.py`

强化测试：

- `tests/test_memory_evolution_e2e.py` 不再手工篡改 regression 状态，实际调用 approve 和 regression gate。
- `tests/test_skill_shadow.py` 增加 Shadow `run_id` 去重。
- Benchmark 的 Memory evolution case 改走统一状态机。

本轮本地结果：

- Repair/Memory/Policy/Provider 聚焦测试：`171 passed`
- 完整测试：`1208 passed, 18 failed`
- 18 项失败均为当前执行沙箱禁止绑定本地 socket，异常为 `PermissionError: [Errno 1] Operation not permitted`
- Benchmark：68 passed，2 not_run；2 项同样依赖本地端口绑定

## 4. 明确未完成

以下工作需要真实外部环境，不属于本轮确定性代码修复：

1. 使用真实讯飞 API 证明 LLM diagnosis 改变了 repair action。
2. 在真实 Conda/GPU 服务器验证 PyTorch CUDA 安装和服务恢复。
3. 在允许本地端口的环境运行 18 项 HTTP/E2E 测试。
4. 使用真实历史 run 完成至少两个不同 `run_id` 的 Shadow evidence。
5. 生成当前 LangGraph 架构下的可复核 evidence package。

## 5. 下一阶段优先级

### P0

1. 增加三组因果 E2E：失败基线、LLM 决策、动作执行、状态变化、新 trace 验证。
2. 重做 LLM necessity evaluator，禁止用预设 expected status 代替实际运行结果。
3. 为 lifecycle JSONL 增加独立校验命令和 readiness 检查。

### P1

1. [已完成] LangGraph checkpoint 故障注入和跨进程恢复证据。
2. [已完成] Docker install 与 runtime/verify 命令权限分层；未声明已完成持久化镜像 build。
3. [已完成] 统一采集 LLM 调用、Policy reject、repair effectiveness、Skill gain 指标。

P1 本地验收：

- 三个故障窗口、稳定幂等键、durable result 和独立 Python 子进程恢复测试已接入。
- Docker install/runtime/verify phase profile 已接入命令构造和测试。
- 统一 `AgentMetricEvent`、JSONL、聚合和 provenance 已接入 `AgentMetricsCollector`。
- 新增 3 个 benchmark case，均通过；完整 benchmark 为 `71 passed, 2 not_run`。
- 完整测试为 `1222 passed, 18 failed`；18 项均因当前执行沙箱禁止本地 socket bind。
- 未运行真实 LLM API、Docker daemon、联网模型部署或 GPU smoke。

### 不做

- 不开放任意 shell。
- 不允许 LLM 修改源码后自动执行。
- 不允许 LLM 直接判定部署成功。
- 不为提高“Agent 纯度”引入无必要的多 Agent、RAG 或向量数据库。
