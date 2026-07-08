# LLM 必要性与真实 Agent 化优化方案

## 0. 目标

本方案目标是把 AI-Auto-Harness 从“部署自动化流水线 + LLM advisor”优化为“面向未知 AI Demo 的受控探索型部署 Agent”。

核心不是扩大 LLM 权限，而是让 LLM 进入不可替代的不确定决策闭环：

- LLM 负责项目理解、实验规划、失败诊断、修复策略选择和验证路径推断。
- Python runtime 负责工具执行、权限控制、状态恢复、审计记录和证据化 verify。
- 最终成功只能由 evidence-based verify 判定，不能由 LLM 判定。

本计划不要求立即执行真实联网、真实 GPU、真实 Docker 或真实大模型测试。真实 E2E 作为后续证据补强项。

## 1. 当前问题判断

当前项目已有 Agentic 成分，但 LLM 必要性不足：

1. 主流程仍是固定 pipeline：`analyze -> resource_plan -> env_solve -> env_deploy -> model_prepare -> runner -> verify -> report`。
2. LLM 多数作为 planner/advisor/diagnoser，不是每轮决策主路径。
3. deterministic analyzer 已能完成大量项目识别，LLM 的增量贡献没有被量化。
4. repair 里存在 metadata-only rerun，容易被质疑为“重跑脚本”，不是自修复。
5. memory promotion 有代码，但缺少真实 verified success 数据闭环。
6. Docker/GPU/HF/ModelScope 能力目前更偏 plan/mock/local benchmark，不应作为 Agent 纯度依据。
7. 当前缺少 baseline vs LLM agent 对照评估，无法证明“没有 LLM 做不到”。

## 2. 新项目定位

建议新的项目定位：

> 面向未知开源 AI Demo 的 evidence-driven deployment agent，通过 LLM 进行项目理解、实验规划、失败诊断和修复策略选择，并由 Python runtime 负责工具执行、权限控制、状态恢复和证据验证。

不建议定位：

- 生产级 Agent 平台。
- 多 Agent 协作系统。
- 大规模 GPU 调度平台。
- 开源大模型自动部署平台。
- LangChain/Dify/CrewAI 类套壳 Agent。

## 3. 目标架构

### 3.1 从 Pipeline 改为 Agent Loop

当前主流程是固定 pipeline。优化后应引入显式 Agent loop：

```text
Goal
  -> Observe repo/state/logs/evidence
  -> Plan next action
  -> Policy gate
  -> Execute tool
  -> Observe tool result
  -> Update belief/state
  -> Revise plan
  -> Decide continue/repair/escalate/success
  -> Verify evidence
```

### 3.2 新增核心对象

建议新增数据结构：

```text
AgentGoal
AgentObservation
AgentBeliefState
AgentPlan
AgentPlanRevision
AgentAction
AgentToolCall
AgentToolResult
AgentCritique
AgentTerminationReason
```

每一轮 Agent step 至少记录：

```json
{
  "step_id": 1,
  "goal": "...",
  "observation": {},
  "belief_state_before": {},
  "llm_decision": {},
  "policy_result": {},
  "tool_call": {},
  "tool_result": {},
  "belief_state_after": {},
  "next_step": "continue|verify|repair|escalate|stop",
  "termination_reason": ""
}
```

### 3.3 新增产物

每个 run 应生成：

```text
runs/<task-id>/agent_steps.jsonl
runs/<task-id>/agent_state.json
runs/<task-id>/agent_plan.json
runs/<task-id>/agent_plan_revisions.jsonl
runs/<task-id>/reports/agent_contribution.json
```

这些产物是面试中证明“这是 Agent，不是脚本”的核心证据。

## 4. LLM 必要性改造

### 4.1 双轨 Analyzer

保留 deterministic analyzer，但新增 LLM analyzer。两者输出必须分离：

```json
{
  "deterministic_facts": {},
  "deterministic_candidates": [],
  "llm_hypotheses": [],
  "llm_candidates": [],
  "merged_candidates": [],
  "selected_candidate": {},
  "selection_source": "deterministic|llm|hybrid",
  "llm_required_reason": ""
}
```

LLM 应负责以下不确定任务：

- 识别非标准入口文件。
- 从 README 推断真实启动顺序。
- 识别隐式依赖或 README 中未写入 requirements 的依赖。
- 判断 demo 类型：WebUI、API、CLI、OpenAI-compatible server、artifact generator。
- 选择最小可验证启动路径。
- 生成多个 run candidate 并解释置信度。
- 判断 deterministic candidate 是否过于脆弱。

验收标准：

- analyze 阶段必须保留 deterministic 与 LLM 的候选差异。
- report 中必须显示最终选择来源。
- 如果 LLM 没有贡献，必须明确 `llm_required=false`，不能硬凑。

### 4.2 多步 Planning Agent

新增规划阶段，不再只让 LLM 输出一次 action。

LLM 每次必须输出：

```json
{
  "hypothesis": "service failed because missing runtime dependency",
  "confidence": 0.82,
  "next_action": {
    "type": "install_package",
    "payload": {
      "package": "rich"
    }
  },
  "expected_observation": "runner starts and port 8000 becomes ready",
  "fallback_action": {
    "type": "inspect_log",
    "payload": {
      "path": "logs/runner.log"
    }
  },
  "stop_condition": "verify trace response contains current trace_id"
}
```

验收标准：

- 每个 action 都必须绑定 hypothesis。
- 每个 action 都必须绑定 expected observation。
- action 执行后必须检查 observation 是否符合预期。
- 若不符合，必须触发 plan revision。

### 4.3 Tool Registry

将现有 stage 能力包装为 tool，而不是让 Agent 直接操作 stage。

建议新增工具：

```text
inspect_repo_tree
read_selected_files
parse_dependency_files
solve_environment
install_environment
start_service
probe_http
probe_browser_dom
discover_gradio_api
discover_openapi_schema
discover_openai_compatible_model
download_model_asset
inspect_log
classify_failure
propose_repair
apply_repair
resume_from_stage
verify_evidence
```

每个 tool 需要 schema：

```json
{
  "name": "start_service",
  "risk_level": "medium",
  "side_effects": ["process", "network"],
  "requires_policy": true,
  "allowed_modes": ["planner", "gated_actor"],
  "input_schema": {},
  "output_schema": {},
  "success_signal": "process alive and port readiness observed"
}
```

验收标准：

- LLM 只能请求 tool call。
- Python runtime 负责 policy gate 和 tool execution。
- 所有 tool call 都写入 `agent_steps.jsonl`。
- 高风险 tool 必须被 policy 显式允许。

### 4.4 Hypothesis-driven Repair

repair plan 不应只是 action list。必须升级为 hypothesis-driven repair：

```json
{
  "failure_hypothesis": "runner exited because rich is missing",
  "evidence": [
    "runner.log: ModuleNotFoundError: No module named 'rich'"
  ],
  "repair_action": {
    "type": "install_package",
    "payload": {
      "package": "rich"
    }
  },
  "expected_effect": "app.py imports rich successfully and service stays alive",
  "verification_plan": "rerun from env_deploy, then verify HTTP trace",
  "rollback_plan": "discard run workspace",
  "risk": "low"
}
```

repair 成功口径必须统一：

```text
repair proposed
  -> policy accepted
  -> action executed or metadata applied
  -> rerun performed
  -> final verify pass
  -> repair marked effective
```

只有满足以上闭环，才能计入：

```text
repair_verified_success = true
```

metadata-only action 不能计入 executed repair。

### 4.5 Critic / Reflection

新增单 Agent 内部 critic，不包装成多 Agent。

Critic 只做质量控制：

1. 检查 plan 是否违反安全策略。
2. 检查是否过早判成功。
3. 检查是否有更低风险工具可先执行。
4. 检查 repair 是否有 evidence 支撑。
5. 检查 verify 是否真的有当前 trace 证据。

示例输出：

```json
{
  "critique": "The proposed action installs torch nightly, but no evidence requires it.",
  "decision": "reject",
  "safer_alternative": {
    "type": "inspect_log",
    "payload": {
      "path": "logs/runner.log"
    }
  }
}
```

验收标准：

- Critic 不能执行 tool。
- Critic 不能判定部署成功。
- Critic 的 reject/approve 必须写入 trace。

## 5. Agent 贡献度评估

必须新增 baseline vs LLM agent 对照，否则无法证明 LLM 必要性。

### 5.1 三种运行模式

```text
agent_mode=off
```

只跑 deterministic baseline。

```text
agent_mode=planner
```

LLM 只能规划、选择候选、更新 verify hint，不能执行 repair。

```text
agent_mode=gated_actor
```

LLM 可请求低风险 repair action，但必须通过 policy gate。

### 5.2 指标

每个 target 至少统计：

```text
success_rate
verify_pass_rate
false_positive_blocked
repair_attempt_count
repair_executed_count
repair_verified_success_count
manual_intervention_required
avg_iterations_to_success
unsafe_action_rejected_count
time_to_first_success
llm_action_acceptance_rate
llm_action_rejection_rate
baseline_failed_agent_passed_count
```

### 5.3 新增报告

新增：

```text
runs/<eval-id>/comparison_report.json
runs/<eval-id>/comparison_report.md
```

报告示例：

```json
{
  "baseline": {
    "total": 20,
    "verify_pass": 11,
    "failed": 9
  },
  "agent": {
    "total": 20,
    "verify_pass": 16,
    "failed": 4
  },
  "llm_helped_cases": [
    {
      "target_id": "wrong-entrypoint-gradio",
      "help_type": "selected non-default entrypoint",
      "evidence": "agent_steps.jsonl step 3"
    }
  ]
}
```

没有真实数据前，不要在简历里写具体提升百分比。

## 6. Evaluation Target 设计

不要求当前阶段真实联网执行，但需要先设计 target manifest。

建议新增：

```text
eval_targets/manifest.json
eval_targets/fixtures/
```

target 类型：

| 类型 | 数量建议 | 目标 |
|---|---:|---|
| Gradio tiny demo | 3 | 真实启动、trace verify |
| Streamlit demo | 2 | DOM/browser verify |
| FastAPI/Flask API | 2 | OpenAPI verify |
| OpenAI-compatible API | 2 | `/v1/models` + chat verify |
| Hugging Face tiny model | 2 | 下载/cache/etag/sha256 |
| ModelScope tiny model | 1 | 下载/cache |
| 缺依赖故障仓库 | 3 | LLM repair |
| 错误入口/端口仓库 | 3 | LLM planner 修正 |
| CUDA/PyTorch 冲突样例 | 2 | dry-run/env_solve/GPU smoke plan |

最低可交付：

```text
10 个本地 fixture target
agent off vs planner vs gated_actor 对照框架
comparison report 结构完整
```

后续补强：

```text
20 个真实/半真实 target
至少 5 个 LLM helped success case
至少 3 个 unsafe action rejected case
至少 3 个 false-positive blocked case
```

## 7. 安全模型升级

新增文档：

```text
docs/agent-threat-model.md
docs/tool-policy.md
docs/prompt-injection-eval.md
```

必须覆盖：

- Tool risk level。
- Command allowlist。
- Network allowlist。
- Secret redaction。
- Source edit 默认禁止。
- Repair approval gate。
- Prompt injection 标注和拒绝。
- Action replay audit。
- Unsafe action rejection metrics。

安全原则：

1. LLM 不直接执行 shell。
2. LLM 不直接修改源代码。
3. LLM 不直接读取 secret value。
4. LLM 不判定最终 success。
5. 所有 side effect tool 必须经过 policy。
6. 所有 rejected action 必须保留审计记录。

## 8. 代码结构建议

新增目录：

```text
src/auto_harness/agent_runtime/
  runtime.py
  loop.py
  state.py
  schemas.py
  planner.py
  critic.py
  evaluator.py
  contribution.py

src/auto_harness/tools/
  registry.py
  schemas.py
  inspect.py
  environment.py
  service.py
  verify.py
  repair.py
  model_assets.py

src/auto_harness/evals/
  runner.py
  baseline.py
  comparison.py
  metrics.py
```

现有模块建议降级为 tool implementation：

```text
ProjectAnalyzer -> inspect_repo / analyze_project tool
EnvSolveModule -> solve_environment tool
EnvDeployModule -> install_environment tool
RunnerModule -> start_service tool
VerifyModule -> verify_evidence tool
RepairApplier -> apply_repair tool
ModelPrepareModule -> prepare_model_assets tool
```

## 9. 分阶段开发计划

### Phase 1：Agent Runtime 骨架

目标：让项目形态从 pipeline 变成 Agent loop。

任务：

1. 新增 `agent_runtime` package。
2. 定义 AgentGoal、AgentObservation、AgentBeliefState、AgentAction、AgentToolResult schema。
3. 新增 `AgentRuntime.run(goal)`。
4. 将现有 TaskRunner pipeline 包装为默认 tools。
5. 写入 `agent_steps.jsonl`、`agent_state.json`、`agent_plan.json`。

验收：

- dry-run 部署时生成 agent step trace。
- 每个 step 可追踪 observation、action、tool result。
- 不改变现有 CLI 行为。

### Phase 2：LLM Planner 进入主路径

目标：LLM 不再只是 advisor。

任务：

1. analyze 阶段拆分 deterministic facts 与 LLM hypotheses。
2. LLM 必须输出 run candidate 选择理由。
3. verify uncertain 时 LLM 必须输出 next probe 或说明无法推进。
4. failure 时 LLM 必须输出 hypothesis + expected observation。
5. report 生成 `agent_contribution.json`。

验收：

- 每个 LLM action 都有 accepted/rejected 记录。
- report 能显示 LLM 是否影响最终路径。
- 如果 LLM 未帮助，明确显示 `llm_helped=false`。

### Phase 3：Tool Registry 与 Policy Gate

目标：让 Agent 行为单位变成 tool call。

任务：

1. 新增 ToolRegistry。
2. 为现有 stage 封装 tool schema。
3. 每个 tool 标注 risk level、side effects、policy requirements。
4. LLM 只能请求 tool call。
5. policy 统一处理 tool call，而不是散落在各模块。

验收：

- `agent_steps.jsonl` 中出现 tool call。
- 高风险 tool 未授权时被拒绝。
- rejected action 不影响 deterministic pipeline 安全运行。

### Phase 4：Hypothesis-driven Repair

目标：提高自修复可信度。

任务：

1. repair plan schema 增加 hypothesis、evidence、expected_effect、verification_plan、rollback_plan。
2. 区分 `repair_applied`、`repair_executed`、`repair_effective`、`repair_verified`。
3. metadata-only action 不计入 executed repair。
4. verified memory 只接收 final verify pass 的 repair。
5. 修复 live smoke manifest 的 repair 计数口径。

验收：

- repair 成功必须绑定 final verify pass。
- report 中展示 repair effectiveness。
- memory 中不写入未验证成功的 skill promotion 候选。

### Phase 5：Baseline vs Agent Evaluation

目标：证明 LLM 必要性。

任务：

1. 新增 eval manifest。
2. 新增 baseline runner。
3. 新增 planner/gated_actor runner。
4. 新增 comparison report。
5. 统计 LLM helped cases。

验收：

- 可在本地 fixtures 上跑出 comparison report。
- 每个 target 有 baseline 与 agent 的 status 对比。
- 能定位 LLM 帮助成功的具体 step。

### Phase 6：安全与面试证据补强

目标：提高大厂面试抗压能力。

任务：

1. 新增 threat model 文档。
2. 新增 tool policy 文档。
3. 增加 prompt injection benchmark。
4. 增加 unsafe action rejection benchmark。
5. 增加 secret redaction benchmark。

验收：

- 面试时能回答“如何防止 Agent 执行恶意项目代码”。
- 能展示 rejected action trace。
- 能展示 secret 没有进入 prompt 和 report。

## 10. 不做事项

本阶段不要做：

- 不做生产级多租户。
- 不做真正分布式调度。
- 不做 Kubernetes。
- 不做复杂 GPU 调度。
- 不做真实大模型长耗时部署。
- 不做多 Agent 包装。
- 不做让 LLM 直接执行 shell。
- 不做让 LLM 修改源码。
- 不做为了通过 verify 而降低验证标准。

## 11. 当前执行状态（2026-07-08）

本轮已按本方案完成本地可验证的受控探索型 Agent 升级，不执行真实联网、真实 GPU、真实 Docker 或真实大模型测试。

已完成：

- Phase 1 Agent Runtime 骨架：
  - 新增 `src/auto_harness/agent_runtime/`。
  - 每个 run 生成 `agent_steps.jsonl`、`agent_state.json`、`agent_plan.json`、`agent_plan_revisions.jsonl`。
  - `AgentRuntime` 将现有 pipeline stage 映射为 tool call，并记录 observation、belief state、tool result、policy result 和 critique。
- Phase 2 LLM Planner 主路径证据：
  - analyze 输出拆分为 `deterministic_facts`、`deterministic_candidates`、`llm_hypotheses`、`llm_candidates`、`merged_candidates`、`selected_candidate`、`selection_source`、`llm_required_reason`。
  - 新增 `reports/agent_contribution.json`，明确 `llm_required` / `llm_helped` / `help_type`，LLM 未贡献时不硬凑。
- Phase 3 Tool Registry 与 Policy Gate：
  - 新增 `src/auto_harness/tools/`。
  - tool schema 包含 `risk_level`、`side_effects`、`requires_policy`、`allowed_modes`、`success_signal`。
  - side-effect tool 只能由 Python runtime 根据 policy 执行。
- Phase 4 Hypothesis-driven Repair：
  - repair plan 增加 `failure_hypothesis`、`evidence`、`expected_effect`、`verification_plan`、`rollback_plan`、`risk`、`repair_effectiveness_criteria`。
  - repair apply 区分 `repair_applied`、`repair_executed`、`repair_effective`、`repair_verified`。
  - metadata-only action 不计入 executed repair。
- Phase 5 Baseline vs Agent Evaluation：
  - 新增 `src/auto_harness/evals/`。
  - 新增 CLI `eval-compare`。
  - 新增 `eval_targets/manifest.json`，包含 10 个本地 target。
  - 可生成 `comparison_report.json` / `comparison_report.md`。
- Phase 6 安全与面试证据：
  - 新增 `docs/agent-threat-model.md`。
  - 新增 `docs/tool-policy.md`。
  - 新增 `docs/prompt-injection-eval.md`。
  - Benchmark manifest 新增 `agent_runtime_artifacts`、`tool_registry_policy_gate`、`agent_comparison_report`。

静态验收建议：

```bash
python3 -m py_compile \
  src/auto_harness/agent_runtime/*.py \
  src/auto_harness/tools/*.py \
  src/auto_harness/evals/*.py \
  src/auto_harness/orchestrator.py \
  src/auto_harness/modules/analyzer.py \
  src/auto_harness/repair/planner.py \
  src/auto_harness/repair/apply.py \
  src/auto_harness/memory/success.py \
  src/auto_harness/cli.py
```

可选本地 benchmark 子集：

```bash
PYTHONPATH=src python3 -m auto_harness.cli benchmark \
  --case-id agent_runtime_artifacts \
  --case-id tool_registry_policy_gate \
  --case-id agent_comparison_report
```

## 11. 简历口径

优化前不建议写：

- 生产级 Agent 平台。
- 开源模型自动部署 Agent。
- 长期记忆和技能进化已落地。
- Docker/GPU 隔离执行已验证。
- 大规模并发部署。

优化后可写：

> 构建面向未知 AI Demo 的 evidence-driven deployment agent，将 LLM 用于项目理解、实验规划、失败诊断和修复策略选择；Python runtime 负责 tool execution、policy gate、state resume 和 trace-based verification。

有 baseline vs agent 数据后可写：

> 构建 deterministic baseline 与 LLM agent 对照评估，统计 verify pass rate、repair verified success、unsafe action rejection 和 false-positive block，并通过 agent step trace 定位 LLM 贡献。

## 12. 最优先的 5 个开发任务

1. 修复现有 unittest 失败，恢复测试可信度。
2. 新增 `agent_steps.jsonl` 和 `agent_state.json`。
3. 新增 ToolRegistry，将现有 stage 包装为 tools。
4. 新增 `agent_contribution.json`，记录 LLM 是否真正影响结果。
5. 新增 baseline vs agent comparison report。

## 13. 最小可交付版本

MVP 不要求真实联网测试，但必须具备：

```text
AgentRuntime
ToolRegistry
agent_steps.jsonl
agent_state.json
agent_contribution.json
baseline vs agent comparison report
hypothesis-driven repair schema
统一 repair success 口径
```

MVP 面试表述：

> 我把原先固定部署 pipeline 重构为 goal-driven Agent loop。LLM 不直接执行命令，而是基于 observation 生成 hypothesis、选择 tool action、预测 expected observation；Python runtime 负责 policy gate 和执行，最终由 trace-based verify 裁决成功。为了证明 LLM 必要性，我设计了 baseline vs agent 对照评估，并记录每个 LLM action 对最终结果的贡献。
