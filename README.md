# auto-deploy-harness

auto-deploy-harness 是一个基于 LangGraph 编排、由 LLM 负责不确定性决策的自动部署与验证 Agent。

**默认采用 LangGraph Plan-first 编排：** LLM 负责部署规划、候选选择、失败诊断与差异化 Replan；所有模型输出经 Schema 和 Policy Gate 裁决为受控动作，由确定性 stage executor 执行，最终成功仅由 Evidence Gate 裁决。Legacy deterministic pipeline 作为显式 baseline 和无 LLM 降级路径保留。

**恢复能力：** 结合 SQLite Checkpoint、Operation Journal 和资源 Reconciler，对下载、依赖环境、服务进程及 Docker 容器执行恢复前状态对账，避免 checkpoint resume 导致重复副作用。

核心架构原则：

- LLM 不直接执行 shell、Docker、文件修改、模型下载或进程启动。
- 所有 LLM 输出经 parser → schema → policy → compiler → typed executor 管道。
- VerifyModule Evidence Gate 是唯一部署成功裁判。
- Checkpoint 只描述图状态；外部副作用恢复必须经过 Operation Journal 和 Reconciler。
- 禁止副作用发生后自动从 LangGraph 切到 legacy。

项目发布名和主命令为 `auto-deploy-harness`。为避免破坏历史 import 和已有脚本，Python 包名继续保留为 `auto_harness`，旧命令 `auto-harness` 也继续兼容。

## 当前 MVP

当前仓库已经包含：

- CLI：`init`、`deploy`、`resume`、`status`、`report`、`package`、`dashboard`、`queue`、`readiness`、`llm-test`、`benchmark`、`eval-compare`、`live-smoke-plan`、`agent-live-smoke`、`docker-smoke`、`repair-approve`、`memory-evolve`；`memory-promote` 仅保留为旧 proposal 只读兼容入口。
- 任务状态存储：`task.json`、`state.json`、`events.jsonl`。
- 确定性项目分析器。
- 安全默认的 `env_solve`、`env_deploy`、`runner`、`verify`、report 模块。
- Mock LLM provider 和讯飞 Anthropic-compatible provider。
- Claude Code executor wrapper。
- Policy-constrained LLM Agent：`agent_mode=planner` 时 LLM 可通过 schema 化 action 影响 analyze plan；`agent_mode=gated_actor` 且显式开关打开时，LLM repair action 可在 policy gate 后执行安全动作；LLM 永远不能直接执行 shell、修改源码或判定成功。
- Provider protocol：当前 Mock 与讯飞 Provider 使用 `json_action`（模型输出 Schema 化 JSON，再由 Python 校验和执行），不是 provider-native function/tool calling；运行证据会显式记录 `provider_protocol`，避免把 JSON Action 包装成原生 Tool Calling。
- Self-healing 主流程：`deploy/resume --agent-self-heal` 会打开 bounded repair/resume loop，普通 `TaskRunner.run_existing()` 能在 policy-approved repair 后自动从 `env_deploy` / `model_prepare` / `runner` / `verify` 安全阶段恢复，最终仍由 verify trace 判定成功。
- Agent runtime 证据：每个 run 会生成 `agent_steps.jsonl`、`agent_state.json`、`agent_plan.json`、`agent_plan_revisions.jsonl` 和 `reports/agent_contribution.json`，记录 observe / plan / policy gate / tool call / observe / critique 的受控探索过程。
- Tool registry：LLM 只能请求命名 tool，Python runtime 根据 `risk_level`、`side_effects`、`requires_policy` 和 `allowed_modes` 进行执行控制；side-effect tool 默认需要 policy gate。
- HTTP trace evidence：`verify` 支持 GET 和 POST JSON；响应必须证明当前 trace 被处理，HTTP 200 本身不算成功；文件产物证据必须可读、非空并记录 size/sha256。
- 可选 Claude Code analyzer advisor：通过 `AUTO_HARNESS_USE_AGENT_ANALYZER=1` 启用。
- 仓库内置 skill：位于 `skills/*/SKILL.md`，使用统一 schema（name/version/type/stages/risk_level/allowed_tools 等），按阶段选择并记录 hash。
- Skill-driven Agent System：`SkillSchemaParser` 解析校验 skill frontmatter；`SkillRouter` 根据 stage/framework/failure_category/history 选择最相关 skills；`SkillContextBuilder` 将 skill 压缩为 LLM 可用上下文；Plan-first / Replan / Verify / Repair 均接入 skill context；`SkillEffectRecorder` 记录 skill 对 plan 的影响；`SkillOutcomeRecorder` 评估 skill 是否带来 trace-verified success；`SkillMetricsReporter` 汇总 selection/influence/pass/harm 指标；report 展示 Skill Usage / Effects / Outcomes。
- 结构化问题记忆：位于 `memory/deployment_issues.jsonl`，用于检索历史相似失败。
- Memory evolution：`memory-evolve` 将 verified memory 生成 Skill candidate，并强制经过 `proposed -> approved -> regression_passed -> shadow_passed -> active -> rolled_back/rejected` 生命周期；每次迁移写入哈希链审计。旧 `memory-promote --apply` 已禁用，避免绕过主状态机修改 Skill。
- `resource_plan` 阶段：识别模型资产、GPU/CUDA 信号、磁盘风险和外部 token 需求。
- `env_solve` 阶段：在安装前生成更稳定的依赖方案，识别老 Gradio 与 `numpy<2` / `pydantic<2` 兼容风险、headless OpenCV 替换建议，解析 `environment.yml` 并选择 `venv` / `conda` / `mamba` backend；根据本机 CUDA/Python 生成 PyTorch pip wheel 或 conda `pytorch-cuda` 方案、fallback 和 `xformers` / `flash-attn` / `bitsandbytes` / `triton` / `deepspeed` / `accelerate` GPU 包兼容矩阵。
- Verified long-term memory：自修复后只有最终 verify pass、repair policy 通过、repair action 有效且存在 trace id 时，才会写入 `memory_type=verified_success` 的长期记忆；report 会展示 memory id、repair action hash 和 verification trace id。
- Docker backend：`env_deploy` 和 `runner` 支持把安装/启动命令包装为 `docker run` 计划，包含 GPU 参数、模型缓存挂载、容器日志命令和清理命令元数据；默认仍是本地 backend，真实执行仍受 `--execute` 和命令白名单保护。
- Git LFS / submodule 检测：识别 `.gitattributes`、LFS pointer 文件和 `.gitmodules`，估算 pointer size，缺工具时给出诊断，并生成 `git lfs` / `git submodule` 准备命令。
- Git LFS / submodule 受控准备：`model_prepare` 在 `--execute` 且 `git` 通过命令白名单时执行 `git lfs install/pull` 和 `git submodule sync/update`，并记录命令结果、stderr/stdout tail、进度和最终状态。
- `model_prepare` 阶段：生成模型资产 manifest、缓存 key 和 `model_cache` 路径；执行模式下支持 Hugging Face / ModelScope 文件清单解析、断点续传、并发下载、sha256/etag 校验元数据和缓存写入。
- `verify` 增强：支持 Gradio `/config` discovery、Gradio queue `/call/<api_name>` follow-up、FastAPI/Flask `/openapi.json` schema discovery、vLLM/OpenAI-compatible `/v1/models` + `/v1/chat/completions`、streaming SSE trace、Streamlit DOM/HTML 证据探测、可选 Playwright 浏览器 DOM probe，以及长耗时首次推理验证过程中的状态刷新。
- 日志规则分类器：对缺依赖、CUDA OOM、磁盘不足、token 权限、wheel 构建失败、numpy/pydantic/protobuf 冲突等常见错误生成结构化诊断；诊断会包含 package、constraint、recommended action 和 rerun stage，token 权限问题只提取所需环境变量名，不记录密钥值。
- Report 会汇总 `resource_plan`、diagnosis 和 repair plan 中的 token 变量名，提示 operator/secret manager 注入，报告中不保存任何 token value。
- Repair loop：失败或 uncertain 阶段会生成结构化修复建议，经过 policy 和 loop gate 校验后写入受控 repair artifacts；同一问题有最大尝试次数，不安全的 `rerun_from` 会回退到安全阶段，`resume` 会按 `rerun_from_effective` 从安全阶段重跑，需要人工确认的 action 可通过 `repair-approve` 批准。
- Resume execution audit：当 repair resume 从中间阶段恢复时，会生成 `reports/execution_audit.json`，并在报告中展示复用阶段、重跑阶段和 fallback 信息。
- Dashboard：`dashboard` 命令会从本地 `runs/`、任务状态和可选 benchmark report 生成静态 HTML/JSON；`dashboard --serve` 可启动只读本地 HTTP 服务，暴露 HTML、JSON 和 health check。
- Persistent queue：`queue submit/list/run` 提供本地持久化任务队列，入队与执行分离，前台 worker 显式消费任务；`queue run --max-jobs N` 会用线程池并发运行多个队列项，并通过原子 claim lock 防止多 worker 重复执行同一个 job，过期 lock 会按 TTL 回收。
- Deployment package：`package --task-id` 会导出 `tar.gz` 审计产物包和 sidecar manifest，包含 task/state/events/reports/evidence/repairs，默认排除 workspace、模型缓存和日志。
- Readiness audit：`readiness` 命令会生成机器可读完成度审计，区分本地已完成能力和真实联网/GPU/Docker/vLLM 外部验收门，不保存任何密钥值。
- Agent 设计与评估文档：`docs/agent-architecture.md`、`docs/agent-safety-model.md`、`docs/agent-evaluation-report.md`。
- Agent runtime 安全文档：`docs/agent-threat-model.md`、`docs/tool-policy.md`、`docs/prompt-injection-eval.md`。
- 真实 Agent smoke 证据：`docs/evidence/live-agent-smoke-manifest.json` 记录了无密钥 Xunfei repair-mode live smoke manifest。
- Benchmark fixtures：`tests/fixtures/benchmarks` 覆盖下载续传、缓存命中、并发下载、etag 缓存失效、缓存清理、按来源/repo/keep-list 清理、服务启动后退出、历史 artifact 干扰、Gradio API shape 变化、token 缺失、token report 提示、Gradio `/config` discovery、OpenAPI schema discovery、OpenAI-compatible model discovery/stream verify、浏览器 DOM trace、Streamlit 错误页面、HTTP 200 false-positive 防护、repair policy 拒绝、repair loop 限流、repair resume 阶段跳转、人工审批、checksum 失败、本地 E2E fixture matrix、memory promotion proposal/审批/回归绑定、LLM planner policy merge、LLM repair execute loop、LLM verify hint recovery、静态 dashboard 导出、只读 HTTP dashboard、持久化任务队列 dry-run 调度、队列并发 worker pool、队列 claim lock、stale claim lock recovery、GPU 探测调度、部署产物包导出、readiness audit、Docker backend plan/GPU/cache/log 元数据、GPU 包矩阵、verify progress、Git LFS progress parse、主流程 self-healing、conda/PyTorch backend、verified memory、skill evolution、agent runtime artifacts、tool registry policy 和 baseline vs agent comparison report。
- 本地 E2E trace smoke：`tests/test_deployment_e2e.py` 会对 `tests/fixtures/e2e/http_trace_echo` 真实执行 `TaskRunner.deploy(... execute=True ...)`，完成本地 repo 复制、venv 安装、服务启动、HTTP trace verify、evidence/report/events 落盘和进程清理，证明部署闭环不是只停留在 dry-run 或 HTTP 200 健康检查。
- 开发进度报告：`docs/progress.md`。

## 快速开始

```bash
python3 -m auto_harness.cli init
python3 -m auto_harness.cli deploy --repo https://github.com/example/demo --name demo --dry-run
python3 -m auto_harness.cli status --task-id <task-id>
python3 -m auto_harness.cli report --task-id <task-id>
```

本地源码运行：

```bash
PYTHONPATH=src python3 -m auto_harness.cli init
```

如果要真正执行依赖安装和服务启动，需要显式打开执行开关：

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo ./demo --name demo --execute --allow-install --allow-start
```

`--repo` 也可以传入本地项目路径，系统会复制到隔离的 run workspace。

开启自修复和自动环境 backend：

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy \
  --repo ./demo \
  --name demo \
  --execute \
  --allow-install \
  --allow-start \
  --agent-self-heal \
  --env-backend auto
```

生成 baseline vs agent 对照报告结构：

```bash
PYTHONPATH=src python3 -m auto_harness.cli eval-compare \
  --manifest eval_targets/manifest.json \
  --output-dir runs/evals/local-fixture-eval
```

## 讯飞配置

密钥只能通过环境变量注入：

```bash
export XUNFEI_APP_ID="..."
export XUNFEI_API_KEY="..."
export XUNFEI_API_SECRET="..."
export XUNFEI_MODEL="..."
export XUNFEI_API_BASE="..."
# 或者 export XUNFEI_API_URL="..."
```

不要提交真实 `.env` 文件。密钥应放在本地 shell、secret manager 或 CI secret settings 中。

当前讯飞 provider 默认使用 Anthropic-compatible messages payload。如果 Claude Code 无法直接使用讯飞，可以走本项目的本地 LLM provider 路径，把 Claude Code 作为可选 executor。

Provider smoke test：

```bash
PYTHONPATH=src python3 -m auto_harness.cli llm-test --provider xunfei --prompt "Return JSON only: {\"status\":\"ok\"}"
```

## 可选 Claude Code Analyzer Advisor

确定性 analyzer 默认开启。若要让 Claude Code 在 analyze 阶段提供可选 JSON 建议：

```bash
export AUTO_HARNESS_USE_AGENT_ANALYZER=1
export CLAUDE_CODE_CMD="claude --print --output-format json"
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo ./demo --name demo --dry-run
```

Agent advice 会写入 `analyze_result.json` 作为可选元数据，但不能绕过确定性 pipeline。

## LLM Agent Planner

LLM Agent 默认关闭，不影响 deterministic pipeline：

```json
{
  "agent_mode": "off"
}
```

开启结构化 planner：

```bash
export AUTO_HARNESS_AGENT_MODE=planner
export AUTO_HARNESS_AGENT_PROVIDER=mock
export AUTO_HARNESS_ENABLE_ANALYZE_PLANNER=1
export AUTO_HARNESS_ENABLE_VERIFY_PLANNER=1
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo ./demo --name demo --dry-run
```

`planner` 模式下，LLM 只能输出 JSON decision/action。允许合并的动作包括 `add_run_candidate`、`select_run_candidate`、`update_verify_hint` 和 `add_dependency_constraint`；所有 action 都会经过 schema parsing、`AgentActionPolicy` 和 trace 写入，trace 会记录 policy accepted / rejected 结果。决策 trace 写入：

```text
runs/<task-id>/logs/agent_calls/
```

受控 repair action 需要更高权限：

```bash
export AUTO_HARNESS_AGENT_MODE=gated_actor
export AUTO_HARNESS_ENABLE_LOG_DIAGNOSIS=1
export AUTO_HARNESS_ENABLE_REPAIR_ACTIONS=1
```

即使在 `gated_actor` 下，LLM 也不能直接执行 shell 或修改源码。`install_package` 必须满足包名安全正则、runtime policy 允许依赖安装、repair loop 未超限、命令在 `allowed_commands` 白名单内，并且最终仍必须由 verify evidence 证明修复有效。LLM verify planner 的二次 HTTP evidence 会单独保存，不覆盖初始 uncertain evidence。

`AgentLoopController` 会把失败观察、LLM diagnosis、repair plan、policy gate、repair apply、stop reason 和 auto-resume 判定写入 stage result 与 `runs/<task-id>/logs/agent_loop/`。自动恢复默认关闭；只有显式设置 `agent_auto_resume_after_repair=true`，且 policy、runtime 和 repair action 都通过时，才会产生 `should_auto_resume=true` 的恢复建议。

run candidate 会记录统一 ranking 信息：`score`、`score_reasons` 和 `selected_by`。LLM planner 只能添加候选或提升已有候选排序，不能删除 deterministic candidate；最终选择原因会进入 runner result 和 report。

verify planner 可以输出多个 `verify_candidates`。Python 会过滤 external URL、缺少 `{{trace_id}}` 或包含 token 的请求，并最多尝试前三个合法候选；每个候选都会写独立 evidence，成功仍必须由当前 trace id 证明。

repair 阶段会区分 LLM 提议和 Python 裁决的 rerun stage：`rerun_from_proposed`、`rerun_from_required`、`rerun_from_effective` 都会进入 repair plan/report；过晚或非法的 LLM 提议会降级到安全阶段。

发送给 LLM 的 selected files 会先经过 `AgentInputSanitizer`：secret value 会替换为 `[REDACTED_SECRET]`，prompt injection 风险会写入 observation metadata，trace 写入前也会脱敏。恶意 README 诱导出的 shell/network run candidate 会被 policy 拒绝。

每次 run 会生成 `reports/agent_metrics.json`，记录 LLM 调用数、accepted/rejected action、执行动作数、repair attempts、verify candidates、final status 和 `help_type`。可用 `agent-metrics --runs-dir runs` 汇总本地所有任务。

## LLM-driven Agent Mode

auto-deploy-harness 支持 gated LLM-driven deployment mode。LLM 不直接执行 shell 命令，只能提出 typed tool call，如选择 runner candidate、选择 verify probe 或提出 repair。Python 通过 schema、policy、command allowlist 和 evidence gate 校验每个 action。

### 运行 Primary Loop E2E

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy \
  --repo tests/fixtures/e2e/http_trace_echo \
  --name agent-primary-e2e \
  --execute \
  --allow-install \
  --allow-start \
  --agent-runtime-loop \
  --agent-runtime-loop-position primary
```

### 查看 Agent Artifacts

```text
runs/<task-id>/agent_steps.jsonl          # 逐步执行日志
runs/<task-id>/agent_state.json           # Agent 状态快照
runs/<task-id>/agent_plan.json            # 部署计划
runs/<task-id>/reports/agent_loop_result.json  # Agent 循环结果
runs/<task-id>/reports/pipeline_results.json   # 阶段结果
runs/<task-id>/reports/report.md          # 部署报告
runs/<task-id>/evidence/*trace*.json      # HTTP trace 证据
```

### 安全边界

- LLM 不能直接执行 shell
- LLM 只能请求 typed tool
- side-effect tool 必须经过 policy gate
- 命令必须经过 allowlist
- verify 成功只能由 Python evidence gate 判定
- HTTP 200、端口开放、进程存活都不能单独判定成功
- 当前 trace 被服务处理，才算强证据

## Skill-driven Agent System

Skill 是阶段化、版本化、可审计、可评估、可回归的部署经验单元。LLM 可以读取 skill 并据此提出 plan/tool call，但执行权和成功判定仍由框架控制。

核心流程：

```text
项目观察
  -> SkillRouter 选择阶段相关 skills
  -> SkillContextBuilder 压缩成 LLM 可用上下文
  -> LLM plan/replan 参考 skill context 生成部署方案
  -> PlanPolicyGate 校验方案
  -> PlanCompiler 编译成 effective plan
  -> 分阶段执行
  -> evidence verify
  -> SkillEffectRecorder 记录 skill 影响
  -> SkillOutcomeRecorder 评估 skill 是否真的帮助
  -> verified memory 推动 skill evolution / rollback
```

### Skill Schema

每个 `skills/<skill-name>/SKILL.md` 使用统一 frontmatter：

```yaml
---
name: deploy-python-webui
version: 1.0.0
type: execution_skill
stages:
  - runner
  - plan_first
  - replan
frameworks:
  - gradio
  - streamlit
  - fastapi
risk_level: low
side_effects: false
allowed_tools:
  - add_runner_candidate
  - select_runner_candidate
  - set_stage_hint
success_signals:
  - runner process alive
  - verify trace response pass
regression_cases:
  - gradio_tiny_local
  - streamlit_tiny_local
---
```

必填字段：`name`、`version`、`type`、`stages`、`risk_level`、`side_effects`、`allowed_tools`、`success_signals`、`regression_cases`。

可选字段：`frameworks`、`failure_categories`、`model_sources`、`env_backends`、`owners`、`deprecated`、`replacement`。

Skill 类型：

| 类型 | 用途 |
|---|---|
| `analysis_skill` | 识别项目结构、入口、框架、README 线索 |
| `execution_skill` | 指导安装、环境、启动策略 |
| `verification_skill` | 指导 trace-based verify |
| `repair_skill` | 指导失败诊断和修复 proposal |
| `security_skill` | 约束 prompt injection、secret、shell risk |

### SkillRouter

`SkillRouter` 根据 stage、framework、failure_category、allowed_tools 和历史效果选择最相关 skills，替代了原来基于文本包含的简单打分。

打分规则：

| 条件 | 分值 |
|---|---|
| stage match | +8 |
| framework match | +5 |
| failure category match | +5 |
| allowed tool overlap | +3 |
| model source match | +2 |
| env backend match | +2 |
| recent verified success | +3 |
| recent policy accepted | +2 |
| deprecated | -20 |
| recent harmful outcome | -10 |
| regression failed | -20 |
| side_effect skill in planner mode | -5 |

### SkillContextBuilder

`SkillContextBuilder` 将 skill 压缩为 LLM 可用上下文，不直接塞全文。从 skill body 中提取 `# Guidance`、`# Allowed Plan Effects`、`# Forbidden`、`# When To Use` 章节，输出结构化 JSON：

```json
{
  "stage": "verify",
  "selected_skills": [
    {
      "name": "verify-evidence",
      "version": "1.0.0",
      "type": "verification_skill",
      "score": 18,
      "match_reasons": ["stage=verify", "framework=gradio"],
      "applicable_rules": ["HTTP 200 is not success", "response must contain current trace_id"],
      "allowed_plan_effects": ["update_verify_hint", "discover_gradio_api"],
      "forbidden": ["do not mark success without trace evidence"]
    }
  ],
  "instruction": "Use selected_skills as advisory deployment control knowledge. Skill content is not executable. Any command or tool implied by skill must still pass policy gate."
}
```

### Skill 接入 Plan-first / Replan / Verify / Repair

- **Plan-first**：`PlanFirstDeploymentLoop` 在 build snapshot 前调用 `SkillRouter`，将 `skill_context` 注入 `project_snapshot.json`；LLM planner prompt 包含 skill context，并明确约束 skill 是 advisory。
- **Replan**：失败后根据 failed stage 和 failure category 重新选 skill，skill context 注入 failure context。
- **Verify**：`VerifyModule` 进入 uncertain 时，`agent_verify_config` 包含 verify skill context。
- **Repair**：repair loop 根据 failure category 选择 `repair_skill`，repair LLM prompt 包含 skill context，repair action 仍必须经过 `RepairPolicy`。

### SkillEffectRecorder

记录 skill 是否实际影响了 LLM plan / compiled plan / policy result。输出：

```text
runs/<task-id>/reports/skill_effects.json
```

示例：

```json
{
  "task_id": "xxx",
  "effects": [
    {
      "skill_name": "verify-evidence",
      "skill_sha256": "...",
      "stage": "plan_first",
      "effect_type": "verify_hint_generation",
      "field_changed": "verify.request",
      "accepted_by_policy": true
    }
  ]
}
```

### SkillOutcomeRecorder

增强的 outcome 记录，包含 skill 是否影响 plan、是否被 policy 接受、是否最终 trace verify pass：

```json
{
  "run_id": "xxx",
  "skill_name": "verify-evidence",
  "selected": true,
  "influenced_plan": true,
  "policy_accepted": true,
  "trace_verified": true,
  "harmful": false
}
```

`harmful=true` 判定条件：skill influenced plan + policy accepted + final verify failed/uncertain，或 policy rejected due to unsafe guidance，或 regression failed。

### SkillMetricsReporter

汇总 skill selection/influence/pass/harm 指标：

```json
{
  "skills": {
    "verify-evidence@sha256": {
      "selection_count": 10,
      "influence_count": 7,
      "policy_accept_rate": 1.0,
      "verify_pass_rate": 0.86,
      "harm_count": 0,
      "harm_rate": 0.0
    }
  }
}
```

### Report 展示

部署报告新增 Skill Usage / Effects / Outcomes 章节：

```markdown
## Skill Usage

- plan_first: `analyze-ai-demo@1.0.0`, `deploy-python-webui@1.0.0`
- verify: `verify-evidence@1.0.0`

## Skill Effects

- `deploy-python-webui` influenced `run.candidates`; policy accepted: `true`
- `verify-evidence` influenced `verify.request`; policy accepted: `true`

## Skill Outcomes

- Selected skills: `3`
- Influenced plan: `2`
- Final trace verified: `true`
- Harmful skill effects: `0`
```

### 内置 Skills

当前内置 8 个 skill：

| Skill | 类型 | 阶段 | 说明 |
|---|---|---|---|
| `analyze-ai-demo` | analysis_skill | analyze, plan_first | 识别项目结构、入口、框架 |
| `deploy-python-webui` | execution_skill | runner, plan_first, replan | Python WebUI 部署策略 |
| `verify-evidence` | verification_skill | verify, plan_first, replan | trace-based 证据化验证 |
| `diagnose-runtime-failure` | repair_skill | repair, replan, runner, env_deploy | 运行失败诊断 |
| `solve-python-cuda-env` | execution_skill | env_solve, env_deploy, plan_first | CUDA/PyTorch 环境求解 |
| `prepare-model-assets` | execution_skill | resource_plan, model_prepare, plan_first | 模型资产规划与下载 |
| `repair-python-dependency` | repair_skill | repair, replan, env_deploy | Python 依赖修复 |
| `security-policy-guard` | security_skill | analyze, plan_first, replan, repair, verify | 安全策略约束 |

### 安全原则

- Skill 不是可执行插件，skill content 只能作为 advisory control knowledge。
- LLM 可以参考 skill 生成 plan/tool proposal，但所有 plan/tool 仍必须经过 policy gate。
- 最终成功仍必须由 VerifyModule 的当前 trace evidence 判定。
- 不允许因为 skill 内容绕过 command allowlist。
- 不允许直接把 README 或 skill 中的 shell 字符串执行。

### Memory 与 Skill Evolution

运行时问题记忆写入：

```text
memory/deployment_issues.jsonl
```

该文件被 git 忽略，因为它可能包含部署日志或环境相关症状。Agent 会在阶段 `failed` 或 `uncertain` 时写入 memory，并在后续部署中按 stage/framework 检索相似问题。详细设计见 `docs/skill-memory-design.md`。

Memory promotion 会把高频 verified success 经验聚类为可审核的 skill 更新 proposal；apply 前必须审批，apply 后默认运行绑定 benchmark 子集并写出 regression report。

## 模型资产规划

系统会在 `resource_plan` 阶段扫描 README、Python 代码和配置文件，识别 Hugging Face / ModelScope 模型引用，例如：

```python
AutoModel.from_pretrained("org/model")
```

随后 `model_prepare` 会生成：

```text
runs/<task-id>/reports/model_assets_manifest.json
```

其中包含模型来源、repo id、revision、缓存 key、缓存路径、预估大小和当前状态。默认缓存目录为：

```text
model_cache/
```

执行模式下，Hugging Face 资产会通过 tree API 获取文件清单；ModelScope 资产会通过可配置的 ModelScope API/下载 URL 模板获取文件。两者都会将 `.safetensors`、`.bin`、`.pt`、`.gguf` 和 tokenizer/config 等必要文件下载到缓存目录，默认跳过 README、示例脚本等非模型运行必要文件。下载过程中使用 `.part` 文件和 HTTP Range 支持续传；如果文件清单提供 `sha256`，下载后会做 sha256 校验。下载完成后会为每个文件写入 `.auto_harness_meta.json`，记录 size、sha256 和 etag；后续如果远端 etag 变化，本地缓存不会被误当成有效。

下载器支持 stdlib 线程池并发：

```python
HuggingFaceDownloader(max_workers=4)
ModelScopeDownloader(max_workers=4)
```

生产部署时可以通过 `configs/default.json` 控制下载并发、失败重试和缓存清理阈值：

```json
{
  "model_download_max_workers": 2,
  "model_download_retry_count": 2,
  "model_download_retry_backoff_seconds": 1.0,
  "model_cache_cleanup_max_total_bytes": 500000000000,
  "model_cache_cleanup_older_than_days": 30,
  "model_cache_cleanup_source": "huggingface",
  "model_cache_cleanup_repo_id": null,
  "model_cache_cleanup_keep_repo_ids": ["org/critical-model"]
}
```

`deploy` / `resume` 也支持临时覆盖下载参数：

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo <repo> --execute --model-download-workers 2 --download-retries 3 --download-retry-backoff 2
PYTHONPATH=src python3 -m auto_harness.cli resume --task-id <task-id> --execute --model-download-workers 2
```

执行 backend 默认是 `local`。如果需要把依赖安装和服务启动包装进 Docker，可通过配置或 CLI 指定：

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo <repo> --execute --allow-install --allow-start --execution-backend docker --docker-image python:3.11-slim --docker-gpus all
```

Docker backend 会生成 `docker run --rm -v <repo>:/workspace/repo -w /workspace/repo ...` 形式的 effective command。可通过 `--docker-gpus all` 添加 GPU 参数，通过 `--docker-model-cache-dir <path>` 挂载模型缓存到容器内 `/workspace/model_cache`。runner 阶段还会记录容器 `log_command` 和 `cleanup_command` 元数据，便于失败后审计和清理。真正执行仍必须让 `docker` 进入 `allowed_commands`，否则会被 policy 拒绝。

Docker/GPU runtime 可先做 smoke 计划或本机探测：

```bash
PYTHONPATH=src python3 -m auto_harness.cli docker-smoke
PYTHONPATH=src python3 -m auto_harness.cli docker-smoke --probe --image python:3.11-slim
PYTHONPATH=src python3 -m auto_harness.cli docker-smoke --probe --require-gpu
```

默认不执行 Docker 命令；只有 `--probe` 才会运行 `docker version/info/run`。GPU 检查需要 NVIDIA container runtime。

缓存清理由 `ModelCache.cleanup(...)` 提供，默认 `dry_run=True`，会先返回候选列表、候选大小和预计删除项；只有显式传入 `dry_run=False` 才会删除缓存目录。

CLI 缓存清理同样默认 dry-run：

```bash
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup --max-total-bytes 500000000000
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup --max-total-bytes 500000000000 --apply
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup --source huggingface --repo-id org/demo --keep-repo-id org/critical-model
```

每个缓存目录会写入 `.auto_harness_asset.json`，记录 source、repo id、revision、origin 和 cache key。清理计划支持 `--source`、`--repo-id`、`--keep-cache-key`、`--keep-repo-id`，用于只清理某个模型源或某个 repo，并保护关键模型。

阶段进度会写入 `state.json` 的 `model_prepare.progress`，包括当前文件、已下载字节、总字节和状态。`verify` 阶段也会写入长耗时状态，例如 `service_discovered`、`first_inference_probe_started`、`http_trace_request_sent`、`browser_probe_completed` 和 `verify_completed`，避免首次加载模型时看不到进展。

如果项目使用 Git LFS 保存权重，`resource_plan.git_lfs` 会记录：

- `.gitattributes` 中的 LFS patterns。
- 当前仓库内尚未拉取的 LFS pointer 文件、oid 和 size。
- `prepare_commands`: `git lfs install` 与 `git lfs pull`。
- 缺少 `git-lfs` 时的 `git_lfs_missing` diagnosis。

执行模式下，`model_prepare.git_lfs` 会记录：

- `executed`: 是否真实执行。
- `status`: `planned` / `ready` / `failed`。
- 每条 LFS 准备命令的 exit code、stdout/stderr tail、timeout 状态和解析出的 progress。
- `progress`: 从 `git lfs pull` 输出解析出的 `percent`、`files_done`、`files_total`、`downloaded_bytes`、`total_bytes`。
- 命令被白名单拒绝时的 `command_rejected` diagnosis。

如果项目使用 Git submodule，`resource_plan.git_submodules` 会记录：

- `.gitmodules` 中的 submodule name、path、url、branch 和 initialized 状态。
- `prepare_commands`: `git submodule sync --recursive` 与 `git submodule update --init --recursive`。
- 缺少 `git` 时的 `git_missing` diagnosis。

执行模式下，`model_prepare.git_submodules` 会记录：

- `executed`: 是否真实执行。
- `status`: `planned` / `ready` / `failed`。
- 每条 submodule 准备命令的 exit code、stdout/stderr tail 和 timeout 状态。
- 命令被白名单拒绝时的 `command_rejected` diagnosis。

## Browser Verify

`VerifyModule` 会对 Gradio、Streamlit 或 `verify_hint.service_type=webui` 的服务增加 `browser_dom_probe`。该 probe 会给页面 URL 注入当前 trace id，并检查浏览器渲染后的 DOM：

- DOM 中包含当前 trace id 时，可作为强证据通过。
- DOM 中包含 traceback、import error、runtime error 等错误标记时，判定为失败证据。
- DOM 加载成功但没有 trace 时保持 `uncertain`，不会因为页面能打开就误判成功。

浏览器 backend 使用可选 Python Playwright：

```bash
python3 -m pip install playwright
python3 -m playwright install chromium
```

如果环境没有安装 Playwright，`browser_dom_probe` 会记录 `uncertain` 证据，不会阻塞已有 HTTP trace 或 artifact evidence。

真实浏览器 backend smoke test 文档见：

```text
docs/playwright-smoke.md
```

仓库还提供手动触发的 GitHub Actions workflow：`.github/workflows/playwright-smoke.yml`。

## 优化路线图

面向真实开源模型自动部署的长期优化计划见：

```text
docs/optimization-roadmap.md
```

该文档覆盖模型下载与缓存、断点续传、资源预估、CUDA/PyTorch 环境求解、长任务状态机、自动诊断修复、增强 verify、安全沙箱和 benchmark 体系。

## Dashboard

可以从本地 run 状态生成静态 dashboard，不需要启动 Web 服务：

```bash
PYTHONPATH=src python3 -m auto_harness.cli dashboard --output runs/dashboard.html
PYTHONPATH=src python3 -m auto_harness.cli dashboard --output runs/dashboard.html --benchmark-report benchmark_report.json
PYTHONPATH=src python3 -m auto_harness.cli dashboard --serve --host 127.0.0.1 --port 8765
```

该命令会同时写出：

```text
runs/dashboard.html
runs/dashboard.json
```

HTML 展示任务状态、当前阶段、各阶段状态、报告路径和可选 benchmark 概览；JSON 用于后续接入真正的 Web dashboard 或任务队列。

`--serve` 会以前台只读方式启动本地 HTTP dashboard：

```text
GET /              HTML dashboard
GET /dashboard.json JSON summary
GET /healthz       health check
```

## Queue

可以先把多个部署任务写入本地持久化队列，再由显式前台 worker 消费：

```bash
PYTHONPATH=src python3 -m auto_harness.cli queue submit --repo ./demo --name demo
PYTHONPATH=src python3 -m auto_harness.cli queue list
PYTHONPATH=src python3 -m auto_harness.cli queue run --max-jobs 1
```

`queue submit` 默认仍是 dry-run，不安装依赖、不启动服务；只有传入 `--execute --allow-install --allow-start`，并且 worker 执行时命令白名单允许，才会进入真实执行路径。`queue run --max-jobs N` 使用线程池并发消费已选队列项，返回结果仍按调度顺序排列。每个 job 运行前会在 `queue/locks/` 下做原子 claim；如果另一个 worker 已经 claim，同一 job 会被跳过并保持 queued，不会重复部署。崩溃 worker 遗留的 lock 会按 `queue_claim_ttl_seconds` 回收，调度结果会写入 `recovered_locks`。GPU 任务可以用 `--require-gpu` 标记；worker 未显式传入 `--gpu-slots` 时会先读取 `AUTO_HARNESS_GPU_SLOTS`，再尝试 `nvidia-smi`，并把 `gpu_probe` 写入调度结果。普通 Mac 开发机没有 GPU 时，GPU 任务会保持 queued 并记录跳过原因。

## Deployment Package

可以把单次部署的审计材料导出为压缩包：

```bash
PYTHONPATH=src python3 -m auto_harness.cli package --task-id <task-id>
PYTHONPATH=src python3 -m auto_harness.cli package --task-id <task-id> --output dist/packages/demo.tar.gz
```

默认产物包含 `task.json`、`state.json`、`events.jsonl`、`reports/`、`evidence/` 和 `repairs/`，并写出同名 `.manifest.json`，记录每个文件的 size 和 sha256。默认不打包 `workspace/`、模型缓存和运行日志；如确实需要日志，可显式加 `--include-logs`。

## Readiness Audit

可以生成项目完成度审计报告：

```bash
PYTHONPATH=src python3 -m auto_harness.cli readiness --output reports/readiness_audit.json
PYTHONPATH=src python3 -m auto_harness.cli readiness --benchmark-report benchmark_report.json --output reports/readiness_audit.json
```

`readiness` 会检查关键代码文件、进度文档和 benchmark manifest，输出 `local_readiness_percent`、本地 gate 结果、外部真实 smoke gate 和 operator next steps。普通 Mac 开发机不默认下载真实模型、不启动 GPU/Docker/vLLM 大模型服务；这些会被标记为 `external_required`，用于后续在具备网络、token、磁盘和 GPU 的环境执行。

## Benchmark

本地 benchmark fixtures 可直接执行，不访问外网：

```bash
PYTHONPATH=src python3 -m auto_harness.cli benchmark --manifest tests/fixtures/benchmarks/manifest.json --output benchmark_report.json
```

可选真实联网 E2E smoke 不在默认 benchmark 中执行。需要网络、磁盘、token 和时间窗口时，先生成计划：

```bash
PYTHONPATH=src python3 -m auto_harness.cli live-smoke-plan --include-long-running
PYTHONPATH=src python3 -m auto_harness.cli live-smoke-plan --execution-backend docker
PYTHONPATH=src python3 -m auto_harness.cli agent-live-smoke --provider xunfei --execute --output runs/live_smoke/manual
```

该命令只输出 Hugging Face、ModelScope、Git LFS 等目标的建议执行命令、环境变量和预计耗时，不会触发下载或启动服务。

当前 benchmark 覆盖：

- 模型下载断点续传。
- 缓存命中避免重复下载。
- 多文件并发下载。
- 临时下载错误会按有限次数重试，不会无限 retry。
- 远端 etag 变化时本地缓存失效并重新下载。
- 模型缓存 dry-run 清理和受控删除。
- 模型缓存可以按 source / repo id 限定清理范围，并用 keep-list 保护关键模型。
- Gradio `/config` discovery 构造 `/api/predict` trace 请求。
- Gradio `/config` shape 变化时仍能选中正确 backend API。
- Gradio queue 模式会先 POST `/call/<api_name>` 获取 `event_id`，再 GET `/call/<api_name>/<event_id>`，只有 follow-up 结果包含当前 trace 才通过。
- FastAPI/Flask `/openapi.json` 会自动选择 POST JSON endpoint，并按 schema 构造 trace 请求。
- 浏览器 DOM 中出现当前 trace 时可以作为强证据。
- Streamlit HTTP 200 错误页面不能通过 verify。
- HTTP 200 但无当前 trace 不能判定成功。
- 历史 artifact 不能作为本次 verify 的新鲜证据。
- 本次 trace 后产生的新文件产物必须可读、非空并记录 sha256 后，才能作为 artifact 强证据。
- Git LFS pointer 和 `.gitattributes` 会被识别；缺 `git-lfs` 时 resource plan 进入诊断态，并输出 LFS 准备命令；执行阶段仍受命令白名单控制。
- 老 Gradio / 未 pin 依赖项目会在 `env_solve` 中生成 `numpy<2`、`pydantic<2`、`opencv-python-headless` 等兼容约束，真正安装仍由 `env_deploy` 受控执行。
- PyTorch 项目会在 `env_solve` 中根据本机 CUDA 版本选择 `cu121` / `cu118` / `cpu` wheel index，并保留 CPU fallback 方案。
- `xformers`、`flash-attn`、`bitsandbytes`、`triton` 会按 Python/CUDA/Torch/平台生成兼容矩阵，阻塞不兼容组合并给出建议动作。
- Docker backend 会记录 GPU 参数、模型缓存挂载、容器日志命令和清理命令元数据。
- Memory promotion 必须先审批，并绑定 apply 后建议运行的 benchmark case。
- LLM planner 可以通过结构化 action 追加 run candidate、更新 verify hint，并记录 accepted/rejected actions。
- LLM diagnoser 可以在未知失败日志下生成 repair action；`install_package` 只有在 policy 允许时才执行并记录命令结果。
- LLM verify planner 可以在首次 verify uncertain 后生成新的 trace request hint，但 Python verify 仍要求响应、DOM 或 artifact 包含当前 trace。
- 长耗时 verify 会持续刷新首次推理探针和完成状态。
- vLLM/OpenAI-compatible server 会优先通过 `/v1/models` 发现模型，再使用 `/v1/chat/completions` 发送 trace prompt；普通 JSON 或 streaming SSE 响应中包含当前 trace 才通过。
- 本地 E2E fixture matrix 会把小型 Gradio demo、Streamlit demo 和 Git LFS 权重仓库跑完整 dry-run pipeline，并检查阶段结果。
- Dashboard 可以作为只读本地 HTTP 服务提供 HTML、JSON 和 health check。
- 持久化任务队列可以入队、列出并前台消费 dry-run 部署任务。
- 队列 worker pool 可以并发消费多个已选队列任务，并保持结果顺序稳定。
- 队列 claim lock 可以阻止多个 worker 重复执行同一个 queued job。
- 崩溃 worker 遗留的过期 claim lock 可以按 TTL 回收并继续执行任务。
- 队列可以根据 GPU 探测结果调度 `--require-gpu` 任务，无可用 slot 时保留 queued 并写入跳过原因。
- 部署任务可以导出不包含 workspace 的审计产物包和 manifest。
- Readiness audit 可以把本地完成度标记为 100%，并列出真实联网/GPU/Docker/vLLM 外部验收门。
- 服务进程启动后快速退出不能判定为启动成功。
- token 缺失或 401 日志会被诊断为 `auth_required`。
- token 缺失时 report 只展示 `HF_TOKEN` / `MODELSCOPE_TOKEN` 等变量名，不记录密钥值。
- 权限不足时 repair action 被 policy 拒绝。
- 同一问题的 repair loop 超过次数后会拒绝继续自动修复。
- `resume` 会按 `rerun_from_effective` 从安全阶段重跑，避免每次全量 pipeline。
- 需要人工确认的 repair action 只有审批后才能通过。
- 模型文件 checksum 不一致时下载失败，不写入成功缓存态。

## Repair Loop

失败或不确定阶段会写入：

```text
runs/<task-id>/repairs/
```

其中可能包含 `repair_plan.json`、`repair_loop_state.json`、`repair_install_plan.json`、`repair_verify_hints.json`、`required_env_vars.json`、`operator_approval.json` 和 `repair_apply_result.json`。下一次 `resume` 时，`RepairOverlay` 只消费 policy 与 loop gate 均允许的非执行型 artifact：

- 将 `repair_install_plan.json` 中的命令追加到 `env_deploy` 的 install plan，真正执行仍需要 `--execute --allow-install` 和命令白名单通过。
- 将 `repair_verify_hints.json` 合并到 `verify_hint`，用于修正 endpoint、请求路径或 POST JSON 模板。
- 如果 repair 被 policy、attempt limit 或人工审批要求拒绝，overlay 不会生效，只保留审计记录。
- 如果 plan 中的 `rerun_from` 不在安全阶段集合中，loop 会记录 `rerun_from_requested`，并生成 `rerun_from_effective` 作为安全回退阶段。
- `resume` 会读取已允许的 `repair_apply_result.json`，从 `rerun_from_effective` 开始重跑，并复用该阶段之前的 `reports/*_result.json` / `pipeline_results.json`。如果前置结果缺失或损坏，会记录 `resume_stage_fallback` 事件并回退到 `analyze`。
- 中间阶段恢复会额外写入 `reports/execution_audit.json` 和 `resume_execution_plan` 事件，最终报告的 `Execution Audit` 小节会列出 reused stages、rerun stages、requested/effective start stage，便于面试或生产复盘解释本次恢复执行为什么没有全量重跑。

人工审批入口：

```bash
PYTHONPATH=src python3 -m auto_harness.cli repair-approve --task-id <task-id> --note "approved cache dir change"
```

该命令只写入 action 类型、审批时间和备注，不记录任何 token、key 或 secret 值。

## Memory Evolution

当 `memory/deployment_issues.jsonl` 中同类已验证成功经验反复出现，通过统一主链路生成 Skill candidate：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve --propose --min-verified-count 3 --provider mock
```

默认只写入 candidate，不修改 Skill：

```text
memory/skill_candidates/candidate_<candidate_id>.json
memory/skill_candidates/candidate_<candidate_id>.md
memory/skill_candidates/candidate_<candidate_id>.lifecycle.jsonl
```

普通失败或未验证的 LLM diagnosis 仍可用于相似问题检索，但不能进入 Skill evolution。只有满足 verified memory quality gate 的记录才会生成 candidate。

候选项必须先写入显式审批：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve --approve --candidate memory/skill_candidates/candidate_<candidate_id>.json --reviewer <name> --note "reviewed"
```

然后运行绑定回归，并用真实历史 run 做 shadow 评估：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve --regression --candidate memory/skill_candidates/candidate_<candidate_id>.json
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve --shadow --candidate memory/skill_candidates/candidate_<candidate_id>.json --run-dir runs/<task-id>
```

只有状态达到 `shadow_passed` 后才允许修改 Skill：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve --promote --candidate memory/skill_candidates/candidate_<candidate_id>.json
```

每次状态迁移都写入带 `previous_event_hash` / `event_hash` 的 JSONL 审计。晋升记录保留修改前 SHA、修改后 SHA 和 rollback artifact。`memory-promote` 仍可读取并生成旧格式 proposal，但 `memory-promote --apply` 固定失败，不能再作为 Skill 写入口。

也可以单独运行某些 benchmark case：

```bash
PYTHONPATH=src python3 -m auto_harness.cli benchmark --case-id gradio_config_discovery --case-id gradio_queue_call_followup
```

## 安全默认值

默认情况下，`deploy` 是 dry-run，不会安装依赖，也不会启动长驻服务，除非显式传入执行参数。

执行模式还受命令白名单保护。`env_deploy` 和 `runner` 会拒绝 `configs/default.json` 的 `allowed_commands` 之外的命令。

## LLM Plan-first Agent Mode

auto-deploy-harness 支持 Plan-first 部署模式。在此模式下，LLM 先读取经过脱敏的项目快照，生成结构化部署方案（install 命令、runner 候选、model asset 策略、verify request），框架将其当作不可信 proposal：经过 schema 校验、policy gate 验证、编译为 effective plan，再分阶段执行。最终是否部署成功仍由 trace-based evidence 判定，不由 LLM 自行声称。

运行：

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy \
  --repo tests/fixtures/e2e/llm_plan_first_http_trace \
  --name plan-first-http \
  --execute \
  --allow-install \
  --allow-start \
  --agent-plan-first \
  --agent-plan-first-provider mock \
  --agent-plan-first-mode gated_actor
```

核心流程：

```text
项目代码/文件观察
  -> LLM 生成部署方案和命令候选
  -> 框架校验/编译（schema + policy gate + command allowlist + path boundary + secret redaction）
  -> 分阶段执行
  -> evidence verify（必须包含当前 trace_id）
  -> 失败后 LLM replan
```

检查产出：

```text
runs/<task-id>/reports/project_snapshot.json
runs/<task-id>/reports/llm_deployment_plan.raw.json
runs/<task-id>/reports/llm_deployment_plan.parsed.json
runs/<task-id>/reports/llm_plan_policy.json
runs/<task-id>/reports/effective_deployment_plan.json
runs/<task-id>/reports/plan_revisions.jsonl
runs/<task-id>/reports/llm_contribution_evidence.json
runs/<task-id>/evidence/*trace*.json
```

关键安全属性：

- raw LLM plan 和 effective plan 必须分开。LLM 给的是 proposal，框架执行的是 policy-normalized plan。
- LLM 不直接执行 shell，只能提出命令候选。
- 所有命令必须通过 command allowlist、path boundary、secret redaction 和 verify trace 检查。
- 失败后 LLM 可基于真实日志和 evidence replan，但新 plan 仍需通过 policy gate。
- 最终成功只能由 VerifyModule 的当前 trace evidence 判定。
