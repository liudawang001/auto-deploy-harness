# 真正 LLM 驱动自动部署 Agent 优化开发文档

本文档用于交给其他 AI 编码模型执行开发。目标不是写概念方案，而是把当前项目从“部署 workflow + 事后 Agent 审计”升级为“在关键不确定环节由 LLM 运行时选择工具、经策略门禁后真实执行、并改变后续状态”的 Agentic deployment system。

本文档中的“必须”是开发验收要求；“可以暂缓”不是本轮目标。

## 0. 项目当前事实与核心问题

当前项目已经具备以下基础：

- deterministic deployment pipeline：`analyze -> resource_plan -> env_solve -> env_deploy -> model_prepare -> runner -> verify -> report`
- `VerifyModule`：包含 HTTP trace、浏览器/框架相关验证逻辑
- `RepairPlanner` / `RepairApplier` / repair loop：具备 repair plan 与 action apply 雏形
- `AgentRuntime`：可生成 `agent_steps.jsonl`、`agent_state.json`、`agent_plan.json`
- `ToolRegistry`：已有工具 schema / 风险等级 / policy 字段
- `AgentCritic`：已有 critic 输出，但目前主要用于记录
- `eval-compare`：已有命令入口，但目前不是严格真实对照评估

当前最大问题：

```text
AgentRuntime 主要在 pipeline 执行完成后运行。
它记录了类似 Agent step 的 artifact，但没有真实控制主流程。
LLM 多数时候是 advisor / planner，而不是运行时决策者。
ToolRegistry 多数是声明式 registry，不是被 Agent loop 调用的真实工具执行边界。
Critic / policy reject 不一定阻断真实执行。
eval-compare 不能证明 LLM 让 baseline uncertain/failed 变成 passed。
```

因此，当前不能直接声称“实现真正 LLM 驱动的自动部署 Agent”。本轮优化要补的不是概念，而是运行时控制权证据。

## 1. 总目标

### 1.1 最终定位

更稳妥的项目定位：

> 一个面向 AI demo 部署的 LLM-driven verification / repair sub-agent。系统保留确定性部署 pipeline，但在 verify uncertain 和 failure diagnosis 等不确定环节，由 LLM 基于 evidence 选择下一步 tool，经 policy / critic 校验后执行，并把 tool result 纳入最终 evidence-based verification。

不要把目标写成：

> 让 LLM 完全自主完成所有部署动作。

这会带来安全风险，也不符合大厂工程评审对 Agent 系统的要求。合理架构应是：

```text
Deterministic runtime handles execution.
LLM handles uncertainty.
Policy handles safety.
Verifier handles truth.
Artifacts prove contribution.
```

### 1.2 本轮必须达成的 Agent 标准

本轮完成后，至少要满足：

```text
1. LLM 在运行时选择 tool_call，而不是事后总结。
2. tool_call 经过 schema validation。
3. tool_call 经过 critic / policy gate。
4. policy 或 critic reject 时，原 tool_call 不得执行。
5. approved tool_call 由 ToolExecutor 真实执行。
6. tool_result 写入 run artifact。
7. tool_result 会改变 verify 或 repair 后续状态。
8. Agent loop 发生在 final report 之前。
9. 至少一个 fixture 能证明 baseline uncertain/failed，agent mode passed。
10. artifact 能证明 LLM 的实际贡献，而不是只证明调用过 LLM。
```

缺少任何一项，都只能叫 Agentic Workflow 或 LLM-enhanced Automation，不建议叫真正 LLM-driven Agent。

## 2. 本轮范围控制

### 2.1 必做

优先做一个高价值、边界清晰、容易验证的子 Agent：

```text
LLM-driven Verify Agent
```

场景：

```text
服务已经启动，进程存活，端口可访问。
deterministic verify 无法证明成功，状态是 uncertain。
LLM 根据 README / app.py / 当前 evidence / 服务状态选择下一步 probe tool。
系统执行 probe tool 后获得新的 trace evidence。
最终 verify 从 uncertain 变成 passed。
```

同时保留 repair Agent 的接口设计，但本轮不强制完成复杂源码修复。

### 2.2 不做或暂缓

本轮不要做：

- 不让 LLM 直接执行任意 shell
- 不让 LLM 直接修改用户源码
- 不做多 Agent 协作
- 不做大规模并发调度
- 不做真实 GPU 调度平台
- 不做多租户权限系统
- 不做生产 dashboard
- 不做 Docker backend 大改
- 不强制真实 Hugging Face 大模型部署
- 不把所有 pipeline stage 改成 LLM 决策

原因：这些范围会稀释当前最关键目标，即证明 LLM 在关键不确定环节“必要且有效”。

## 3. 新架构总览

### 3.1 目标架构

```text
TaskRunner / Orchestrator
  -> deterministic analyze
  -> deterministic resource_plan
  -> deterministic env_solve
  -> deterministic env_deploy
  -> deterministic model_prepare
  -> deterministic runner
  -> deterministic verify
       -> if passed:
            finish
       -> if uncertain and agent verify enabled:
            AgentRuntime.act_verify(...)
              -> observe
              -> LLM planner chooses tool_call
              -> schema validation
              -> critic gate
              -> policy gate
              -> ToolExecutor executes
              -> observe tool_result
              -> update evidence
              -> repeat until pass / reject / max_steps
       -> final report
```

### 3.2 关键改动

当前：

```text
pipeline 完成后 AgentRuntime.run(...) 生成审计记录
```

目标：

```text
verify uncertain 时，AgentRuntime.act_verify(...) 插入主路径，并影响 final verify result
```

### 3.3 模式划分

必须支持三种模式：

| 模式 | 作用 | 是否执行 LLM tool | 是否改变最终结果 | 用途 |
|---|---|---:|---:|---|
| `off` | 纯 deterministic pipeline | 否 | 否 | baseline |
| `audit` | 事后生成 Agent artifacts | 否 | 否 | 兼容旧行为 |
| `gated_actor` | LLM 选择 tool，经 policy 后执行 | 是 | 是 | 真 Agent 模式 |

可选支持：

| 模式 | 作用 |
|---|---|
| `planner` | LLM 只生成 plan，不执行 tool，用于对比 LLM 建议质量 |

## 4. 需要新增或重构的文件

### 4.1 Agent runtime

新增或重构：

```text
src/auto_harness/agent_runtime/runtime.py
src/auto_harness/agent_runtime/planner.py
src/auto_harness/agent_runtime/policy.py
src/auto_harness/agent_runtime/state.py
src/auto_harness/agent_runtime/schemas.py
```

职责：

| 文件 | 职责 |
|---|---|
| `runtime.py` | Agent loop controller，提供 `audit()` 和 `act_verify()` |
| `planner.py` | 调用 LLM provider，要求输出严格 JSON tool_call |
| `policy.py` | Agent-level tool policy，判断 tool_call 是否允许执行 |
| `state.py` | Agent belief state、step persistence、artifact writer |
| `schemas.py` | dataclass 或 typed dict schema，定义 observation / decision / result |

### 4.2 Tools

新增或重构：

```text
src/auto_harness/tools/executor.py
src/auto_harness/tools/verify_tools.py
src/auto_harness/tools/repair_tools.py
```

职责：

| 文件 | 职责 |
|---|---|
| `executor.py` | 根据 approved tool_call 分发到具体 tool |
| `verify_tools.py` | 实现 verify probe tools |
| `repair_tools.py` | 后续 repair actions 的 executor 边界，本轮可只放 skeleton |

### 4.3 Verify 集成点

需要修改：

```text
src/auto_harness/modules/verify.py
src/auto_harness/orchestrator.py
src/auto_harness/config.py
src/auto_harness/cli.py
```

目标：

- `VerifyModule` 默认 verify 得到 uncertain 后触发 `AgentRuntime.act_verify()`
- `orchestrator` final report 读取 agent verify result
- config / CLI 支持开关
- 默认行为保持向后兼容

### 4.4 Eval

需要修改：

```text
src/auto_harness/evals/comparison.py
eval_targets/manifest.json
tests/fixtures/...
```

目标：

- `eval-compare` 不再只生成 unknown skeleton
- 至少支持本地 fixture 的真实 off vs gated_actor 对比
- 报告必须能证明 `llm_helped=true` 的 case

## 5. 配置与 CLI 设计

### 5.1 Config 字段

在项目配置对象中增加或确认以下字段：

```json
{
  "agent_mode": "off|audit|planner|gated_actor",
  "agent_enable_verify": true,
  "agent_enable_repair": false,
  "agent_verify_max_steps": 3,
  "agent_repair_max_steps": 2,
  "agent_allow_network": false,
  "agent_allowed_hosts": ["127.0.0.1", "localhost"],
  "agent_require_trace_id": true,
  "agent_artifacts_enabled": true
}
```

默认建议：

```json
{
  "agent_mode": "audit",
  "agent_enable_verify": false,
  "agent_enable_repair": false
}
```

原因：不能让升级后默认执行 LLM tool，避免改变旧用户行为。

### 5.2 CLI 参数

建议支持：

```bash
python -m auto_harness.cli deploy \
  --repo tests/fixtures/e2e/gradio_tiny_model \
  --name agent-verify-smoke \
  --agent-mode gated_actor \
  --agent-enable-verify \
  --agent-verify-max-steps 3
```

如已有参数体系不同，按当前 CLI 风格接入，但语义必须一致。

### 5.3 模式行为

| 参数组合 | 行为 |
|---|---|
| `--agent-mode off` | 不调用 AgentRuntime |
| `--agent-mode audit` | 只调用 post-hoc artifact 生成 |
| `--agent-mode planner` | LLM 输出 tool_call，但不执行，记录 would_execute |
| `--agent-mode gated_actor --agent-enable-verify` | verify uncertain 时真实执行 approved tool |

## 6. 核心数据结构

建议使用 dataclass，或沿用项目现有模型风格。字段语义必须保持。

### 6.1 AgentObservation

```python
@dataclass
class AgentObservation:
    run_id: str
    stage: str
    attempt: int
    service: dict
    failed_checks: list[dict]
    evidence_summary: dict
    selected_files: dict[str, str]
    allowed_tools: list[str]
    constraints: list[str]
    previous_steps: list[dict]
```

要求：

- `selected_files` 必须截断，避免把整个仓库塞进 prompt
- 不允许包含 secret value
- `constraints` 必须包含 trace 约束、host 约束、tool 约束
- `previous_steps` 最多保留最近 3 步摘要

### 6.2 AgentDecision

```python
@dataclass
class AgentDecision:
    status: str  # ok|no_action|invalid
    hypothesis: str
    confidence: float
    tool_call: dict | None
    expected_observation: str
    fallback_tool_call: dict | None
    stop_reason: str | None
    raw_response: str
```

LLM 输出必须是 JSON，并能解析成此结构。禁止自由文本计划直接进入 executor。

### 6.3 ToolCall

```python
@dataclass
class ToolCall:
    name: str
    input: dict
    idempotency_key: str
```

`idempotency_key` 用于防重复执行：

```text
sha256(run_id + step_index + tool_name + canonical_json(input))
```

### 6.4 PolicyDecision

```python
@dataclass
class PolicyDecision:
    allowed: bool
    reason: str
    risk: str
    normalized_input: dict | None
```

`normalized_input` 是 policy 校验后的输入。executor 只能使用 normalized input，不能直接用 LLM 原始 input。

### 6.5 ToolResult

```python
@dataclass
class ToolResult:
    status: str  # passed|failed|uncertain|error|rejected
    tool_name: str
    evidence: dict
    evidence_path: str | None
    strong_verify_pass: bool
    error: str | None
    started_at: str
    ended_at: str
```

### 6.6 AgentStepRecord

每一步写入 JSONL：

```json
{
  "step_index": 1,
  "stage": "verify",
  "observation_hash": "sha256:...",
  "decision": {
    "hypothesis": "The app exposes a Gradio named API.",
    "confidence": 0.76,
    "tool_call": {
      "name": "discover_gradio_api",
      "input": {
        "endpoint": "http://127.0.0.1:7860",
        "trace_template": "{{trace_id}}"
      }
    }
  },
  "critic": {
    "allowed": true,
    "reason": "Tool is relevant to Gradio endpoint discovery."
  },
  "policy": {
    "allowed": true,
    "reason": "Localhost endpoint and trace template present.",
    "risk": "low"
  },
  "execution": {
    "executed": true,
    "status": "passed",
    "strong_verify_pass": true,
    "evidence_path": "runs/.../evidence/trace_agent_probe_1.json"
  },
  "state_delta": {
    "verify_status_before": "uncertain",
    "verify_status_after": "passed"
  }
}
```

## 7. LLM-driven Verify Agent 详细设计

### 7.1 触发条件

在 deterministic verify 之后触发：

```text
verify_status == uncertain
service.process_alive == true
service.port_ready == true
agent_mode in ["planner", "gated_actor"]
agent_enable_verify == true
strong_verify_pass == false
```

不触发条件：

```text
verify_status == passed
process not alive
port not ready
agent_mode == off
agent_enable_verify == false
max retry exhausted
```

### 7.2 Observation 构建

LLM 输入示例：

```json
{
  "stage": "verify",
  "goal": "Find an evidence-producing local probe that proves the current service handles the current trace_id.",
  "service": {
    "endpoint_candidates": ["http://127.0.0.1:7860"],
    "process_alive": true,
    "port_ready": true,
    "framework_hint": "gradio"
  },
  "failed_checks": [
    {
      "name": "http_trace_response",
      "status": "uncertain",
      "reason": "HTTP 200 did not contain current trace_id."
    }
  ],
  "evidence_summary": {
    "trace_id": "trace-abc",
    "http_status": 200,
    "response_excerpt": "<html>..."
  },
  "selected_files": {
    "README.md": "truncated content",
    "app.py": "truncated content"
  },
  "allowed_tools": [
    "discover_gradio_api",
    "discover_openapi_schema",
    "probe_http",
    "probe_browser_dom"
  ],
  "constraints": [
    "Only call localhost or 127.0.0.1 endpoints.",
    "Do not call external URLs.",
    "Do not include secret values.",
    "Any verification request must include {{trace_id}} or explain why discovery-only tool is used.",
    "Success requires current trace evidence, not HTTP 200 alone."
  ]
}
```

### 7.3 Prompt 要求

Planner prompt 必须强制输出 JSON：

```text
You are a deployment verification agent.
Choose exactly one next tool call from allowed_tools.
Do not return prose outside JSON.
Do not mark success yourself.
The runtime verifier decides success from tool_result.
```

输出 schema：

```json
{
  "status": "ok",
  "hypothesis": "string",
  "confidence": 0.0,
  "tool_call": {
    "name": "string",
    "input": {}
  },
  "expected_observation": "string",
  "fallback_tool_call": {
    "name": "string",
    "input": {}
  }
}
```

如果 LLM 认为没有合法动作：

```json
{
  "status": "no_action",
  "hypothesis": "No safe local probe can prove trace handling.",
  "confidence": 0.4,
  "tool_call": null,
  "expected_observation": "",
  "fallback_tool_call": null,
  "stop_reason": "no_safe_tool"
}
```

### 7.4 可执行 Verify Tools

本轮至少实现以下工具：

| Tool | 输入 | 行为 | 成功标准 | 风险 |
|---|---|---|---|---|
| `probe_http` | `endpoint`, `method`, `path`, `body`, `headers`, `trace_template` | 对本地服务发起请求 | response 或 structured output 含当前 trace_id | low |
| `discover_gradio_api` | `endpoint`, `trace_template` | 读取 `/config`，推断 Gradio callable endpoint，再发 trace probe | trace_id 出现在 API response | low |
| `discover_openapi_schema` | `endpoint`, `trace_template` | 读取 `/openapi.json`，选取安全 POST/GET 路径 probe | trace_id 出现在 response | low |
| `probe_browser_dom` | `endpoint`, `trace_template` | 访问页面并检查 DOM / 可选填入 trace | DOM 或响应含 trace_id | medium |

可以暂缓：

| Tool | 原因 |
|---|---|
| `discover_openai_compatible_model` | 对当前 MVP 不是必要 |
| browser 自动点击复杂 UI | 成本高、稳定性差 |
| shell tool | 安全风险高 |

### 7.5 ToolExecutor 行为

`ToolExecutor.execute(tool_call, context)` 必须满足：

```text
1. 只接收 policy normalized input。
2. 不直接使用 LLM 原始 input。
3. 只允许调用 registry 中存在且当前 stage 允许的 tool。
4. 每次执行都写 evidence file。
5. 返回 ToolResult。
6. 不能自己调用 LLM。
```

伪代码：

```python
class ToolExecutor:
    def execute(self, tool_call: ToolCall, context: AgentExecutionContext) -> ToolResult:
        if tool_call.name == "probe_http":
            return verify_tools.probe_http(tool_call.input, context)
        if tool_call.name == "discover_gradio_api":
            return verify_tools.discover_gradio_api(tool_call.input, context)
        if tool_call.name == "discover_openapi_schema":
            return verify_tools.discover_openapi_schema(tool_call.input, context)
        if tool_call.name == "probe_browser_dom":
            return verify_tools.probe_browser_dom(tool_call.input, context)
        return ToolResult(status="rejected", error="unknown tool")
```

## 8. Policy 与 Critic 详细设计

### 8.1 执行顺序

必须是：

```text
LLM raw response
  -> JSON parse
  -> schema validation
  -> critic gate
  -> policy gate
  -> executor
```

不能是：

```text
executor 先执行，然后 critic 事后评价
```

### 8.2 Schema validation

必须拒绝：

- 非 JSON 输出
- 缺少 `tool_call.name`
- tool 不在 `allowed_tools`
- input 类型不正确
- `confidence` 不是数字
- `status=ok` 但 `tool_call=null`

### 8.3 Critic gate

Critic 判断“这个动作是否和当前目标相关、是否有明显 hallucination、是否试图越权”。

Critic 输出：

```json
{
  "allowed": true,
  "reason": "The selected tool matches the Gradio framework hint.",
  "issues": [],
  "safer_alternative": null
}
```

Reject 示例：

```json
{
  "allowed": false,
  "reason": "The decision proposes installing packages during verify stage.",
  "issues": ["stage_mismatch"],
  "safer_alternative": {
    "name": "probe_http",
    "input": {
      "endpoint": "http://127.0.0.1:7860",
      "path": "/",
      "trace_template": "{{trace_id}}"
    }
  }
}
```

规则：

- critic reject 时，原 tool_call 不得执行
- 如果使用 `safer_alternative`，必须再次经过 schema validation 和 policy
- critic 不负责判定 success

### 8.4 ToolPolicy

Policy 判断“这个动作是否允许执行”。

必须校验：

```text
tool exists
tool allowed in current stage
tool allowed in current agent_mode
risk <= configured max risk
endpoint host in localhost / 127.0.0.1 / allowlist
no external URL
no shell metacharacter for command-like fields
no secret value or secret-looking field
verify request contains trace_template when required
path traversal forbidden
input normalized
```

Policy reject 示例：

```json
{
  "allowed": false,
  "reason": "External host is not allowed: https://example.com",
  "risk": "high",
  "normalized_input": null
}
```

Policy allow 示例：

```json
{
  "allowed": true,
  "reason": "Local endpoint and trace template are valid.",
  "risk": "low",
  "normalized_input": {
    "endpoint": "http://127.0.0.1:7860",
    "path": "/api/predict",
    "method": "POST",
    "trace_template": "{{trace_id}}"
  }
}
```

## 9. AgentRuntime.act_verify 详细实现

### 9.1 函数签名建议

```python
class AgentRuntime:
    def audit(self, run_dir: Path, results: dict, contribution: dict | None = None) -> dict:
        ...

    def act_verify(
        self,
        *,
        run_dir: Path,
        repo_path: Path,
        initial_verify_result: dict,
        service_context: dict,
        trace_id: str,
        config: dict,
        provider: object | None,
        max_steps: int = 3,
    ) -> dict:
        ...
```

### 9.2 返回值

```json
{
  "triggered": true,
  "final_status": "passed",
  "llm_helped": true,
  "step_count": 2,
  "accepted_tool_count": 1,
  "rejected_tool_count": 1,
  "strong_verify_pass": true,
  "evidence_paths": [
    "runs/.../evidence/trace_agent_probe_1.json"
  ],
  "stop_reason": "strong_verify_pass"
}
```

### 9.3 act_verify 伪代码

```python
def act_verify(...):
    state = AgentVerifyState.from_initial_result(...)
    writer = AgentStepWriter(run_dir)

    for step_index in range(max_steps):
        observation = observation_builder.build(state)

        decision = planner.plan_verify(observation)
        parse_result = parse_and_validate(decision)
        if not parse_result.ok:
            writer.write_rejected(...)
            state.record_reject("invalid_llm_output")
            break

        critic_result = critic.evaluate(observation, parse_result.decision)
        if not critic_result.allowed:
            writer.write_rejected(...)
            state.record_reject("critic_rejected")
            break

        policy_result = policy.validate(parse_result.decision.tool_call, observation)
        if not policy_result.allowed:
            writer.write_rejected(...)
            state.record_reject("policy_rejected")
            break

        tool_result = executor.execute(policy_result.normalized_tool_call, context)
        writer.write_executed(...)
        state.apply_tool_result(tool_result)

        if tool_result.strong_verify_pass:
            return state.to_result(final_status="passed", stop_reason="strong_verify_pass")

        if not service_is_alive(service_context):
            return state.to_result(final_status="uncertain", stop_reason="service_dead")

    return state.to_result(final_status="uncertain", stop_reason="max_steps")
```

### 9.4 Artifact 写入要求

必须写：

```text
runs/<run_id>/agent_verify_steps.jsonl
runs/<run_id>/agent_state.json
runs/<run_id>/reports/agent_verify_result.json
runs/<run_id>/evidence/<trace_id>_agent_probe_<step>.json
```

建议继续写或兼容：

```text
runs/<run_id>/agent_steps.jsonl
runs/<run_id>/agent_plan.json
runs/<run_id>/agent_plan_revisions.jsonl
```

要求：

- artifact 中必须能区分 `planned`、`rejected`、`executed`
- `llm_helped=true` 只能在 tool executed 且 final status 改善时设置
- planner 模式不得设置 `executed=true`
- audit 模式不得设置 `llm_helped=true`

## 10. VerifyModule 集成方式

### 10.1 接入位置

在 `src/auto_harness/modules/verify.py` 中：

```text
run deterministic verify
  -> build VerifyResult
  -> if passed: return
  -> if uncertain and config allows agent verify:
       call AgentRuntime.act_verify(...)
       merge agent evidence
       recompute final status
  -> return final VerifyResult
```

### 10.2 合并规则

Agent verify result 不能绕过 evidence-based verification。

允许：

```text
agent tool produced evidence containing current trace_id
evidence saved to file
tool_result.strong_verify_pass == true
VerifyResult.status becomes passed
```

禁止：

```text
LLM said it is successful
HTTP 200 only
old trace_id matched
artifact missing
tool planned but not executed
policy rejected but still pass
```

### 10.3 VerifyResult 扩展字段

建议加入：

```json
{
  "agent_verify": {
    "triggered": true,
    "mode": "gated_actor",
    "final_status": "passed",
    "llm_helped": true,
    "step_count": 1,
    "accepted_tool_count": 1,
    "rejected_tool_count": 0,
    "evidence_paths": []
  }
}
```

如果项目现有 `VerifyResult` 是 dataclass / dict 混合，按现有风格最小侵入扩展。

## 11. Repair Agent 设计，第二优先级

Repair 可以先保留接口，不要抢本轮主线。

### 11.1 触发条件

```text
stage failed or uncertain
deterministic diagnosis low confidence
agent_mode == gated_actor
agent_enable_repair == true
repair action allowed by policy
```

### 11.2 LLM 输出

```json
{
  "status": "ok",
  "failure_hypothesis": "runner failed because package rich is missing",
  "evidence": [
    "runner.log contains ModuleNotFoundError: No module named 'rich'"
  ],
  "tool_call": {
    "name": "apply_repair",
    "input": {
      "action": {
        "type": "install_package",
        "package": "rich"
      }
    }
  },
  "expected_observation": "rerun runner starts service and port becomes ready",
  "verification_plan": {
    "rerun_from": "env_deploy",
    "verify_required": true
  },
  "risk": "low"
}
```

### 11.3 Repair 成功定义

必须同时满足：

```text
LLM proposed repair
policy accepted repair
ToolExecutor executed repair action
pipeline resumed from safe stage
final verify passed
repair_verified == true
```

禁止把以下情况算作 self-healing：

```text
metadata-only action
只生成建议
只 rerun 没有实际修复
修复后没有 verify pass
LLM 给出原因但没有执行 action
```

## 12. Eval 对照评估设计

### 12.1 eval-compare 目标

`eval-compare` 必须证明：

```text
off mode 做不到或 uncertain
gated_actor mode 通过 LLM-selected tool 得到 pass
artifact 能追溯是哪一步 tool 改变了结果
```

### 12.2 命令建议

```bash
PYTHONPATH=src python -m auto_harness.cli eval-compare \
  --manifest eval_targets/manifest.json \
  --output-dir runs/evals/agent-verify-mvp \
  --run
```

### 12.3 输出结构

```text
runs/evals/<eval_id>/
  comparison_report.json
  target-001/
    off/
      run_summary.json
    gated_actor/
      run_summary.json
      agent_verify_steps.jsonl
      reports/agent_verify_result.json
```

### 12.4 comparison_report schema

```json
{
  "eval_id": "agent-verify-mvp",
  "targets": [
    {
      "target_id": "gradio-named-api-trace",
      "baseline": {
        "mode": "off",
        "verify_status": "uncertain",
        "run_dir": "..."
      },
      "agent": {
        "mode": "gated_actor",
        "verify_status": "passed",
        "llm_helped": true,
        "accepted_tool_count": 1,
        "rejected_tool_count": 0,
        "evidence": "..."
      },
      "delta": {
        "status_improved": true,
        "reason": "agent_selected_discover_gradio_api"
      }
    }
  ],
  "summary": {
    "total": 1,
    "baseline_passed": 0,
    "agent_passed": 1,
    "helped_cases": 1,
    "policy_reject_cases": 0
  }
}
```

### 12.5 最小 eval targets

至少添加 3 个 fixture：

| target | baseline | agent | 目的 |
|---|---|---|---|
| `gradio_named_api_trace` | uncertain | passed | 证明 LLM probe 选择有效 |
| `policy_reject_external_url` | uncertain | uncertain/rejected | 证明 policy 真阻断 |
| `invalid_llm_json` | uncertain | uncertain/rejected | 证明 schema validation 生效 |

可以用 mock provider 控制 LLM 输出，但必须让 ToolExecutor 真实执行本地 probe。不要只 mock 整个 Agent result。

## 13. 测试要求

用户若限制 token 或时间，可以不跑完整测试，但开发文档要求如下。

### 13.1 单元测试

新增或修改：

```text
tests/test_core.py
tests/test_agent_runtime.py       # 如果愿意拆分
tests/test_tool_policy.py         # 如果愿意拆分
tests/test_eval_compare.py        # 如果愿意拆分
```

必须覆盖：

```text
valid tool_call passes schema
invalid JSON rejected
unknown tool rejected
external URL rejected
missing trace_template rejected for verify probe
critic reject prevents execution
planner mode does not execute
gated_actor mode executes approved tool
tool_result strong_verify_pass changes final verify status
llm_helped false when no execution
llm_helped false when execution did not improve status
```

### 13.2 本地 smoke

最小 smoke：

```bash
PYTHONPATH=src python -m auto_harness.cli deploy \
  --repo tests/fixtures/e2e/gradio_tiny_model \
  --name agent-verify-smoke \
  --dry-run \
  --agent-mode gated_actor \
  --agent-enable-verify
```

如果 `--dry-run` 无法启动真实服务，则需要准备一个不依赖真实模型的 local fixture 或 test server。关键不是部署大模型，而是证明 Agent verify loop 真实执行 tool。

### 13.3 验收检查

开发完成后检查：

```text
agent_verify_steps.jsonl 存在
step 中有 LLM decision
step 中有 critic decision
step 中有 policy decision
approved step 中 execution.executed == true
rejected step 中 execution.executed == false
evidence file 存在
evidence 内含 current trace_id
final verify status 从 uncertain 变成 passed
agent_verify.llm_helped == true
```

## 14. 实现阶段拆分

### Phase 1：保留旧行为，拆出 Audit Mode

目标：

- 当前 `AgentRuntime.run()` 不要直接删除
- 可以重命名为 `audit()`，或保留 `run()` 并内部调用 `audit()`
- 新增 mode 字段，明确 artifact 是 audit 还是 act

具体任务：

```text
1. 在 AgentRuntime 中新增 audit(...)
2. 保持旧 orchestrator 调用不坏
3. agent_steps.jsonl 中标记 mode="audit"
4. 不把 audit 产物计入 llm_helped
```

验收：

```text
旧 deploy 流程仍可生成原有 agent artifacts
不会因为没有 LLM provider 崩溃
```

### Phase 2：实现 schema / planner / parser

目标：

- LLM 输出从 advisory text 改成 tool_call JSON
- 所有输出先 parse，再进入 gate

具体任务：

```text
1. 新增 AgentDecision schema
2. 新增 parse_agent_decision(raw_response)
3. planner.plan_verify(observation) 调用 provider
4. mock provider 支持返回固定 JSON
5. invalid JSON 产生 rejected step
```

验收：

```text
合法 JSON -> AgentDecision
非法 JSON -> rejected，不执行 tool
unknown tool -> rejected，不执行 tool
```

### Phase 3：实现 ToolPolicy

目标：

- policy 是 executor 前的硬门禁

具体任务：

```text
1. 新增 agent_runtime/policy.py
2. 实现 validate_tool_call(...)
3. 校验 host allowlist
4. 校验 trace_template
5. 校验 tool/stage/mode/risk
6. 返回 normalized input
```

验收：

```text
localhost probe allowed
external URL rejected
missing trace_template rejected
policy rejected 时 executor 不被调用
```

### Phase 4：实现 ToolExecutor 与 verify tools

目标：

- approved tool 能真实执行并产生 evidence

具体任务：

```text
1. 新增 tools/executor.py
2. 新增 tools/verify_tools.py
3. 实现 probe_http
4. 实现 discover_gradio_api
5. 实现 discover_openapi_schema
6. 每个 tool 写 evidence file
```

验收：

```text
probe_http 对本地服务真实发请求
discover_gradio_api 能读取 /config
tool_result.strong_verify_pass 只在 current trace_id 命中时为 true
```

### Phase 5：实现 AgentRuntime.act_verify

目标：

- Agent loop 真正插入 verify 主路径

具体任务：

```text
1. build observation
2. call planner
3. schema validation
4. critic gate
5. policy gate
6. execute tool
7. update state
8. persist artifacts
9. return AgentVerifyResult
```

验收：

```text
max_steps 生效
rejected step 记录清楚
executed step 有 evidence
strong_verify_pass 时提前停止
```

### Phase 6：接入 VerifyModule

目标：

- deterministic verify uncertain 后进入 Agent verify

具体任务：

```text
1. VerifyModule 支持 agent config/context
2. uncertain 时调用 act_verify
3. 合并 agent evidence
4. 扩展 VerifyResult metadata
5. report 展示 agent_verify 字段
```

验收：

```text
agent disabled 时行为不变
agent enabled 且 evidence 命中 trace 时 final status passed
agent planned but not executed 时 final status 不得变 passed
```

### Phase 7：实现真实 eval-compare

目标：

- 用 fixture 证明 LLM 必要性

具体任务：

```text
1. eval-compare 增加真实运行模式
2. 同一 target 分别跑 off / gated_actor
3. 汇总 run_summary
4. 标记 helped cases
5. 输出 comparison_report.json
```

验收：

```text
comparison_report 不是 unknown 占位
至少一个 target: off uncertain, gated_actor passed
至少一个 target: policy rejected
```

## 15. 防止过度包装的工程约束

### 15.1 llm_helped 判定

只能在以下条件全部满足时为 true：

```text
agent_mode == gated_actor
LLM produced valid tool_call
critic allowed
policy allowed
executor executed tool
tool_result improved final status
evidence path exists
```

以下情况必须为 false：

```text
audit mode
planner mode
LLM only produced suggestion
tool rejected
tool executed but没有改变状态
tool evidence missing
final status 没改善
```

### 15.2 self-healing 判定

只能在以下条件全部满足时为 true：

```text
repair action executed
pipeline resumed
final verify passed
repair_verified == true
```

以下情况不能叫 self-healing：

```text
生成 repair plan
只写 metadata
只 rerun
只诊断
人工执行修复
```

### 15.3 LLM-driven 判定

只能在以下条件满足时说 LLM-driven：

```text
LLM decision occurs before execution
LLM decision selects among multiple legal tools/actions
selected tool is actually executed
execution result changes subsequent state
artifact proves the chain
```

如果 LLM 只给建议或事后总结，只能说 LLM-assisted。

## 16. 面试表达口径

### 16.1 MVP 完成前

只能说：

> 项目已有 deployment pipeline、evidence-based verification、repair planning、memory 和 Agent audit artifacts。我正在把 verify uncertain recovery 改造成 runtime LLM tool loop，让 LLM 在不确定验证场景中选择 probe tool，并由 policy / critic 控制执行。

### 16.2 MVP 完成后

可以说：

> 实现了一个 LLM-driven verification sub-agent：当 deterministic verify 无法证明 AI demo 部署成功时，Agent 会基于服务状态、失败 evidence 和仓库文件选择下一步 probe tool；tool call 必须经过 schema、critic 和 policy 校验；执行结果写入 evidence，并且只有当前 trace-based proof 命中时才会把 verify 从 uncertain 升级为 passed。

### 16.3 不要说

不要写：

```text
实现了完全自主部署 Agent
实现了生产级 Agent 平台
实现了多 Agent 协作
支持大规模并发部署
支持真实 GPU 隔离调度
实现长期记忆和技能进化
实现端到端自修复 Agent
```

除非对应代码、测试和运行证据都补齐。

## 17. 简历 bullet 建议

MVP 完成后可写：

```text
Built an LLM-driven verification sub-agent for AI demo deployment: when deterministic checks were uncertain, the agent selected schema-validated local probe tools, passed them through critic/policy gates, executed evidence collection, and only marked success on current trace-based proof.
```

更工程化版本：

```text
Designed a gated Agent runtime for deployment verification, separating LLM decision-making from deterministic execution with typed tool-call schemas, localhost-only policy checks, critic rejection, evidence artifacts, and baseline-vs-agent evaluation.
```

不要写：

```text
Built a production-grade autonomous deployment Agent.
```

## 18. 最终验收清单

开发完成后，逐项检查：

- [ ] `AgentRuntime.audit()` 保留旧 post-hoc artifact 能力
- [ ] `AgentRuntime.act_verify()` 存在并由 verify uncertain 主路径调用
- [ ] LLM planner 输出严格 JSON tool_call
- [ ] invalid JSON 被拒绝
- [ ] unknown tool 被拒绝
- [ ] external URL 被拒绝
- [ ] missing trace template 被拒绝
- [ ] critic reject 会阻断执行
- [ ] policy reject 会阻断执行
- [ ] planner mode 不执行 tool
- [ ] gated_actor mode 执行 approved tool
- [ ] tool_result 写入 evidence file
- [ ] evidence file 含 current trace_id
- [ ] final verify pass 不依赖 LLM 自述
- [ ] `agent_verify_steps.jsonl` 可追溯完整链路
- [ ] `agent_verify.llm_helped` 只在状态改善时为 true
- [ ] `eval-compare` 至少有一个 off uncertain / gated_actor passed case
- [ ] `comparison_report.json` 不再是 unknown skeleton

## 19. 给执行模型的开发顺序

推荐严格按以下顺序执行，不要并行大改：

```text
1. 先读 runtime.py / verify.py / orchestrator.py / cli.py / config.py
2. 拆 AgentRuntime audit / act_verify，不改行为
3. 加 schema 和 parser
4. 加 ToolPolicy，并写 rejection tests
5. 加 ToolExecutor 和最小 probe_http
6. 加 discover_gradio_api
7. 接入 VerifyModule uncertain branch
8. 写 agent artifacts
9. 做一个 fixture 证明 uncertain -> passed
10. 再改 eval-compare
```

每一步都要保证：

```text
旧 CLI 不破坏
agent disabled 时行为不变
artifact 字段可解释
不要把 LLM 输出直接当 truth
```

## 20. 一句话成功标准

本轮优化真正成功的标准只有一个：

> 在一次真实运行中，deterministic verify 不能证明成功；LLM 在运行时选择了一个合法 probe tool；该 tool 经 critic / policy 后真实执行并产生当前 trace evidence；最终 verify 因这份 evidence 从 uncertain 变成 passed；完整链路可由 artifact 复现。

达不到这条，不建议声称“真正 LLM 驱动的自动部署 Agent”。
