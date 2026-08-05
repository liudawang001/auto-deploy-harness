# DeepSeek Provider 深度适配执行方案

> 文档状态：P0～P2 已实施并完成静态闭环修复；真实 API smoke 待外部执行，P3 未启用
> 编写日期：2026-08-05
> 适用项目：`auto-deploy-harness`
> 当前基线：项目已通过 `ProviderRegistry` 支持通用 OpenAI-compatible DeepSeek 调用，但尚未实现 DeepSeek V4 专用协议治理。
> 原始方案编写说明：当时仅定义执行方案，未修改业务代码、未运行测试、未调用真实 DeepSeek API。

> 2026-08-05 实施补记：专用 Provider、purpose 路由、Thinking、JSON Output、空响应单次恢复、结构化重试、HTTPS、deadline、上下文预算、Usage 与安全审计已经接入。`native_tool_calling=true` 在完整 P3 多轮协议实现前保持 fail-closed；readiness 不会把“已注册”误报为“已配置”或“live smoke 通过”。本补记只描述代码状态，不代表真实 API 验证已经完成。

---

## 0. 执行前结论

当前系统已经具备“能调用 DeepSeek”的基础能力：

```text
ProviderRegistry
  -> deepseek
  -> OpenAICompatibleProvider
  -> POST /chat/completions
  -> JSON Action
  -> Schema / Policy / ToolExecutor / Evidence Gate
```

但当前 `deepseek` 只是通用 OpenAI-compatible 别名，尚未覆盖：

1. DeepSeek V4 模型名称和能力注册。
2. Thinking Mode 与 `reasoning_effort`。
3. JSON Output 和空响应处理。
4. DeepSeek 结构化错误、限流重试和请求追踪。
5. `reasoning_content`、`finish_reason` 和精细 Usage。
6. 按 Plan、Agent、Memory 等用途选择模型。
7. DeepSeek Native Tool Calling 多轮协议。

本方案按两级完成标准实施：

```text
P0～P2：DeepSeek 稳定 JSON Action 适配
P3：DeepSeek Native Tool Calling 适配
```

P0～P2 完成前，不应宣称“深度适配 DeepSeek”；P3 完成前，不应宣称“使用 DeepSeek 原生 Tool Calling”。

---

## 1. 必读文件

执行前必须阅读并理解：

```text
src/auto_harness/providers/base.py
src/auto_harness/providers/registry.py
src/auto_harness/providers/openai_compatible.py
src/auto_harness/providers/xunfei.py
src/auto_harness/providers/interactive.py
src/auto_harness/config.py
src/auto_harness/cli.py
src/auto_harness/orchestrator.py
src/auto_harness/controllers/langgraph_deps.py
src/auto_harness/context/capabilities.py
src/auto_harness/context/executor.py
src/auto_harness/context/tokens.py
src/auto_harness/agent_runtime/plan_first_loop.py
src/auto_harness/agent_runtime/decision_gate.py
src/auto_harness/agent_runtime/policy.py
src/auto_harness/tools/registry.py
src/auto_harness/tools/executor.py
src/auto_harness/memory/curator.py
src/auto_harness/readiness.py
```

当前关键事实：

- `ProviderRegistry` 已把 `deepseek` 注册为内置名称。
- Agent、Plan-first、Memory Evolution、`llm-test` 和 Live Smoke 已通过 Registry 创建 Provider。
- 当前 `OpenAICompatibleProvider` 使用同步、非流式 `urllib` 请求。
- 当前 Provider 协议为 `json_action`，不是原生工具调用。
- `LLMCallExecutor` 已提供上下文预算和一次上下文溢出压缩重试。
- Schema、Policy、Tool Executor 和 Evidence Gate 不应随 Provider 更换而绕过。

---

## 2. 官方协议基线

实现时以 DeepSeek 官方文档为准：

- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [Create Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion)
- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)
- [Error Codes](https://api-docs.deepseek.com/quick_start/error_codes/)
- [Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit/)

截至本文日期，执行方案使用以下基线：

```text
OpenAI-compatible base URL: https://api.deepseek.com
Models:
  - deepseek-v4-flash
  - deepseek-v4-pro
Context length: 1M
Thinking Mode: 默认 enabled
```

旧模型名：

```text
deepseek-chat
deepseek-reasoner
```

已于 2026-07-24 停止使用。新实现必须对旧模型名给出明确配置错误，不能等到真实请求时才返回模糊 HTTP 错误。

---

## 3. 严格目标

### 3.1 P0～P2 目标

```text
1. DeepSeek 使用专用 Provider，而不是只作为通用别名。
2. 所有 DeepSeek 请求显式声明模型、Thinking Mode 和输出协议。
3. JSON Action 调用优先启用 JSON Output。
4. JSON 空响应、非法 JSON、截断和上下文溢出可被明确区分。
5. 401/402/422 不重试；429/500/503 和网络异常有界重试。
6. Plan、Agent、Memory Evolution 可以按 purpose 选择模型。
7. 调用证据包含 provider/model/purpose/protocol/usage/retry，不包含密钥、完整 Prompt 和完整推理内容。
8. DeepSeek 不可用时安全停止，不自动回退到 Mock 后继续执行副作用。
9. Provider 更换不得改变 Schema、Policy、ToolExecutor 和 Evidence Gate。
```

### 3.2 P3 目标

```text
1. 真正发送 DeepSeek tools schema。
2. 真正解析 tool_calls 和 tool_call_id。
3. Tool 参数仍需通过本地 Schema 与 Policy。
4. Tool 结果以 role=tool 返回模型。
5. Thinking Tool Calling 正确回传 reasoning_content。
6. 原生工具调用失败时不得降级为任意 Shell。
7. 最终成功仍只能由 Evidence Gate 判定。
```

---

## 4. 明确非目标

本轮不做：

```text
1. 不删除 Xunfei Provider，保留兼容和历史证据读取能力。
2. 不把 DeepSeek 设置为仓库默认真实 Provider；默认仍可保持 mock，避免无密钥启动失败和意外费用。
3. 不开放任意 Shell、任意文件写入或任意网络访问。
4. 不允许 LLM 直接判定部署成功。
5. 不保存完整 reasoning_content。
6. 不因 DeepSeek 支持 1M 上下文而取消现有上下文预算。
7. 不在第一阶段依赖 DeepSeek Beta Strict Tool Calls。
8. 不在 Provider 异常时自动切换 Mock 并继续执行。
9. 不把 Prompt 中要求 JSON 当成唯一结构化保障。
10. 不把“能解析 tool_calls 字段”描述成“已支持原生 Tool Calling”。
```

---

## 5. 核心架构决策

### 5.1 保留通用传输，增加厂商语义层

目标结构：

```text
ProviderRegistry
  -> DeepSeekProvider
       -> OpenAI-compatible HTTP transport
       -> DeepSeek request builder
       -> DeepSeek response parser
       -> DeepSeek error classifier
       -> DeepSeek capability profile
```

不要复制整份网络请求代码。优先把 `OpenAICompatibleProvider` 拆成可复用的小方法：

```python
_build_payload(...)
_build_headers(...)
_perform_request(...)
_parse_response(...)
_classify_http_error(...)
```

`DeepSeekProvider` 继承或组合通用 Transport，只覆盖厂商差异。

### 5.2 P0～P2 继续使用 JSON Action

现有安全链路已经围绕 JSON Action 建立：

```text
LLM text
  -> JSON parser
  -> stage schema
  -> critic
  -> policy
  -> typed executor
```

第一阶段不要同时迁移 Provider 和 Agent 协议，否则难以区分故障来自网络、DeepSeek 输出还是 Tool Calling 状态机。

### 5.3 Purpose 驱动模型与推理模式

当前 Factory 已接收 `purpose`，但通用适配器没有利用。DeepSeekProvider 必须保存 purpose，并据此解析：

```text
agent
plan_first
memory_evolution
llm_test
live_smoke
```

建议默认映射：

| Purpose | 模型 | Thinking | JSON Mode |
|---|---|---:|---:|
| `plan_first` | `deepseek-v4-pro` | enabled/high | true |
| `agent` | `deepseek-v4-flash` | disabled | true |
| `memory_evolution` | `deepseek-v4-flash` | disabled | true |
| `llm_test` | `deepseek-v4-flash` | disabled | false |
| `live_smoke` | `deepseek-v4-flash` | disabled | true |

默认值必须允许配置覆盖。

### 5.4 明确成功判定边界

```text
DeepSeek 请求成功 ≠ Plan 合法
Plan 合法 ≠ Policy 允许
Policy 允许 ≠ Tool 执行成功
Tool 执行成功 ≠ 部署成功
部署成功 = Evidence Gate 证明当前 trace 被真实处理
```

---

## 6. 配置契约

### 6.1 推荐配置

```json
{
  "agent_provider": "deepseek",
  "agent_plan_first_provider": "deepseek",
  "memory_evolution_provider": "deepseek",
  "provider_configs": {
    "deepseek": {
      "api_base": "https://api.deepseek.com",
      "api_key_env": "DEEPSEEK_API_KEY",
      "models": {
        "agent": "deepseek-v4-flash",
        "plan_first": "deepseek-v4-pro",
        "memory_evolution": "deepseek-v4-flash",
        "llm_test": "deepseek-v4-flash",
        "live_smoke": "deepseek-v4-flash"
      },
      "thinking": {
        "agent": "disabled",
        "plan_first": "enabled",
        "memory_evolution": "disabled",
        "llm_test": "disabled",
        "live_smoke": "disabled"
      },
      "reasoning_effort": {
        "plan_first": "high"
      },
      "json_mode": {
        "agent": true,
        "plan_first": true,
        "memory_evolution": true,
        "llm_test": false,
        "live_smoke": true
      },
      "context_window_tokens": 65536,
      "max_tokens": 4096,
      "timeout_seconds": 60,
      "max_retries": 2,
      "retry_base_seconds": 1.0,
      "retry_max_seconds": 8.0,
      "require_api_key": true,
      "allow_beta": false,
      "native_tool_calling": false
    }
  }
}
```

### 6.2 密钥规则

只允许：

```bash
export DEEPSEEK_API_KEY="..."
```

禁止：

```json
{"api_key": "sk-..."}
```

现有 `HarnessConfig` 的明文 Secret 拒绝规则必须继续保留。

### 6.3 配置校验

在 `src/auto_harness/config.py` 增加 DeepSeek 专项校验：

```text
api_base:
  - 必须为 https
  - 默认只允许 api.deepseek.com
  - /beta 需要 allow_beta=true

models:
  - 每个值必须是非空字符串
  - 显式拒绝 retired model names

thinking:
  - 仅 enabled / disabled

reasoning_effort:
  - 仅 high / max

json_mode/native_tool_calling/allow_beta:
  - 必须为 bool

timeout/max_tokens/max_retries/retry interval:
  - 必须为合法正数或允许的零值
```

未知模型默认不直接硬编码拒绝，但必须满足以下之一：

```text
1. 已存在于 DeepSeek 能力表；或
2. 操作者显式配置 context_window_tokens 和 max_tokens，并设置 allow_unknown_model=true。
```

---

## 7. 数据契约调整

### 7.1 Message

修改：

```text
src/auto_harness/providers/base.py
```

从：

```python
@dataclass
class Message:
    role: str
    content: str
```

演进为兼容旧调用的结构：

```python
@dataclass
class Message:
    role: str
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
```

P0～P2 只使用 `role/content`；新增字段为 P3 预留。

### 7.2 LLMResult

增加：

```python
reasoning_content: str = ""
finish_reason: str = ""
request_id: str = ""
provider_name: str = ""
provider_model: str = ""
retry_count: int = 0
```

保留：

```python
text
usage
latency_ms
protocol
tool_calls
context
```

约束：

- `text` 只保存最终 `message.content`。
- `reasoning_content` 默认不写入运行 artifact。
- `raw` 不得未经脱敏直接进入报告。
- `protocol` 只能根据真实请求标记为 `json_action` 或 `native_tool_call`。

### 7.3 ProviderError

新增文件或定义：

```text
src/auto_harness/providers/errors.py
```

建议结构：

```python
class ProviderError(RuntimeError):
    provider_name: str
    status_code: int | None
    error_code: str
    category: str
    retryable: bool
    request_id: str
    safe_detail: str
```

`safe_detail` 必须截断并脱敏。

错误类别至少包含：

```text
invalid_request
authentication_failed
insufficient_balance
invalid_parameter
rate_limited
server_error
server_overloaded
network_timeout
network_error
context_overflow
empty_content
invalid_response
```

---

## 8. Phase 0：基线与兼容边界冻结

### 8.1 目标

在修改 Provider 前冻结现有契约，避免把原有缺陷和 DeepSeek 改造混在一起。

### 8.2 任务

- [ ] 记录现有 Provider 名称、配置优先级和调用 purpose。
- [ ] 记录 `OpenAICompatibleProvider` 当前请求与返回字段。
- [ ] 记录所有 `provider.complete()` 调用点。
- [ ] 记录 `LLMCallExecutor` 的上下文溢出重试契约。
- [ ] 明确哪些 artifact 会保存 `LLMResult.raw`。
- [ ] 明确默认配置仍为 `mock`，不在本阶段更改。

### 8.3 产物

```text
docs/evidence/deepseek-provider-baseline.json
```

至少记录：

```json
{
  "provider_protocol": "json_action",
  "default_provider": "mock",
  "deepseek_adapter": "openai_compatible",
  "native_tool_calling": false,
  "real_api_called": false
}
```

### 8.4 验收

- 不修改运行行为。
- 不包含任何 Secret。
- 基线明确区分代码能力和真实 API 证据。

---

## 9. Phase 1：DeepSeek 专用 Provider 与模型能力

### 9.1 新增文件

```text
src/auto_harness/providers/deepseek.py
src/auto_harness/providers/errors.py
```

### 9.2 修改文件

```text
src/auto_harness/providers/registry.py
src/auto_harness/providers/__init__.py
src/auto_harness/providers/openai_compatible.py
src/auto_harness/config.py
src/auto_harness/context/capabilities.py
configs/default.json
README.md
```

### 9.3 任务

- [ ] 从通用适配器提取可复用 HTTP 请求方法。
- [ ] 实现 `DeepSeekProvider`，保存 `purpose`。
- [ ] 按 purpose 解析模型、Thinking 和 JSON Mode。
- [ ] 默认 API Base 为 `https://api.deepseek.com`，仍允许显式覆盖。
- [ ] 对 retired 模型名返回确定性配置错误。
- [ ] 注册 V4 Flash/Pro 能力。
- [ ] `ProviderRegistry` 将 `deepseek` 映射到 `DeepSeekProvider`。
- [ ] 其他 OpenAI-compatible Provider 行为保持不变。

### 9.4 输入、处理、输出

```text
输入：HarnessConfig + provider name + purpose + environment secret
处理：解析配置 -> 校验模型 -> 解析能力 -> 创建 DeepSeekProvider
输出：已绑定 purpose/model/capability 的 Provider
异常：配置缺失或退休模型在请求前失败
```

### 9.5 验收

- `deepseek` 不再实例化为普通 `OpenAICompatibleProvider`。
- `qwen/vllm/ollama` 等现有别名不受影响。
- 缺少 key、model、context 能力时给出具体字段名。
- 旧 DeepSeek 模型名不发起网络请求。

---

## 10. Phase 2：Thinking、JSON Output 与响应解析

### 10.1 请求构造

DeepSeek JSON Action 请求建议：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [],
  "max_tokens": 4096,
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high",
  "response_format": {"type": "json_object"}
}
```

规则：

```text
thinking=enabled:
  - 不发送 temperature
  - reasoning_effort 必须为 high/max

thinking=disabled:
  - 可发送 temperature

json_mode=true:
  - Prompt 必须明确包含 json 要求
  - 请求增加 response_format

json_mode=false:
  - 不增加 response_format
```

### 10.2 响应解析

解析：

```text
choices[0].message.content
choices[0].message.reasoning_content
choices[0].message.tool_calls
choices[0].finish_reason
usage
response id / request id
model
```

状态转换：

```text
HTTP 200
  -> JSON body 合法
     -> content 非空
        -> LLMResult
     -> content 为空
        -> empty_content
        -> 最多重试一次
  -> JSON body 非法
     -> invalid_response
     -> 不进入 Parser/Policy
```

### 10.3 空内容与非法 JSON

JSON Mode 空内容重试必须满足：

```text
1. 只重试一次。
2. 使用更短的 retry prompt。
3. 记录第一次 finish_reason 和 usage。
4. 第二次仍失败则安全停止。
5. 不切换 mock。
6. 不复用旧 Plan 冒充新响应。
```

### 10.4 Reasoning 隐私

禁止将完整 `reasoning_content` 写入：

```text
runs/*/events.jsonl
runs/*/reports/*.json
runs/*/logs/agent_calls/*
dashboard
deployment package
```

只允许记录：

```json
{
  "thinking_enabled": true,
  "reasoning_present": true,
  "reasoning_chars": 1234,
  "reasoning_sha256": "..."
}
```

P3 Tool Calling 多轮处理中，`reasoning_content` 只能在内存中的当前 Provider 会话里暂存。

### 10.5 验收

- Thinking 开关在请求体中显式存在。
- Thinking 开启时不依赖 temperature 控制确定性。
- JSON Action 使用 JSON Output。
- 空响应不会进入业务 Parser。
- 完整 reasoning 不进入磁盘 artifact。

---

## 11. Phase 3：错误、超时、重试与限流

### 11.1 错误矩阵

| HTTP/异常 | Category | Retry | 任务行为 |
|---|---|---:|---|
| 400 普通格式错误 | `invalid_request` | 否 | 停止并记录安全摘要 |
| 400 上下文溢出 | `context_overflow` | 由 Context Executor 最多一次 | 压缩后重试 |
| 401 | `authentication_failed` | 否 | 立即停止 |
| 402 | `insufficient_balance` | 否 | 立即停止 |
| 422 | `invalid_parameter` | 否 | 立即停止 |
| 429 | `rate_limited` | 是 | 有界退避 |
| 500 | `server_error` | 是 | 有界退避 |
| 503 | `server_overloaded` | 是 | 有界退避 |
| Connect/Read Timeout | `network_timeout` | 是 | 有界退避 |
| DNS/连接失败 | `network_error` | 是 | 有界退避 |

### 11.2 重试算法

```text
max_retries = 2
delay = min(retry_max, retry_base * 2^attempt) + jitter
优先使用 Retry-After
```

约束：

- 每次重试记录新的 transport attempt。
- 同一逻辑调用保留稳定 `call_id`。
- 记录潜在重复计费风险。
- 400/401/402/422 禁止退避重试。
- Provider 重试与 Context Overflow 重试分别计数。
- 总尝试次数必须受调用 deadline 限制。

### 11.3 超时治理

统一：

```text
Agent call deadline
  >= Provider total timeout
     >= 单次 network timeout
```

禁止出现：

```text
agent_decision_timeout_seconds = 60
provider timeout_seconds = 180
```

但外层 60 秒提前取消、Provider 配置实际失效的情况。

建议引入：

```python
ProviderRequestContext(
    call_id,
    purpose,
    deadline_at,
    task_id_hash,
)
```

Provider 每次重试前计算剩余预算。

### 11.4 自动 Fallback

默认：

```text
DeepSeek 失败 -> provider_unavailable / provider_rejected -> 当前 LLM 分支停止
```

禁止：

```text
DeepSeek 401/402/429 -> 自动切 mock -> 继续执行真实副作用
```

如果未来加入多 Provider fallback，必须：

1. 显式配置 fallback 列表。
2. 记录 provider 切换原因。
3. 新 Provider 重新生成 Plan。
4. 新 Plan 重新经过完整 Policy。
5. 禁止复用旧 Provider 未完成的 Tool Call。

### 11.5 验收

- 所有 DeepSeek 错误转换为 `ProviderError`。
- 上层可以区分 auth、balance、rate-limit、server、context。
- 限流与服务异常有界重试。
- Secret 不进入错误信息。
- Fallback 不会绕过 Policy。

---

## 12. Phase 4：上下文能力、Usage 与成本遥测

### 12.1 ProviderCapabilities

DeepSeekProvider 应暴露：

```python
ProviderCapabilities(
    provider_name="deepseek",
    model="deepseek-v4-pro",
    context_window_tokens=1_000_000,
    max_output_tokens=384_000,
    supports_tool_calling=False,
    supports_json_mode=True,
    supports_thinking=True,
    supports_streaming=False,
    source="deepseek_model_registry",
)
```

这里报告的是当前适配器可直接调用的能力，不是模型 API 的理论能力。只有实现并接通 `complete_with_tools()`/流式入口后，才能分别改为 `True`。

需要扩展 `ProviderCapabilities` 数据类以容纳新字段。

### 12.2 能力上限与运行预算分离

```text
Provider capability: 1M
Project operational budget: 例如 64K
Effective budget: min(provider capability, project budget)
```

默认不要把项目输入预算直接提升到 1M。

推荐保留：

```json
{
  "agent_context_window_tokens": 65536,
  "agent_context_reserved_output_tokens": 4096,
  "agent_context_skill_budget_tokens": 2000,
  "agent_context_memory_budget_tokens": 2000
}
```

### 12.3 Usage 归一化

继续支持：

```text
prompt_tokens -> input_tokens
completion_tokens -> output_tokens
total_tokens
```

DeepSeek 特有缓存字段如果响应提供，则扩展保存：

```text
prompt_cache_hit_tokens
prompt_cache_miss_tokens
```

记录：

```json
{
  "provider": "deepseek",
  "model": "deepseek-v4-pro",
  "purpose": "plan_first",
  "input_tokens": 1000,
  "output_tokens": 300,
  "cache_hit_tokens": 500,
  "latency_ms": 1500,
  "retry_count": 0
}
```

### 12.4 Token 估算

当前 UTF-8 字节上界估算可以保留为 fail-safe，但要明确：

- 它不是 DeepSeek 官方 tokenizer。
- 它可能过度压缩中文和代码。
- Provider 返回 Usage 后必须用真实值做离线校准。
- 没有可靠 tokenizer 时不要声称 Token 预算精确。

### 12.5 验收

- DeepSeek 模型能力不需要每次人工重复填写。
- 未知模型 fail closed 或需要显式能力配置。
- 运行预算不会因 1M 能力自动放大。
- Usage 字段进入统一指标但不泄露 Prompt。

---

## 13. Phase 5：调用点、Purpose 与可观测性闭环

### 13.1 调用点治理

检查并统一：

```text
TaskRunner._create_plan_first_provider()
TaskRunner._create_agent_provider()
MemoryEvolutionManager Provider
llm-test
agent-live-smoke
LangGraphControllerDependencies
Decision Gate
Verify Planner
```

不得在业务模块内出现：

```python
if provider == "deepseek": ...
```

厂商差异只应存在于 Provider、Capabilities 和 Config 层。

### 13.2 ProviderRequestContext

建议让 `LLMCallExecutor.execute()` 向 Provider 传入安全上下文：

```python
ProviderRequestContext(
    call_id="...",
    call_site="agent.plan_first",
    stage="plan_first",
    purpose="plan_first",
    task_scope_hash="...",
    requested_output_tokens=4096,
    deadline_at="...",
)
```

不要传真实仓库 URL、用户名或租户隐私。

### 13.3 user_id

如启用 DeepSeek `user_id`：

```python
user_id = sha256(stable_installation_id + task_scope).hexdigest()[:32]
```

约束：

- 仅包含 `[a-zA-Z0-9-_]`。
- 不含姓名、邮箱、仓库地址、Token。
- 同一任务稳定，不同任务不可反推。

### 13.4 运行证据

每次调用记录：

```text
provider_name
provider_model
purpose
protocol
thinking mode
reasoning effort
json mode
request id
call id
latency
finish reason
retry count
HTTP status category
token usage
context telemetry
```

禁止记录：

```text
API Key
Authorization Header
完整 Prompt
完整仓库源码
完整 reasoning_content
未脱敏 HTTP error body
```

### 13.5 Readiness

修改：

```text
src/auto_harness/readiness.py
```

增加 DeepSeek readiness：

```json
{
  "provider": "deepseek",
  "configured": true,
  "model_supported": true,
  "protocol": "json_action",
  "thinking_configured": true,
  "json_mode_configured": true,
  "native_tool_calling": false,
  "live_smoke_status": "not_run"
}
```

`configured=true` 不能等同于 `live_smoke=passed`。

---

## 14. Phase 6：DeepSeek Native Tool Calling

> 本阶段为独立里程碑。P0～P2 稳定前不得开始。

### 14.1 协议入口

实现：

```python
DeepSeekProvider.complete_with_tools(
    messages,
    tools,
    tool_choice="auto",
    request_context=None,
)
```

只有真实发送 `tools` 且解析真实 `tool_calls` 后，`protocol` 才能写：

```text
native_tool_call
```

### 14.2 Tool Schema 转换

输入必须来自现有 ToolRegistry：

```text
ToolRegistry schema
  -> DeepSeek function tool schema
  -> request tools[]
```

禁止 LLM 自己声明工具。

每个 Tool 必须包含：

```text
name
description
parameters JSON Schema
risk metadata（仅本地）
allowed stages（仅本地）
```

DeepSeek 请求只获得最小必要 Schema；风险、权限和内部实现不能交给模型修改。

### 14.3 单轮状态机

```text
LLM response
  -> tool_calls
  -> preserve tool_call_id
  -> parse arguments JSON
  -> local Schema
  -> Critic
  -> Policy
  -> ToolExecutor
  -> role=tool result
  -> next DeepSeek request
  -> final content or next tool call
```

### 14.4 Thinking Tool Calling

Thinking 模式有工具调用时，必须在当前轮后续请求中回传：

```text
assistant.content
assistant.reasoning_content
assistant.tool_calls
tool result
```

但 `reasoning_content` 只允许保留在内存会话中，不写普通日志。

### 14.5 并行工具调用

第一版：

```text
parallel_tool_calls = false
每轮最多一个 Tool Call
```

原因：

- 当前 Policy 和 Agent State 主要按串行 effect 设计。
- 并行副作用会扩大幂等、资源锁和状态合并复杂度。
- 部署工具存在共享 workspace、环境、端口和进程资源。

### 14.6 Tool 错误反馈

返回模型的 Tool Result 必须是结构化、脱敏和有界内容：

```json
{
  "status": "failed",
  "category": "dependency_conflict",
  "summary": "...",
  "artifact_ref": "...",
  "retryable": false
}
```

不得发送完整 stdout/stderr、Secret 或任意本地文件。

### 14.7 停止条件

```text
max tool turns
max repeated tool signature
max provider calls
max total deadline
Policy reject
Evidence pass
no progress
```

### 14.8 验收

- 请求真实包含 `tools`。
- 返回真实包含 `tool_calls` 和 `tool_call_id`。
- Tool 参数通过本地 Schema/Policy。
- Tool 结果正确回传。
- Thinking Tool Call 回传 reasoning_content。
- 重复 Tool Call 被循环检测阻断。
- 最终成功仍由 Evidence Gate 裁决。

---

## 15. Phase 7：真实 API 验证与发布

### 15.1 验证层级

```text
Level 1：离线请求/响应契约
Level 2：本地 Fake HTTP Server 集成
Level 3：真实 DeepSeek llm-test
Level 4：真实 DeepSeek Agent live smoke
Level 5：固定仓库矩阵对照评测
```

### 15.2 真实 Smoke 最小闭环

目标 Fixture：

```text
本地轻量仓库
  -> DeepSeek 生成 Plan 或 Repair
  -> Schema/Policy 通过
  -> 类型化 Tool 执行
  -> 服务启动
  -> 新 trace Evidence pass
```

Manifest 至少保存：

```json
{
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "protocol": "json_action",
  "thinking": "disabled",
  "request_count": 2,
  "retry_count": 0,
  "policy_gated": true,
  "trace_verified": true,
  "secret_persisted": false
}
```

### 15.3 对照评测

固定相同：

```text
仓库
硬件
Runtime Policy
Skill 版本
Memory 版本
Prompt profile
上下文预算
超时
```

比较：

```text
deterministic baseline
DeepSeek JSON Action
DeepSeek Native Tool Calling（P3 后）
```

指标：

```text
Plan Schema 通过率
Policy 接受率
非法动作率
首次验证通过率
Repair 后验证通过率
无进展循环率
平均 LLM 调用数
P50/P95 延迟
输入/输出 Token
估算成本
429/5xx 比例
```

### 15.4 发布顺序

```text
1. 默认关闭 DeepSeek。
2. llm-test 手工启用。
3. live-smoke 固定 fixture。
4. dry-run Plan-first。
5. 低风险本地 execute。
6. 固定真实仓库集。
7. 再考虑设为个人默认 Provider。
```

---

## 16. 文件改动矩阵

| 文件 | 主要改动 | 阶段 |
|---|---|---|
| `src/auto_harness/providers/deepseek.py` | DeepSeek 请求、响应、能力、Thinking、JSON Mode | P0 |
| `src/auto_harness/providers/errors.py` | 结构化 ProviderError | P0 |
| `src/auto_harness/providers/openai_compatible.py` | 提取通用 Transport 与可覆盖方法 | P0 |
| `src/auto_harness/providers/registry.py` | `deepseek -> DeepSeekProvider` | P0 |
| `src/auto_harness/providers/__init__.py` | 导出 DeepSeekProvider/ProviderError | P0 |
| `src/auto_harness/providers/base.py` | 扩展 Message、LLMResult、Tool Calling 契约 | P0/P3 |
| `src/auto_harness/config.py` | DeepSeek 专项配置校验 | P0 |
| `configs/default.json` | 示例字段或保持空配置并补注释文档 | P0 |
| `src/auto_harness/context/capabilities.py` | DeepSeek 模型能力解析 | P1 |
| `src/auto_harness/context/executor.py` | ProviderError、deadline、overflow 协调 | P1 |
| `src/auto_harness/context/tokens.py` | DeepSeek Usage/Cache 字段归一化 | P1 |
| `src/auto_harness/orchestrator.py` | purpose/request context 传递 | P1 |
| `src/auto_harness/controllers/langgraph_deps.py` | 调用 purpose 和 Provider context 对齐 | P1 |
| `src/auto_harness/agent_runtime/*` | 可选 native tool call loop | P3 |
| `src/auto_harness/readiness.py` | DeepSeek readiness 与证据边界 | P1 |
| `src/auto_harness/cli.py` | DeepSeek smoke/配置摘要，不输出密钥 | P1 |
| `README.md` | 当前模型、配置、协议边界 | P0/P3 |

---

## 17. 测试规划

> 本节定义后续执行时需要编写的测试；本文编写过程中不运行测试。

### 17.1 单元测试

建议新增：

```text
tests/test_deepseek_provider.py
tests/test_deepseek_errors.py
tests/test_deepseek_config.py
tests/test_deepseek_context.py
tests/test_deepseek_tool_calling.py
```

覆盖：

```text
1. V4 Flash/Pro 模型选择。
2. 退休模型拒绝。
3. Purpose-specific 模型。
4. Thinking enabled/disabled 请求体。
5. Thinking enabled 时不发送 temperature。
6. JSON Mode 请求体。
7. content/reasoning_content/finish_reason 解析。
8. 空 content 有界重试。
9. 非 JSON 响应拒绝。
10. Secret 不进入 payload 和 error。
11. 401/402/422 不重试。
12. 429/500/503 重试。
13. Retry-After 优先。
14. Context Overflow 交给 Context Executor。
15. Usage/Cache token 归一化。
16. reasoning 不进入 trace artifact。
```

### 17.2 Registry 和主链测试

```text
1. deepseek 创建 DeepSeekProvider。
2. 其他 provider 不受影响。
3. Plan-first 使用 purpose=plan_first。
4. Agent 使用 purpose=agent。
5. Memory Evolution 使用 purpose=memory_evolution。
6. 未配置 DeepSeek 时 readiness 准确显示 configured=false。
7. Provider 失败不切换 Mock。
```

### 17.3 Tool Calling 测试

P3 后覆盖：

```text
1. 请求包含真实 tools schema。
2. tool_call_id 保留。
3. 参数 JSON 解析失败。
4. 未注册工具拒绝。
5. Policy reject 不执行。
6. Tool Result 回传。
7. Thinking reasoning_content 回传。
8. 同一 Tool Call 循环阻断。
9. 并行 Tool Call 默认拒绝或串行化。
10. Evidence pass 后停止调用。
```

### 17.4 Live 测试边界

真实 API 测试必须：

- 通过环境变量注入 Key。
- 默认手工触发，不进入普通离线测试。
- 设置调用次数和 Token 上限。
- 生成无 Secret Manifest。
- 失败时标记 `failed/skipped`，不能伪造 `passed`。

---

## 18. 安全要求

### 18.1 Secret

```text
1. API Key 只在进程环境或交互式内存中存在。
2. 配置只保存 api_key_env。
3. HTTP headers 不写日志。
4. Error body 先脱敏再截断。
5. Deployment package 排除所有 Secret 来源。
```

### 18.2 Prompt 与仓库数据

```text
1. README、源码、日志继续视为不可信数据。
2. Provider 切换不能降低 Prompt Injection Policy。
3. 发送前继续经过 Context Governance 与 Redaction。
4. 不因为 DeepSeek 支持长上下文而上传整个仓库。
5. user_id 不包含个人隐私或仓库信息。
```

### 18.3 Tool 权限

```text
DeepSeek Tool Call
  != 已批准工具调用

必须继续经过：
ToolRegistry -> Schema -> Critic -> Policy -> Executor
```

---

## 19. 迁移策略

### 19.1 兼容旧配置

继续支持：

```json
{
  "provider_configs": {
    "deepseek": {
      "api_base": "https://api.deepseek.com",
      "model": "deepseek-v4-flash",
      "api_key_env": "DEEPSEEK_API_KEY",
      "context_window_tokens": 65536
    }
  }
}
```

如果只有单个 `model`，将其作为所有 purpose 的默认模型。

### 19.2 配置优先级

建议保持当前原则：

```text
DeepSeek 专用环境变量
  > 通用 AUTO_HARNESS_LLM 环境变量
  > provider_configs.deepseek
  > 内置非敏感默认值
```

API Key 不允许内置或从配置明文读取。

### 19.3 Xunfei

- 保留 `XunfeiSparkProvider`。
- 不自动迁移历史任务的 provider 字段。
- 新 DeepSeek 任务不得要求存在任何 `XUNFEI_*` 环境变量。
- 历史 Xunfei evidence 继续按原 provider/protocol 展示。

---

## 20. 回滚方案

### 20.1 P0～P2 回滚

保留通用适配器路径：

```text
deepseek 专用实现异常
  -> 配置切换为 openai_compatible
  -> 显式配置相同 DeepSeek endpoint
```

该回滚只能由操作者修改配置触发，不能在单次运行中自动发生。

### 20.2 P3 回滚

配置：

```json
{
  "native_tool_calling": false
}
```

回到：

```text
json_action -> local schema/policy/executor
```

运行中的任务固定启动时的 Provider protocol，不允许中途从 native 切回 JSON Action 后继续同一 Tool 会话。

### 20.3 数据兼容

- 新字段全部使用默认值，旧 `LLMResult`/Message 构造保持兼容。
- Artifact 增加 `schema_version`。
- 未识别的新 Provider artifact 不影响旧报告读取。

---

## 21. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| DeepSeek 模型名再次变化 | 配置失效 | 模型能力表、清晰错误、README 更新 |
| Thinking 默认行为变化 | 输出不稳定 | 每次请求显式发送 thinking |
| JSON Mode 返回空内容 | Plan 缺失 | 一次有界重试，随后安全停止 |
| 429/503 | 任务中断 | Retry-After、退避、并发限制 |
| 非流式长等待 | Worker 长时间占用 | deadline、超时协调、后续可选 streaming |
| reasoning 泄露 | 隐私与安全风险 | 只保存在内存，artifact 仅存 hash/长度 |
| Native Tool Call 绕过 Policy | 高风险副作用 | ToolRegistry/Schema/Policy 强制复用 |
| 自动 fallback 行为不一致 | 错误 Plan 被执行 | 默认禁止自动 fallback |
| 1M 上下文导致成本失控 | 延迟和费用上升 | 保持项目 64K 等业务预算 |
| 通用 Provider 回归 | 其他厂商受影响 | Transport 重构与厂商适配分层 |

---

## 22. 实施顺序与提交边界

建议拆分提交，禁止一个大提交同时完成全部阶段。

### Commit 1：数据契约与错误类型

```text
providers/base.py
providers/errors.py
兼容性单元测试
```

### Commit 2：DeepSeekProvider 与 Registry

```text
providers/deepseek.py
providers/openai_compatible.py
providers/registry.py
providers/__init__.py
```

### Commit 3：配置与模型能力

```text
config.py
context/capabilities.py
README.md
```

### Commit 4：Thinking、JSON Mode、响应解析

```text
deepseek.py
相关 Parser/Telemetry
```

### Commit 5：错误、重试、Deadline

```text
providers/errors.py
deepseek.py
context/executor.py
```

### Commit 6：Usage、Readiness、证据

```text
context/tokens.py
readiness.py
cli.py
```

### Commit 7：真实 JSON Action Live Smoke

```text
docs/evidence/deepseek-*.json
README.md
```

### Commit 8～N：Native Tool Calling

单独开发、单独开关、单独验证，不能和 JSON Action 稳定适配混在同一提交。

---

## 23. Definition of Done

### 23.1 DeepSeek 稳定 JSON Action 适配完成

- [ ] `deepseek` 创建专用 DeepSeekProvider。
- [ ] 使用有效 V4 模型。
- [ ] Purpose-specific 模型配置生效。
- [ ] Thinking Mode 显式配置。
- [ ] JSON Action 启用 JSON Output。
- [ ] 空内容和非法响应安全处理。
- [ ] DeepSeek 错误结构化。
- [ ] 429/500/503 有界重试。
- [ ] 上下文溢出接入已有压缩重试。
- [ ] Provider Usage 和调用元数据可审计。
- [ ] API Key、完整 Prompt、reasoning 不落盘。
- [ ] Agent/Plan-first/Memory 全部可选择 DeepSeek。
- [ ] DeepSeek 失败不自动切 Mock。
- [ ] 真实 Live Smoke 生成无密钥证据。
- [ ] 最终成功由 Evidence Gate 判定。

### 23.2 DeepSeek Native Tool Calling 完成

- [ ] 请求真实发送 `tools`。
- [ ] 返回真实解析 `tool_calls`。
- [ ] 保留并回传 `tool_call_id`。
- [ ] Thinking Tool Call 回传 `reasoning_content`。
- [ ] Tool 参数经过本地 Schema 与 Policy。
- [ ] Tool 执行结果结构化回传。
- [ ] 并行调用策略明确且默认安全。
- [ ] Tool 循环和预算限制生效。
- [ ] Protocol 证据准确标记 `native_tool_call`。
- [ ] Native 模式可通过配置回滚到 JSON Action。

---

## 24. 最终项目口径

P0～P2 完成后可以表述：

> 项目通过专用 DeepSeek Provider 接入 V4 模型，支持按 Agent 用途选择模型、Thinking Mode、JSON Output、上下文预算、结构化错误、限流重试和调用审计；模型输出仍需经过本地 Schema、Policy 和 Tool Executor，最终部署成功由 Evidence Gate 判定。

P3 完成后才可以补充：

> 项目同时支持 DeepSeek 原生 Tool Calling，完整维护 tool_call_id、工具结果消息和 Thinking reasoning 上下文，但工具权限与执行仍由本地确定性控制层裁决。

在真实 Live Smoke 和固定仓库评测完成前，不能宣称：

```text
生产稳定运行
大规模成功率提升
DeepSeek 原生 Tool Calling 已完整落地
DeepSeek V4 在所有仓库上优于其他 Provider
```
