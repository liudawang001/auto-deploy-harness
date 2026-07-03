# AI-Auto-Harness 开发进度

## 2026-07-03

### 已完成

- 通过 GitHub CLI 克隆 `liudawang001/ai-auto-harness`。
- 初始化 Python package 和 `pyproject.toml`。
- 添加 README、`.gitignore`、`.env.example` 和默认配置。
- 实现核心数据模型：
  - `TaskSpec`
  - `TaskState`
  - `StageResult`
  - `VerifyResult`
- 实现状态持久化：
  - `task.json`
  - `state.json`
  - `events.jsonl`
  - 各阶段 result 文件
- 实现 `auto-harness` CLI：
  - `init`
  - `deploy`
  - `resume`
  - `status`
  - `report`
  - `llm-test`
- 实现 LLM provider 抽象：
  - `MockLLMProvider`
  - `XunfeiSparkProvider`，支持 Anthropic-compatible payload，并通过环境变量读取配置。
- 实现 Agent executor 抽象：
  - `AgentExecutor`
  - `ClaudeCodeExecutor`
- 实现第一版核心模块：
  - `ProjectAnalyzer`
  - `EnvDeployModule`
  - `RunnerModule`
  - `VerifyModule`
  - `ReportGenerator`
- 实现 `TaskRunner` 编排，支持 dry-run pipeline。
- 添加初始 stdlib 单测，覆盖：
  - state store roundtrip
  - analyzer 框架识别
  - verify 在缺少证据时不会 false pass
  - mock LLM provider
- 实现安全默认行为：`deploy` 默认 dry-run，除非显式传入 `--execute`。
- 添加显式执行开关：
  - `--allow-install`
  - `--allow-start`
- 支持本地路径作为 `--repo` 输入，并复制到隔离 run workspace。
- 使用临时 Gradio 风格 demo 验证本地路径 dry-run：
  - 成功识别 `gradio`
  - 生成 venv/pip install plan
  - 生成 `app.py` run candidate
  - `verify` 保持 `uncertain`，因为 dry-run 没有真实 trace 证据
- 实现第一版 HTTP trace verify：
  - 从 `verify_hint.endpoint` 或 runner endpoint candidate 选择 endpoint
  - 为请求追加 `_auto_harness_trace=<trace_id>`
  - 将 request/response evidence 写入 `evidence/`
  - 只有响应体证明当前 trace 被处理，才允许通过
  - 不把 HTTP 200 当成成功
- 扩展 HTTP trace verify：
  - 支持 GET query trace
  - 支持 POST JSON trace template
  - 为 Gradio 生成默认 `/api/predict` POST verify hint
- 将 `ClaudeCodeExecutor` 作为可选 advisor 接入 `ProjectAnalyzer`。
  - 默认关闭
  - 通过 `AUTO_HARNESS_USE_AGENT_ANALYZER=1` 启用
  - advice 只作为元数据存储，不能绕过确定性 analyzer 输出
- 为 `env_deploy` 和 `runner` 增加命令策略检查。
  - 如果 executable 不在 `allowed_commands` 中，执行前直接拒绝。
  - 这是启用更广泛 `--execute` 前必须具备的安全能力。
- 增加仓库内置 skill 加载：
  - skill 位于 `skills/*/SKILL.md`
  - `SkillRegistry` 按 stage、framework、service hint 选择相关 skill
  - 选中的 skill 会写入阶段 `control_context`，包含 path 和 SHA-256
- 增加结构化问题记忆：
  - `MemoryStore` 将 failed/uncertain 阶段写入 `memory/deployment_issues.jsonl`
  - memory 按 signature 去重
  - 后续阶段可按 stage/framework 检索历史相似问题
  - runtime memory JSONL 被 git 忽略，避免提交日志或环境相关信息
- 增加 `docs/skill-memory-design.md`，说明 skill-driven、memory-augmented Agent 设计。
- 将 README、进度报告、skill 文档和 skill/memory 设计文档改为中文，保留必要英文技术关键词。

### 当前行为

系统可以创建任务、扫描仓库目录、生成安装/启动计划、执行 dry-run env/runner 阶段、运行证据化 `verify`，并生成 Markdown 报告。

每个阶段现在还会记录选中的 skill 文档和相关 memory hits。失败或不确定阶段会自动生成结构化 memory entry，供未来部署复用。

### 重要设计说明

- `verify` 当前只有在具备真实 artifact 或 trace evidence 时才会通过。这是故意设计：false pass 比 uncertain 更危险。
- 讯飞集成已经抽象化。当前 provider 支持通过环境变量配置的 Anthropic-compatible HTTP messages 接口。真实密钥不会写入仓库文件。
- Claude Code 通过 `CLAUDE_CODE_CMD` 配置，是可选能力。当前 dry-run MVP 不依赖 Claude Code。
- Skill 是建议性控制文档，不能覆盖 Python 执行策略、命令白名单或源码修改限制。
- Memory 使用机器可读 JSONL，而不是 Markdown，这样后续部署可以检索、打分和去重。

### 下一步

1. 扩展 provider parsing、命令安全、CLI 行为和报告生成测试。
2. 在 `tests/fixtures` 下添加 demo 项目。
3. 扩展 Gradio verify，支持真实 API discovery 和文件/download artifact 检查。
4. 为 Agent/LLM 输出增加 JSON schema validation。
5. 使用本地环境变量私下执行一次讯飞 smoke test，确认真实响应格式。
6. 在允许广泛 `--execute` 前，扩展命令策略，增加参数级检查和危险模式检测。
7. 将 `ClaudeCodeExecutor` 进一步接入 analyzer 或 verify 的可选执行阶段。
8. 增加 benchmark cases：
   - HTTP 200 但没有输出。
   - 历史输出文件干扰。
   - 缺失依赖。
   - 服务启动后立刻退出。
9. 增加 repair-loop，让 Agent 使用 selected skills 和 memory hits 提出或执行受控修复。
10. 增加 memory promotion 工作流，把反复出现的问题记忆提升为稳定 `SKILL.md` 规则。

### 已知限制

- 真实依赖安装和服务启动默认关闭。
- `VerifyModule` 已支持 GET query trace 和 POST JSON trace template，但还不支持 Gradio API discovery、浏览器/UI 操作、文件下载验证或 CLI trace 执行。
- `RunnerModule` 尚未持久化进程句柄，后续需要支持清理。
- `XunfeiSparkProvider` 当前假设 Anthropic-compatible HTTP messages 接口；如果选定的 Spark API 变体需要 WebSocket 签名，需要新增 transport。
- 测试套件仍较小，目前主要覆盖 dry-run 核心路径。
- Memory 会自动记录，但还没有 human review/promotion 命令来把重复 memory 转成 skill 更新。
