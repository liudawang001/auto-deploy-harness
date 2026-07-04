# AI-Auto-Harness

AI-Auto-Harness 是一个面向 AI 开源 demo 项目的自动部署与验证 Agent。

项目采用“确定性 Python 编排 + 受控 Agent 执行”的架构：

- Python 负责工作流状态、重试、超时、文件边界和证据校验。
- Claude Code headless 或其他 executor 负责项目阅读、日志诊断等不确定任务。
- 讯飞 Spark 等 LLM provider 通过统一接口接入。
- `verify` 是证据驱动的，不能把端口开放或 HTTP 200 直接当成部署成功。

## 当前 MVP

当前仓库已经包含：

- CLI：`init`、`deploy`、`resume`、`status`、`report`、`llm-test`、`benchmark`。
- 任务状态存储：`task.json`、`state.json`、`events.jsonl`。
- 确定性项目分析器。
- 安全默认的 `env_deploy`、`runner`、`verify`、report 模块。
- Mock LLM provider 和讯飞 Anthropic-compatible provider。
- Claude Code executor wrapper。
- HTTP trace evidence：`verify` 支持 GET 和 POST JSON；响应必须证明当前 trace 被处理，HTTP 200 本身不算成功。
- 可选 Claude Code analyzer advisor：通过 `AUTO_HARNESS_USE_AGENT_ANALYZER=1` 启用。
- 仓库内置 skill：位于 `skills/*/SKILL.md`，按阶段选择并记录 hash。
- 结构化问题记忆：位于 `memory/deployment_issues.jsonl`，用于检索历史相似失败。
- `resource_plan` 阶段：识别模型资产、GPU/CUDA 信号、磁盘风险和外部 token 需求。
- `model_prepare` 阶段：生成模型资产 manifest、缓存 key 和 `model_cache` 路径；执行模式下支持 Hugging Face / ModelScope 文件清单解析、断点续传、sha256 校验字段和缓存写入。
- `verify` 增强：支持 Gradio `/config` discovery 和 Streamlit DOM/HTML 证据探测。
- 日志规则分类器：对缺依赖、CUDA OOM、磁盘不足、token 权限、wheel 构建失败等常见错误生成结构化诊断。
- Repair plan：失败或 uncertain 阶段会生成结构化修复建议，经过 policy 校验后写入受控 repair artifacts；下一次 `resume` 会在 policy 允许时把安装建议和 verify hint 回灌到 pipeline 输入，但仍不会绕过 `--execute`、命令白名单或源码修改限制。
- Benchmark fixtures：`tests/fixtures/benchmarks` 覆盖下载续传、缓存命中、Gradio `/config` discovery、Streamlit 错误页面、HTTP 200 false-positive 防护、repair policy 拒绝和 checksum 失败。
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

## Skill 与 Memory

Agent 控制文档写在仓库内：

```text
skills/<skill-name>/SKILL.md
```

当前内置 skill 覆盖项目分析、Python WebUI 部署、证据化验证和运行失败诊断。部署过程中，orchestrator 会为每个阶段选择相关 skill，并把名称、路径和 SHA-256 写入阶段结果的 `control_context`。

运行时问题记忆写入：

```text
memory/deployment_issues.jsonl
```

该文件被 git 忽略，因为它可能包含部署日志或环境相关症状。Agent 会在阶段 `failed` 或 `uncertain` 时写入 memory，并在后续部署中按 stage/framework 检索相似问题。详细设计见 `docs/skill-memory-design.md`。

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

执行模式下，Hugging Face 资产会通过 tree API 获取文件清单；ModelScope 资产会通过可配置的 ModelScope API/下载 URL 模板获取文件。两者都会将 `.safetensors`、`.bin`、`.pt`、`.gguf` 和 tokenizer/config 等必要文件下载到缓存目录，默认跳过 README、示例脚本等非模型运行必要文件。下载过程中使用 `.part` 文件和 HTTP Range 支持续传；如果文件清单提供 `sha256`，下载后会做 sha256 校验。

阶段进度会写入 `state.json` 的 `model_prepare.progress`，包括当前文件、已下载字节、总字节和状态。

## 优化路线图

面向真实开源模型自动部署的长期优化计划见：

```text
docs/optimization-roadmap.md
```

该文档覆盖模型下载与缓存、断点续传、资源预估、CUDA/PyTorch 环境求解、长任务状态机、自动诊断修复、增强 verify、安全沙箱和 benchmark 体系。

## Benchmark

本地 benchmark fixtures 可直接执行，不访问外网：

```bash
PYTHONPATH=src python3 -m auto_harness.cli benchmark --manifest tests/fixtures/benchmarks/manifest.json --output benchmark_report.json
```

当前 benchmark 覆盖：

- 模型下载断点续传。
- 缓存命中避免重复下载。
- Gradio `/config` discovery 构造 `/api/predict` trace 请求。
- Streamlit HTTP 200 错误页面不能通过 verify。
- HTTP 200 但无当前 trace 不能判定成功。
- 权限不足时 repair action 被 policy 拒绝。
- 模型文件 checksum 不一致时下载失败，不写入成功缓存态。

## Repair Overlay

失败或不确定阶段会写入：

```text
runs/<task-id>/repairs/
```

其中可能包含 `repair_install_plan.json`、`repair_verify_hints.json`、`required_env_vars.json` 和 `repair_apply_result.json`。下一次 `resume` 时，`RepairOverlay` 只消费 policy 已允许的非执行型 artifact：

- 将 `repair_install_plan.json` 中的命令追加到 `env_deploy` 的 install plan，真正执行仍需要 `--execute --allow-install` 和命令白名单通过。
- 将 `repair_verify_hints.json` 合并到 `verify_hint`，用于修正 endpoint、请求路径或 POST JSON 模板。
- 如果 repair 被拒绝，overlay 不会生效，只保留审计记录。

## 安全默认值

默认情况下，`deploy` 是 dry-run，不会安装依赖，也不会启动长驻服务，除非显式传入执行参数。

执行模式还受命令白名单保护。`env_deploy` 和 `runner` 会拒绝 `configs/default.json` 的 `allowed_commands` 之外的命令。
