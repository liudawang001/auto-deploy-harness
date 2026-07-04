# AI-Auto-Harness

AI-Auto-Harness 是一个面向 AI 开源 demo 项目的自动部署与验证 Agent。

项目采用“确定性 Python 编排 + 受控 Agent 执行”的架构：

- Python 负责工作流状态、重试、超时、文件边界和证据校验。
- Claude Code headless 或其他 executor 负责项目阅读、日志诊断等不确定任务。
- 讯飞 Spark 等 LLM provider 通过统一接口接入。
- `verify` 是证据驱动的，不能把端口开放或 HTTP 200 直接当成部署成功。

## 当前 MVP

当前仓库已经包含：

- CLI：`init`、`deploy`、`resume`、`status`、`report`、`llm-test`、`benchmark`、`repair-approve`。
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
- `model_prepare` 阶段：生成模型资产 manifest、缓存 key 和 `model_cache` 路径；执行模式下支持 Hugging Face / ModelScope 文件清单解析、断点续传、并发下载、sha256/etag 校验元数据和缓存写入。
- `verify` 增强：支持 Gradio `/config` discovery、Streamlit DOM/HTML 证据探测，以及可选 Playwright 浏览器 DOM probe。
- 日志规则分类器：对缺依赖、CUDA OOM、磁盘不足、token 权限、wheel 构建失败等常见错误生成结构化诊断；token 权限问题只提取所需环境变量名，不记录密钥值。
- Report 会汇总 `resource_plan`、diagnosis 和 repair plan 中的 token 变量名，提示 operator/secret manager 注入，报告中不保存任何 token value。
- Repair loop：失败或 uncertain 阶段会生成结构化修复建议，经过 policy 和 loop gate 校验后写入受控 repair artifacts；同一问题有最大尝试次数，不安全的 `rerun_from` 会回退到安全阶段，`resume` 会按 `rerun_from_effective` 从安全阶段重跑，需要人工确认的 action 可通过 `repair-approve` 批准。
- Benchmark fixtures：`tests/fixtures/benchmarks` 覆盖下载续传、缓存命中、并发下载、etag 缓存失效、缓存清理、服务启动后退出、历史 artifact 干扰、Gradio API shape 变化、token 缺失、token report 提示、Gradio `/config` discovery、浏览器 DOM trace、Streamlit 错误页面、HTTP 200 false-positive 防护、repair policy 拒绝、repair loop 限流、repair resume 阶段跳转、人工审批和 checksum 失败。
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
  "model_cache_cleanup_older_than_days": 30
}
```

`deploy` / `resume` 也支持临时覆盖下载参数：

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo <repo> --execute --model-download-workers 2 --download-retries 3 --download-retry-backoff 2
PYTHONPATH=src python3 -m auto_harness.cli resume --task-id <task-id> --execute --model-download-workers 2
```

缓存清理由 `ModelCache.cleanup(...)` 提供，默认 `dry_run=True`，会先返回候选列表、候选大小和预计删除项；只有显式传入 `dry_run=False` 才会删除缓存目录。

CLI 缓存清理同样默认 dry-run：

```bash
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup --max-total-bytes 500000000000
PYTHONPATH=src python3 -m auto_harness.cli cache --cleanup --max-total-bytes 500000000000 --apply
```

阶段进度会写入 `state.json` 的 `model_prepare.progress`，包括当前文件、已下载字节、总字节和状态。

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
- 多文件并发下载。
- 临时下载错误会按有限次数重试，不会无限 retry。
- 远端 etag 变化时本地缓存失效并重新下载。
- 模型缓存 dry-run 清理和受控删除。
- Gradio `/config` discovery 构造 `/api/predict` trace 请求。
- Gradio `/config` shape 变化时仍能选中正确 backend API。
- 浏览器 DOM 中出现当前 trace 时可以作为强证据。
- Streamlit HTTP 200 错误页面不能通过 verify。
- HTTP 200 但无当前 trace 不能判定成功。
- 历史 artifact 不能作为本次 verify 的新鲜证据。
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

人工审批入口：

```bash
PYTHONPATH=src python3 -m auto_harness.cli repair-approve --task-id <task-id> --note "approved cache dir change"
```

该命令只写入 action 类型、审批时间和备注，不记录任何 token、key 或 secret 值。

## 安全默认值

默认情况下，`deploy` 是 dry-run，不会安装依赖，也不会启动长驻服务，除非显式传入执行参数。

执行模式还受命令白名单保护。`env_deploy` 和 `runner` 会拒绝 `configs/default.json` 的 `allowed_commands` 之外的命令。
