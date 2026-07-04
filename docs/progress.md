# AI-Auto-Harness 开发进度

## 2026-07-04

### 已完成

- 五阶段优化任务 - 阶段 1：
  - `TaskRunner.run_existing` 在 repair resume 中间阶段恢复时生成 `reports/execution_audit.json`，记录 requested/effective start stage、dry-run 标记、复用阶段、重跑阶段和 fallback 状态。
  - 新增 `resume_execution_plan` 事件，便于后续从 `events.jsonl` 审计本次恢复执行为什么从某个阶段开始。
  - `ReportGenerator` 新增 `Execution Audit` 小节，展示 reused stages / rerun stages，避免 report 只给阶段结果而无法解释 resume 行为。
  - Benchmark cases 从 20 个扩展到 21 个，新增 `repair_resume_audit_report`。
  - 定向单测与 benchmark 已通过，覆盖 report 审计摘要和阶段跳转审计文件。
- 五阶段优化任务 - 阶段 2：
  - `VerifyModule` 的 Gradio `/config` discovery 新增 queue 模式识别：当 config 或 dependency 显示 queue 启用且存在 named API 时，会使用 `/call/<api_name>`。
  - HTTP trace evidence 支持 follow-up 请求：先 POST `/call/<api_name>` 获取 `event_id`，再 GET `/call/<api_name>/<event_id>` 读取 SSE/JSON 输出。
  - verify 只有在初始响应或 follow-up 响应包含当前 trace id 时才判定 `http_trace_response=pass`，不会因为 event_id 存在就误判成功。
  - Benchmark cases 从 21 个扩展到 22 个，新增 `gradio_queue_call_followup`。
  - 定向单测与 benchmark 已通过，覆盖 `/call/predict` 和 follow-up trace 证据。
- 继续执行 P0/P1 优化任务：
  - `ModelCache.reserve` 会为缓存目录写入 `.auto_harness_asset.json`，记录 source、repo id、revision、origin、asset id 和 cache key。
  - `ModelCache.entries()` 会读取缓存元数据，返回 repo id、revision、origin，便于后续审计和精细清理。
  - `ModelCache.cleanup()` 新增 `source`、`repo_id`、`keep_cache_keys`、`keep_repo_ids` 参数，支持按模型源 / repo 限定清理范围，并保护关键模型。
  - `cache --cleanup` CLI 新增 `--source`、`--repo-id`、`--keep-cache-key`、`--keep-repo-id`，仍默认 dry-run，只有 `--apply` 才删除。
  - 配置文件新增缓存清理过滤和 keep-list 默认值。
  - Benchmark cases 从 19 个扩展到 20 个，新增 `cache_cleanup_scoped_keep`。
  - 单测扩展到 57 个，覆盖缓存 metadata、source/repo filter、keep-list、旧缓存 fallback 和 CLI 参数。
- 继续执行 P0/P1 优化任务：
  - `LogClassifier` 在 `auth_required` 诊断中提取 `HF_TOKEN`、`MODELSCOPE_TOKEN` 等所需环境变量名，并显式标记 `values_recorded=false`。
  - `RepairPlanner` 的 `set_env_var_name_only` action 会优先使用诊断提取到的变量名，不写入任何密钥值。
  - `ReportGenerator` 会汇总 repair artifact、repair plan、阶段 diagnosis 和 `resource_plan.external_tokens` 中的变量名，生成 `Required Environment Variables` 小节。
  - Report 只展示变量名和“由 operator/secret manager 提供”的提示，测试覆盖不会把伪造 token value 写入报告。
  - Benchmark cases 从 18 个扩展到 19 个，新增 `token_report_required_env`。
  - 单测扩展到 53 个，覆盖 token 变量名提取、repair plan 变量名传递和 report 无密钥值输出。
- 继续执行 P0/P1 优化任务：
  - `TaskRunner.run_existing` 支持 `start_stage`，可以复用目标阶段之前的历史 `pipeline_results.json` / `*_result.json`，只重跑目标阶段及其后续阶段。
  - `resume` 会读取已允许的 `repair_apply_result.json` / `repair_plan.json`，按 `rerun_from_effective` 从 `analyze`、`resource_plan`、`env_deploy`、`model_prepare`、`runner` 或 `verify` 中的安全阶段继续。
  - 如果前置 stage result 缺失或结构不完整，会写入 `resume_stage_fallback` 事件，并安全回退到 `analyze` 全量重跑，避免复用坏状态。
  - Repair overlay 仍只影响阶段输入，不直接执行 shell；例如 `repair_verify_hints.json` 可以让 resume 只从 `verify` 重跑并应用新的 endpoint/request hint。
  - Benchmark cases 从 17 个扩展到 18 个，新增 `repair_resume_stage_jump`。
  - 单测扩展到 51 个，覆盖 verify 阶段跳转、缺失前置结果回退和 rejected repair 不触发阶段跳转。
- 继续执行 P0/P1 优化任务：
  - 将下载并发、下载重试和缓存清理阈值暴露到 `HarnessConfig` 与 `configs/default.json`。
  - `TaskRunner` 会按配置创建 Hugging Face / ModelScope 下载器，`model_prepare` 不再只能使用硬编码下载参数。
  - `deploy` / `resume` 新增 `--model-download-workers`、`--download-retries`、`--download-retry-backoff`，支持单次任务临时覆盖下载策略。
  - 新增 CLI 子命令 `cache`，支持查看模型缓存和执行缓存清理计划；清理默认 dry-run，只有 `--apply` 才删除缓存目录。
  - 下载器新增有限重试机制，覆盖文件清单请求和单文件下载；checksum、无可下载文件等确定性失败不会无限重试。
  - 单测扩展到 48 个，覆盖配置读取、CLI 参数覆盖、临时下载错误重试和缓存清理 dry-run。
- 继续执行 P0/P1 优化任务：
  - Benchmark cases 从 13 个扩展到 17 个，新增 `service_exits_after_start`、`stale_artifact_ignored`、`gradio_api_shape_variation`、`token_missing_diagnosis`。
  - 新增服务启动后快速退出回归用例，确保 `RunnerModule` 不会把瞬时退出进程误判为启动成功。
  - 新增历史 artifact 干扰回归用例，确保 verify 只接受本次 trace 后的新鲜 artifact 或 trace evidence。
  - 新增 Gradio API shape 变化回归用例，覆盖 `backend_fn=false` 跳过和 `api_name="/predict"` 归一化。
  - 新增 token 缺失诊断回归用例，确保 401 / Repository Not Found 会归类为 `auth_required`。
  - 修复 `RunnerModule` 父进程未关闭 runner log 文件句柄的问题。
  - 单测扩展到 44 个，覆盖上述新增场景。
- 继续执行 P0/P1 优化任务：
  - 新增 `RepairLoopController`，对同一问题 signature 记录 repair attempts，并通过 `max_repair_attempts` 控制自动修复尝试次数。
  - Repair plan 会记录 `rerun_from_requested` 和 `rerun_from_effective`；当 plan 给出不安全阶段时，会回退到当前安全阶段或 `analyze`。
  - `RepairPolicy` 支持 `operator_approval.json`，需要人工确认的 action 在审批前会拒绝，审批后可通过；需要 secret 的 action 仍不会记录密钥值。
  - 新增 CLI 子命令 `repair-approve`，用于为最新 repair plan 写入人工审批文件和事件日志。
  - 被拒绝的 repair 现在也会写入 `repair_apply_result.json`，避免旧 overlay artifact 被误复用。
  - Benchmark cases 从 11 个扩展到 13 个，新增 `repair_loop_attempt_limit` 和 `operator_repair_approval`。
  - 单测扩展到 40 个，覆盖 attempt limit、安全回退、审批入口、rejected overlay 失效。
- 继续执行 P0/P1 优化任务：
  - Hugging Face / ModelScope 下载器新增 `max_workers`，支持 stdlib 线程池并发下载多个模型文件，并保持 manifest 文件顺序稳定。
  - 下载完成后为每个文件写入 `.auto_harness_meta.json`，记录 size、sha256 和 etag；缓存命中时会校验远端 etag，etag 不一致会重新下载，避免旧缓存误用。
  - `ModelCache` 新增 `entries()` 和 `cleanup()`，支持按总大小或时间生成清理候选；默认 `dry_run=True`，只有显式关闭 dry-run 才删除缓存目录。
  - Benchmark cases 从 8 个扩展到 11 个，新增 `parallel_model_download`、`etag_cache_invalidation`、`cache_cleanup_plan`。
  - 单测扩展到 36 个，覆盖并发下载、etag 缓存失效和缓存清理。
- 继续执行 P0/P1 优化任务：
  - 新增 `BrowserVerifier` 和可选 `PlaywrightBrowserBackend`，用于 Gradio/Streamlit/webui 的浏览器 DOM probe。
  - `VerifyModule` 会对 webui 服务写入 `browser_dom_probe` evidence；DOM 包含当前 trace id 可作为强证据，DOM 出现 traceback/import/runtime error 会作为失败证据。
  - Playwright 是可选依赖；未安装时只记录 `uncertain`，不会让 HTTP trace 或 artifact evidence 被误判失败。
  - Benchmark cases 从 7 个扩展到 8 个，新增 `browser_dom_trace`。
  - 单测扩展到 33 个，覆盖浏览器 DOM trace 通过和错误标记失败。
- 继续执行 P0/P1 优化任务：
  - 新增 `RepairOverlay`，可读取 policy 允许后生成的 repair artifacts。
  - 下一次 `run_existing` / `resume` 会把 `repair_install_plan.json` 合并到 `env_deploy` 的 install plan，把 `repair_verify_hints.json` 合并到 `verify_hint`。
  - Overlay 只改变阶段输入，不直接执行 shell，不绕过 `--execute`、RuntimePolicy 或命令白名单；被拒绝的 repair artifact 不会生效。
  - Benchmark cases 从 4 个扩展到 7 个，新增 Gradio `/config` discovery、repair policy reject、checksum failure。
  - 单测扩展到 31 个，覆盖 repair overlay 合并、policy reject 和新增 benchmark cases。
- 继续执行 P0 优化任务：
  - 新增 `BenchmarkRunner`，可读取 `tests/fixtures/benchmarks/manifest.json` 并执行本地 benchmark cases。
  - 新增 CLI 子命令 `benchmark`，支持 `--manifest` 和 `--output`。
  - 当前 benchmark 不访问外网，覆盖下载续传、缓存命中、Streamlit 错误页、HTTP 200 false-positive 防护。
  - 单测扩展到 29 个，覆盖 benchmark runner 和 benchmark CLI 输出。
- 继续执行 P0 优化任务：
  - 新增 `RepairPolicy` 和 `RepairApplier`，repair plan 会先做权限校验，再生成受控 artifacts。
  - 受控 apply 当前只写入 `repairs/` 下的计划文件、依赖安装建议、verify hint 建议或所需环境变量名，不直接执行 shell，不修改源码。
  - 新增 `ModelFileSelector`，默认只下载模型权重和 tokenizer/config 等必要文件，跳过 README 和项目脚本。
  - 新增 benchmark fixtures：模型下载续传、缓存命中、Streamlit 错误页面、HTTP 200 false-positive 防护。
  - 单测扩展到 27 个，覆盖 repair policy/apply、文件选择策略和 benchmark fixtures。
- 继续执行 P0 优化任务：
  - 新增 `ModelScopeDownloader`，支持可配置 ModelScope API base / download base、文件清单解析、`.part` 和 Range 续传。
  - 抽取 `ResumableDownloadMixin`，复用断点续传、缓存命中和 sha256 校验逻辑。
  - Hugging Face 文件记录增加 `etag` / `sha256` / `verified` 字段；当清单提供 sha256 时会校验完整性。
  - `model_prepare` 支持外部 progress callback，orchestrator 会在下载过程中持续刷新 `state.json` 的 `model_prepare.progress`。
  - 新增 `RepairPlanner` 和 repair plan schema，失败或 uncertain 阶段会在事件日志中记录结构化修复建议；当前只 propose，不自动 apply。
  - 新增 Streamlit DOM/HTML verify 第一版，能识别 Streamlit 页面标记、错误标记和 trace evidence。
  - 单测扩展到 24 个，覆盖 ModelScope 续传、HF sha256/etag、repair plan、Streamlit verify 和 model_prepare progress callback。

### 下一步

1. 增加真实 Playwright smoke test 文档和可选 CI job，在安装 Playwright 的环境中验证浏览器 backend。
2. 增加文件产物下载验证。
3. 增加 Git LFS 检测和下载准备，覆盖权重在 Git 仓库/LFS 中的项目。

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
- 增加 `docs/optimization-roadmap.md`，形成面向真实开源模型全自动部署的详细优化计划，覆盖模型下载、缓存、资源预估、环境求解、长任务状态、诊断修复、verify、安全和 benchmark。
- 开始执行 P0 优化任务：
  - 新增 `resource_plan` 阶段，输出 Python/GPU/CUDA/磁盘/token/模型资产风险信息。
  - 新增 `model_prepare` 阶段，生成模型资产 manifest 和 cache path。
  - 新增 `src/auto_harness/assets/`，包含 `ModelAssetDetector`、`ModelCache`、`AssetManifest`。
  - 支持从 README、Python 代码和配置中识别 Hugging Face / ModelScope 模型引用。
  - 新增 `model_cache_dir` 配置，并将 `model_cache/` 加入 `.gitignore`。
  - 新增 `prepare-model-assets` skill，覆盖 resource_plan 和 model_prepare 阶段。
  - 单测扩展到 15 个，覆盖模型资产识别、资源规划和 manifest 生成。
- 继续执行 P0 优化任务：
  - 新增 `HuggingFaceDownloader`，支持 Hugging Face tree API 文件发现。
  - 支持 `.part` 文件和 HTTP Range 断点续传。
  - `model_prepare` 在执行模式下可以下载 Hugging Face 资产，并把进度写入 stage result。
  - `StateStore.update_stage` 支持保存 stage progress，`state.json` 可展示下载进度。
  - `VerifyModule` 增加 Gradio `/config` discovery，能基于 dependency `api_name` / `fn_index` 构造 trace 请求。
  - 新增 `LogClassifier`，覆盖缺依赖、CUDA OOM、磁盘不足、token 权限、Git LFS、wheel 构建、numpy/pydantic/protobuf 冲突和端口占用。
  - `env_deploy` 和 `runner` 失败时会附带结构化 diagnosis。
  - 单测扩展到 19 个，覆盖 HF 续传、Gradio config discovery、log classifier 和 progress 写入。
- 继续执行 P0 优化任务：
  - 新增 `ModelScopeDownloader`，支持可配置 ModelScope API base / download base、文件清单解析、`.part` 和 Range 续传。
  - 抽取 `ResumableDownloadMixin`，复用断点续传、缓存命中和 sha256 校验逻辑。
  - Hugging Face 文件记录增加 `etag` / `sha256` / `verified` 字段；当清单提供 sha256 时会校验完整性。
  - `model_prepare` 支持外部 progress callback，orchestrator 会在下载过程中持续刷新 `state.json` 的 `model_prepare.progress`。
  - 新增 `RepairPlanner` 和 repair plan schema，失败或 uncertain 阶段会在事件日志中记录结构化修复建议；当前只 propose，不自动 apply。
  - 新增 Streamlit DOM/HTML verify 第一版，能识别 Streamlit 页面标记、错误标记和 trace evidence。
  - 单测扩展到 24 个，覆盖 ModelScope 续传、HF sha256/etag、repair plan、Streamlit verify 和 model_prepare progress callback。

### 当前行为

系统可以创建任务、扫描仓库目录、生成安装/启动计划、规划模型资产和资源风险、执行 dry-run env/model_prepare/runner 阶段、运行证据化 `verify`，并生成 Markdown 报告。

每个阶段现在还会记录选中的 skill 文档和相关 memory hits。失败或不确定阶段会自动生成结构化 memory entry，供未来部署复用。

### 重要设计说明

- `verify` 当前只有在具备真实 artifact 或 trace evidence 时才会通过。这是故意设计：false pass 比 uncertain 更危险。
- 讯飞集成已经抽象化。当前 provider 支持通过环境变量配置的 Anthropic-compatible HTTP messages 接口。真实密钥不会写入仓库文件。
- Claude Code 通过 `CLAUDE_CODE_CMD` 配置，是可选能力。当前 dry-run MVP 不依赖 Claude Code。
- Skill 是建议性控制文档，不能覆盖 Python 执行策略、命令白名单或源码修改限制。
- Memory 使用机器可读 JSONL，而不是 Markdown，这样后续部署可以检索、打分和去重。
- `model_prepare` 已具备 Hugging Face / ModelScope 下载执行能力，但真实联网下载仍应在私有环境中配合 token、磁盘和网络策略验证。

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
11. 继续执行 P0：增强 Hugging Face/ModelScope 文件选择策略、并发下载、etag 强一致校验、repair apply policy、Playwright 真浏览器 verify 和更多 benchmark fixtures。

### 已知限制

- 真实依赖安装和服务启动默认关闭。
- `VerifyModule` 已支持 GET query trace、POST JSON trace template、Gradio `/config` discovery 和 Streamlit DOM/HTML probe，但还不支持 Playwright 真浏览器交互、文件下载验证或 CLI trace 执行。
- `RunnerModule` 尚未持久化进程句柄，后续需要支持清理。
- `XunfeiSparkProvider` 当前假设 Anthropic-compatible HTTP messages 接口；如果选定的 Spark API 变体需要 WebSocket 签名，需要新增 transport。
- 测试套件仍较小，目前主要覆盖 dry-run 核心路径。
- Memory 会自动记录，但还没有 human review/promotion 命令来把重复 memory 转成 skill 更新。
- `model_prepare` 已接入 Hugging Face 和 ModelScope 下载器。
- 下载器目前使用 stdlib HTTP 实现，尚未支持并发下载和 etag 强一致校验；sha256 仅在远端清单提供该字段时校验。
- Repair plan 当前会生成受控 artifacts，但不会直接执行 shell 或修改源码。
