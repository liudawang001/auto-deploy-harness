# AI-Auto-Harness 开发进度

## 2026-07-04

### 已完成

- 下一阶段优化任务：
  - `DeploymentQueue` 新增跨进程 claim lock，worker 运行 job 前会用 `queue/locks/<job_id>.lock` 做原子 claim。
  - claim 成功后队列项立即写为 `running`，记录 pid、claimed_at 和 lock_path；完成或失败后释放 lock，避免同一个 queued job 被多个 worker 重复部署。
  - 如果另一个 worker 已 claim，当前 worker 会跳过该 job 并记录 `job already claimed`，任务保持 queued，便于后续 worker 继续处理。
  - Benchmark cases 从 46 个扩展到 47 个，新增 `queue_claim_lock_prevents_duplicate`。
- 下一阶段优化任务：
  - 新增 `GpuResourceProbe`，支持 `AUTO_HARNESS_GPU_SLOTS` 覆盖、`nvidia-smi --query-gpu` 探测和无 GPU fallback，输出可审计的 `available_slots`、source、GPU 列表和错误信息。
  - `DeploymentQueue.run_next` 在未显式传入 `gpu_slots` 时会自动探测 GPU slot，并把 `gpu_probe` 写入调度结果；GPU slot 不足时保留任务 queued 并记录 `gpu slot unavailable`。
  - 配置 `queue_gpu_slots` 改为 `null` 表示自动探测；CLI `queue run --gpu-slots <n>` 仍可手工覆盖。
  - Benchmark cases 从 45 个扩展到 46 个，新增 `queue_gpu_probe_scheduling`。
- 下一阶段优化任务：
  - `DeploymentQueue.run_next` 从顺序执行升级为线程池 worker pool；`queue run --max-jobs N` 会并发消费多个已选队列项。
  - 并发结果会按调度选择顺序返回，队列项各自持久化 running/completed/failed 状态，仍复用原有 `TaskRunner.deploy` 安全边界。
  - 新增 barrier 型并发回归测试，顺序执行会失败，只有两个 worker 同时进入 deploy 才能通过。
  - Benchmark cases 从 44 个扩展到 45 个，新增 `queue_parallel_worker_pool`。
- 下一阶段优化任务：
  - 新增 `DeploymentPackageExporter`，可为单个 run 导出 tar.gz 审计产物包，并生成同名 `.manifest.json`。
  - CLI 新增 `package --task-id <id> --output <path> --include-logs`；默认输出到 `dist/packages/<task-id>.tar.gz`。
  - 产物包默认包含 `task.json`、`state.json`、`events.jsonl`、`reports/`、`evidence/` 和 `repairs/`，记录每个文件 size/sha256，默认排除 `workspace/`、模型缓存和日志，避免大文件或环境敏感信息进入交付物。
  - Benchmark cases 从 43 个扩展到 44 个，新增 `deployment_package_export`。
- 下一阶段优化任务：
  - 新增 `DeploymentQueue` 本地持久化任务队列，队列项写入 `queue/items/*.json`，记录 repo、dry-run/execute 策略、GPU 需求、优先级、attempt、task_id、错误和时间戳。
  - CLI 新增 `queue submit/list/run`：submit 只入队，run 由显式前台 worker 消费任务；默认仍是 dry-run，不会安装依赖或启动服务。
  - `queue run --max-jobs` 支持一次消费多个队列项；`--require-gpu` 与 `--gpu-slots` 支持在无 GPU slot 时跳过 GPU 任务，避免普通 Mac 开发机误跑 GPU 部署。
  - Benchmark cases 从 42 个扩展到 43 个，新增 `deployment_queue_dry_run`。
- 下一阶段优化任务：
  - 新增 `DashboardGenerator`，可从本地 `runs/`、`state.json`、`task.json` 和可选 benchmark report 生成静态 HTML dashboard 与 JSON 摘要。
  - CLI 新增 `dashboard --output <path> --benchmark-report <path>`，默认写入 `runs/dashboard.html` 和同名 `.json`，不启动 Web 服务。
  - Dashboard 展示 task count、状态统计、benchmark 概览、任务当前阶段、各 stage 状态和 report 路径，适合 mac 开发机与面试演示。
  - Benchmark cases 从 41 个扩展到 42 个，新增 `static_dashboard_export`。
- 下一阶段优化任务：
  - `BenchmarkRunner.run` 支持按 `case_ids` 执行 benchmark 子集，CLI 新增 `benchmark --case-id`。
  - `memory-promote --apply` 审批通过后默认运行 proposal `regression_binding.case_ids` 对应的 benchmark 子集，并写出 `memory/promotions/<proposal_id>.regression.json`。
  - 回归结果会写回 proposal JSON；若回归失败，CLI 返回非 0，避免 skill promotion 变成无验证的规则追加。
  - Benchmark cases 从 40 个扩展到 41 个，新增 `memory_promotion_apply_regression_run`。
- 下一阶段优化任务：
  - `LogClassifier` 增强结构化诊断字段：缺失依赖会提取 package，numpy/pydantic/protobuf 冲突会输出兼容 constraint，wheel build 失败会提取失败包名。
  - 诊断结果新增 `root_cause`、`requires`、`rerun_from` 和 `recommended_actions`，便于后续 repair planner 直接消费。
  - `RepairPlanner` 优先使用诊断中的 recommended action，能把 `numpy.dtype size changed` 转换为受控 `pip install numpy<2` 建议，并从 `env_deploy` 安全重跑。
  - Benchmark cases 从 39 个扩展到 40 个，新增 `structured_dependency_diagnosis`。
- 下一阶段优化任务：
  - 新增 `GitSubmoduleDetector`，可解析 `.gitmodules` 中的 submodule name、path、url、branch，并记录目标目录是否已初始化。
  - `ResourcePlanner` 接入 `git_submodules`，会输出 `prepare_commands`、submodule count 和风险原因；缺少 `git` 时进入 `git_missing` 诊断态。
  - `ModelPrepareModule` 支持 Git submodule 受控执行，执行模式下按白名单运行 `git submodule sync --recursive` 和 `git submodule update --init --recursive`，命令结果写入 `model_prepare.git_submodules`。
  - Benchmark cases 从 38 个扩展到 39 个，新增 `git_submodule_prepare_execute`。
- 下一阶段优化任务：
  - OpenAI-compatible verify 新增 `/v1/models` discovery：当 verify hint 未显式提供 `model` / `model_id` 时，会先读取模型列表并选择第一个 model id。
  - `/v1/chat/completions` 支持 `stream=true` 场景，SSE body 中包含当前 trace 即可作为强证据；evidence 会标记 `response.stream_detected=true`。
  - Benchmark cases 从 37 个扩展到 38 个，新增 `openai_model_discovery_stream_verify`。
- 下一阶段优化任务：
  - `VerifyModule` 新增 FastAPI/Flask `/openapi.json` discovery，可选择不含 path parameter、带 JSON requestBody 的 POST endpoint。
  - OpenAPI schema 请求体支持 `$ref` 解析、object required 字段、string/number/boolean/array 最小值生成，并自动注入当前 `trace_id`。
  - HTTP 证据仍要求响应体包含当前 trace 才能通过，不会因为 OpenAPI 文档或 POST 200 本身误判成功。
  - Benchmark cases 从 36 个扩展到 37 个，新增 `openapi_schema_verify`。
- 继续执行 90% 后三阶段开发 - 阶段 3：
  - `ProjectAnalyzer` 新增 `vllm` / `openai_compatible` 识别，能根据依赖和 README 中的 `/v1/chat/completions` 信号生成 OpenAI-compatible verify hint。
  - `VerifyModule` 支持 OpenAI-compatible `/v1/chat/completions` POST trace 请求，自动替换 `{{model}}`，仍要求响应体包含当前 trace 才算强证据。
  - Benchmark 新增 `openai_compatible_verify`，覆盖 vLLM/OpenAI-compatible trace 验证。
- 继续执行 90% 后三阶段开发 - 阶段 2：
  - 新增 `DockerSmokeChecker`，可生成 Docker/GPU runtime smoke 检查计划，默认不执行任何 Docker 命令。
  - 新增 CLI `docker-smoke`，支持默认 plan 模式和 `--probe` 本机探测模式；探测项包括 `docker version`、`docker info`、Python 镜像运行和可选 `--gpus all`。
  - GPU 检查在 `require_gpu=false` 时可跳过，避免普通开发机没有 GPU 导致 smoke 失败；需要真实 GPU 时用 `--require-gpu` 强制检查。
- 继续执行 90% 后三阶段开发 - 阶段 1：
  - 新增 `LiveSmokePlanner`，生成可选真实联网 E2E smoke 矩阵，覆盖 Hugging Face tiny model、ModelScope public model、真实 Git LFS 仓库和可选中等 Hugging Face 模型。
  - 新增 CLI `live-smoke-plan`，只输出可审计计划，不触发网络下载或服务启动；计划包含目标 repo、命令、预计耗时、所需环境变量和期望验证信号。
  - README 补充 live smoke 用法，明确默认 benchmark 仍不访问外网。
- 继续执行到 90% 阶段：
  - `EnvSolveModule` 新增 `gpu_package_matrix`，对 `xformers`、`flash-attn`、`bitsandbytes`、`triton` 按 Python 版本、平台、CPU 架构、CUDA 可用性和已选 Torch wheel 输出 `compatible` / `risky` / `blocked`、原因和建议动作。
  - GPU 依赖风险不再只是自然语言提示，会写入结构化阶段结果；CPU Torch fallback 下的 `flash-attn` / `xformers` / `bitsandbytes` 会进入 blocked，非 Linux 的 `triton` 会进入 blocked，便于后续 repair/resume 自动换方案。
  - Docker backend 支持 `--gpus`、`model_cache` 挂载、容器名、`docker logs` 日志命令和 `docker rm -f` 清理命令元数据；CLI 新增 `--docker-gpus` 和 `--docker-model-cache-dir`，配置新增 `docker_gpus` / `docker_model_cache_dir`。
  - `TaskRunner` 默认会把项目 `model_cache` 作为 Docker 挂载源；真实执行仍受 `--execute`、权限开关和 `allowed_commands` 中的 `docker` 白名单约束。
  - `memory-promote` proposal 新增 `approval` 审批元数据和 `regression_binding` 回归 case 绑定；审批前 `--apply` 返回 `approval_required`，必须先执行 `memory-promote --approve --proposal <path>`。
  - `VerifyModule` 支持长耗时 verify 进度回调，写入 `service_discovered`、`first_inference_probe_started`、`http_trace_request_sent`、`browser_probe_completed`、`verify_completed` 等状态，避免首次模型加载期间状态不可见。
  - Benchmark cases 从 31 个扩展到 35 个，新增 `gpu_package_matrix_rules`、`docker_gpu_cache_backend`、`memory_promotion_approval_regression`、`verify_progress_refresh`。
  - README、优化路线图、skill memory 设计和相关 skill 控制文档已同步。
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
- 五阶段优化任务 - 阶段 3：
  - `VerifyModule` 新增 `artifact_download_validation` check，对本次 trace 后新增或修改的文件产物做可读性、非空、size 和 sha256 校验。
  - `artifact_freshness` 仍记录本次文件变化，但不再单独作为强通过证据；真正的 artifact 强证据必须来自 `artifact_download_validation=pass`。
  - 空文件、不可读文件或非文件变化会记录为 invalid artifact，并阻止 verify 误判成功。
  - Benchmark cases 从 22 个扩展到 23 个，新增 `artifact_download_validation`。
  - 定向单测与 benchmark 已通过，覆盖非空产物通过和空产物拒绝。
- 五阶段优化任务 - 阶段 4：
  - 新增 `GitLFSDetector`，可解析 `.gitattributes` 中的 `filter=lfs` pattern，并识别仓库内 Git LFS pointer 文件的 oid 与 size。
  - `ResourcePlanner` 接入 Git LFS 检测，`resource_plan.git_lfs` 会记录 required、available、patterns、pointers、total pointer size 和 `git lfs install/pull` 准备命令。
  - 检测到 LFS 但本机缺少 `git-lfs` 时，`resource_plan` 返回 `uncertain`，并附带 `git_lfs_missing` diagnosis，避免继续假装权重已存在。
  - Git LFS pointer size 会纳入磁盘估算，风险原因中会标记 `Git LFS model files detected`。
  - Benchmark cases 从 23 个扩展到 24 个，新增 `git_lfs_detection`。
  - 定向单测与 benchmark 已通过，覆盖 LFS pointer 解析和缺工具诊断。
- 五阶段优化任务 - 阶段 5：
  - 新增 `docs/playwright-smoke.md`，给出真实 Python Playwright backend 的本地 smoke test 步骤和期望证据。
  - 新增 `.github/workflows/playwright-smoke.yml`，提供手动触发的 GitHub Actions workflow，避免普通 push/PR 默认下载 Chromium。
  - README 的 Browser Verify 小节补充 smoke test 文档和 workflow 入口。
  - 该 smoke test 使用极小 HTTP server 将 `_auto_harness_trace` 渲染到 DOM，再调用 `BrowserVerifier` 验证 `browser_dom_probe=pass`。
- 下一阶段优化任务：
  - `ModelPrepareModule` 新增 Git LFS 受控执行层，在 `execute=True` 时根据 `resource_plan.git_lfs.prepare_commands` 执行 `git lfs install` 和 `git lfs pull`。
  - Git LFS 执行仍受 `allowed_commands` 白名单约束；未允许 `git` 时直接返回 `command_rejected` diagnosis，不会绕过策略执行命令。
  - `TaskRunner` 会把 `repo_dir`、`allowed_commands` 和默认 timeout 传入 `model_prepare`，让真实 `deploy --execute` 路径也遵循同一策略。
  - `model_prepare.progress` 会在 Git LFS 执行期间刷新为 `git_lfs_running` / `git_lfs_ready`，命令结果写入 `model_prepare.git_lfs.commands`。
  - Benchmark cases 从 24 个扩展到 25 个，新增 `git_lfs_prepare_execute`。
  - 定向单测与 benchmark 已通过，覆盖允许执行和白名单拒绝两条路径。
- 下一阶段优化任务：
  - 新增正式 pipeline stage `env_solve`，位于 `resource_plan` 和 `env_deploy` 之间；`StateStore`、resume stage、report、benchmark 均已接入。
  - 新增 `EnvSolveModule`，读取 analyzer install plan、requirements 和 resource plan，输出带约束的 install plan、constraints、constraint reasons 和 risk reasons。
  - 第一版兼容规则覆盖老 Gradio / 未 pin Gradio 项目的 `numpy<2`、`pydantic<2`，headless 部署的 `opencv-python-headless`，以及 GPU/CUDA/Torch、`flash-attn`、`bitsandbytes` 风险提示。
  - 新增 `skills/solve-python-cuda-env/SKILL.md`，让 agent 在环境求解阶段有明确中文控制文档。
  - `env_deploy` 改为消费 `env_solve` 输出后的 `install_plan`；真正安装仍受 `--execute --allow-install` 和命令白名单控制。
  - Benchmark cases 从 25 个扩展到 26 个，新增 `env_solve_legacy_gradio_constraints`。
  - 定向单测与 benchmark 已通过，覆盖约束生成、CUDA/Torch 风险和 dry-run pipeline 中的 env_solve 结果。
- 下一阶段优化任务：
  - `EnvSolveModule` 新增 `LocalEnvironmentProbe`，可读取 `AUTO_HARNESS_CUDA_VERSION`、`nvidia-smi` 或 `nvcc`，记录 Python、平台、架构和 CUDA 探测来源。
  - `env_solve.torch_solution` 会识别 `torch` / `torchvision` / `torchaudio` 依赖，并根据本机 CUDA 映射 PyTorch wheel index：CUDA 12.1+ -> `cu121`，CUDA 11.8+ -> `cu118`，无兼容 CUDA -> `cpu`。
  - 安装计划会在 pip upgrade 后插入受控的 Torch 预安装命令，例如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`；真正执行仍由 `env_deploy` 和命令白名单控制。
  - `torch_solution.fallbacks` 保留 CPU fallback，CUDA 12.1 场景额外保留 `cu118` fallback，便于后续 repair/resume 自动换方案。
  - GPU 需求但只能选择 CPU wheel 时，会写入明确 risk reason；`flash-attn` 在 CPU fallback 下会额外标记不可兼容风险。
  - Benchmark cases 从 26 个扩展到 27 个，新增 `env_solve_torch_cuda_wheel`。
  - 定向单测已通过，覆盖 CUDA `cu121` 选择、CPU fallback、无 CUDA 风险提示和 benchmark manifest 执行。
- 下一阶段优化任务：
  - 新增 `tests/fixtures/e2e/` 本地端到端 fixture，覆盖 `gradio_tiny_model`、`streamlit_tiny_demo` 和 `git_lfs_weight_repo` 三类开源模型项目形态。
  - `gradio_tiny_model` 模拟小型 Gradio 推理 demo，带本地 `model/config.json` 占位模型资产；`streamlit_tiny_demo` 模拟 Streamlit 推理页面；`git_lfs_weight_repo` 使用标准 Git LFS pointer 模拟 safetensors 权重。
  - Benchmark 新增 `local_e2e_fixture_matrix`，逐个调用 `TaskRunner.deploy(..., dry_run=True)` 跑完整 pipeline，并检查 `analyze`、`resource_plan`、`env_solve`、`env_deploy`、`model_prepare`、`runner`、`verify`、`report` 阶段结果。
  - E2E matrix 会断言 framework 识别、run candidate 端口、env_solve 约束、Git LFS pointer/size、Torch solution、manifest 和 report 输出，防止 pipeline 接线回归。
  - Benchmark cases 从 27 个扩展到 28 个。
- 下一阶段优化任务：
  - 新增 `MemoryPromoter`，读取 `memory/deployment_issues.jsonl`，按 stage / category / frameworks 聚类高频失败记忆。
  - 新增 CLI 子命令 `memory-promote`：默认生成 `memory/promotions/<proposal_id>.json` 和 `.md` 审核稿，包含 cluster、目标 skill、建议追加片段和 `review_required=true`。
  - `memory-promote --apply --proposal <path>` 才会把审核后的片段追加到目标 `skills/*/SKILL.md`，并使用 marker 防止重复应用；默认 proposal 模式不会修改 skill。
  - Promotion 会按阶段选择目标 skill：verify -> `verify-evidence`，resource/model -> `prepare-model-assets`，env_solve/dependency -> `solve-python-cuda-env`，env_deploy/runner -> `deploy-python-webui`。
  - Benchmark cases 从 28 个扩展到 29 个，新增 `memory_promotion_proposal`，验证 proposal JSON/Markdown 生成和默认不修改 skill 的安全边界。
- 下一阶段优化任务：
  - 新增 Docker sandbox backend 第一版，`env_deploy` 和 `runner` 可根据 `execution_backend=docker` 将安装/启动命令包装成 `docker run` effective command。
  - Docker backend 支持 workspace volume mount、workdir、网络参数和服务端口映射；默认仍为 `local` backend，真实执行仍需要 `--execute`、权限开关和 `allowed_commands` 中显式允许 `docker`。
  - `deploy` / `resume` 新增 `--execution-backend`、`--docker-image`、`--docker-network` 覆盖参数，`configs/default.json` 新增对应默认配置。
  - Git LFS 准备阶段新增进度解析，可从 `git lfs pull` 输出提取 `percent`、`files_done`、`files_total`、`downloaded_bytes` 和 `total_bytes`，写入 `git_lfs.commands[].progress` 与 `model_prepare.progress`。
  - Benchmark cases 从 29 个扩展到 31 个，新增 `docker_backend_plan` 和 `git_lfs_progress_parse`。

### 五阶段总结与总进度

本轮 5 个阶段分别完成：

1. 恢复执行审计：resume 从中间阶段重跑时，report 能解释复用了哪些阶段、重跑了哪些阶段。
2. Gradio queue verify：支持 `/call/<api_name>` + `event_id` follow-up，不因 event id 存在而误判成功。
3. 文件产物验证：新产物必须可读、非空并记录 sha256，空文件不会作为强证据。
4. Git LFS 检测：识别 LFS pointer 和 `.gitattributes`，缺 `git-lfs` 时进入诊断态，并给出准备命令。
5. Playwright smoke 治理：补齐真实浏览器 backend 的本地 smoke 文档和手动 CI workflow。

按 `docs/optimization-roadmap.md` 的“全自动开源模型部署 Agent”目标估算，当前总项目进度约为 **97%**。

估算依据：

- 已完成约 95% 的 P0/P1 核心控制链路：下载缓存、断点续传、verify 防误判、OpenAPI/OpenAI-compatible verify、repair plan/policy/resume、memory promotion 审批与回归、benchmark 回归体系已经成型。
- 已完成约 95% 的真实开源模型部署能力：Hugging Face / ModelScope 下载、Git LFS / submodule 检测与受控准备、Gradio/Streamlit/browser verify、PyTorch CPU/CUDA wheel 求解、GPU 包兼容矩阵、本地 E2E fixture matrix、Docker GPU/cache backend、静态 dashboard、本地持久化队列、并发 worker pool、跨进程 claim lock、GPU 探测调度和部署产物包已具备，但真实联网/真实大模型端到端矩阵仍需扩大。
- 尚未完成的关键 3% 主要是：真实联网长耗时 E2E、真实 Docker/GPU smoke、真实 vLLM 服务 smoke、更多模型仓库源、分布式资源锁和常驻 Web dashboard。
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

1. 扩展真实联网 E2E：选择小型 Hugging Face / ModelScope demo 和一个真实 Git LFS 权重仓库做可选长耗时 smoke。
2. 做真实 Docker/GPU smoke：验证 `--gpus all`、模型缓存挂载、容器日志和清理命令在有 Docker/GPU 的机器上可用。
3. 增加 vLLM / OpenAI-compatible server 识别与 verify。
4. 扩展 GPU 包矩阵到具体版本：Torch 2.1/2.2/2.3、CUDA 11.8/12.1/12.4、Python 3.10/3.11/3.12。
5. 为 memory promotion 增加 apply 后自动运行绑定 benchmark case 的执行器和 dashboard。

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
- `VerifyModule` 已支持 GET query trace、POST JSON trace template、Gradio `/config` discovery、Gradio queue `/call` follow-up、Streamlit DOM/HTML probe、可选 Playwright DOM probe 和文件产物校验；后续仍需补 CLI inference trace 与更复杂的多模态交互。
- `RunnerModule` 尚未持久化进程句柄，后续需要支持清理。
- `XunfeiSparkProvider` 当前假设 Anthropic-compatible HTTP messages 接口；如果选定的 Spark API 变体需要 WebSocket 签名，需要新增 transport。
- 测试套件已覆盖核心 dry-run、verify、repair、下载和 benchmark 路径；后续仍需增加真实联网和真实浏览器矩阵。
- Memory 会自动记录；`memory-promote` 已支持生成 human review proposal 和显式 apply，后续还需要把 apply 与 fixture 回归、审批人元数据绑定。
- `model_prepare` 已接入 Hugging Face 和 ModelScope 下载器。
- 下载器目前使用 stdlib HTTP 实现，已支持并发下载、有限重试、etag 缓存失效和远端提供 sha256 时的校验；后续仍需扩展更多模型源和真实大文件 E2E。
- Repair plan 当前会生成受控 artifacts，但不会直接执行 shell 或修改源码。
