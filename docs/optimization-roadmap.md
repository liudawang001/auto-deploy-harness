# AI-Auto-Harness 全自动开源模型部署优化计划

## 1. 目标定位

AI-Auto-Harness 的目标不是做一个简单的部署脚本，而是做一个面向开源 AI 模型项目的全自动部署 Agent。

它需要处理的真实场景包括：

- GitHub / Hugging Face / ModelScope 上的开源模型项目。
- Gradio、Streamlit、FastAPI、Flask、CLI inference、notebook 等多种入口。
- 大模型权重下载，可能达到数 GB 到数十 GB。
- pip、CUDA、PyTorch、Transformers、Diffusers、vLLM 等复杂依赖。
- 首次启动和首次推理耗时很长。
- 项目文档不完整、依赖缺失、启动命令不准、verify API 不明确。
- 部署过程中失败后需要自动诊断、修复、重试，并沉淀经验。

最终系统应具备：

```text
可恢复下载 + 可审计部署 + 可控自动修复 + 证据化验证 + 跨任务问题记忆
```

## 2. 总体架构升级

当前 MVP pipeline：

```text
clone -> analyze -> env_deploy -> runner -> verify -> report
```

目标 pipeline：

```text
source_acquire
-> analyze
-> resource_plan
-> env_solve
-> env_deploy
-> model_prepare
-> runner
-> verify
-> diagnose
-> repair_plan
-> repair_apply
-> rerun_from_safe_stage
-> report
```

其中 Python controller 仍然负责确定性控制：

- 状态机
- 权限策略
- 命令执行
- 文件边界
- 进度持久化
- 证据判断
- 失败重试
- 结果审计

LLM/Agent 只进入不确定环节：

- 复杂项目阅读
- 日志摘要与 root cause 判断
- 修复方案生成
- skill/memory 更新建议

LLM 不能直接绕过 Python policy 执行命令或修改源码。

## 3. 新增核心阶段

### 3.1 source_acquire

职责：获取项目源码，并记录来源与版本。

需要支持：

- GitHub HTTPS clone。
- 本地路径复制。
- Hugging Face Space 仓库。
- ModelScope 仓库。
- Git LFS 项目检测。已完成第一版，支持 `.gitattributes` / pointer 识别、size 估算、缺工具诊断、`git lfs install/pull` 准备命令，以及 `model_prepare` 阶段的白名单受控执行。
- submodule 检测。

输出：

```json
{
  "source_type": "github",
  "repo_url": "...",
  "commit": "...",
  "branch": "main",
  "git_lfs_required": true,
  "submodules_required": false,
  "workspace_path": "runs/<task>/workspace/repo"
}
```

验收标准：

- clone/copy 失败可恢复。
- 记录 commit hash。
- 如果检测到 Git LFS 但本机缺失 `git-lfs`，进入诊断而不是继续假装成功。

### 3.2 resource_plan

职责：在真正安装和下载前预估资源需求。

需要分析：

- Python 版本要求。
- CUDA/PyTorch 版本要求。
- 是否需要 GPU。
- 预估显存。
- 预估磁盘。
- 模型权重来源。
- 是否需要外部 token。
- 是否有 CPU fallback。

输出：

```json
{
  "python_range": ">=3.10,<3.12",
  "gpu_required": true,
  "cuda_required": ">=11.8",
  "torch_variant": "cu121",
  "estimated_vram_gb": 16,
  "estimated_disk_gb": 35,
  "external_tokens": ["HF_TOKEN"],
  "risk_level": "high",
  "risk_reasons": [
    "large safetensors weights",
    "flash-attn build may require CUDA toolchain"
  ]
}
```

验收标准：

- 在下载大模型前给出磁盘需求判断。
- 在安装 CUDA 相关包前给出兼容性判断。
- 缺 token 时明确提示变量名，不记录密钥值。

### 3.3 env_solve

职责：从项目依赖和本机环境中推导稳定安装方案。

需要支持：

- Python 版本选择。
- venv / conda / docker backend 选择。
- CPU/GPU PyTorch wheel 选择。
- 高频依赖冲突规则。
- 特殊包处理：`flash-attn`、`xformers`、`bitsandbytes`、`triton`、`opencv-python-headless`。

输出：

```json
{
  "backend": "local_venv",
  "python": "3.10",
  "install_plan": [
    ["python3.10", "-m", "venv", ".venv"],
    [".venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"],
    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"]
  ],
  "constraints": [
    "numpy<2",
    "pydantic<2"
  ],
  "reason": "old gradio project detected"
}
```

验收标准：

- 不直接盲装 requirements。
- 对老项目自动识别 `numpy 2.x`、`pydantic v2`、`gradio v4` 等兼容风险。
- 安装失败后能把错误分类给 diagnose。

当前状态：已完成第一版 pipeline stage `env_solve`，可生成带约束的 install plan，覆盖老 Gradio 的 `numpy<2` / `pydantic<2`、headless OpenCV 替换建议，以及 GPU/CUDA/Torch 构建风险提示；已补充本机 CUDA 探测、PyTorch `cu121` / `cu118` / `cpu` wheel index 求解和 CPU fallback 生成。后续继续补 Docker backend、更多 GPU 包规则和真实 CUDA E2E。

### 3.4 model_prepare

职责：下载、缓存、校验、挂载模型资产。

需要支持来源：

- Hugging Face model repo。
- Hugging Face Space 附带模型。
- ModelScope model repo。
- Git LFS 权重。
- GitHub Release asset。
- README 中的外部链接。
- 本地 checkpoint 路径。

核心能力：

- 大文件断点续传。
- 下载进度持久化。
- 缓存复用。
- checksum / size 校验。
- token 缺失诊断。
- 磁盘空间预检。
- 并发下载限制。

建议目录：

```text
src/auto_harness/assets/
  detector.py
  manifest.py
  cache.py
  downloader.py
  huggingface.py
  modelscope.py
  git_lfs.py
  checksum.py
```

manifest 示例：

```json
{
  "asset_id": "huggingface:Qwen/Qwen2.5-7B-Instruct@main",
  "source": "huggingface",
  "repo_id": "Qwen/Qwen2.5-7B-Instruct",
  "revision": "main",
  "expected_size_bytes": 15200000000,
  "downloaded_bytes": 8300000000,
  "cache_path": "model_cache/huggingface/...",
  "status": "downloading",
  "resume_supported": true,
  "files": [
    {
      "name": "model-00001-of-00008.safetensors",
      "size_bytes": 1900000000,
      "sha256": null,
      "status": "done"
    }
  ],
  "last_error": null
}
```

验收标准：

- 中断后可继续下载。
- 同一个模型第二次部署命中缓存。
- 缺磁盘空间时提前失败。
- 缺 token 时只记录变量名。
- 运行目录通过 symlink 或引用使用缓存，不重复复制大文件。

### 3.5 diagnose

职责：把失败日志转成结构化 root cause。

需要识别：

- `ModuleNotFoundError`
- `ImportError`
- `CUDA out of memory`
- `torch not compiled with CUDA enabled`
- `No space left on device`
- `Repository Not Found`
- `401 Unauthorized`
- `git-lfs: command not found`
- `subprocess-exited-with-error`
- `Could not build wheels`
- `flash_attn` 编译失败
- `numpy.dtype size changed`
- `pydantic` v1/v2 冲突
- `protobuf` 版本冲突
- `gradio` API shape 变化

建议目录：

```text
src/auto_harness/diagnostics/
  patterns.py
  log_classifier.py
  root_cause.py
  evidence_reader.py
```

输出：

```json
{
  "category": "dependency_missing",
  "signal": "ModuleNotFoundError: No module named 'gradio'",
  "root_cause": "requirements.txt does not include gradio",
  "confidence": 0.91,
  "suggested_fix": "add gradio to dependency plan or install it explicitly",
  "requires": {
    "source_edit": false,
    "dependency_install": true,
    "network": true
  }
}
```

验收标准：

- 对常见错误优先使用规则分类，不把整段日志直接丢给 LLM。
- LLM 只在规则无法分类或低置信度时介入。
- 每次诊断都写入 evidence 和 memory。
- token 权限问题只抽取所需环境变量名，并在 report 中提示 operator/secret manager 注入，不记录密钥值。已完成第一版，覆盖 `auth_required` 诊断、repair plan 和 report 输出。

### 3.6 repair_plan / repair_apply

职责：根据 diagnose、skill、memory 生成并执行受控修复。

repair plan 必须结构化：

```json
{
  "root_cause": "numpy 2.x incompatible with old gradio project",
  "confidence": 0.84,
  "actions": [
    {
      "type": "modify_dependency",
      "file": "requirements.txt",
      "change": "add numpy<2",
      "requires_source_edit": true,
      "reason": "old gradio imports fail with numpy 2.x"
    }
  ],
  "rollback": {
    "type": "restore_file",
    "file": "requirements.txt"
  },
  "rerun_from": "env_deploy",
  "verification_required": true
}
```

action 类型：

- `install_package`
- `modify_dependency`
- `set_env_var_name_only`
- `switch_run_candidate`
- `change_port`
- `download_model_asset`
- `use_cached_model`
- `skip_optional_extension`
- `patch_source`

策略：

- Python controller 校验 action 是否允许。
- `allow_source_edit=false` 时禁止源码 patch。
- 每个 repair 必须有 rollback。
- repair 后必须从 policy/loop 生成的 `rerun_from_effective` 安全阶段重新运行；如果前置阶段结果缺失或损坏，则回退到 `analyze`。
- repair 成功后将模式写入 memory。

验收标准：

- 不允许 LLM 直接执行 shell。
- 不允许无上限 retry。
- 每个修复都有 evidence、diff、rerun 和 verify。

## 4. Verify 模块重点增强计划

Verify 是本项目最重要的差异化模块。它要证明模型链路真的处理了当前输入，而不是只证明服务打开了。

### 4.1 Gradio API Discovery

需要支持：

- 读取 `/config`。
- 解析 dependencies、api_name、fn_index。
- 识别 text/image/audio/video 输入组件。
- 生成 `/api/predict` 或 `/call/<api_name>` 请求。
- 等待 queue 结果。

验收标准：

- 不依赖固定 `/api/predict`。
- 支持 Gradio 3.x 和 4.x 常见接口。
- 返回体或日志必须包含当前 trace，或产物必须是本次新生成。

当前状态：已完成第一版 `/config` discovery、`api_name` / `fn_index` 解析、queue `/call/<api_name>` + `event_id` follow-up 验证。

### 4.2 FastAPI / Flask OpenAPI Verify

需要支持：

- 探测 `/openapi.json`。
- 选择可调用 POST endpoint。
- 根据 schema 构造最小请求。
- 若 endpoint 是 OpenAI-compatible，例如 `/v1/chat/completions`，使用 chat prompt trace。

验收标准：

- OpenAPI 存在时不盲猜 endpoint。
- 不把 docs 页面 200 当成成功。

### 4.3 Streamlit Browser Verify

Streamlit 通常没有稳定 API，需要浏览器验证。

需要支持：

- 打开页面。
- 等待 app load。
- 检测报错框。
- 输入 trace prompt。
- 点击运行按钮。
- 截图和 DOM evidence。

验收标准：

- 至少能区分“页面打开但报错”和“页面完成一次交互”。
- 截图、DOM、日志都写入 evidence。

当前状态：已完成可选 Playwright backend、DOM trace / error marker 判定，并补充真实浏览器 smoke test 文档和手动 CI workflow。

### 4.4 文件产物 Verify

适用于：

- 文生图
- 图生图
- TTS
- ASR
- 视频生成
- embedding 导出

需要校验：

- 文件是否在 trace 后新生成。
- 文件大小是否合理。
- 格式是否正确。
- 图片/音频/视频能否被解析。

验收标准：

- 历史产物不能误判为本次成功。
- 空文件不能通过。

当前状态：已完成第一版 `artifact_download_validation`，会校验本次新增/修改文件的可读性、非空、size 和 sha256。

### 4.5 长耗时 Verify

首次推理可能很慢，需要状态化等待。

策略：

- 按模型加载、请求发送、推理中、产物生成分阶段记录。
- 日志持续有新输出时，不应立即超时。
- 总超时、静默超时分开。

配置示例：

```json
{
  "verify": {
    "max_wait_seconds": 1800,
    "idle_timeout_seconds": 180,
    "require_trace_id": true,
    "allow_artifact_proof": true
  }
}
```

## 5. 模型下载与缓存计划

### 5.1 缓存目录

建议：

```text
model_cache/
  huggingface/
  modelscope/
  git-lfs/
  external/
  manifests/
```

默认加入 `.gitignore`。

### 5.2 缓存 key

缓存 key 由以下字段生成：

- source
- repo_id 或 URL
- revision / commit / etag
- file path
- size

当前已完成第一版：缓存目录会写入 `.auto_harness_asset.json`，记录 source、repo id、revision、origin、asset id 和 cache key；缓存清理可按 source / repo id 限定范围，并通过 cache key 或 repo id keep-list 保护关键模型。

### 5.3 下载策略

- 小文件直接下载。
- 大文件使用支持 resume 的下载器。
- Hugging Face 优先使用官方 snapshot_download 能力。
- ModelScope 使用官方 SDK 或 HTTP fallback。
- Git LFS 使用 `git lfs pull` 并记录依赖。
- 外链下载必须记录来源、大小、文件名和 hash。

### 5.4 进度展示

状态文件应持续更新：

```json
{
  "stage": "model_prepare",
  "status": "waiting_download",
  "progress": {
    "current_file": "model-00003-of-00008.safetensors",
    "downloaded_bytes": 8200000000,
    "total_bytes": 15200000000,
    "speed_bps": 45000000,
    "eta_seconds": 154
  }
}
```

## 6. 状态机升级

当前状态较粗，需要支持长任务。

新增状态：

```text
pending
running
waiting_network
waiting_download
waiting_build
retrying
paused
passed
failed
uncertain
cancelled
```

每个 stage 应包含：

```json
{
  "status": "waiting_download",
  "attempt": 2,
  "started_at": "...",
  "updated_at": "...",
  "progress": {},
  "result_path": "...",
  "error": null
}
```

验收标准：

- 用户能区分“卡死”和“正在下载”。
- resume 可以从未完成 stage 继续。
- stage 进度写入 `state.json` 和 `events.jsonl`。

## 7. 安全与沙箱计划

运行网上开源项目本质上是在执行陌生代码，必须提高安全等级。

### 7.1 local_venv 后端

短期继续支持，但必须加强：

- 命令白名单。
- 参数级危险模式检测。
- 禁止写入 workspace 外路径。
- secrets redaction。
- 进程清理。
- timeout。

### 7.2 docker 后端

中期增加 Docker backend：

```text
src/auto_harness/backends/
  base.py
  local_venv.py
  docker.py
  conda.py
```

Docker backend 需要支持：

- volume mount workspace。
- volume mount model_cache。
- 端口映射。
- GPU 参数，例如 `--gpus all`。
- 网络策略。
- 容器日志收集。
- 容器清理。

### 7.3 secret 保护

规则：

- 密钥只能通过环境变量传入。
- report、memory、events、logs 写入前做 redaction。
- memory 只记录变量名，不记录变量值。

## 8. Skill 与 Memory 进一步增强

### 8.1 新增 skill

建议新增：

```text
skills/prepare-huggingface-model/SKILL.md
skills/prepare-modelscope-model/SKILL.md
skills/solve-python-cuda-env/SKILL.md
skills/verify-gradio-api/SKILL.md
skills/verify-streamlit-browser/SKILL.md
skills/diagnose-cuda-torch/SKILL.md
skills/repair-dependency-conflict/SKILL.md
```

### 8.2 memory promotion

当同类 memory 出现多次，应提升为 skill 规则。

流程：

```text
memory cluster
-> human review
-> generate skill patch
-> test on fixture
-> commit skill update
```

验收标准：

- memory 不是无限堆积。
- 高频问题会沉淀为稳定部署策略。

## 9. 测试与 Benchmark 计划

### 9.1 Fixture 项目

需要创建 `tests/fixtures`：

```text
tests/fixtures/
  gradio_echo/
  fastapi_echo/
  streamlit_echo/
  missing_dependency/
  bad_requirements_numpy2/
  service_exits/
  fake_large_model/
  artifact_output/
```

### 9.2 Benchmark 场景

必须覆盖：

- HTTP 200 但未处理 trace。
- 历史产物干扰。
- 缺依赖。
- 端口冲突。
- 服务启动后退出。
- 模型下载中断后 resume。
- 缓存命中。
- 缺 HF token。
- 磁盘空间不足模拟。
- CUDA 不匹配。

### 9.3 验收指标

建议指标：

- 部署成功率。
- false pass 数量必须为 0。
- 平均诊断准确率。
- repair 成功率。
- 缓存命中节省时间。
- resume 后重复下载字节数。
- 每个 stage evidence 完整度。

## 10. 实施优先级

### P0：真实部署闭环

目标：能稳定部署中小型 Gradio/FastAPI 开源模型 demo。

任务：

1. 增加 `resource_plan` 阶段。
2. 增加 `model_prepare` 阶段。
3. 实现模型 asset detector 和 manifest。
4. 实现本地 `model_cache`。
5. 实现 Hugging Face 下载与缓存。
6. 增强 stage 状态和进度。
7. 增强 Gradio API discovery。
8. 增加 diagnose log classifier。
9. 增加结构化 repair plan。
10. 添加基础 fixture 和 benchmark。已完成第一版，提供 `benchmark` CLI 执行入口。

验收：

- 一个需要下载模型的 Gradio demo 可以自动完成部署。
- 中断后 resume 不重新下载已完成文件。
- verify 能证明 trace 被处理。
- 常见依赖缺失能诊断并给出 repair plan。

### P1：大模型与 GPU 部署

目标：能处理较大模型和 GPU 环境。

任务：

1. CUDA/PyTorch compatibility solver。已完成 env_solve 第一版风险识别、本机 CUDA 探测、Torch wheel index URL 和 CPU/CUDA fallback 生成；后续补真实 CUDA E2E、`xformers` / `flash-attn` / `bitsandbytes` 的版本矩阵和 Docker fallback。
2. 磁盘/显存预估。
3. ModelScope 下载支持。
4. Git LFS 支持。已完成检测、准备命令和白名单受控执行第一版；后续补真实 LFS 大文件 E2E 与下载进度解析。
5. Docker backend。
6. 长耗时 verify。
7. vLLM/OpenAI-compatible server 识别与 verify。
8. 更多 dependency conflict rules。

验收：

- 能部署至少一个 Hugging Face 大模型 demo。
- 能识别 GPU 缺失、CUDA 不匹配、显存不足。
- Docker backend 可以隔离运行陌生项目。

### P2：产品化与规模化

目标：多任务、可视化、可运营。

任务：

1. Web dashboard。
2. 任务队列。
3. 多任务并发。
4. GPU 调度。
5. 缓存清理策略。已完成第一版，支持 dry-run / apply、按 source/repo id 过滤和 keep-list。
6. memory promotion 命令。
7. 自动生成部署产物包。
8. CI benchmark。已完成第一版 benchmark CLI；Playwright browser smoke 已提供手动 GitHub Actions workflow，后续再扩展为完整 CI matrix。

验收：

- 多个部署任务可排队执行。
- 可查看下载进度、日志、verify evidence 和 memory。
- 重复部署相似项目时成功率提升。

## 11. 建议代码目录演进

目标目录：

```text
src/auto_harness/
  assets/
    detector.py
    manifest.py
    cache.py
    downloader.py
    huggingface.py
    modelscope.py
    git_lfs.py
  backends/
    base.py
    local_venv.py
    docker.py
    conda.py
  diagnostics/
    patterns.py
    log_classifier.py
    root_cause.py
  env/
    python.py
    pip_solver.py
    cuda.py
    torch.py
    conflict_rules.py
  repair/
    plan.py
    policy.py
    apply.py
    rollback.py
  verify/
    gradio.py
    fastapi.py
    streamlit.py
    artifacts.py
```

短期不要一次性大重构。建议按 P0 阶段逐步迁移：

1. 先新增 `assets/` 和 `diagnostics/`。
2. 再新增 `model_prepare` 阶段。
3. 再把 verify 拆分为子模块。
4. 最后引入 backend abstraction。

## 12. 面试讲法

可以这样描述项目升级路线：

> 我把自动部署拆成确定性控制和不确定性 Agent 两部分。Python controller 负责状态、安全、下载、执行和证据；LLM 只处理复杂阅读、诊断和修复建议。针对开源模型部署的痛点，我设计了模型资产管理、断点续传、缓存复用、资源预估、CUDA/PyTorch 兼容判断、结构化 repair plan 和 evidence-based verify。系统不会因为端口打开或 HTTP 200 就判成功，而是要求 trace、产物或日志证明模型链路真实处理了当前输入。

核心亮点：

- 不是 LangChain/Dify 套壳，而是部署系统。
- 不是只会调用 LLM，而是用 LLM 处理不确定问题。
- 不是简单 health check，而是证据化 verify。
- 不是一次性脚本，而是可恢复、可记忆、可演进的 Agent pipeline。

## 13. 最近两周建议执行计划

### 第 1 周

1. 添加 `resource_plan` 和 `model_prepare` stage skeleton。已完成。
2. 添加 `AssetManifest`、`ModelAsset`、`ModelCache` 数据结构。已完成。
3. 实现本地缓存目录和 manifest 持久化。已完成。
4. 支持从 README/requirements/code 中识别 Hugging Face repo id。已完成第一版，覆盖 README、Python 代码和常见配置文件。
5. 实现 dry-run model_prepare，输出待下载资产和预估大小。已完成。
6. 新增 fake large model fixture。

### 第 2 周

1. 实现 Hugging Face 下载器。已完成第一版，使用 stdlib HTTP 和 Hugging Face tree API。
2. 支持 resume 和缓存命中。已完成第一版，使用 `.part` 文件和 Range 续传。
   - 下载并发、有限重试、重试 backoff 和缓存清理阈值已暴露到配置文件；`deploy` / `resume` 支持 CLI 临时覆盖，`cache` 子命令支持 dry-run / apply 清理。
3. 将 state 扩展为可记录 download progress。已完成第一版，stage progress 写入 `state.json`。
4. 增强 Gradio `/config` API discovery。已完成第一版，支持 dependency `api_name` / `fn_index`；当前已补充 queue `/call/<api_name>` + `event_id` follow-up 验证。
5. 增加 log classifier 第一批规则。已完成第一版，覆盖常见依赖、CUDA、磁盘、权限和版本冲突错误；token 权限错误会提取变量名并在 report 中生成无密钥值提示。
6. 设计 repair plan schema，并在失败时生成建议但暂不自动 apply。已完成第一版；当前已增加 policy 校验、artifact 级受控 apply、按 `rerun_from_effective` 的阶段级 resume，以及写入 report 的 resume execution audit，但仍不会直接执行 shell 或修改源码。

第 2 周结束时应达到：

```text
能自动识别一个需要 Hugging Face 权重的 Gradio 项目，
完成缓存式模型准备，
启动服务，
通过 Gradio API 发送 trace，
验证 trace 响应或可读非空文件产物，
并生成完整 evidence/report。
```
