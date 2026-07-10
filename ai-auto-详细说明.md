# AI-Auto-Harness 详细说明

本文用于面试前系统学习 AI-Auto-Harness。阅读目标不是背 API，而是能讲清楚：为什么要做这个 Agent、架构怎么拆、每个模块解决什么真实问题、失败后如何自动修复、为什么 verify 模块是项目的核心壁垒。

## 1. 项目定位

AI-Auto-Harness 是一个面向开源 AI demo / 模型项目的自动部署 Agent。它要解决的问题不是“执行一个脚本”，而是把 GitHub、Hugging Face、ModelScope 上形态不统一的项目，自动分析、下载模型、求解环境、启动服务、验证结果、诊断失败、生成修复方案，并把高频失败沉淀为可复用经验。

一句话介绍：

> 我把开源模型部署拆成确定性控制链路和受控 Agent 推理链路。Python controller 负责状态机、安全、下载、执行和证据校验；LLM/Claude Code 只参与项目阅读、日志理解和修复建议，不能绕过 policy 直接执行命令。

项目区别于普通部署脚本的关键点：

- 不是 health check，而是 evidence-based verify。
- 不是盲装 requirements，而是先做 env_solve。
- 不是下载大模型到工作区，而是 model_cache + manifest + 断点续传。
- 不是失败后全量重跑，而是 repair plan + safe rerun stage。
- 不是让 LLM 直接操作系统，而是 Python policy 控制执行边界。
- 不是把经验写在 prompt 里，而是 skill + memory + promotion。

## 2. 当前整体进度

按照 `docs/progress.md`，项目当前约 90%：

- P0/P1 主链路基本完成：状态机、下载缓存、verify 防误判、repair/resume、benchmark 已成型。
- 真实开源模型部署能力已覆盖 Hugging Face、ModelScope、Git LFS、Gradio、Streamlit、Browser verify、Docker backend、PyTorch CUDA wheel 求解。
- 剩余主要是：真实联网长耗时 E2E、真实 Docker/GPU smoke、vLLM/OpenAI-compatible server verify、更多具体版本矩阵和 dashboard。

## 3. 总体架构

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

当前实现中的核心 pipeline 在 `TaskRunner.PIPELINE_STAGES`：

```text
analyze -> resource_plan -> env_solve -> env_deploy -> model_prepare -> runner -> verify -> report
```

核心文件：

- `src/auto_harness/orchestrator.py`：总编排器。
- `src/auto_harness/modules/analyzer.py`：项目分析。
- `src/auto_harness/modules/resource_plan.py`：资源规划。
- `src/auto_harness/modules/env_solve.py`：Python/CUDA/Torch 依赖求解。
- `src/auto_harness/modules/env_deploy.py`：环境部署。
- `src/auto_harness/modules/model_prepare.py`：模型资产准备。
- `src/auto_harness/modules/runner.py`：服务启动。
- `src/auto_harness/modules/verify.py`：证据化验证。
- `src/auto_harness/repair/*`：修复计划、策略、应用和循环控制。
- `src/auto_harness/assets/*`：模型资产检测、缓存、下载器。
- `src/auto_harness/runtime/sandbox.py`：Docker sandbox 命令包装。
- `src/auto_harness/memory/*`：问题记忆与 skill promotion。
- `tests/fixtures/benchmarks/manifest.json`：回归基准入口。

## 4. 设计原则

### 4.1 确定性控制优先

所有会影响机器状态的行为都由 Python controller 决定：

- 是否安装依赖。
- 是否启动服务。
- 是否执行 Git LFS。
- 是否应用 repair artifact。
- 是否复用缓存。
- 是否从中间阶段 resume。

LLM/Agent 只给建议。即便 Claude Code analyzer advisor 提供了建议，也只能写入 `agent_advice` 元数据，不能覆盖确定性 analyzer 输出。

### 4.2 默认安全

默认 `deploy` 是 dry-run。真实执行必须同时满足：

- CLI 显式传 `--execute`。
- 依赖安装要有 `--allow-install`。
- 服务启动要有 `--allow-start`。
- 命令必须在 `allowed_commands` 白名单中。

Docker backend 也是这样。即使生成了 `docker run` effective command，没有把 `docker` 加入白名单也会被拒绝。

### 4.3 证据化成功

项目不接受：

```text
进程存在 + 端口开放 + HTTP 200 = 成功
```

真正通过 verify 必须有强证据：

- HTTP 响应包含本次 `trace_id`。
- Gradio queue follow-up 响应包含本次 `trace_id`。
- 浏览器 DOM 包含本次 `trace_id`。
- 本次 trace 后生成了新的、非空、可读、带 sha256 的产物。

## 5. 阶段详解

### 5.1 analyze

文件：`src/auto_harness/modules/analyzer.py`

职责：

- 收集 repo 文件列表。
- 从 README、requirements、pyproject、package.json 中检测框架。
- 生成基础 install plan。
- 生成 run candidates。
- 生成 verify_hint。
- 可选调用 Claude Code advisor。

当前支持识别：

- `gradio`
- `streamlit`
- `fastapi`
- `flask`
- `torch`
- `transformers`
- `node`
- `unknown`

Gradio 项目默认 verify hint：

```json
{
  "service_type": "webui",
  "request": {
    "method": "POST",
    "path": "/api/predict",
    "json": {"data": ["{{trace_id}}"]}
  }
}
```

面试要点：

- analyzer 是确定性 baseline，不依赖 LLM。
- LLM advice 是增强项，不参与安全决策。
- run candidate 是候选，不代表最终成功。

### 5.2 resource_plan

文件：`src/auto_harness/modules/resource_plan.py`

职责是在真实下载/安装前做资源评估：

- 检测模型资产来源。
- 识别 GPU/CUDA 信号。
- 估算磁盘和显存需求。
- 检测外部 token 需求。
- 检测 Git LFS。

输出包括：

- `python_range`
- `gpu_required`
- `cuda_required`
- `torch_variant`
- `estimated_vram_gb`
- `estimated_disk_gb`
- `external_tokens`
- `risk_level`
- `risk_reasons`
- `model_assets`
- `git_lfs`

Git LFS 是关键工程点。很多模型权重在 GitHub 中只是 pointer 文件，如果不检测就会误以为权重已存在。项目会解析 `.gitattributes` 和 pointer 文件，记录 oid/size，并在缺少 `git-lfs` 时返回 `uncertain`，而不是继续假装成功。

### 5.3 env_solve

文件：`src/auto_harness/modules/env_solve.py`

职责是在 `pip install` 前生成更稳的依赖方案。

已实现能力：

- 老 Gradio 项目自动加 `numpy<2`、`pydantic<2`。
- headless 部署中优先建议 `opencv-python-headless`。
- 探测 `AUTO_HARNESS_CUDA_VERSION`、`nvidia-smi`、`nvcc`。
- 根据 CUDA 选择 PyTorch wheel：
  - CUDA 12.1+ -> `cu121`
  - CUDA 11.8+ -> `cu118`
  - 无兼容 CUDA -> `cpu`
- 为 `torch` / `torchvision` / `torchaudio` 生成预安装命令。
- 保留 CPU fallback。
- 生成 GPU 包兼容矩阵：
  - `xformers`
  - `flash-attn`
  - `bitsandbytes`
  - `triton`

面试重点：

> 真实部署中很多失败不是代码问题，而是 Python/CUDA/Torch/包版本组合问题。所以我把 env_solve 独立成阶段，在安装前把风险结构化，避免盲装 requirements。

### 5.4 env_deploy

文件：`src/auto_harness/modules/env_deploy.py`

职责：

- 消费 env_solve 后的 install_plan。
- dry-run 时只生成计划。
- execute 时逐条执行命令。
- 每条命令走白名单校验。
- 失败日志进入 LogClassifier。
- 支持 Docker backend。

Docker backend 不直接改变业务逻辑，而是把原始命令包装为 effective command：

```text
docker run --rm
  -v <repo>:/workspace/repo
  -w /workspace/repo
  --network <network>
  --gpus <gpus>
  -v <model_cache>:/workspace/model_cache
  <image>
  <original command>
```

### 5.5 model_prepare

职责：

- 生成模型资产 manifest。
- 使用模型缓存。
- 支持 Hugging Face / ModelScope 下载。
- 支持 `.part` + Range 断点续传。
- 支持并发下载。
- 支持 sha256 / etag 校验。
- 支持缓存清理。
- 支持 Git LFS 受控执行。

工程价值：

- 大模型下载可能数 GB 到数十 GB，必须支持中断恢复。
- 同一模型多次部署必须复用缓存，不能重复下载。
- etag 变化要让缓存失效，否则可能使用旧权重。
- token 缺失只记录变量名，不能记录值。

### 5.6 runner

文件：`src/auto_harness/modules/runner.py`

职责：

- 从 run_candidates 里选择启动命令。
- dry-run 只返回候选。
- execute 时启动服务。
- 等待端口就绪。
- 如果进程很快退出，不能误判成功。
- 写 runner log。
- 支持 Docker backend。

关键细节：

- 服务启动成功不等于部署成功。
- runner 只证明进程和端口，verify 才证明业务链路。

### 5.7 verify

文件：`src/auto_harness/modules/verify.py`

verify 是项目最重要的模块。

它解决的是“自动部署假成功”问题。很多 Agent 看到 HTTP 200 就认为成功，但 AI demo 可能只是首页能打开，模型没有加载、API 没处理输入、输出文件是旧的、Streamlit 页面报错但状态码仍是 200。

当前 verify 能力：

- 生成唯一 `trace_id`。
- GET query trace。
- POST JSON trace。
- Gradio `/config` discovery。
- Gradio queue `/call/<api_name>` + event_id follow-up。
- Streamlit DOM/HTML probe。
- Browser DOM probe。
- 文件产物 freshness + 非空 + sha256 校验。
- 长耗时进度刷新。

强通过证据：

- `http_trace_response=pass`
- `browser_dom_probe=pass`
- `artifact_download_validation=pass`

失败或 uncertain 会进入 memory 和 repair plan。

### 5.8 repair loop

文件：`src/auto_harness/repair/*`

职责：

- 根据失败阶段生成结构化 repair plan。
- 用 policy 检查修复动作是否允许。
- 用 loop controller 限制同一问题尝试次数。
- 生成受控 repair artifacts。
- resume 时从安全阶段重跑。

重要设计：

- repair_apply 不直接执行 shell。
- overlay 只改变阶段输入。
- 需要人工审批的 action 必须先 approve。
- 不安全的 `rerun_from` 会被回退。
- 中间阶段 resume 会生成 execution audit。

### 5.9 skill 与 memory

Skill：

```text
skills/<skill-name>/SKILL.md
```

部署时每个阶段会选择相关 skill，把路径和 SHA-256 写入 `control_context`。这保证 agent 运行时有可审计控制文档。

Memory：

```text
memory/deployment_issues.jsonl
```

阶段 failed/uncertain 后写入结构化问题记忆。后续任务按 stage/framework 检索相似问题。

Memory promotion：

- 聚类高频 issue。
- 生成 proposal。
- 绑定目标 skill。
- 绑定 regression cases。
- 必须审批后才能 apply。

这体现了“部署经验可沉淀”的 Agent 能力。

## 6. 面试中怎么讲这个项目

推荐讲法：

1. 先讲业务痛点：开源模型部署失败率高，问题集中在依赖、权重、CUDA、入口、验证。
2. 再讲架构拆分：确定性 controller + 受控 LLM advisor。
3. 讲主 pipeline：analyze 到 report。
4. 重点讲 verify：不接受 HTTP 200，必须 trace/DOM/artifact 证据。
5. 讲 repair：policy、attempt limit、safe rerun。
6. 讲 memory：问题沉淀到 skill。
7. 讲 benchmark：35 个本地 case 防回归。

## 7. 你负责 verify 模块时的表达

你可以这样说：

> 我主要负责 verify 模块。这个模块的目标是阻止自动部署中的假成功。我没有把端口开放或 HTTP 200 当作成功，而是为每次验证生成唯一 trace_id，通过 Gradio API、queue follow-up、Streamlit DOM、浏览器 DOM 或新产物 sha256 来证明当前请求被模型链路处理。对于大模型首次加载，我还加入了进度刷新，避免长时间 verify 看起来像卡死。verify 失败会输出结构化 diagnosis，并进入 memory/repair 链路。

可展开细节：

- Gradio `/config`：识别 dependency、api_name、fn_index。
- Queue 模式：POST `/call/<api>` 得到 event_id，再 GET follow-up。
- Artifact：只接受本次 trace 后新增/修改的非空文件，并记录 sha256。
- Browser DOM：页面能打开不够，DOM 必须包含 trace 或出现错误标记。
- False positive 防护：HTTP 200 无 trace 保持 uncertain。

## 8. 项目不足与后续方向

当前不足：

- 真实联网 E2E 还不充分。
- 真实 Docker/GPU smoke 需要在有 GPU 的机器上验证。
- vLLM/OpenAI-compatible server verify 还需增强。
- GPU 包矩阵还需要具体版本规则。
- dashboard 还没有。

后续可以做：

- 添加 live smoke manifest。
- 增加 vLLM `/v1/chat/completions` verify。
- 增加真实 HF/ModelScope 小模型端到端。
- 增加 Docker image 预热。
- 增加 memory promotion dashboard。

## 9. 面试总结话术

> 这个项目本质上是一个自动部署 Agent 的工程控制系统。我把 LLM 能力限制在不确定性强的分析和诊断环节，把状态、安全、执行、下载、验证都交给确定性 Python 模块。核心难点不是会不会调用模型，而是如何在不可信开源项目和长耗时大模型部署场景下，保证过程可恢复、结果可验证、失败可诊断、修复可审计、经验可复用。
