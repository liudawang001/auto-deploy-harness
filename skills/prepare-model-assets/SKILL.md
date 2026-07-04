---
name: prepare-model-assets
description: 规划和准备开源模型资产。用于 resource_plan 和 model_prepare 阶段，识别 Hugging Face、ModelScope、Git LFS、大文件 checkpoint、safetensors、bin 权重，估算磁盘/GPU 资源，生成可恢复下载 manifest 和本地 model_cache 缓存策略。
---

# 模型资产准备

目标：在真正下载或启动服务前，先识别模型资产、评估资源风险，并生成可恢复的资产清单。

## 识别来源

优先从以下位置识别模型：

1. README 中的 Hugging Face / ModelScope 链接。
2. Python 代码中的 `from_pretrained("org/model")`。
3. `snapshot_download(repo_id="org/model")`。
4. Git LFS 权重文件，例如 `.safetensors`、`.bin`、`.ckpt`。
5. 配置文件中的 model id、checkpoint path、revision。

## 资源判断

需要输出：

- 模型来源和 repo id。
- revision 或 commit。
- 是否需要 token，例如 `HF_TOKEN`。
- 预估磁盘空间。
- 是否可能需要 GPU/CUDA。
- 是否支持 resume。
- 缓存路径和 cache key。

## 安全边界

不要在 skill 中写密钥。不要在未检查磁盘空间和 token 的情况下直接下载大文件。下载计划必须写入 manifest，便于中断后恢复。

## 当前实现状态

当前阶段已经支持 Hugging Face 资产下载的第一版实现：

- 通过 Hugging Face tree API 获取文件清单。
- 使用 `resolve` URL 下载模型文件和配置文件。
- 通过可配置 ModelScope API base / download base 获取 ModelScope 文件清单并下载。
- 使用 `.part` 文件保存未完成下载。
- 已有 `.part` 时通过 HTTP Range 续传。
- 文件清单包含 `sha256` 时，下载后必须校验 sha256。
- 进度写入 stage result 和 `state.json`。

并发下载和 etag 强一致校验尚未接入。sha256 只在远端清单提供该字段时可用，不要假定所有模型文件都有 checksum。
