---
name: solve-python-cuda-env
description: 求解 Python/CUDA/Torch 依赖环境。用于 env_solve 阶段，识别 Python 版本范围、Torch CUDA wheel、老 Gradio 与 numpy/pydantic 兼容风险、flash-attn/bitsandbytes/xformers 等 GPU 依赖风险，生成可审计 install plan 和 constraints。
---

# Python / CUDA 环境求解

目标：在真正执行 `pip install` 前，先把依赖安装方案变得可解释、可审计、可回滚。

## 输入证据

优先读取：

1. `requirements.txt`
2. `pyproject.toml`
3. `setup.py`
4. `README.md`
5. `resource_plan` 中的 Python、CUDA、GPU 和模型资产信息

## 求解策略

- 不直接盲装依赖。
- 老 Gradio 或未 pin Gradio 项目优先加 `numpy<2` 和 `pydantic<2` 约束。
- Headless 部署中出现 `opencv-python` 时，优先建议 `opencv-python-headless`。
- 检测到 `flash-attn`、`xformers`、`bitsandbytes`、`triton` 时，标记 CUDA/toolchain 风险。
- 检测到 GPU/CUDA 信号但 Torch 未 pin 时，标记 Torch wheel variant 风险。
- 只生成计划和风险说明，真正执行仍由 `env_deploy` 根据命令白名单完成。

## 输出要求

输出结构应包含：

- `backend`: 例如 `local_venv`
- `python`: 例如 `python3`、`3.10`
- `install_plan`: 约束后的安装命令
- `constraints`: 自动增加的依赖约束
- `constraint_reasons`: 每条约束的原因
- `risk_reasons`: GPU/CUDA/Torch/构建相关风险

## 安全边界

不要修改源码。不要执行 shell。不要为了提高成功率随意升级核心依赖；所有新增约束必须有可解释原因，并写入阶段结果。
