---
name: analyze-ai-demo
description: 分析 AI demo 仓库并生成自动部署计划。用于 analyze/project classification 阶段，识别 Gradio、Streamlit、FastAPI、Flask、PyTorch、Transformers 项目的框架、入口文件、依赖文件、模型资产、服务形态以及安全的 install/run/verify 方案。
---

# AI Demo 项目分析

目标：基于仓库证据生成保守、可执行、可审计的部署计划，不依赖主观猜测。

## 必查项

1. 优先检查依赖文件：`requirements.txt`、`pyproject.toml`、`setup.py`、`environment.yml`、`package.json`、`Dockerfile`。
2. 检查常见入口文件：`app.py`、`main.py`、`server.py`、`webui.py`、`demo.py`、`api.py`。
3. 识别框架与服务形态：
   - Gradio：出现 `gr.Interface`、`gr.Blocks`、`.launch`，默认端口通常为 `7860`。
   - Streamlit：通过 `streamlit run` 启动，默认端口通常为 `8501`。
   - FastAPI：出现 `FastAPI()`、`uvicorn`，默认端口通常为 `8000`。
   - Flask：出现 `Flask(__name__)`，默认端口通常为 `5000`。
4. 识别模型与运行时约束：大模型权重、GPU-only 依赖、本地 checkpoint 路径、外部 API key、数据集下载。
5. 默认不建议修改源代码。若确实需要修改，必须作为“修复建议”输出，不能作为默认部署动作。

## 输出要求

输出结构化建议，至少包含：

- `frameworks`：识别出的框架标签。
- `install_plan`：按顺序执行的安装命令。
- `run_candidates`：启动命令、预期端口、置信度和依据。
- `verify_hint`：HTTP method/path/body 模板，或文件产物验证策略。
- `risks`：缺失文件、未知环境变量、重型模型下载、GPU 需求等风险。

## 安全边界

不要建议破坏性命令。不要写入或输出密钥。除非同时存在入口文件和可行的验证路径，否则不要把项目判定为“可部署成功”。
