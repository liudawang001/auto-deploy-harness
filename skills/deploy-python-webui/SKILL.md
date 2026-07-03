---
name: deploy-python-webui
description: 部署 Python AI WebUI 项目。用于 env_deploy 和 runner 阶段，覆盖 Gradio、Streamlit、Flask、FastAPI、PyTorch、Transformers、requirements 项目的虚拟环境创建、依赖安装、启动命令选择和端口就绪检查。
---

# Python WebUI 部署

目标：在隔离的运行目录中构建 Python 环境，并用受控权限启动最可能正确的服务命令。

## 执行流程

1. 在复制后的 run workspace 中创建 `.venv`。
2. 按证据强度选择依赖安装方式：
   - `requirements.txt`：`.venv/bin/python -m pip install -r requirements.txt`
   - `pyproject.toml`：`.venv/bin/python -m pip install .`
   - `setup.py`：`.venv/bin/python -m pip install .`
3. 按框架选择启动命令：
   - Gradio：优先运行包含 `.launch` 的入口文件，例如 `.venv/bin/python app.py`。
   - Streamlit：使用 `.venv/bin/streamlit run app.py`。
   - FastAPI：若能明确 `app` 对象，优先使用 `uvicorn module:app --host 127.0.0.1 --port 8000`。
4. 将 stdout/stderr 写入 run log。
5. 如果进程在就绪前退出，即使日志中曾打印 URL，也必须判定为失败。

## 常见失败处理

- 安装后缺失模块：记录缺失包名和依赖文件；只有在允许修改源代码时，才建议补充或 pin 依赖。
- 端口被占用：只有框架支持无源码改动切换端口时，才尝试备用端口。
- GPU 或模型下载失败：记录环境需求；不能用空白 UI 伪造成功。
- API key 缺失：只记录变量名，不记录密钥值。

## 证据要求

部署阶段不等于验证成功。runner 阶段至少需要活跃进程、端口就绪证据以及可供后续诊断的日志路径。
