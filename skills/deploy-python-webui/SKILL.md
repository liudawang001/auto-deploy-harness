---
name: deploy-python-webui
version: "1.0.0"
type: execution_skill
stages: [runner, plan, plan_first, replan]
frameworks: [gradio, streamlit, fastapi, flask]
risk_level: low
side_effects: false
allowed_tools: [add_runner_candidate, select_runner_candidate, set_stage_hint]
success_signals: [process_alive, port_ready, run_log_written]
regression_cases: [gradio_launch_success, streamlit_launch_success, fastapi_uvicorn_launch, flask_launch, docker_plan_generated, process_exit_detected]
---

# Purpose

部署 Python AI WebUI 项目。用于 env_deploy 和 runner 阶段，覆盖 Gradio、Streamlit、Flask、FastAPI、PyTorch、Transformers、requirements 项目的虚拟环境创建、依赖安装、启动命令选择和端口就绪检查。

在隔离的运行目录中构建 Python 环境，并用受控权限启动最可能正确的服务命令。

# When To Use

- Pipeline 处于 runner、plan_first 或 replan 阶段时自动激活。
- 需要为 Python WebUI 项目创建虚拟环境、安装依赖、启动服务时。
- 需要生成 Docker 部署计划时。

# Guidance

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
6. 当 `execution_backend=docker` 时，将安装和启动命令包装为 `docker run` 计划，并记录原始命令与 effective command。Docker 计划应包含 workspace mount、端口映射、网络参数、可选 `--gpus`、`model_cache` 挂载、容器日志命令和清理命令元数据。Docker 真实执行仍必须经过 `--execute`、权限开关和 `allowed_commands` 中的 `docker` 白名单。

## 常见失败处理

- 安装后缺失模块：记录缺失包名和依赖文件；只有在允许修改源代码时，才建议补充或 pin 依赖。
- 端口被占用：只有框架支持无源码改动切换端口时，才尝试备用端口。
- GPU 或模型下载失败：记录环境需求；不能用空白 UI 伪造成功。
- API key 缺失：只记录变量名，不记录密钥值。
- Docker 未授权：返回 command rejected，不要绕过白名单改成本地执行。
- Docker 运行失败：优先保留 `docker logs <container>` 和 `docker rm -f <container>` 元数据，不要吞掉容器侧 stdout/stderr。

## 证据要求

部署阶段不等于验证成功。runner 阶段至少需要活跃进程、端口就绪证据以及可供后续诊断的日志路径。

# Allowed Plan Effects

- 向 pipeline 提交 runner candidate（启动命令、端口、环境）。
- 选择并确认 runner candidate 执行。
- 设置 stage hint 指导后续验证行为。
- 生成 Docker 部署计划（需经权限校验后执行）。

# Forbidden

- 不要用空白 UI 伪造成功。
- 不要绕过白名单执行 Docker 命令。
- 不要吞掉容器侧 stdout/stderr。
- 不要在进程退出后仅凭日志中的 URL 判定成功。
- 不要记录密钥值。
