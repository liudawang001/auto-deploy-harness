---
name: diagnose-runtime-failure
version: "1.0.0"
type: repair_skill
stages: [repair, replan, runner, env_deploy]
frameworks: []
failure_categories: [dependency_missing, port_conflict, import_error, process_exited]
risk_level: medium
side_effects: false
allowed_tools: [classify_failure, propose_repair, apply_dependency_constraint]
success_signals: [root_cause_classified, repair_plan_generated, issue_memory_written]
regression_cases: [missing_dependency_detected, numpy_pydantic_conflict_resolved, wheel_build_failure_flagged, port_conflict_identified, api_key_missing_recorded]
---

# Purpose

诊断并沉淀自动部署中的复发问题。用于 env_deploy、runner、verify 阶段失败或 uncertain 时，处理依赖错误、进程退出、模型文件缺失、端口冲突、API key 缺失、验证证据不足，并写入可复用 issue memory。

把失败或不确定阶段转化为可复用的问题诊断，避免泄漏密钥，也不伪造成功。

# When To Use

- Pipeline 处于 repair、replan、runner 或 env_deploy 阶段且出现失败或 uncertain 时自动激活。
- 需要对运行失败进行根因分类和修复建议时。
- 需要将可复用问题模式写入 issue memory 时。

# Guidance

## 排查顺序

1. 确认失败阶段和最新 evidence 路径。
2. 只读取有界日志尾部和阶段 JSON 结果。
3. 分类根因：
   - 依赖安装失败；
   - 启动命令或入口文件不匹配；
   - 服务就绪失败；
   - 模型资产或硬件需求缺失；
   - 环境变量缺失；
   - 验证链路证据不足。
4. 优先使用内置 log classifier 的结构化结果；只有规则无法覆盖或置信度较低时，才把裁剪后的日志交给 LLM 进一步分析。
   - 缺失依赖应提取 package，并生成 `install_package` 建议。
   - numpy/pydantic/protobuf 冲突应生成兼容 constraint，例如 `numpy<2`、`pydantic<2`、`protobuf<=3.20.3`。
   - wheel build 失败应提取失败包名；GPU 可选扩展构建失败优先建议兼容 wheel、跳过可选扩展或回到 `env_solve`。
   - token 权限问题只记录环境变量名和 `values_recorded=false`。
5. 判断修复是否被当前策略允许：
   - 是否允许安装依赖；
   - 是否允许启动服务；
   - 是否允许修改源代码；
   - 是否允许联网或下载模型。
6. 写入 issue memory，包含 symptom、root cause、affected framework、evidence 和 next action。
7. 生成结构化 repair plan，并经过 policy 校验后写入受控 repair artifacts；直接执行 shell 或修改源码仍必须经过后续 pipeline 权限开关。

## 记忆写入规则

记录可复用模式，不记录一次性噪声。不要写入密钥值。环境变量只记录变量名。优先形成可检索签名，例如 `gradio api shape unknown` 或 `torch wheel incompatible with python version`。

## 修复边界

当 `allow_source_edit` 为 false 时，不要静默修改源码。不要无限重试。任何 workaround 都必须重新经过 verify 阶段并产生强证据后，才能视为成功。

当前 repair apply 只会写入可审计 artifacts，例如依赖安装建议、verify hint 建议、所需环境变量名。不要把 repair artifacts 的存在视为修复已经执行。

# Allowed Plan Effects

- 分类失败根因并输出结构化诊断。
- 生成修复建议（依赖安装、约束调整、verify hint 更新等）。
- 写入 issue memory 记录可复用问题模式。
- 写入受控 repair artifacts。

# Forbidden

- 不要泄漏密钥值。
- 不要伪造成功。
- 不要在 `allow_source_edit` 为 false 时静默修改源码。
- 不要无限重试。
- 不要把 repair artifacts 的存在视为修复已经执行。
- 不要直接执行 shell 或修改源码（必须经过 pipeline 权限开关）。
