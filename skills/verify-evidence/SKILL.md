---
name: verify-evidence
description: 对 AI 自动部署结果做证据化验证。用于 verify 阶段，覆盖 WebUI/API 服务，尤其是 Gradio、FastAPI、Flask、Streamlit 项目；要求发送可追踪 trace 请求、检查响应/产物/日志，诊断 uncertain，避免把 HTTP 200 误判为成功。
---

# 证据化 Verify

目标：证明本次部署的服务处理了当前 run 生成的新 trace；如果无法证明，必须返回 `uncertain`。

## 验证策略

1. 每次 verify 都生成唯一 `trace_id`。
2. 优先使用框架 API，而不是只检查页面是否能打开：
   - Gradio：优先读取 `/config`，根据 dependency 的 `api_name` / `fn_index` 构造请求；无法 discovery 时再回退 POST `/api/predict`，请求体可用 `{"data":["{{trace_id}}"]}`。
   - FastAPI/Flask：优先读取 `/openapi.json`，选择不含 path parameter、带 JSON requestBody 的 POST endpoint，根据 schema 构造最小 trace JSON 请求。
   - vLLM/OpenAI-compatible：调用 `/v1/chat/completions`，把 `trace_id` 放入 user message；只有 response body 返回当前 trace 才通过。
   - Streamlit：当前支持 DOM/HTML probe 第一版，能识别 Streamlit 页面标记、错误标记和 trace；后续可升级为 Playwright 真浏览器交互。
3. 端口开放或 HTTP 200 只能证明服务可能活着，不能证明业务链路成功。
4. 只有出现至少一种强证据时才能通过：
   - 响应体包含当前 `trace_id`；
   - trace 执行后生成了新的输出产物；
   - 框架事件或日志能证明当前 trace 被处理。
5. 保存请求、响应尾部、状态码、body 模板和 evidence 文件路径。
6. 长耗时首次推理或模型加载期间必须持续刷新阶段进度，至少记录 service discovery、HTTP trace 请求开始/结束、follow-up、browser/Streamlit probe 和最终 verify 状态。

## 诊断分类

- `service_unreachable`：服务不可达，或端口未就绪。
- `api_shape_unknown`：服务存在，但可调用 API 形态未知。
- `trace_not_observed`：请求成功，但响应或产物没有证明 trace 被处理。
- `artifact_missing`：预期输出文件不存在或不是本次新生成。
- `dry_run_missing_evidence`：当前是 dry-run，天然缺少真实执行证据。

## 修复方向

当 verify 为 `uncertain` 时，下一步应检查服务 API 形态和 runner 日志，然后更新 `verify_hint` 或新增框架专用 verify skill。不要为了让 pipeline 变绿而降低验证标准。

当前 Streamlit DOM probe 不是完整浏览器自动化。它可以作为 readiness/DOM evidence，但复杂表单输入、按钮点击、文件上传仍需要后续 browser backend。

对于大模型首次推理，不要因为等待时间长就提前判失败；应通过进度状态和日志证据区分“仍在加载模型”和“已经抛出异常”。
