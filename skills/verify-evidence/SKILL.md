---
name: verify-evidence
description: 对 AI 自动部署结果做证据化验证。用于 verify 阶段，覆盖 WebUI/API 服务，尤其是 Gradio、FastAPI、Flask、Streamlit 项目；要求发送可追踪 trace 请求、检查响应/产物/日志，诊断 uncertain，避免把 HTTP 200 误判为成功。
---

# 证据化 Verify

目标：证明本次部署的服务处理了当前 run 生成的新 trace；如果无法证明，必须返回 `uncertain`。

## 验证策略

1. 每次 verify 都生成唯一 `trace_id`。
2. 优先使用框架 API，而不是只检查页面是否能打开：
   - Gradio：优先 POST `/api/predict` 或已发现的 API endpoint，请求体可用 `{"data":["{{trace_id}}"]}`。
   - FastAPI/Flask：调用能 echo 或处理 trace 输入的 GET/POST endpoint。
   - Streamlit：单纯 HTTP readiness 证据较弱；若无 API，需要 DOM、日志或文件产物证据。
3. 端口开放或 HTTP 200 只能证明服务可能活着，不能证明业务链路成功。
4. 只有出现至少一种强证据时才能通过：
   - 响应体包含当前 `trace_id`；
   - trace 执行后生成了新的输出产物；
   - 框架事件或日志能证明当前 trace 被处理。
5. 保存请求、响应尾部、状态码、body 模板和 evidence 文件路径。

## 诊断分类

- `service_unreachable`：服务不可达，或端口未就绪。
- `api_shape_unknown`：服务存在，但可调用 API 形态未知。
- `trace_not_observed`：请求成功，但响应或产物没有证明 trace 被处理。
- `artifact_missing`：预期输出文件不存在或不是本次新生成。
- `dry_run_missing_evidence`：当前是 dry-run，天然缺少真实执行证据。

## 修复方向

当 verify 为 `uncertain` 时，下一步应检查服务 API 形态和 runner 日志，然后更新 `verify_hint` 或新增框架专用 verify skill。不要为了让 pipeline 变绿而降低验证标准。
