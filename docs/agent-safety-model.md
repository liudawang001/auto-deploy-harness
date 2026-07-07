# AI-Auto-Harness Agent 安全模型

## Threat Model

主要威胁来自四类输入：

- 开源项目仓库中的 README、脚本、配置和日志可能包含 prompt injection。
- 用户或环境可能包含 token、password、API key 等 secret。
- LLM 可能输出越权 action、错误修复、任意命令或外部网络访问。
- Memory promotion 可能把未验证 workaround 固化成长期 skill 规则。

系统假设：LLM 输出不可信，仓库内容不可信，日志不可信。可信边界在 Python controller、policy gate、runtime policy、evidence verifier 和人工审批。

## Prompt Injection 防护

`AgentInputSanitizer` 会在 selected files 进入 LLM prompt 前扫描：

- `ignore previous instructions`
- `disregard system prompt`
- `run shell` / `execute shell` / `rm -rf`
- `print secrets` / `exfiltrate token`
- `curl http(s)://` / `wget http(s)://`
- `base64 decode execute`

命中后不会让文本直接获得控制权；风险会写入 observation extra 和 trace，用于审计。文件名包含 `.env`、`secret`、`credential`、`token`、`key` 的文件默认不送入 LLM prompt。

## Secret Redaction

进入 prompt 或 trace 前会脱敏常见 secret：

- Hugging Face token
- OpenAI-style key
- Bearer token
- `api_key=`
- `api_secret=`
- `password=`
- 环境变量风格的 TOKEN/SECRET/KEY/PASSWORD
- AWS access key
- GitHub token

`AgentTraceWriter` 写入 raw output、parsed decision、policy result 前也会做 redaction。项目文档和 sample manifest 不保存真实密钥值。

## Command Policy

LLM 不能直接执行 shell。run candidate 必须满足：

- `cmd` 是字符串列表。
- 不包含 `;`、`&&`、管道、重定向、反引号、`$()` 等 shell metachar。
- executable 不能是 `bash`、`sh`、`zsh`、`fish`、`cmd`、`powershell`、`curl`、`wget`。
- 真正执行还要经过全局 `allowed_commands`。

这使得 LLM 可以建议“选择已有 Python 启动入口”，但不能把 README 中的恶意 shell 指令变成执行动作。

## Network Policy

默认 benchmark 不访问外网。真实联网下载和 live smoke 是显式外部 gate：

- 模型下载通过 Hugging Face / ModelScope downloader 的受控 URL 和缓存逻辑。
- 下载支持 `.part`、Range 续传、etag/sha256 元数据。
- `agent-live-smoke` 需要操作者通过环境变量注入 provider secret。
- readiness 会把真实联网/GPU/Docker/vLLM smoke 标记为 `external_required` 或 `future_scale_gate`，避免把未执行外部验收误报为已通过。

## Source Edit Policy

当前 Agent 默认不直接修改用户项目源码。`propose_source_patch` 或 action `requires.source_edit=true` 必须通过 runtime policy 的 `allow_source_edit`，否则被拒绝。

Skill promotion 也不是直接自动学习：

- 先生成 proposal JSON/Markdown。
- 必须人工 approve。
- apply 后默认运行绑定 benchmark case。
- 回归失败返回非 0。

## Repair Action Policy

repair action 由 `RepairPlanner` 生成，再经过 `RepairPolicy`：

- `dependency_install` 需要 runtime policy 允许。
- `service_restart` 需要 runtime policy 允许。
- `operator_secret` 永远不能由 LLM 代填，只能请求环境变量名。
- `operator_approval` 需要显式审批记录。
- `install_package` 的 package spec 不能包含 URL、路径、`git+`、extra index、trusted host 等。

`RepairApplier` 只有在 `gated_actor`、runtime policy、repair policy、command policy 都通过时才会执行受控命令，并记录 stdout/stderr tail、exit code、executed flag 和脱敏结果。

## Memory Promotion Safety

普通 issue memory 默认：

```json
{
  "verified_success": false,
  "verification_trace_id": "",
  "repair_action_hash": "",
  "regression_case_ids": [],
  "policy_rejected_high_risk": false
}
```

只有满足以下条件才会参与 skill promotion：

- `verified_success=true`
- 存在 `verification_trace_id`
- 存在 `repair_action_hash`
- `regression_case_ids` 非空
- verify/regression 状态为 pass
- 没有 high-risk policy reject

这避免把“LLM 说可能可行”的建议长期写入 skill。

## Audit Trail

关键审计产物：

- `task.json`
- `state.json`
- `events.jsonl`
- `reports/pipeline_results.json`
- `reports/report.md`
- `reports/agent_metrics.json`
- `logs/agent_calls/*.json`
- `logs/agent_loop/*.json`
- `repairs/repair_plan.json`
- `repairs/repair_apply_result.json`
- `evidence/*.json`
- `memory/promotions/*.json`
- `memory/promotions/*.regression.json`

这些文件用于回答面试中的核心追问：LLM 为什么这么建议、系统为什么接受或拒绝、是否真的执行、执行后是否通过 verify、是否有回归证据。
