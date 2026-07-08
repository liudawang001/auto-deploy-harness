# Live Agent Smoke

`agent-live-smoke` 是可选的联网 smoke 命令，用于证明 LLM Agent 路径能产出真实可审计 artifact。密钥只能通过环境变量注入，不能写入仓库文件、命令历史、报告或 manifest。

## Provider 配置

Mock provider 不需要任何环境变量，适合本地 dry-run 验证命令和 manifest 结构：

```bash
PYTHONPATH=src python3 -m auto_harness.cli agent-live-smoke \
  --provider mock \
  --output runs/live_smoke/mock_dry_run
```

Xunfei provider 读取以下环境变量：

```text
XUNFEI_API_BASE 或 XUNFEI_API_URL
XUNFEI_API_KEY
XUNFEI_APP_ID
XUNFEI_MODEL
XUNFEI_TIMEOUT_SECONDS
XUNFEI_MAX_TOKENS
XUNFEI_ANTHROPIC_VERSION
```

只在本机 shell 私下 export 这些变量。不要把真实值放进 README、进度文档、JSON 配置、测试 fixture 或提交记录。

## 运行命令

```bash
PYTHONPATH=src python3 -m auto_harness.cli agent-live-smoke \
  --repo tests/fixtures/live/llm_repair_missing_dependency \
  --provider xunfei \
  --execute \
  --output runs/live_smoke/manual
```

`--execute` 会允许安装 fixture 所需的小型依赖并启动本地服务；不带 `--execute` 时只做 dry-run，不应把 dry-run 的 `uncertain` verify 当成真实 smoke pass。

如果要证明 repair/resume 闭环，而不是让 analyze planner 提前规避失败，可使用 repair-mode：

```bash
PYTHONPATH=src python3 -m auto_harness.cli agent-live-smoke \
  --repo tests/fixtures/live/llm_repair_missing_dependency \
  --provider xunfei \
  --execute \
  --disable-analyze-planner \
  --resume-attempts 1 \
  --output runs/live_smoke/xunfei_repair_real
```

2026-07-08 已用真实 Xunfei provider 跑通该 repair-mode，归档 manifest 在 `docs/evidence/live-agent-smoke-manifest.json`。该 manifest 不包含密钥值、完整 prompt 或原始大日志。

## 成功判断

真实 smoke 通过应同时满足：

- 命令返回 exit code 0。
- `live-agent-smoke-manifest.json` 存在。
- `provider_name` 是真实 provider 名称。
- `repair_executed_count` 大于 0，证明受控 repair action 执行过。
- `resume_attempt_count` 大于 0，证明 repair 后确实触发过受控 resume。
- `final_verify_status` 为 `passed` 或 `pass`。
- `artifact_paths` 至少包含 task/state/events、agent call trace、repair plan/apply、pipeline results 和 verify evidence。
- manifest 只包含 metadata、状态、路径和 sha256，不包含 prompt、token、完整日志或 workspace 副本。

## Expected Artifact Classes

- `task.json`, `state.json`, `events.jsonl`
- `logs/agent_calls/*.json`
- `repairs/repair_plan.json`
- `repairs/repair_apply_result.json`
- `reports/pipeline_results.json`
- `evidence/*verify*.json`
- `live-agent-smoke-manifest.json`

## 跳过和失败条件

以下情况应记录为 skipped 或 external_required，不能误报为 passed：

- 未配置 `XUNFEI_API_BASE` / `XUNFEI_API_URL`。
- 未配置 provider key 或模型名。
- 当前网络不能访问 provider endpoint。
- 依赖安装被 runtime policy 禁止。
- 本地端口被占用，服务未启动。
- verify 没有观测到当前 trace id。

当真实 provider 的必要环境变量缺失时，`agent-live-smoke` 会直接写出 skipped manifest，其中 `final_verify_status=skipped`，`external_gate.status=external_required`，`missing_env` 只包含环境变量名，不包含任何密钥值。

已提交的样例 manifest 位于 `docs/evidence/live-agent-smoke-manifest.json`，只保留 metadata、artifact paths、计数、状态和哈希。
