# AI-Auto-Harness Agent 评估报告

## 评估口径

本文只记录当前仓库已有代码和本地可验证证据，不把外部未跑通的长耗时任务写成已完成。

当前静态统计：

- 测试函数数量：146 个，按 `rg -n "^\\s+def test_" tests | wc -l` 统计。
- Benchmark case 数量：56 个，来自 `tests/fixtures/benchmarks/manifest.json`。
- Readiness：本地 readiness audit 设计为区分 local gates 与 external gates。

## 本地 Benchmark 覆盖

覆盖能力包括：

- 下载续传、缓存命中、并发下载、etag 缓存失效、缓存清理。
- Hugging Face / ModelScope / Git LFS / Git submodule 准备。
- Gradio、Streamlit、browser DOM、OpenAPI、OpenAI-compatible verify。
- HTTP 200 false-positive 防护、trace evidence、artifact evidence。
- repair policy、repair loop attempt limit、operator approval、resume stage。
- memory promotion proposal、审批、apply 后 regression。
- LLM planner policy merge、LLM repair execute loop、LLM verify hint recovery。
- Agent loop self-repair fixture。
- prompt injection defense。
- agent metrics paired comparison。
- dashboard、queue、GPU probe、Docker backend metadata、package export、readiness audit。

## Agent vs Workflow 对照

当前对照不是大规模线上 A/B，而是本地 fixture paired comparison：

- baseline：deterministic workflow 关闭 Agent 介入。
- agent：启用 planner / diagnoser / verify planner / repair action 的 gated mode。
- 输出：`reports/agent_metrics.json` 和 `agent-metrics` 汇总，记录 LLM calls、accepted/rejected actions、executed actions、repair attempts、verify candidates、final status、agent_helped、help_type。

该对照能证明 Agent 在受控场景中能产生额外诊断、repair 和 verify candidate 价值，但不能证明对所有开源模型仓库都有同等成功率。

## 本次已执行验证

Phase 6 本次执行过的验证：

```bash
PYTHONPATH=src python3 -m unittest \
  tests.test_core.CoreTests.test_memory_store_records_and_queries_issue \
  tests.test_core.CoreTests.test_memory_promoter_generates_review_proposal_and_apply \
  tests.test_core.CoreTests.test_memory_promotion_requires_verified_agent_success \
  tests.test_core.CoreTests.test_memory_promotion_rejects_unverified_llm_suggestion \
  tests.test_core.CoreTests.test_memory_promote_cli_outputs_proposal
```

结果：5 个测试通过。

```bash
PYTHONPATH=src python3 -m auto_harness.cli benchmark \
  --case-id memory_promotion_proposal \
  --case-id memory_promotion_approval_regression \
  --case-id memory_promotion_apply_regression_run \
  --output /tmp/ai_auto_harness_memory_phase6_benchmark.json
```

结果：3 个 selected benchmark case 通过。

本次也执行了 secret scan，未命中已知敏感值。

## Live Smoke 状态

`agent-live-smoke` 入口已实现，并有 live fixture 与 sample manifest：

- `tests/fixtures/live/llm_repair_missing_dependency/`
- `docs/live-agent-smoke.md`
- `docs/evidence/live-agent-smoke-manifest.json`

真实 provider live smoke 需要操作者在本机通过环境变量注入密钥后执行。本次没有运行真实联网 provider smoke，因此不能声称真实外部 provider 已通过。

## External Gates

以下 gate 当前属于外部验收，不在普通本地 benchmark 中默认执行：

- 真实 Hugging Face / ModelScope 大模型下载。
- 真实 GPU Docker smoke。
- 真实 vLLM / OpenAI-compatible 大模型服务。
- 长耗时多仓库部署稳定性测试。
- 分布式队列和多机资源锁。

readiness report 会把这些标记为 `external_required` 或 `future_scale_gate`。

## 已知限制

- Agent 当前主要在 fixture 和本地 mock provider 场景完成验证，真实复杂仓库仍需要更多 live smoke 矩阵。
- auto-resume 已有受控判定，但默认仍强调审计和安全门禁。
- LLM advice 已 schema 化进入 planner/diagnoser/verify planner，但价值取决于日志质量、项目文档质量和 provider 输出稳定性。
- Memory promotion 现在只吸收 verified success，但自动写入 verified success memory 的更完整闭环仍可继续加强。
- 当前不是生产级平台，没有宣称大规模支持或完全自动化无人值守。

## 面试可展示结论

这个项目的亮点不是“让 LLM 执行命令”，而是把 Agent 工程化成可控闭环：

- typed action schema
- deterministic pipeline
- policy-gated action merge
- evidence-based verify
- repair/resume state machine
- prompt injection 与 secret redaction
- trace/audit/metrics
- verified memory promotion
- benchmark/readiness 分层验收

这些能力对应大厂 Agent 开发中的核心问题：可靠性、安全边界、可解释性、可评估性和长期经验沉淀。
