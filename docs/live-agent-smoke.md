# Live Agent Smoke

`agent-live-smoke` is an optional networked smoke command for proving that the
LLM agent path can produce real agent artifacts. Secrets must be injected only
through environment variables.

Example:

```bash
PYTHONPATH=src python3 -m auto_harness.cli agent-live-smoke \
  --repo tests/fixtures/live/llm_repair_missing_dependency \
  --provider xunfei \
  --execute \
  --output runs/live_smoke/manual
```

Expected artifact classes:

- `task.json`, `state.json`, `events.jsonl`
- `logs/agent_calls/*.json`
- `repairs/repair_plan.json`
- `repairs/repair_apply_result.json`
- `reports/pipeline_results.json`
- `evidence/*verify*.json`
- `live-agent-smoke-manifest.json`

The committed sample manifest in `docs/evidence/live-agent-smoke-manifest.json`
contains only metadata, artifact paths, counts, statuses and hashes. It must not
contain prompts, token values, raw logs or workspace copies.
