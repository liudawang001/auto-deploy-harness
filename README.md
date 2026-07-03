# AI-Auto-Harness

AI-Auto-Harness is a Python-controlled Agent pipeline for deploying and verifying AI open-source demo projects.

The project is intentionally built around deterministic orchestration plus bounded Agent execution:

- Python owns workflow state, retries, timeouts, filesystem boundaries, and evidence checks.
- Claude Code headless or another executor handles uncertain stage work such as project reading and log diagnosis.
- LLM providers, including Xunfei Spark, are abstracted behind a provider interface.
- `verify` is evidence-driven and must not treat a live port or HTTP 200 as final success.

## Current MVP

This repository currently includes:

- CLI: `init`, `deploy`, `resume`, `status`, `report`, `llm-test`.
- Task state store: `task.json`, `state.json`, `events.jsonl`.
- Deterministic project analyzer.
- Safe `env_deploy`, `runner`, `verify`, and report modules.
- Mock LLM provider and Xunfei Anthropic-compatible provider.
- Claude Code executor wrapper.
- HTTP trace evidence in `verify`: GET and POST JSON responses must prove the current trace was handled; HTTP 200 alone is not enough.
- Optional Claude Code analyzer advisor via `AUTO_HARNESS_USE_AGENT_ANALYZER=1`.
- Repo-local deployment skills under `skills/*/SKILL.md`, selected per stage and recorded with hashes.
- Structured issue memory under `memory/deployment_issues.jsonl`, used to retrieve similar historical failures.
- Progress report in `docs/progress.md`.

## Quick Start

```bash
python3 -m auto_harness.cli init
python3 -m auto_harness.cli deploy --repo https://github.com/example/demo --name demo --dry-run
python3 -m auto_harness.cli status --task-id <task-id>
python3 -m auto_harness.cli report --task-id <task-id>
```

For local source execution:

```bash
PYTHONPATH=src python3 -m auto_harness.cli init
```

To execute installation/startup rather than dry-run planning, be explicit:

```bash
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo ./demo --name demo --execute --allow-install --allow-start
```

You can also pass a local project path as `--repo`; it will be copied into the run workspace.

## Xunfei Configuration

Set secrets through environment variables only:

```bash
export XUNFEI_APP_ID="..."
export XUNFEI_API_KEY="..."
export XUNFEI_API_SECRET="..."
export XUNFEI_MODEL="..."
export XUNFEI_API_BASE="..."
# or export XUNFEI_API_URL="..."
```

Do not commit a real `.env` file. Keep secrets in your local shell, secret manager, or CI secret settings.

The Xunfei provider defaults to an Anthropic-compatible messages payload. If Claude Code cannot directly use Xunfei, use the local LLM provider path and keep Claude Code as an optional executor.

Provider smoke test:

```bash
PYTHONPATH=src python3 -m auto_harness.cli llm-test --provider xunfei --prompt "Return JSON only: {\"status\":\"ok\"}"
```

## Optional Claude Code Analyzer Advisor

The deterministic analyzer runs by default. To let Claude Code provide optional JSON advice during analysis:

```bash
export AUTO_HARNESS_USE_AGENT_ANALYZER=1
export CLAUDE_CODE_CMD="claude --print --output-format json"
PYTHONPATH=src python3 -m auto_harness.cli deploy --repo ./demo --name demo --dry-run
```

Agent advice is stored under `analyze_result.json` as optional metadata and does not bypass the deterministic pipeline.

## Skills and Memory

Write Agent control documents as repo-local skills:

```text
skills/<skill-name>/SKILL.md
```

Current built-in skills cover project analysis, Python WebUI deployment, evidence-driven verification, and runtime failure diagnosis. During a deployment, the orchestrator selects relevant skills for each stage and stores their names, paths, and SHA-256 hashes in the stage result under `control_context`.

Runtime issue memory is stored as JSONL:

```text
memory/deployment_issues.jsonl
```

This file is ignored by git because it can contain deployment logs or environment-specific symptoms. The Agent writes memory when a stage is `failed` or `uncertain`, then retrieves similar issues by stage/framework in later deployments. See `docs/skill-memory-design.md` for the detailed design.

## Safety Defaults

By default, `deploy` runs as a dry run and does not install dependencies or start long-running services unless explicit execution flags are passed.

Execution mode is additionally guarded by command policy. `env_deploy` and `runner` reject commands whose executable name is not listed in `allowed_commands` in `configs/default.json`.
