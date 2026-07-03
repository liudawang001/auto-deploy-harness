---
name: diagnose-runtime-failure
description: Diagnose and repair recurring automatic deployment failures. Use when env_deploy, runner, or verify is failed/uncertain, when logs contain dependency errors, process exits, missing model files, port conflicts, API key issues, or verification gaps, and when writing reusable issue memory.
---

# Diagnose Runtime Failure

Goal: turn a failed or uncertain stage into a reusable diagnosis without leaking secrets or inventing success.

## Triage order

1. Identify the failing stage and latest evidence path.
2. Read only bounded log tails and stored stage JSON.
3. Classify root cause:
   - dependency failure;
   - command or entrypoint mismatch;
   - service readiness failure;
   - model asset or hardware requirement;
   - missing environment variable;
   - verification gap.
4. Decide whether the fix is safe under current policy:
   - dependency install allowed;
   - service start allowed;
   - source edit allowed;
   - network/model download allowed.
5. Write the issue memory with symptom, root cause, affected framework, evidence, and next action.

## Memory writing rules

Record reusable patterns, not one-off noise. Keep secret values out. Include variable names only when useful. Prefer actionable signatures such as `gradio api shape unknown` or `torch wheel incompatible with python version`.

## Repair boundaries

Do not silently modify source code when `allow_source_edit` is false. Do not retry indefinitely. Do not treat a workaround as verified until the verify stage produces strong evidence.
