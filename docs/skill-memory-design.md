# Skill and Memory Design

## Why this Agent needs skill documents

AI-Auto-Harness is not a chat-only Agent. It is an automatic deployment system that must make repeatable decisions under safety constraints. Hardcoding all deployment knowledge in Python would make the system rigid, while putting all knowledge in a single prompt would make behavior hard to audit.

The project therefore separates control knowledge into three layers:

- Python pipeline: owns state, permissions, command execution, evidence, and persistence.
- Skill Markdown: stage-specific deployment playbooks that an Agent can read before acting.
- Memory JSONL: historical failure patterns learned from prior deployments.

Skills answer: "How should this stage be handled?"

Memory answers: "Have we seen this kind of problem before, and what did we learn?"

## Where to write skills

Write repository-owned skills under:

```text
skills/<skill-name>/SKILL.md
```

Examples in this repo:

```text
skills/analyze-ai-demo/SKILL.md
skills/deploy-python-webui/SKILL.md
skills/verify-evidence/SKILL.md
skills/diagnose-runtime-failure/SKILL.md
```

This location is intentional:

- It is versioned with the deployment Agent.
- It can be reviewed like code.
- It is independent from a developer's local Codex/Claude personal skills.
- It can be loaded inside CI or a remote worker without relying on a user home directory.

Do not put runtime deployment secrets in skills. Do not put one-off run logs in skills. Skills should be stable playbooks.

## How to write a skill

Each skill is a folder with a required `SKILL.md`:

```markdown
---
name: verify-evidence
description: Verify automatic AI deployment with hard evidence. Use during verify stages for web UI/API services...
---

# Verify Evidence

Goal: prove the deployed service handled this run's fresh trace, or explicitly return `uncertain`.

## Verification policy

1. Generate a unique trace id.
2. Send a framework-specific request.
3. Pass only when response/artifact/log proves current trace handling.
```

The frontmatter should stay small. The `description` is important because the registry uses it for routing. The body should contain only durable operating rules, expected outputs, failure modes, and safety boundaries.

## How the Agent reads skills

The loader is implemented in `src/auto_harness/skills/registry.py`.

For each stage, the orchestrator calls:

```python
skills.select_for_stage(stage, analysis, limit=3)
```

The selected skills are stored in the stage result under:

```json
{
  "control_context": {
    "selected_skills": [
      {
        "name": "verify-evidence",
        "path": "skills/verify-evidence/SKILL.md",
        "sha256": "...",
        "content": "..."
      }
    ],
    "memory_hits": []
  }
}
```

The `sha256` is critical for auditability. If a future deployment behaves differently, the report can show which skill version was read.

## Where to write memory

Write cross-task issue memory under:

```text
memory/deployment_issues.jsonl
```

This file is append-only and machine-readable. Each line records one reusable failure pattern:

```json
{
  "id": "mem_...",
  "stage": "verify",
  "category": "api_shape_unknown",
  "frameworks": ["gradio"],
  "symptom": "HTTP response did not contain trace id",
  "root_cause": "service API shape differs from default /api/predict",
  "fix_status": "unresolved",
  "suggested_next_action": "Inspect service API shape and add a trace-producing verification request."
}
```

Markdown is not used as the source of truth for memory because retrieval, deduplication, and scoring are easier and safer with JSONL. Human-readable summaries can be generated into reports.

## How memory is used

The memory store is implemented in `src/auto_harness/memory/store.py`.

Before each stage, the orchestrator queries memory by stage and framework. After a stage returns `failed` or `uncertain`, the orchestrator writes a deduplicated memory entry. This prevents repeated failures from disappearing into logs.

The important rule is that memory is advisory. It can influence diagnosis and repair, but it cannot override execution policy. For example, a memory entry may recommend editing a Gradio launch parameter, but the pipeline still must check `allow_source_edit`.

## Verify module design focus

The verify module should be the strongest part of this project because it prevents false positives.

A weak deployment Agent checks:

```text
process exists + port open + HTTP 200 = success
```

AI-Auto-Harness should check:

```text
fresh trace generated + service invoked + response/artifact/log proves trace handling = success
```

This is why the `verify-evidence` skill says HTTP 200 is only readiness evidence. The verify stage must produce an evidence file containing request, response, trace id, status, and reason. If the trace cannot be observed, the result should stay `uncertain`, and the issue should be written to memory for later improvement.

## Interview framing

For interviews, describe this as a "skill-driven, memory-augmented deployment Agent":

- The orchestrator is deterministic and policy-bound.
- Skills are versioned operational knowledge.
- Memory is structured postmortem data, not vague chat history.
- Verify is evidence-driven and intentionally refuses false success.
- LLM/Claude/讯飞 can be plugged into uncertain stages, but the Python controller remains the source of truth for state, safety, and audit.
