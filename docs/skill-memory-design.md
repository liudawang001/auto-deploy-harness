# Skill 与 Memory 设计

## 为什么这个 Agent 需要技能文档

AI-Auto-Harness 不是一个单纯聊天式 Agent，而是一个自动部署系统。它需要在安全约束下做可复现、可审计的决策。

如果把所有部署知识都硬编码到 Python 里，系统会很僵硬；如果把所有知识都塞进一个大 prompt，行为又很难审计和复现。因此本项目把控制知识拆成三层：

- Python pipeline：负责状态机、权限控制、命令执行、证据校验和持久化。
- Skill Markdown：负责每个阶段的部署作战手册。
- Memory JSONL：负责沉淀历史失败模式和修复经验。

Skill 回答的问题是：当前阶段应该怎么做？

Memory 回答的问题是：类似问题以前是否出现过，当时学到了什么？

## Skill 应该写在哪里

本项目的内置技能文档统一写在：

```text
skills/<skill-name>/SKILL.md
```

当前仓库内置示例：

```text
skills/analyze-ai-demo/SKILL.md
skills/deploy-python-webui/SKILL.md
skills/verify-evidence/SKILL.md
skills/diagnose-runtime-failure/SKILL.md
```

这样设计的原因：

- skill 跟部署 Agent 的代码一起版本化。
- 每次变更都可以 code review。
- 不依赖开发者本机的个人 Codex/Claude skill 目录。
- CI、远程 worker 或其他机器克隆仓库后，也能读取同一套控制文档。

不要把运行时密钥写进 skill。不要把一次性部署日志写进 skill。Skill 应该是稳定的作战规则，而不是运行记录。

## Skill 应该怎么写

每个 skill 是一个文件夹，必须包含 `SKILL.md`：

```markdown
---
name: verify-evidence
description: 对 AI 自动部署结果做证据化验证。用于 verify 阶段...
---

# 证据化 Verify

目标：证明部署服务处理了当前 run 的新 trace；如果无法证明，必须返回 uncertain。

## 验证策略

1. 生成唯一 trace_id。
2. 发送框架专用请求。
3. 只有响应、产物或日志证明当前 trace 被处理，才能通过。
```

Frontmatter 保持简洁。`name` 建议使用英文短标识，方便作为稳定 ID；`description` 可以写中文，但应保留关键英文技术词，例如 `Gradio`、`FastAPI`、`verify`、`trace_id`。

正文可以使用中文。正文只写稳定规则、输出要求、失败模式和安全边界，不写临时想法。

## Agent 如何读取 Skill

加载逻辑在：

```text
src/auto_harness/skills/registry.py
```

每个阶段开始前，orchestrator 会调用：

```python
skills.select_for_stage(stage, analysis, limit=3)
```

选中的技能会写入阶段结果：

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

这里的 `sha256` 很重要。它保证后续排查时能知道某次部署到底读的是哪个版本的 skill。

## Memory 应该写在哪里

跨任务的问题记忆统一写到：

```text
memory/deployment_issues.jsonl
```

这是 append-only 的机器可读文件。每一行是一条可复用失败模式：

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

Memory 不用 Markdown 作为源数据，是因为 JSONL 更适合检索、去重、打分和自动化处理。给人看的总结可以在 report 里生成。

## Memory 如何参与下一次部署

记忆模块在：

```text
src/auto_harness/memory/store.py
```

每个阶段开始前，orchestrator 会按 stage 和 framework 检索相似问题。阶段返回 `failed` 或 `uncertain` 后，orchestrator 会写入去重后的 memory entry。

重要原则：memory 只能作为建议，不能覆盖执行策略。

例如，一条 memory 可以建议“修改 Gradio launch 参数”，但如果当前 `allow_source_edit=false`，pipeline 仍然不能自动改源码。

## Memory 如何提升为 Skill

当同一类 memory 多次出现，可以用 `memory-promote` 生成可审核的 skill 更新建议：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-promote --min-count 2
```

默认只生成审核稿：

```text
memory/promotions/<proposal_id>.json
memory/promotions/<proposal_id>.md
```

proposal 会记录：

- 聚类条件：stage、category、frameworks、count、memory ids。
- 目标 skill，例如 verify 问题默认指向 `skills/verify-evidence/SKILL.md`。
- 建议追加的 Markdown 规则片段。
- `review_required=true`，表示必须人工确认后才能进入 skill。

只有显式执行下面的命令才会修改 skill：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-promote --apply --proposal memory/promotions/<proposal_id>.json
```

这条命令会用 marker 包裹追加内容，避免重复应用。默认 proposal 模式不会改 `skills/*/SKILL.md`，也不会执行 shell。任何从 memory 提升到 skill 的内容都不能包含密钥值、一次性路径或未经验证的临时 workaround。

## Verify 模块为什么要重点设计

Verify 是这个项目最关键的工程价值点，因为它负责阻止“假成功”。

弱部署 Agent 往往这样判断：

```text
进程存在 + 端口打开 + HTTP 200 = 成功
```

AI-Auto-Harness 应该这样判断：

```text
生成新 trace + 调用服务 + 响应/产物/日志证明 trace 被处理 = 成功
```

所以 `verify-evidence` skill 明确规定：HTTP 200 只能说明服务可能活着，不能说明业务链路成功。

Verify 阶段必须产生 evidence 文件，记录 request、response、trace_id、status 和 reason。如果无法观测到 trace，结果应该保持 `uncertain`，并把问题写入 memory，供后续类似项目复用。

## 面试表达方式

这个设计可以概括为：

```text
skill-driven, memory-augmented deployment Agent
```

也就是“技能文档驱动、问题记忆增强的自动部署 Agent”。

面试中可以这样讲：

- 编排器是确定性的，负责状态、权限、命令执行和证据。
- Skill 是版本化的运维知识，把部署经验从代码中解耦出来。
- Memory 是结构化 postmortem，不是模糊聊天历史。
- Verify 是证据驱动的，宁可返回 uncertain，也不能产生 false positive。
- Claude/讯飞可以接入不确定阶段，但 Python controller 始终掌握状态、安全边界和审计链路。
