# Memory 与 Skill 自动进化执行方案

本文档用于交给 AI 编程工具直接执行开发。目标是在现有 LLM-driven 自动部署 Agent 基础上，实现一个受控、可验证、可回滚的 memory-to-skill evolution loop。

本方案不是“让 LLM 自动改技能并上线”。正确目标是：

> Agent 从真实部署结果中沉淀 verified memory；LLM 负责归纳可复用经验并生成 skill patch candidate；Python runtime 负责质量门槛、回归验证、shadow evaluation、promotion 与 rollback。

## 0. 当前项目基础

执行前先阅读以下文件：

```text
src/auto_harness/memory/store.py
src/auto_harness/memory/success.py
src/auto_harness/memory/promotion.py
src/auto_harness/skills/registry.py
src/auto_harness/cli.py
src/auto_harness/config.py
src/auto_harness/orchestrator.py
src/auto_harness/agent_runtime/runtime.py
docs/skill-memory-design.md
docs/true-llm-driven-agent-design.md
```

当前已有能力：

```text
MemoryStore:
  - 记录失败和 uncertain issue memory
  - 按 stage / framework 查询 memory_hits

VerifiedMemoryRecorder:
  - 在 final verify passed 后记录 verified success memory
  - 写入 memory/deployment_issues.jsonl

MemoryPromoter:
  - 从 verified memory 聚类生成 proposal
  - 支持 approve
  - 支持 apply 到 skills/<skill>/SKILL.md
  - apply 前可运行 regression

SkillRegistry:
  - 加载 repo-local skills
  - 按 stage / framework / verify_hint 选择 skill

CLI:
  - 已有 memory-promote
```

当前主要缺口：

```text
1. VerifiedMemoryRecorder._effective_repair() 仍可能把 metadata_only 当成 effective repair。
2. Memory promotion 主要是模板化规则，不是真正 LLM 归纳。
3. 没有独立 skill candidate 生命周期。
4. 没有 shadow evaluation。
5. 没有 skill outcome tracking。
6. 没有自动 rollback 命令。
7. 没有 evidence-gated evolution 的统一 CLI。
```

## 1. 严格目标

本轮目标：

```text
实现 evidence-gated memory-to-skill evolution：
verified memory -> LLM curator -> skill candidate -> quality gate -> regression -> shadow evaluation -> controlled promotion -> outcome tracking -> rollback
```

必须满足：

```text
1. LLM 只能生成候选 skill patch，不能直接修改正式 skill。
2. 只有 verified memory 可以进入 evolution。
3. metadata_only repair 不得作为 verified success。
4. candidate 必须带 source memory ids、base skill sha、risk analysis、regression binding。
5. promotion 必须由 deterministic gate 决定，不由 LLM 决定。
6. shadow mode 不得改变真实部署行为。
7. active skill patch 必须可回滚。
8. skill outcome 必须可追踪到 run_id / skill_sha / candidate_id。
```

不要实现：

```text
1. 不要让 LLM 直接写正式 skills/*/SKILL.md。
2. 不要让 LLM 删除或覆盖 memory。
3. 不要用聊天历史当长期记忆。
4. 不要把失败 memory 直接 promotion。
5. 不要跳过 regression。
6. 不要把 shadow candidate 用于真实执行。
7. 不要把 memory evolution 包装成多 Agent 协作。
```

## 2. 新增模块规划

新增文件：

```text
src/auto_harness/memory/quality.py
src/auto_harness/memory/curator.py
src/auto_harness/memory/evolution.py
src/auto_harness/memory/outcomes.py
src/auto_harness/skills/patch.py
src/auto_harness/skills/shadow.py
src/auto_harness/skills/rollback.py
```

可选新增测试文件：

```text
tests/test_memory_quality.py
tests/test_memory_curator.py
tests/test_memory_evolution.py
tests/test_skill_shadow.py
tests/test_skill_rollback.py
```

如果项目测试风格集中在 `tests/test_core.py`，也可以把测试放进去，但建议拆分，避免单文件过大。

## 3. 数据目录与产物

新增产物目录：

```text
memory/
  deployment_issues.jsonl
  skill_candidates/
    candidate_<id>.json
    candidate_<id>.md
    candidate_<id>.regression.json
    candidate_<id>.shadow.json
  promotions/
  skill_outcomes.jsonl
```

正式 skill 仍然位于：

```text
skills/<skill-name>/SKILL.md
```

history 继续使用现有 promotion apply 风格：

```text
skills/<skill-name>/history/
```

## 4. Phase 1：修复 Memory 质量门槛

### 4.1 修改目标

先修复数据污染问题，否则后续自动进化没有可信输入。

修改：

```text
src/auto_harness/memory/success.py
```

当前风险：

```python
def _effective_repair(self, apply_result):
    if any(item.get("executed") and int(item.get("exit_code") or 0) == 0 for item in apply_result.get("action_results", [])):
        return True
    return any(item.get("status") == "metadata_only" for item in apply_result.get("action_results", []))
```

目标语义：

```text
metadata_only 不得算 effective repair。
必须有真实 executed action，或者明确 tool_result strong_verify_pass。
```

建议实现：

```python
def _effective_repair(self, apply_result: Dict) -> bool:
    for item in apply_result.get("action_results", []):
        if item.get("executed") is True and int(item.get("exit_code") or 0) == 0:
            return True
        if item.get("tool_result", {}).get("strong_verify_pass") is True:
            return True
    return False
```

如果现有 action result schema 不含 `tool_result`，先只支持 `executed + exit_code=0`。

### 4.2 新增 MemoryQualityGate

新增：

```text
src/auto_harness/memory/quality.py
```

接口：

```python
class MemoryQualityGate:
    def classify(self, entry: dict) -> dict:
        ...

    def eligible_for_evolution(self, entry: dict) -> bool:
        ...

    def filter_verified(self, entries: list[dict]) -> list[dict]:
        ...
```

质量等级：

```text
raw_issue:
  failed / uncertain memory，不能 promotion

diagnosed_issue:
  有 root_cause / diagnosis，但没有 verified final pass，不能 promotion

verified_resolution:
  final verify passed + verification_trace_id + repair_action_hash + repair truly executed

regression_proven:
  verified_resolution + regression_status passed + regression_case_ids 非空

production_proven:
  多次 active skill outcome 证明有效，本轮可暂不实现完整统计
```

`classify()` 返回示例：

```json
{
  "level": "verified_resolution",
  "eligible": true,
  "reasons": [
    "verified_success=true",
    "verification_trace_id present",
    "repair_action_hash present"
  ],
  "reject_reasons": []
}
```

拒绝条件：

```text
verified_success != true
verification_trace_id missing
repair_action_hash missing
policy_rejected_high_risk == true
repair_action_status not in executed/success/passed/succeeded
regression_status failed
metadata_only == true
secret-like value present
absolute tmp path in suggested action
```

### 4.3 测试

必须覆盖：

```text
metadata_only 不 eligible
verified_success=false 不 eligible
缺 verification_trace_id 不 eligible
policy_rejected_high_risk=true 不 eligible
valid verified memory eligible
```

建议测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests.test_memory_quality
```

## 5. Phase 2：实现 LLM Memory Curator

### 5.1 新增模块

新增：

```text
src/auto_harness/memory/curator.py
```

职责：

```text
输入 verified memory cluster
调用 LLM provider 归纳可复用模式
输出严格 JSON candidate draft
不修改正式 skill
```

### 5.2 接口

```python
class MemoryCurator:
    def __init__(self, provider=None, max_input_chars: int = 20000):
        ...

    def curate(self, cluster: dict, target_skill_content: str = "") -> dict:
        ...

    def parse_response(self, text: str) -> dict:
        ...
```

### 5.3 LLM 输入

输入必须包含：

```json
{
  "task": "generalize verified deployment memories into a reusable skill patch candidate",
  "cluster": {
    "stage": "verify",
    "category": "verification_gap",
    "frameworks": ["gradio"],
    "memory_ids": [],
    "symptoms": [],
    "root_causes": [],
    "repair_actions": [],
    "verification_trace_ids": [],
    "regression_case_ids": []
  },
  "target_skill_excerpt": "...",
  "constraints": [
    "Do not include secrets.",
    "Do not include one-off absolute paths.",
    "Do not mark HTTP 200 alone as success.",
    "Only propose reusable rules.",
    "Output JSON only."
  ]
}
```

### 5.4 LLM 输出 schema

LLM 必须输出 JSON：

```json
{
  "status": "ok",
  "pattern": {
    "stage": "verify",
    "frameworks": ["gradio"],
    "failure_signature": "HTTP 200 but current trace_id absent",
    "root_cause_generalized": "The app exposes a non-default Gradio API shape."
  },
  "reusable_rule": {
    "when": "verify uncertain and framework_hint=gradio",
    "do": [
      "discover /config",
      "infer callable dependency",
      "send current trace_id through the inferred API"
    ],
    "do_not": [
      "do not mark success on HTTP 200 alone",
      "do not reuse old trace_id"
    ]
  },
  "skill_patch": {
    "target_skill": "verify-evidence/SKILL.md",
    "section_title": "Gradio API shape discovery",
    "markdown": "## Memory Evolution: Gradio API shape discovery\n..."
  },
  "regression_proposal": {
    "case_ids": ["gradio_api_shape_variation"],
    "new_case_suggestions": []
  },
  "risk": {
    "level": "low",
    "overfit_risk": "medium",
    "failure_modes": [
      "wrong endpoint inference",
      "false pass if trace_id is not checked"
    ]
  }
}
```

非法情况：

```text
非 JSON -> rejected
status != ok -> rejected
缺 skill_patch.markdown -> rejected
包含 secret-like 字段 -> rejected
包含 /tmp/、/Users/<name>/、绝对一次性路径 -> rejected
建议绕过 verify trace -> rejected
建议扩大 shell/source edit 权限 -> rejected
```

### 5.5 Mock provider

必须支持 mock provider 测试，不依赖真实 LLM。

如果现有 `MockLLMProvider` 已能返回文本，测试里直接传固定 JSON。

### 5.6 测试

覆盖：

```text
valid curator JSON parsed
invalid JSON rejected
secret-like markdown rejected
absolute local path rejected
HTTP 200 alone success rule rejected
```

## 6. Phase 3：Skill Candidate 生命周期

### 6.1 新增模块

新增：

```text
src/auto_harness/memory/evolution.py
src/auto_harness/skills/patch.py
```

### 6.2 Candidate schema

候选文件：

```text
memory/skill_candidates/candidate_<id>.json
memory/skill_candidates/candidate_<id>.md
```

JSON schema：

```json
{
  "candidate_id": "skillcand_xxx",
  "created_at": "iso",
  "status": "candidate",
  "source_memory_ids": [],
  "target_skill": "verify-evidence/SKILL.md",
  "base_skill_sha256": "...",
  "curator": {
    "provider": "mock|xunfei",
    "raw_response_hash": "..."
  },
  "pattern": {},
  "reusable_rule": {},
  "patch": {
    "section_title": "...",
    "markdown": "..."
  },
  "quality_gate": {
    "passed": true,
    "reasons": [],
    "reject_reasons": []
  },
  "regression_binding": {
    "manifest": "tests/fixtures/benchmarks/manifest.json",
    "case_ids": [],
    "required_before_promote": true
  },
  "shadow": {
    "enabled": false,
    "helped_count": 0,
    "harmful_count": 0
  },
  "promotion": {
    "status": "not_promoted",
    "promoted_at": "",
    "previous_sha256": "",
    "new_sha256": "",
    "rollback_path": ""
  }
}
```

### 6.3 MemoryEvolutionManager

接口：

```python
class MemoryEvolutionManager:
    def __init__(self, memory_dir: Path, skills_dir: Path, provider=None):
        ...

    def propose(self, min_verified_count: int = 3, stage: str = None, category: str = None, output_dir: Path = None) -> dict:
        ...

    def run_regression(self, candidate_path: Path, benchmark_runner=None) -> dict:
        ...

    def promote(self, candidate_path: Path, require_shadow: bool = True) -> dict:
        ...

    def reject(self, candidate_path: Path, reason: str) -> dict:
        ...
```

`propose()` 流程：

```text
read memory/deployment_issues.jsonl
filter with MemoryQualityGate
cluster by stage/category/frameworks
require count >= min_verified_count
load target skill
call MemoryCurator
validate patch
write candidate json + markdown
return summary
```

### 6.4 SkillPatchValidator

新增：

```text
src/auto_harness/skills/patch.py
```

接口：

```python
class SkillPatchValidator:
    def validate(self, markdown: str) -> dict:
        ...

class SkillPatchApplier:
    def apply_candidate(self, candidate: dict, skills_dir: Path) -> dict:
        ...
```

校验：

```text
markdown 非空
长度合理
不含 secret / token / api_key / password
不含绝对一次性路径
不含 “HTTP 200 means success”
不要求扩大 allow_source_edit / shell 权限
不删除原 skill 内容
必须带 candidate marker
```

marker：

```markdown
<!-- auto-harness-skill-evolution:candidate_<id> -->
...
<!-- /auto-harness-skill-evolution:candidate_<id> -->
```

正式 apply 前必须检查 `base_skill_sha256` 是否与当前 skill 匹配。若不匹配：

```text
返回 status=base_changed
不得自动 apply
```

## 7. Phase 4：Regression Gate

### 7.1 目标

candidate promotion 前必须跑绑定 regression。

复用：

```text
src/auto_harness/benchmarks/runner.py
```

### 7.2 行为

`run_regression(candidate_path)`：

```text
读取 candidate.regression_binding.manifest
读取 candidate.regression_binding.case_ids
调用 BenchmarkRunner.run(...)
写 memory/skill_candidates/candidate_<id>.regression.json
更新 candidate.status 或 candidate.regression
```

返回：

```json
{
  "status": "passed",
  "candidate_id": "skillcand_xxx",
  "manifest": "...",
  "case_ids": [],
  "output_path": "...",
  "failed_case_ids": []
}
```

规则：

```text
case_ids 为空 -> regression_failed
benchmark failed -> 不得 promote
regression skipped -> 不得 auto promote
```

## 8. Phase 5：Shadow Skill Evaluation

### 8.1 目标

candidate skill patch 先旁路评估，不影响真实部署。

新增：

```text
src/auto_harness/skills/shadow.py
```

### 8.2 ShadowSkillEvaluator

接口：

```python
class ShadowSkillEvaluator:
    def evaluate_run(self, run_dir: Path, candidate_path: Path, active_context: dict) -> dict:
        ...

    def record(self, candidate_path: Path, result: dict) -> dict:
        ...
```

最小实现不要复杂接入所有 planner。MVP 可以这样做：

```text
读取 run_dir/reports/agent_verify_result.json
读取 run_dir/agent_verify_steps.jsonl
读取 candidate pattern / target skill
判断 candidate 是否匹配当前 stage/framework/failure_signature
如果 candidate 推荐的 tool 与最终 helped tool 一致，记 would_help=true
如果 candidate 推荐绕过 trace 或扩大权限，记 harmful=true
```

输出：

```text
runs/<run_id>/reports/skill_shadow_eval.json
memory/skill_candidates/candidate_<id>.shadow.json
```

示例：

```json
{
  "candidate_id": "skillcand_xxx",
  "run_id": "...",
  "matched": true,
  "would_help": true,
  "would_harm": false,
  "reason": "candidate recommends discover_gradio_api, same as successful agent verify step",
  "active_skill_sha256": "...",
  "candidate_base_sha256": "..."
}
```

### 8.3 Shadow promotion 阈值

默认：

```text
helped_count >= 2
harmful_count == 0
```

MVP 可通过 CLI 参数允许 `--no-require-shadow`，但默认 promote 必须要求 shadow。

## 9. Phase 6：Skill Outcome Tracking

### 9.1 新增模块

新增：

```text
src/auto_harness/memory/outcomes.py
```

### 9.2 SkillOutcomeRecorder

接口：

```python
class SkillOutcomeRecorder:
    def __init__(self, memory_dir: Path):
        ...

    def record_run(self, run_id: str, stage: str, selected_skills: list[dict], result: dict, agent_metadata: dict = None) -> dict:
        ...

    def summarize(self, skill_name: str = None, candidate_id: str = None) -> dict:
        ...
```

写入：

```text
memory/skill_outcomes.jsonl
```

记录格式：

```json
{
  "created_at": "iso",
  "run_id": "...",
  "stage": "verify",
  "skill_name": "verify-evidence",
  "skill_path": "skills/verify-evidence/SKILL.md",
  "skill_sha256": "...",
  "candidate_id": "",
  "selected": true,
  "status": "passed",
  "llm_helped": true,
  "tool_selected": "discover_gradio_api",
  "policy_rejected": false,
  "trace_verified": true
}
```

### 9.3 Orchestrator 接入

修改：

```text
src/auto_harness/orchestrator.py
```

在 stage 完成后记录 selected skill outcome。

要求：

```text
只记录元数据，不改变 stage result
失败不能影响部署主流程
agent disabled 时也可以记录 skill outcome
```

如果接入 orchestrator 风险过大，MVP 可先提供 CLI 从 run_dir 回放生成 outcome。

## 10. Phase 7：Promotion 与 Rollback

### 10.1 Promotion 规则

`MemoryEvolutionManager.promote(candidate_path)` 必须检查：

```text
candidate.status in candidate|shadow_passed|regression_passed
quality_gate.passed == true
regression.status == passed
base_skill_sha256 == current target skill sha
patch validator passed
if require_shadow:
  shadow.helped_count >= 2
  shadow.harmful_count == 0
```

Promotion 行为：

```text
1. 读取 target skill
2. 写 rollback copy 到 skills/<skill>/history/
3. 用 marker 追加 patch
4. 写 previous_sha256 / new_sha256 / rollback_path
5. candidate.status = active
```

禁止：

```text
覆盖整个 skill
删除原有内容
base sha 变化时强行 apply
regression failed 时 apply
shadow harmful 时 apply
```

### 10.2 Rollback

新增：

```text
src/auto_harness/skills/rollback.py
```

接口：

```python
class SkillRollbackManager:
    def rollback_candidate(self, candidate_path: Path) -> dict:
        ...

    def rollback_to_history(self, skill_path: Path, history_path: Path) -> dict:
        ...
```

行为：

```text
读取 candidate.promotion.rollback_path
校验 rollback_path 存在
保存当前 skill 到 history
恢复 rollback_path 内容
更新 candidate.status = rolled_back
写 rollback metadata
```

返回：

```json
{
  "status": "rolled_back",
  "candidate_id": "...",
  "target_skill": "...",
  "restored_sha256": "...",
  "previous_active_sha256": "..."
}
```

## 11. CLI 设计

修改：

```text
src/auto_harness/cli.py
```

新增命令：

```bash
memory-evolve
skill-rollback
skill-outcomes
```

### 11.1 memory-evolve

Parser：

```python
memory_evolve = sub.add_parser("memory-evolve", help="propose, validate, regress, shadow, and promote skill candidates from verified memory")
memory_evolve.add_argument("--propose", action="store_true", default=False)
memory_evolve.add_argument("--regression", action="store_true", default=False)
memory_evolve.add_argument("--shadow", action="store_true", default=False)
memory_evolve.add_argument("--promote", action="store_true", default=False)
memory_evolve.add_argument("--reject", action="store_true", default=False)
memory_evolve.add_argument("--candidate", default="")
memory_evolve.add_argument("--min-verified-count", type=int, default=3)
memory_evolve.add_argument("--stage", default=None)
memory_evolve.add_argument("--category", default=None)
memory_evolve.add_argument("--output-dir", default="")
memory_evolve.add_argument("--provider", choices=["mock", "xunfei"], default=None)
memory_evolve.add_argument("--run-dir", default="")
memory_evolve.add_argument("--no-require-shadow", action="store_true", default=False)
memory_evolve.add_argument("--reason", default="")
```

行为：

```text
--propose:
  生成 candidate，不修改 skill

--regression --candidate:
  跑 candidate regression

--shadow --candidate --run-dir:
  对某个 run 做 shadow eval

--promote --candidate:
  通过 gate 后 apply 到 skill

--reject --candidate:
  标记 rejected
```

返回码：

```text
status in failed/rejected/regression_failed/base_changed -> exit 2
其他 -> exit 0
```

### 11.2 skill-rollback

Parser：

```python
skill_rollback = sub.add_parser("skill-rollback", help="rollback a promoted skill candidate")
skill_rollback.add_argument("--candidate", required=True)
```

行为：

```text
SkillRollbackManager.rollback_candidate(Path(args.candidate))
```

### 11.3 skill-outcomes

Parser：

```python
skill_outcomes = sub.add_parser("skill-outcomes", help="summarize skill outcome records")
skill_outcomes.add_argument("--skill", default="")
skill_outcomes.add_argument("--candidate", default="")
```

行为：

```text
SkillOutcomeRecorder(config.memory_path).summarize(...)
```

## 12. Config 设计

修改：

```text
src/auto_harness/config.py
```

新增字段：

```python
memory_evolution_enabled: bool = False
memory_evolution_min_verified_count: int = 3
memory_evolution_require_regression: bool = True
memory_evolution_require_shadow: bool = True
memory_evolution_shadow_helped_threshold: int = 2
memory_evolution_shadow_harmful_threshold: int = 0
memory_evolution_provider: str = "mock"
skill_candidate_dir: str = "memory/skill_candidates"
```

默认不要自动开启。

可选环境变量：

```text
AUTO_HARNESS_MEMORY_EVOLUTION_ENABLED
AUTO_HARNESS_MEMORY_EVOLUTION_PROVIDER
```

## 13. 与现有 memory-promote 的关系

不要删除 `memory-promote`。

建议关系：

```text
memory-promote:
  旧的 deterministic proposal / approve / apply 流程，保留兼容。

memory-evolve:
  新的 LLM curator + candidate + shadow + rollback 流程。
```

可以在内部复用 `MemoryPromoter._regression_binding()` 逻辑，但不要依赖私有方法太重。如果需要，抽成公共 helper。

## 14. 安全规则

任何 candidate patch 含以下内容必须 reject：

```text
api_key
token=
password
secret
Authorization:
Bearer
/Users/
/tmp/
C:\Users
ssh key
private key
HTTP 200 is enough
disable trace verification
allow arbitrary shell
allow source edit by default
```

任何 LLM 输出含以下意图必须 reject：

```text
直接修改正式 skill
绕过 regression
把 failed memory promotion
扩大 tool 权限
把 LLM 判断当 verify truth
```

## 15. 测试计划

### 15.1 必跑单元测试

建议新增后运行：

```bash
PYTHONPATH=src python3 -m unittest tests.test_memory_quality
PYTHONPATH=src python3 -m unittest tests.test_memory_curator
PYTHONPATH=src python3 -m unittest tests.test_memory_evolution
PYTHONPATH=src python3 -m unittest tests.test_skill_shadow
PYTHONPATH=src python3 -m unittest tests.test_skill_rollback
```

### 15.2 最小 CLI smoke

准备几条 fixture memory 后运行：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve \
  --propose \
  --min-verified-count 1 \
  --provider mock
```

然后：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve \
  --regression \
  --candidate memory/skill_candidates/candidate_xxx.json
```

Shadow：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve \
  --shadow \
  --candidate memory/skill_candidates/candidate_xxx.json \
  --run-dir runs/<run_id>
```

Promote：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve \
  --promote \
  --candidate memory/skill_candidates/candidate_xxx.json \
  --no-require-shadow
```

Rollback：

```bash
PYTHONPATH=src python3 -m auto_harness.cli skill-rollback \
  --candidate memory/skill_candidates/candidate_xxx.json
```

## 16. 分阶段实施顺序

严格按顺序执行。

### Phase 1

```text
1. 修改 VerifiedMemoryRecorder._effective_repair()
2. 新增 memory/quality.py
3. 新增 tests/test_memory_quality.py
```

验收：

```text
metadata_only 不 eligible
valid verified success eligible
```

### Phase 2

```text
1. 新增 memory/curator.py
2. 实现 LLM JSON parser
3. 实现安全 validator
4. 新增 tests/test_memory_curator.py
```

验收：

```text
mock provider 可生成 candidate draft
非法 JSON / secret / path / HTTP 200 success 规则被拒
```

### Phase 3

```text
1. 新增 memory/evolution.py
2. 新增 skills/patch.py
3. 实现 propose()
4. 写 candidate json/md
5. 新增 CLI memory-evolve --propose
```

验收：

```text
--propose 只生成 candidate，不修改 skills/*
candidate 含 source_memory_ids / base_skill_sha256 / regression_binding
```

### Phase 4

```text
1. 实现 run_regression()
2. CLI memory-evolve --regression
3. 写 candidate regression artifact
```

验收：

```text
regression failed 不得 promote
regression skipped 不得 auto promote
```

### Phase 5

```text
1. 新增 skills/shadow.py
2. CLI memory-evolve --shadow
3. 写 shadow artifact
```

验收：

```text
shadow 不改变真实部署结果
candidate helped/harmful 可累计
```

### Phase 6

```text
1. 新增 memory/outcomes.py
2. 可选接入 orchestrator，或先支持从 run_dir 回放
3. CLI skill-outcomes
```

验收：

```text
memory/skill_outcomes.jsonl 可记录 skill_sha 与 run outcome
```

### Phase 7

```text
1. 实现 promote()
2. 新增 skills/rollback.py
3. CLI skill-rollback
4. 测试 promote + rollback
```

验收：

```text
base sha mismatch 不 apply
apply 前写 rollback copy
rollback 可恢复旧 skill
```

## 17. 最终验收清单

完成后逐项检查：

- [ ] `VerifiedMemoryRecorder._effective_repair()` 不再把 `metadata_only` 视为有效修复
- [ ] `MemoryQualityGate` 能拒绝未验证 memory
- [ ] LLM curator 只生成 candidate，不修改正式 skill
- [ ] candidate 包含 `source_memory_ids`
- [ ] candidate 包含 `base_skill_sha256`
- [ ] candidate 包含 `regression_binding`
- [ ] patch validator 能拒绝 secret / path / HTTP 200 false success
- [ ] `memory-evolve --propose` 只写 `memory/skill_candidates/*`
- [ ] `memory-evolve --regression` 会写 regression artifact
- [ ] regression failed 时 promote 被阻断
- [ ] shadow evaluation 不改变真实部署行为
- [ ] `skill_outcomes.jsonl` 能记录 skill version outcome
- [ ] promotion 只追加 marker block，不覆盖原 skill
- [ ] base skill sha mismatch 时 promote 被阻断
- [ ] rollback 能恢复 promotion 前 skill

## 18. 给 AI 编程工具的执行提示词

可以直接使用：

```text
你现在接手 /Users/AQ/agent/ai-auto-harness 项目开发。

请严格按照 docs/memory-skill-evolution-execution-plan.md 实现 memory 与 skill 自动进化能力。

不要重新设计架构。
不要让 LLM 直接修改正式 skills/*/SKILL.md。
不要让 failed/uncertain memory 直接 promotion。
不要把 metadata_only repair 当作 verified success。
不要删除现有 memory-promote 命令。
不要做无关重构。

请按 Phase 1 到 Phase 7 顺序实现：

1. 修复 memory 质量门槛。
2. 实现 MemoryQualityGate。
3. 实现 LLM MemoryCurator。
4. 实现 Skill Candidate 生命周期。
5. 实现 Regression Gate。
6. 实现 Shadow Skill Evaluation。
7. 实现 Skill Outcome Tracking。
8. 实现 Promotion 与 Rollback。
9. 接入 CLI：memory-evolve / skill-rollback / skill-outcomes。

每个 Phase 完成后运行对应最小测试。
如果遇到现有测试失败，先判断是否与本次修改相关。

最终输出：
- 修改文件列表
- 已实现 Phase
- 未完成项
- 运行过的测试命令
- 测试结果
- 是否满足最终验收清单
```

## 19. 面试表述边界

完成本方案前，只能说：

```text
项目已有 memory promotion 雏形，正在补充 LLM curator、candidate、shadow、outcome tracking 和 rollback。
```

完成本方案后，可以说：

```text
实现了 evidence-gated memory-to-skill evolution loop：Agent 只从 trace-verified successful runs 中提取经验，LLM 负责生成 skill patch candidates，系统通过 deterministic quality gate、regression、shadow evaluation 和 rollback 控制技能演进。
```

仍然不要说：

```text
实现了完全自主长期记忆进化。
实现了生产级自进化 Agent 平台。
实现了无需人工或回归的自动技能上线。
```

除非补齐真实线上指标、持续监控和大规模回归体系。
