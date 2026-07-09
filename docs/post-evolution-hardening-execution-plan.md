# Agent 项目证据链与工程硬化执行方案

本文档用于交给 AI 编程工具继续执行开发。前提是假设以下能力已经基本完成：

```text
1. LLM-driven verify sub-agent 已接入 verify uncertain 主路径。
2. memory-to-skill evolution 已有离线工具链：
   - MemoryQualityGate
   - MemoryCurator
   - MemoryEvolutionManager
   - SkillPatchValidator / SkillPatchApplier
   - ShadowSkillEvaluator
   - SkillOutcomeRecorder
   - SkillRollbackManager
   - CLI: memory-evolve / skill-rollback / skill-outcomes
```

本轮目标不是继续堆概念，而是补齐项目最容易被大厂面试官追问的证据链和工程闭环：

```text
真实模型部署证据可审计
skill outcome 自动记录
mock CLI 可复现
shadow evaluation 更接近真实旁路决策
memory evolution 有完整 E2E artifact
promotion/rollback 具备文件级安全
benchmark 能证明增益而不只是防回归
memory/skill evolution 有 threat model
```

不要跑真实大模型测试。真实模型测试已经在服务器上完成，本轮只要求整理脱敏 evidence 和补本地可运行的工程闭环。

## 0. 执行前必须阅读

```text
src/auto_harness/orchestrator.py
src/auto_harness/cli.py
src/auto_harness/providers/mock.py
src/auto_harness/memory/outcomes.py
src/auto_harness/memory/evolution.py
src/auto_harness/memory/curator.py
src/auto_harness/skills/shadow.py
src/auto_harness/skills/patch.py
src/auto_harness/skills/rollback.py
src/auto_harness/modules/verify.py
docs/memory-skill-evolution-execution-plan.md
docs/true-llm-driven-agent-design.md
```

执行前先跑静态搜索：

```bash
rg -n "SkillOutcomeRecorder|memory-evolve|MockLLMProvider|ShadowSkillEvaluator|SkillPatchApplier|agent_verify_result" src tests docs
```

目的：

```text
确认现有实现位置
确认不要重复造模块
确认新增代码接在已有流程上
```

## 1. 当前关键缺口

基于当前代码状态，优先处理以下问题。

| 缺口 | 严重性 | 说明 |
|---|---:|---|
| `SkillOutcomeRecorder` 未接入 orchestrator | 高 | 有 recorder，但部署主流程不会自动写 `skill_outcomes.jsonl` |
| `memory-evolve --provider mock --propose` 可能不可用 | 高 | 通用 `MockLLMProvider` 不返回 curator 所需 `skill_patch` schema |
| Shadow evaluation 偏 artifact overlap | 高 | 只是读历史 artifact 判断 tool overlap，不是真正 candidate skill 旁路 planner |
| 缺少 memory evolution E2E artifact | 高 | 有单元测试，但缺少可展示的 CLI 串联证据 |
| 服务器真实模型测试缺 evidence 包 | 中高 | 口头说服务器跑过不够，需要脱敏 run evidence |
| promotion/rollback 缺少文件锁与 atomic write | 中 | 多进程或中断场景有文件损坏/并发覆盖风险 |
| benchmark 主要防回归，不证明增益 | 中 | 需要 old skill vs candidate/promoted 的对照 |
| 缺 memory-skill threat model | 中 | 自动进化容易被 prompt injection / memory poisoning 攻击 |

## 2. Phase 1：修复 Memory Evolution Mock Provider

### 2.1 目标

保证以下命令在本地可用：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve \
  --propose \
  --min-verified-count 1 \
  --provider mock
```

该命令必须能在存在 verified memory fixture 时生成 candidate，而不是因为通用 mock provider schema 不匹配返回 `no_candidates`。

### 2.2 当前问题

当前通用 mock provider：

```text
src/auto_harness/providers/mock.py
```

只返回：

```json
{
  "status": "ok",
  "summary": "mock provider response",
  "message_count": 1
}
```

但 `MemoryCurator` 需要：

```json
{
  "status": "ok",
  "pattern": {},
  "reusable_rule": {},
  "skill_patch": {}
}
```

### 2.3 推荐实现

新增一个专用 provider，不要破坏现有 `MockLLMProvider` 行为。

新增文件：

```text
src/auto_harness/providers/memory_evolution_mock.py
```

实现：

```python
class MemoryEvolutionMockProvider:
    def complete(self, messages, temperature: float = 0.2):
        ...
```

返回合法 curator JSON：

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
      "discover /config with discover_gradio_api",
      "send current trace_id through inferred API"
    ],
    "do_not": [
      "do not mark success on HTTP 200 alone",
      "do not reuse old trace_id"
    ]
  },
  "skill_patch": {
    "target_skill": "verify-evidence/SKILL.md",
    "section_title": "Gradio API shape discovery",
    "markdown": "When Gradio verify is uncertain, inspect /config and probe the inferred callable endpoint with the current trace_id. Do not mark success on HTTP 200 alone."
  },
  "regression_proposal": {
    "case_ids": ["gradio_config_discovery", "gradio_api_shape_variation"],
    "new_case_suggestions": []
  },
  "risk": {
    "level": "low",
    "overfit_risk": "medium",
    "failure_modes": ["wrong endpoint inference", "trace_id not checked"]
  }
}
```

修改：

```text
src/auto_harness/providers/__init__.py
src/auto_harness/cli.py
```

CLI 行为：

```text
memory-evolve --provider mock
  -> 使用 MemoryEvolutionMockProvider

llm-test --provider mock
  -> 继续使用原 MockLLMProvider
```

### 2.4 测试

新增或修改：

```text
tests/test_memory_curator.py
tests/test_memory_evolution.py
```

覆盖：

```text
MemoryEvolutionMockProvider 返回合法 candidate schema
memory-evolve propose 使用 mock provider 能生成 candidate
通用 MockLLMProvider 行为不变
```

## 3. Phase 2：接入 SkillOutcomeRecorder 到 Orchestrator

### 3.1 目标

每个 stage 完成后自动记录 selected skill 与 stage outcome。

必须写入：

```text
memory/skill_outcomes.jsonl
```

这一步是“skill 自动进化闭环”的核心证据。否则只能说有离线工具，不能说 Agent 会持续追踪 skill 版本效果。

### 3.2 修改文件

```text
src/auto_harness/orchestrator.py
src/auto_harness/memory/outcomes.py
tests/test_skill_outcomes_integration.py
```

### 3.3 Orchestrator 接入点

推荐在 `_save_stage()` 或每个 stage 保存后调用。

当前已有：

```python
def _save_stage(self, task_id: str, stage: str, result) -> None:
    path = self.store.save_result(task_id, stage, result)
    ...
```

建议新增：

```python
def _record_skill_outcome(self, task_id: str, stage: str, result) -> None:
    ...
```

然后在 `_save_stage()` 末尾调用：

```python
self._record_skill_outcome(task_id, stage, result)
```

### 3.4 字段抽取规则

从 `result.data.control_context.selected_skills` 获取：

```json
[
  {
    "name": "verify-evidence",
    "path": "skills/verify-evidence/SKILL.md",
    "sha256": "..."
  }
]
```

从 verify result 获取 agent metadata：

```text
result.data.agent_verify.llm_helped
result.data.agent_verify.accepted_tool_count
result.data.agent_verify.rejected_tool_count
result.data.agent_verify.evidence_paths
result.data.trace_id
```

建议构造：

```python
agent_metadata = {
    "llm_helped": agent_verify.get("llm_helped", False),
    "tool_selected": first executed tool if available else "",
    "policy_rejected": agent_verify.get("rejected_tool_count", 0) > 0,
    "trace_verified": result.status in ("pass", "passed") and bool(result.data.get("trace_id")),
}
```

`tool_selected` 如果不能可靠从 `result.data` 取到，可以先留空，不要编造。

### 3.5 安全要求

`_record_skill_outcome()` 不得影响部署主流程。

要求：

```text
try/except 包裹
失败只写 event 或忽略
不能让 outcome 写入失败导致 stage failed
不能修改 stage result
```

### 3.6 测试

覆盖：

```text
stage result 带 selected_skills 时写入 skill_outcomes.jsonl
stage result 没有 selected_skills 时也不崩
recorder 异常不影响 _save_stage
verify agent metadata 能写入 llm_helped / trace_verified
```

## 4. Phase 3：补 Memory Evolution E2E Evidence Fixture

### 4.1 目标

新增一个本地可运行的 E2E smoke，不依赖真实 LLM、不依赖 GPU、不依赖真实模型。

目标链路：

```text
verified memory fixture
  -> memory-evolve --propose
  -> memory-evolve --regression
  -> memory-evolve --shadow
  -> memory-evolve --promote --no-require-shadow
  -> skill-rollback
  -> evidence artifact
```

### 4.2 新增目录

```text
docs/evidence/memory-skill-evolution-smoke/
  README.md
  commands.sh
  sample-memory.jsonl
  expected-artifacts.md
```

如果需要测试自动生成，可以写到临时目录，不要污染正式 `skills/`。

### 4.3 新增 CLI smoke runner

可选新增：

```text
src/auto_harness/evals/memory_evolution_smoke.py
```

或在测试中直接构造。

建议新增 CLI：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve-smoke \
  --output-dir runs/evals/memory-evolution-smoke
```

如果不想新增 CLI，至少新增测试：

```text
tests/test_memory_evolution_e2e.py
```

### 4.4 验收

必须证明：

```text
candidate json 存在
candidate md 存在
regression json 存在
shadow json 存在
promote 后 skill 文件出现 marker
rollback 后 marker 消失
skill_outcomes 可 summarize
```

## 5. Phase 4：升级 Shadow Evaluation

### 5.1 当前问题

当前 `ShadowSkillEvaluator` 主要逻辑：

```text
读取 run_dir/reports/agent_verify_result.json
读取 run_dir/agent_verify_steps.jsonl
从 candidate.reusable_rule.do 提取 recommended tools
从历史 successful steps 提取 successful tools
做 overlap 判断 would_help
```

这不是严格 shadow planner。它只能说明“候选规则和历史成功工具一致”，不能说明“candidate skill 会让 Agent 做出更好决策”。

### 5.2 目标

新增一个轻量 candidate skill 旁路决策模式：

```text
active skill -> 真实部署已执行
candidate skill -> shadow planner 输入，但不执行 tool
比较 active decision 与 candidate decision
```

### 5.3 修改文件

```text
src/auto_harness/skills/shadow.py
src/auto_harness/agent_runtime/planner.py
src/auto_harness/agent_runtime/runtime.py
tests/test_skill_shadow.py
```

如果不想改 runtime，可先在 `ShadowSkillEvaluator` 内实现 `simulate_candidate_decision()`。

### 5.4 新接口

```python
class ShadowSkillEvaluator:
    def evaluate_candidate_decision(
        self,
        run_dir: Path,
        candidate_path: Path,
        observation: dict = None,
        planner=None,
    ) -> dict:
        ...
```

行为：

```text
1. 读取 candidate.patch.markdown
2. 读取 run 的 agent observation 或 agent_verify_result
3. 构造 shadow prompt context
4. 调 planner 生成 would_tool_call
5. 不执行 tool
6. 和真实 executed tool / final status 比较
7. 写 candidate decision artifact
```

输出：

```json
{
  "candidate_id": "skillcand_xxx",
  "run_id": "...",
  "shadow_mode": "planner_only",
  "would_tool_call": {
    "name": "discover_gradio_api",
    "input": {}
  },
  "actual_tool_call": {
    "name": "probe_http"
  },
  "would_help": true,
  "would_harm": false,
  "reason": "candidate selected a tool that produced current trace evidence in a prior successful run",
  "executed": false
}
```

### 5.5 严格要求

```text
shadow planner 不得执行 tool
shadow planner 不得改变 verify result
shadow artifact 必须标记 executed=false
would_help 不能只因为文本包含 gradio 就为 true
```

### 5.6 MVP fallback

如果接入 planner 成本过高，至少增强当前 shadow：

```text
要求 agent_verify_result.final_status == passed
要求 llm_helped == true
要求 successful tool 的 evidence_path 存在
要求 evidence 中包含 current trace_id
```

否则不能计入 `helped_count`。

## 6. Phase 5：Promotion / Rollback 文件安全

### 6.1 目标

防止并发或中断导致 skill 文件损坏。

### 6.2 修改文件

```text
src/auto_harness/skills/patch.py
src/auto_harness/skills/rollback.py
tests/test_skill_patch_atomic.py
```

### 6.3 实现要求

新增 helper：

```text
src/auto_harness/utils/atomic.py
```

实现：

```python
def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ...

class FileLock:
    ...
```

可以用标准库：

```text
tempfile
os.replace
fcntl  # mac/linux
```

不需要支持 Windows 锁；如果 Windows 上运行，退化为 atomic write 即可。

### 6.4 Apply 加锁

`SkillPatchApplier.apply_candidate()` 必须：

```text
锁定 target SKILL.md.lock
读取当前内容
校验 base sha
写 rollback copy
atomic replace target skill
写 apply audit
释放锁
```

新增 audit：

```text
skills/<skill>/history/<timestamp>_<candidate_id>.apply.json
```

字段：

```json
{
  "candidate_id": "...",
  "target_skill": "...",
  "previous_sha256": "...",
  "new_sha256": "...",
  "rollback_path": "...",
  "applied_at": "..."
}
```

### 6.5 Rollback 加锁

`SkillRollbackManager.rollback_candidate()` 必须：

```text
锁定 target SKILL.md.lock
保存当前 active 内容
atomic restore rollback copy
写 rollback audit
释放锁
```

## 7. Phase 6：真实模型部署 Evidence 整理

### 7.1 目标

服务器上的真实模型部署测试不需要在 Mac 上复现，但必须变成可审计证据。

新增目录：

```text
docs/evidence/real-model-deployment/
  README.md
  server-env.redacted.json
  run-summary.redacted.json
  agent-verify-result.redacted.json
  agent-verify-steps.redacted.jsonl
  verified-memory.redacted.json
  model-download.redacted.json
  commands.redacted.sh
  redaction-policy.md
```

### 7.2 README 必须包含

```text
测试日期
服务器配置
GPU 型号
driver / CUDA / Python / PyTorch 版本
模型来源 Hugging Face / ModelScope
模型名称
模型大小
是否真实下载
是否命中缓存
部署 backend
最终 verify trace_id
是否触发 agent verify
是否触发 repair
失败/修复摘要
哪些字段已脱敏
为什么 Mac 本地不复现
```

### 7.3 禁止提交

```text
HF token
ModelScope token
私有 endpoint
服务器公网 IP
完整 home path
SSH 信息
云账号 ID
原始超长日志
```

### 7.4 新增 redaction checker

可选新增：

```text
src/auto_harness/utils/redaction.py
tests/test_redaction.py
```

检查：

```text
token
Bearer
Authorization
/home/<user>
/Users/<user>
公网 IP
AWS/GCP/Azure key pattern
```

## 8. Phase 7：Benchmark 从防回归升级为证明增益

### 8.1 当前问题

当前 regression gate 主要证明：

```text
candidate 没有破坏已有 benchmark
```

但 evolution 更需要证明：

```text
candidate 让旧 skill 做不好的 case 变好
```

### 8.2 新增 gain report

新增：

```text
src/auto_harness/evals/skill_gain.py
tests/test_skill_gain_eval.py
```

接口：

```python
class SkillGainEvaluator:
    def evaluate_candidate(self, candidate_path: Path, output_path: Path = None) -> dict:
        ...
```

输出：

```json
{
  "candidate_id": "skillcand_xxx",
  "target_skill": "verify-evidence/SKILL.md",
  "baseline": {
    "skill_sha256": "...",
    "status": "uncertain",
    "selected_tool": "probe_http"
  },
  "candidate": {
    "shadow_decision": "discover_gradio_api",
    "would_execute": false,
    "status": "would_help"
  },
  "gain": {
    "improved": true,
    "reason": "candidate selects tool linked to trace-verified successful memory"
  }
}
```

### 8.3 CLI

可选新增：

```bash
PYTHONPATH=src python3 -m auto_harness.cli skill-gain \
  --candidate memory/skill_candidates/candidate_xxx.json \
  --output runs/evals/skill-gain/candidate_xxx.json
```

## 9. Phase 8：Memory / Skill Threat Model

### 9.1 新增文档

```text
docs/memory-skill-threat-model.md
```

必须覆盖：

```text
memory poisoning
prompt injection from README/logs
malicious candidate patch
secret leakage
permission expansion
false success promotion
benchmark overfitting
rollback failure
concurrent promotion
```

每项格式：

```text
Threat
Attack path
Current mitigation
Missing mitigation
Test evidence
Residual risk
```

### 9.2 代码联动

文档里的 mitigation 必须能对应代码：

```text
MemoryQualityGate
MemoryCurator validator
SkillPatchValidator
regression gate
shadow gate
base sha check
rollback
redaction checker
file lock / atomic write
```

不要只写安全文档，不补代码证据。

## 10. Phase 9：Evidence Package Export

### 10.1 目标

提供一个命令，把项目关键证据打包，方便面试/评审展示。

新增：

```text
src/auto_harness/evidence.py
```

CLI：

```bash
PYTHONPATH=src python3 -m auto_harness.cli evidence-package \
  --output dist/evidence/agent-project-evidence.tar.gz
```

包含：

```text
docs/evidence/real-model-deployment/
docs/evidence/memory-skill-evolution-smoke/
runs/evals/agent-verify-mvp/comparison_report.json
memory/skill_candidates/*.json
memory/skill_candidates/*.regression.json
memory/skill_candidates/*.shadow.json
memory/skill_outcomes.jsonl
docs/memory-skill-threat-model.md
```

必须排除：

```text
tokens
large model files
cache
venv
.conda
raw private logs
```

## 11. 推荐执行顺序

严格按顺序：

```text
Phase 1: 修 mock provider，让 memory-evolve propose 本地可跑
Phase 2: 接入 SkillOutcomeRecorder 到 orchestrator
Phase 3: 补 memory evolution E2E artifact
Phase 4: 升级 shadow evaluation
Phase 5: promotion/rollback 加锁与 atomic write
Phase 6: 整理服务器真实模型部署 evidence
Phase 7: 新增 skill gain benchmark
Phase 8: 补 memory-skill threat model
Phase 9: evidence package export
```

如果时间有限，优先做：

```text
1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 6
```

这四项对简历和面试抗压提升最大。

## 12. 最终验收清单

完成后逐项检查：

- [ ] `memory-evolve --provider mock --propose` 能生成 candidate
- [ ] 通用 `llm-test --provider mock` 行为不被破坏
- [ ] 每个 stage 完成后自动写入 `memory/skill_outcomes.jsonl`
- [ ] outcome 写入失败不影响部署主流程
- [ ] memory evolution E2E smoke 有本地 artifact
- [ ] shadow evaluation 标记 `executed=false`
- [ ] shadow helped 需要 evidence trace 支撑
- [ ] promotion 使用 base sha 二次检查
- [ ] promotion 使用 atomic write
- [ ] rollback 使用 atomic restore
- [ ] 服务器真实模型部署 evidence 已脱敏入库
- [ ] 有 redaction policy
- [ ] skill gain report 能证明 candidate 增益
- [ ] threat model 有代码证据映射
- [ ] evidence package 能导出核心材料

## 13. 建议运行命令

不跑真实模型。

最小测试：

```bash
PYTHONPATH=src python3 -m unittest tests.test_memory_curator
PYTHONPATH=src python3 -m unittest tests.test_memory_evolution
PYTHONPATH=src python3 -m unittest tests.test_skill_shadow
PYTHONPATH=src python3 -m unittest tests.test_skill_rollback
```

新增测试后：

```bash
PYTHONPATH=src python3 -m unittest tests.test_skill_outcomes_integration
PYTHONPATH=src python3 -m unittest tests.test_memory_evolution_e2e
PYTHONPATH=src python3 -m unittest tests.test_skill_patch_atomic
PYTHONPATH=src python3 -m unittest tests.test_redaction
PYTHONPATH=src python3 -m unittest tests.test_skill_gain_eval
```

CLI smoke：

```bash
PYTHONPATH=src python3 -m auto_harness.cli memory-evolve --propose --min-verified-count 1 --provider mock
PYTHONPATH=src python3 -m auto_harness.cli skill-outcomes
```

## 14. 禁止事项

不要做：

```text
不要在 Mac 上强行跑真实大模型/GPU 测试
不要提交 token 或私有 endpoint
不要把 shadow overlap 包装成真实 planner shadow
不要让 outcome recorder 失败影响部署
不要让 mock provider 破坏其他 LLM 测试
不要跳过 base sha 检查
不要覆盖整个 skill 文件
不要删除已有 memory-promote
不要把 evidence 文档写成宣传稿
```

## 15. 给 AI 编程工具的直接提示词

可以直接复制：

```text
你现在接手 /Users/AQ/agent/ai-auto-harness 项目。

请严格按照 docs/post-evolution-hardening-execution-plan.md 执行，不要重新设计架构。

本轮目标是补齐 Agent 项目的工程证据链和 memory/skill evolution 闭环：
1. 修复 memory-evolve 的 mock provider，使本地 propose 可复现。
2. 将 SkillOutcomeRecorder 接入 orchestrator，自动记录每个 stage 的 skill outcome。
3. 新增 memory evolution E2E smoke artifact。
4. 升级 ShadowSkillEvaluator，至少保证 helped 判定有 trace evidence，最好实现 planner-only shadow decision。
5. 为 skill promotion/rollback 增加 file lock 和 atomic write。
6. 整理服务器真实模型部署 evidence，必须脱敏。
7. 新增 skill gain evaluator，证明 candidate 是增益而不只是防回归。
8. 新增 memory-skill threat model，并保证 mitigation 有代码证据。
9. 可选新增 evidence-package 导出命令。

约束：
- 不跑真实大模型测试。
- 不提交 token、私有 endpoint、完整 home path 或原始私密日志。
- 不让 LLM 直接修改正式 skill。
- 不让 shadow candidate 真实执行 tool。
- outcome 写入失败不得影响部署主流程。
- 不删除已有 memory-promote。
- 不做无关重构。

执行前先阅读：
src/auto_harness/orchestrator.py
src/auto_harness/cli.py
src/auto_harness/providers/mock.py
src/auto_harness/memory/outcomes.py
src/auto_harness/memory/evolution.py
src/auto_harness/skills/shadow.py
src/auto_harness/skills/patch.py
src/auto_harness/skills/rollback.py

每个 Phase 完成后说明：
- 修改文件
- 实现内容
- 运行命令
- 测试结果
- 未完成项

最终输出必须包含最终验收清单的逐项状态。
```

## 16. 面试表述边界

完成 Phase 1-4 后，可以说：

```text
项目不仅支持 LLM-driven verify，还具备可复现的 memory-to-skill evolution 证据链：每次部署会记录 skill outcome，本地 mock 能生成候选 skill patch，candidate 经过 regression 和 shadow artifact 后才能 promotion。
```

完成 Phase 5-9 后，可以说：

```text
进一步补齐了 skill promotion 的发布安全和评审证据：promotion/rollback 具备 base-sha、atomic write、history/audit；真实模型部署证据脱敏归档；memory/skill threat model 覆盖 memory poisoning、prompt injection 和 false promotion。
```

仍然不要说：

```text
生产级自进化 Agent 平台
完全自动长期记忆进化
大规模并发部署系统
真实 GPU 调度平台
```

除非后续补齐线上运行、并发调度、多租户、权限和 SLA 证据。
