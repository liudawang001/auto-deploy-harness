# AI-Auto-Harness 自修复 Agent、长期记忆、技能进化与 Conda 环境优化开发方案

## 0. 文档目的

本文档用于指导 Codex / AI 开发者继续开发当前 AI-Auto-Harness 项目，目标是把项目从：

```text
policy-constrained LLM-assisted deployment workflow
```

升级为：

```text
self-healing deployment agent with verified long-term memory,
skill evolution, and LLM-guided conda/PyTorch environment provisioning
```

本文档直接面向开发执行，不是简历包装稿。

核心目标：

1. 实现真正接入普通 `deploy/run_existing` 主流程的自修复 Agent。
2. 实现 verified success 级别的长期记忆写回。
3. 实现只基于验证成功经验的技能进化。
4. 支持 LLM 基于项目分析选择 `venv` / `conda` / `mamba` 环境策略。
5. 支持创建 conda 虚拟环境，并在其中配置 PyTorch、CUDA wheel / conda package、常见 AI 依赖。
6. 不要求新增真实大模型部署测试。验收以本地 fixture、fake command runner、manifest、artifact 和可选服务器实践记录为准。

## 1. 当前问题基线

当前仓库已有：

- `AgentLoopController`：可生成 failure observation、diagnosis、repair plan、policy、apply result、`should_auto_resume`。
- `RepairApplier`：可在 gated mode 下执行受控 `install_package`。
- `AgentInputSanitizer`：支持 secret redaction 和 prompt injection 风险标注。
- `AgentMetricsCollector`：可统计 LLM calls、accepted/rejected actions、repair attempts 等。
- `MemoryStore` / `MemoryPromoter`：普通 issue memory 默认 `verified_success=false`；promotion 只接受 verified success 条目。
- `EnvSolveModule`：已有 venv/pip install plan、PyTorch CUDA wheel selection、GPU package matrix。

当前明确缺口：

1. 普通 `TaskRunner.deploy()` / `run_existing()` 没有在 `should_auto_resume=true` 时自动重跑；真正 resume 主要在 `agent-live-smoke` wrapper 中触发。
2. 成功 repair 后没有通用机制自动写回 verified success memory。
3. skill promotion gate 严格，但 verified success memory 的生产链路不完整。
4. 没有真实 conda/mamba backend；当前实际是 `python3 -m venv .venv`。
5. `environment.yml` 没有被解析为 conda environment spec。
6. LLM 还不能正式输出环境策略，例如选择 conda、mamba、Python 版本、channel、PyTorch/CUDA variant。
7. repair action schema 对真实 LLM 输出容错不足，例如 `payload.packages=["rich"]` 和 `payload.package="rich"` 不统一。

## 2. 非目标

本阶段不做：

- 真实大模型下载和部署测试。
- 真实 GPU 服务器长耗时测试。
- 多机分布式调度。
- Kubernetes / Slurm / Ray 集群调度。
- 任意 shell 放权。
- LLM 直接修改源码。
- LLM 直接判定部署成功。

如果开发者已有服务器实践经验，可以在 docs 中记录为外部实践说明，但普通 CI / benchmark 不依赖真实模型部署。

## 3. 目标架构

目标闭环：

```text
deploy
  -> analyze
     -> deterministic analysis
     -> LLM environment/deployment planner
     -> policy merge
  -> resource_plan
  -> env_solve
     -> choose venv/conda/mamba backend
     -> solve Python/PyTorch/CUDA/package strategy
  -> env_deploy
     -> create env
     -> install dependencies
  -> runner
  -> verify
     -> trace evidence
  -> if failed/uncertain:
       AgentLoopController
         -> diagnose
         -> repair plan
         -> policy
         -> apply action
         -> auto resume from safe stage
         -> verify again
       stop by pass / max loop / policy reject / no progress
  -> if verify pass after repair:
       write verified success memory
       optionally create skill promotion proposal
  -> report
```

目标代码结构：

```text
src/auto_harness/
  agent/
    loop.py
    schemas.py
    policy.py
    metrics.py
  env/
    __init__.py
    schemas.py
    selector.py
    conda.py
    venv.py
    torch.py
  memory/
    store.py
    success.py
    promotion.py
  repair/
    actions.py
    apply.py
    planner.py
  modules/
    analyzer.py
    env_solve.py
    env_deploy.py
    runner.py
    reporter.py
```

如果不想新增 `src/auto_harness/env/`，也可以把相关类放在 `runtime/` 或 `modules/` 下；但必须保持 schema、solver、backend 执行分离。

## 4. Phase 1：把自修复 Agent 接入普通主流程

### 4.1 目标

普通 `deploy` 在配置开启时必须能自动完成：

```text
stage failed/uncertain
-> AgentLoopController
-> repair apply
-> resume from safe stage
-> verify
-> repeat until pass or stop
```

不能只在 `agent-live-smoke` 中完成。

### 4.2 新增配置

在 `HarnessConfig` 中确认或新增：

```python
agent_auto_resume_after_repair: bool = False
agent_max_loop_iterations: int = 2
agent_auto_resume_stages: List[str] = None
agent_stop_on_verify_pass: bool = True
```

默认：

```json
{
  "agent_auto_resume_after_repair": false,
  "agent_max_loop_iterations": 2,
  "agent_auto_resume_stages": ["env_deploy", "model_prepare", "runner", "verify"],
  "agent_stop_on_verify_pass": true
}
```

必须保持默认安全：不开启时现有 pipeline 行为不变。

### 4.3 修改 `TaskRunner.run_existing`

当前问题：

- `_remember()` 会写 `agent_loop.should_auto_resume`。
- 但 `run_existing()` 不根据这个结果重跑。

执行要求：

1. 在 `run_existing()` 外层增加 bounded loop。
2. 每次 pipeline pass 后检查是否有 `agent_loop.should_auto_resume=true`。
3. 如果 final verify 是 pass，停止。
4. 如果 should_auto_resume 为 true，计算 `next_rerun_from`，然后调用内部 rerun。
5. 不允许无限递归；必须由 `agent_max_loop_iterations` 控制。

推荐实现：

```python
def run_existing(...):
    return self._run_existing_with_agent_loop(...)

def _run_existing_once(...):
    # 当前 run_existing 主体移动到这里

def _run_existing_with_agent_loop(...):
    current_start_stage = start_stage
    for iteration in range(max_iterations + 1):
        self._run_existing_once(task_id, dry_run, current_start_stage)
        decision = self._next_agent_resume_decision(task_id)
        if not decision["should_resume"]:
            break
        current_start_stage = decision["start_stage"]
    return task_id
```

### 4.4 新增 resume decision

新增方法：

```python
def _next_agent_resume_decision(self, task_id: str) -> Dict:
    ...
```

返回：

```json
{
  "should_resume": true,
  "start_stage": "env_deploy",
  "reason": "agent_loop_requested_resume",
  "source_stage": "runner",
  "loop_iteration": 1
}
```

必须检查：

- config enabled
- max loop not exceeded
- repair apply status is applied
- policy allowed
- action executed or metadata action effective
- no current verify pass
- requested start stage in allowlist
- no repeated no-progress signature

### 4.5 状态和事件

每次自动 resume 写入 events：

```json
{
  "stage": "task",
  "type": "agent_auto_resume",
  "data": {
    "iteration": 1,
    "start_stage": "env_deploy",
    "source_stage": "runner",
    "reason": "agent_loop_requested_resume"
  }
}
```

`reports/execution_audit.json` 增加：

```json
{
  "agent_auto_resume": true,
  "agent_resume_iteration": 1
}
```

### 4.6 验收标准

新增测试：

```text
test_task_runner_auto_resumes_after_agent_repair
test_task_runner_stops_auto_resume_after_verify_pass
test_task_runner_stops_auto_resume_after_max_iterations
test_task_runner_does_not_auto_resume_when_policy_rejected
test_task_runner_does_not_auto_resume_when_config_disabled
```

新增 benchmark：

```text
agent_full_self_healing_pipeline
```

benchmark 不能手动调用 `AgentLoopController` 和 `VerifyModule` 拼结果；必须调用 `TaskRunner.deploy()` 或 `run_existing()`。

## 5. Phase 2：修复 Action Schema 和 Repair 能力

### 5.1 目标

提高真实 LLM 输出的可消费性，同时保持 policy gate。

### 5.2 统一 action payload

当前风险：

- LLM 可能输出 `payload.package="rich"`。
- 也可能输出 `payload.packages=["rich"]`。
- 当前策略更偏单包字段，真实 provider 输出稍偏就被拒绝。

新增规范化器：

```text
src/auto_harness/repair/actions.py
```

接口：

```python
class RepairActionNormalizer:
    def normalize(self, action: Dict) -> Dict:
        ...
```

规则：

- `install_package`:
  - `payload.package: str` -> packages list
  - `payload.packages: List[str]` -> packages list
  - 每个 package 单独 policy check
  - 生成多个 atomic install actions，或一个 action 内多个 results
- `rerun_from_stage`:
  - 接受 `payload.stage`
  - 接受 top-level `rerun_from`
- `update_verify_hint`:
  - 接受 `payload.verify_hint`
  - 接受 payload 本身为 verify hint

### 5.3 RepairPolicy 支持多包但严格校验

要求：

- 每个 package 都走 safe package spec。
- 禁止 `git+`、URL、path、`-e`、`--extra-index-url`、shell metachar。
- 允许版本约束：
  - `numpy<2`
  - `torch==2.3.1`
  - `pydantic>=1.10,<2`
- 不允许把多个包拼成 shell string。

### 5.4 RepairApplier 支持环境 backend

当前 install command 固定：

```text
.venv/bin/python -m pip install <package>
```

优化为：

```python
repair_env_context = {
  "backend": "venv|conda|mamba",
  "python_executable": "...",
  "conda_env_name": "...",
  "conda_prefix": "..."
}
```

生成命令：

- venv:

```text
.venv/bin/python -m pip install <package>
```

- conda:

```text
conda run -p .conda/envs/<name> python -m pip install <package>
```

- mamba:

```text
mamba run -p .conda/envs/<name> python -m pip install <package>
```

不要使用 `source activate`，因为它依赖 shell。

### 5.5 验收标准

新增测试：

```text
test_repair_normalizes_package_and_packages_payload
test_repair_rejects_mixed_shell_package_string
test_repair_applier_uses_conda_python_when_env_backend_is_conda
test_real_llm_packages_payload_is_consumed_safely
```

## 6. Phase 3：支持 LLM 环境策略规划

### 6.1 目标

LLM 应参与环境策略判断，但不能直接执行。它输出结构化建议：

```json
{
  "type": "select_environment_backend",
  "confidence": 0.82,
  "payload": {
    "backend": "conda",
    "python": "3.10",
    "reason": "environment.yml present and project has CUDA/PyTorch dependencies",
    "channels": ["pytorch", "nvidia", "conda-forge"],
    "prefer_mamba": true
  }
}
```

Python 负责：

- policy 校验
- fallback
- env_solve 合并
- env_deploy 执行

### 6.2 新增 action type

在 `AgentActionPolicy` 中新增 planner action：

```text
select_environment_backend
update_environment_spec
select_torch_variant
```

允许阶段：

- analyze planner
- env_solve planner，建议新增

不允许：

- 直接 shell command
- 任意 channel URL
- 任意 pip index URL

### 6.3 ProjectAnalyzer 合并 env strategy

`analysis` 增加：

```json
{
  "environment_strategy": {
    "backend": "conda",
    "preferred_tool": "mamba",
    "python": "3.10",
    "channels": ["pytorch", "nvidia", "conda-forge"],
    "source": "llm_planner",
    "confidence": 0.82,
    "reasons": []
  }
}
```

如果 deterministic 检测到 `environment.yml`，即使 LLM 没有输出，也应默认倾向 `conda`。

### 6.4 Policy 规则

允许 backend：

```text
venv
conda
mamba
docker
```

允许 channels：

```text
defaults
conda-forge
pytorch
nvidia
fastai
```

禁止：

- 任意 URL channel
- `file://`
- 本地 path channel
- shell metachar
- `pip --extra-index-url`

### 6.5 验收标准

新增测试：

```text
test_llm_selects_conda_backend_when_environment_yml_present
test_llm_environment_backend_policy_rejects_unknown_channel
test_deterministic_environment_yml_selects_conda_without_llm
test_llm_selects_torch_cuda_variant_but_python_policy_merges_it
```

## 7. Phase 4：Conda / Mamba Backend

### 7.1 目标

当前项目不支持 conda 创建环境。本阶段实现真实可执行 conda/mamba backend。

### 7.2 新增 Env Schema

新增：

```text
src/auto_harness/env/schemas.py
```

核心 dataclass：

```python
@dataclass
class EnvironmentSpec:
    backend: str  # venv | conda | mamba
    name: str
    prefix: str
    python: str
    channels: List[str]
    conda_dependencies: List[str]
    pip_dependencies: List[str]
    torch: Dict
    source_files: List[str]
```

### 7.3 解析 `environment.yml`

新增：

```text
src/auto_harness/env/conda.py
```

功能：

- 读取 `environment.yml` / `environment.yaml`
- 解析：
  - `name`
  - `channels`
  - `dependencies`
  - nested `pip`
  - `python=3.10`
  - `pytorch`
  - `pytorch-cuda`
  - `cudatoolkit`
- 不使用手写 fragile string split；优先用 `yaml.safe_load`。
- 如果 PyYAML 不可用，fallback 为保守解析并返回 `parser="fallback"`。

### 7.4 Conda 命令生成

不要依赖 shell activation。全部使用 prefix：

```text
conda create -y -p .conda/envs/<safe_name> python=3.10
conda install -y -p .conda/envs/<safe_name> -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
conda run -p .conda/envs/<safe_name> python -m pip install -r requirements.txt
```

如果选择 mamba：

```text
mamba create -y -p .conda/envs/<safe_name> python=3.10
mamba install -y -p .conda/envs/<safe_name> ...
mamba run -p .conda/envs/<safe_name> python -m pip install ...
```

### 7.5 Conda executable discovery

新增 probe：

```python
class CondaProbe:
    def probe(self) -> Dict:
        return {
          "conda": "/path/to/conda" or "",
          "mamba": "/path/to/mamba" or "",
          "micromamba": "/path/to/micromamba" or "",
          "available": True
        }
```

优先级：

```text
mamba > conda > micromamba
```

但如果 LLM 选择 `conda`，且 conda 存在，就用 conda。

### 7.6 EnvDeployModule 接入

当前 `EnvDeployModule.deploy()` 消费 `analysis.install_plan`。

修改为：

- 如果 `analysis.env_solution.backend == "local_venv"` 或 `"venv"`，走现有逻辑。
- 如果 `backend == "conda"` 或 `"mamba"`，调用 CondaBackend 生成 effective plan。
- effective plan 仍走 `allowed_commands`。
- 执行结果必须记录：

```json
{
  "environment_backend": "conda",
  "environment_prefix": ".conda/envs/<name>",
  "environment_python": ".conda/envs/<name>/bin/python",
  "commands": [],
  "executed": true
}
```

### 7.7 RunnerModule 接入

启动命令不能继续固定 `.venv/bin/python`。

新增 `runtime_command_resolver`：

- venv:

```text
.venv/bin/python app.py
```

- conda:

```text
conda run -p .conda/envs/<name> python app.py
```

- mamba:

```text
mamba run -p .conda/envs/<name> python app.py
```

如果 run candidate 由 analyzer 生成 `.venv/bin/python app.py`，在 env_solution 是 conda 时要改写为 conda run。

### 7.8 验收标准

新增测试：

```text
test_conda_environment_yml_parser_extracts_channels_and_pip
test_conda_backend_generates_prefix_create_command
test_conda_backend_uses_mamba_when_selected_and_available
test_env_deploy_conda_dry_run_records_effective_commands
test_env_deploy_conda_execute_respects_allowed_commands
test_runner_rewrites_venv_python_to_conda_run
```

新增 benchmark：

```text
conda_backend_environment_yml_plan
conda_backend_pytorch_cuda_plan
conda_runner_command_rewrite
```

这些 benchmark 可以 fake command runner，不需要真实 conda 安装。

## 8. Phase 5：PyTorch / CUDA / AI 依赖求解增强

### 8.1 目标

让系统能根据 LLM + deterministic analysis 选择合理 PyTorch 安装方案。

### 8.2 支持两种安装策略

#### venv/pip 策略

已有基础，继续保留：

```text
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### conda 策略

新增：

```text
conda install -y -p <prefix> -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

CPU fallback：

```text
conda install -y -p <prefix> -c pytorch pytorch torchvision torchaudio cpuonly
```

### 8.3 CUDA 映射

沿用现有 CUDA probe，但输出 conda spec：

| Local CUDA | Pip variant | Conda package |
|---|---|---|
| >= 12.1 | cu121 | pytorch-cuda=12.1 |
| >= 11.8 | cu118 | pytorch-cuda=11.8 |
| none/unknown | cpu | cpuonly |

注意：

- 如果 project 显式要求 `pytorch-cuda=11.8`，优先尊重项目约束，但记录风险。
- 如果 LLM 建议 CUDA variant 与本机不匹配，Python policy 要降级或给出 fallback。

### 8.4 GPU package matrix

继续增强：

- `xformers`
- `flash-attn`
- `bitsandbytes`
- `triton`
- `deepspeed`
- `accelerate`

输出：

```json
{
  "name": "flash-attn",
  "status": "blocked|risky|compatible",
  "reasons": [],
  "recommended_actions": []
}
```

### 8.5 LLM repair actions

新增安全 action：

```text
switch_torch_variant
switch_environment_backend
pin_dependency
install_conda_package
install_pip_package
```

执行限制：

- `install_conda_package` 只能安装 safe package spec。
- channel 必须在 allowlist。
- 不允许 URL channel。
- 不允许 shell string。
- 不允许 source edit。

### 8.6 验收标准

新增测试：

```text
test_conda_torch_solution_selects_pytorch_cuda_121
test_conda_torch_solution_falls_back_to_cpuonly
test_llm_switch_torch_variant_is_policy_gated
test_flash_attn_blocks_on_cpu_conda_env
test_bitsandbytes_records_linux_cuda_requirement
```

## 9. Phase 6：长期记忆 Verified Success 写回

### 9.1 目标

自修复成功后，系统必须自动写入 verified success memory，而不是只写 unresolved issue。

### 9.2 新增 `VerifiedMemoryRecorder`

新增：

```text
src/auto_harness/memory/success.py
```

接口：

```python
class VerifiedMemoryRecorder:
    def record_if_verified(self, run_dir: Path, pipeline_results: Dict, agent_metrics: Dict) -> Optional[Dict]:
        ...
```

触发条件：

- 最终 verify pass。
- 本次 run 有 agent loop 或 repair apply。
- repair policy allowed。
- repair action executed 或 metadata action 有效。
- 有 verification trace id。
- 没有 high-risk rejected action。

### 9.3 Memory entry schema

新增 verified success entry：

```json
{
  "id": "mem_success_<hash>",
  "memory_type": "verified_success",
  "task_id": "...",
  "stage": "runner",
  "category": "dependency_missing",
  "frameworks": ["gradio"],
  "project_signature": "...",
  "failure_signature": "...",
  "repair_action_hash": "...",
  "repair_actions": [],
  "environment_backend": "conda",
  "environment_spec_hash": "...",
  "torch_variant": "cu121",
  "verification_trace_id": "...",
  "verify_status": "passed",
  "regression_case_ids": ["agent_full_self_healing_pipeline"],
  "verified_success": true,
  "policy_rejected_high_risk": false,
  "created_at": "..."
}
```

### 9.4 Hash 规则

`repair_action_hash` 应由以下内容生成：

- action type
- normalized payload
- environment backend
- selected package/version
- rerun_from_effective

避免同一个问题不同修复混在一起。

### 9.5 写入时机

在 `TaskRunner.run_existing()` 最终 report 前：

```text
AgentMetricsCollector.collect()
VerifiedMemoryRecorder.record_if_verified()
ReportGenerator.generate()
```

report 中显示：

```text
Verified memory recorded: yes/no
Memory id: ...
Repair action hash: ...
Verification trace id: ...
```

### 9.6 验收标准

新增测试：

```text
test_verified_memory_recorded_after_agent_repair_verify_pass
test_verified_memory_not_recorded_when_verify_uncertain
test_verified_memory_not_recorded_when_policy_rejected_high_risk
test_verified_memory_contains_environment_and_torch_signature
```

新增 benchmark：

```text
verified_memory_after_self_healing
```

## 10. Phase 7：技能进化闭环

### 10.1 目标

让 skill promotion 不只是读取人工构造的 verified entries，而是来自真实 self-healing run。

### 10.2 Promotion 输入

`MemoryPromoter` 只读取：

```text
memory_type == verified_success
verified_success == true
verify_status == passed
repair_action_hash exists
verification_trace_id exists
regression_case_ids non-empty
```

### 10.3 Skill proposal 内容

proposal markdown 必须包含：

- failure pattern
- root cause
- normalized repair action
- environment backend
- PyTorch/CUDA strategy
- rerun_from rule
- verification trace rule
- regression binding
- rollback note

### 10.4 Apply 规则

保持人工审批：

```text
proposed -> approved -> regression passed -> applied
```

不能让 LLM 直接改 skill。

### 10.5 Skill versioning

技能文件更新时写入：

```text
skills/<name>/history/<timestamp>_<proposal_id>.md
```

proposal JSON 记录：

```json
{
  "previous_sha256": "...",
  "new_sha256": "...",
  "applied_at": "...",
  "rollback_path": "..."
}
```

### 10.6 验收标准

新增测试：

```text
test_skill_promotion_uses_verified_memory_from_self_healing_run
test_skill_apply_writes_history_and_sha256
test_skill_apply_requires_regression_pass
test_skill_apply_can_generate_rollback_metadata
```

新增 benchmark：

```text
skill_evolution_from_verified_self_healing
```

## 11. Phase 8：Conda + Self-Healing 集成场景

### 11.1 新增 fixture

新增：

```text
tests/fixtures/e2e/conda_pytorch_demo/
  README.md
  environment.yml
  app.py
```

`environment.yml` 示例：

```yaml
name: auto-harness-demo
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - pytorch-cuda=12.1
  - pip:
      - gradio
```

`app.py` 不需要下载真实模型，只需：

- import torch
- 输出 `torch.__version__`
- HTTP echo trace 或 Gradio echo trace

### 11.2 不做真实模型测试

CI / benchmark 不执行真实 GPU 或大模型。

测试方式：

- dry-run 验证命令生成。
- fake command runner 验证执行顺序。
- fake local environment CUDA probe。
- fake verify response 包含 trace id。

### 11.3 benchmark

新增：

```text
conda_pytorch_env_solve_plan
conda_pytorch_env_deploy_fake_execute
conda_self_healing_missing_package_resume
conda_verified_memory_skill_promotion
```

### 11.4 验收标准

必须证明：

- `environment.yml` 被识别。
- backend 选择 conda/mamba。
- conda create 命令正确。
- PyTorch CUDA package 命令正确。
- runner 使用 conda run，而不是 `.venv/bin/python`。
- 缺依赖时 repair 使用 conda env 内 python/pip。
- verify pass 后写入 verified success memory。
- memory promotion 生成 skill proposal。

## 12. CLI 与配置

### 12.1 新增 CLI

建议新增：

```bash
python -m auto_harness.cli env-plan --repo <path> --backend auto
python -m auto_harness.cli memory-verified --task-id <id>
python -m auto_harness.cli agent-self-heal --repo <path> --execute --allow-install --allow-start
```

如果不新增命令，也必须扩展现有：

```bash
deploy --agent-self-heal
deploy --env-backend auto|venv|conda|mamba
deploy --prefer-mamba
```

### 12.2 配置项

新增：

```python
env_backend: str = "auto"  # auto | venv | conda | mamba | docker
conda_envs_dir: str = ".conda/envs"
conda_prefer_mamba: bool = True
conda_allowed_channels: List[str] = [...]
conda_python_default: str = "3.10"
torch_cuda_preference: str = "auto"  # auto | cpu | cu118 | cu121
```

环境变量覆盖：

```text
AUTO_HARNESS_ENV_BACKEND
AUTO_HARNESS_CONDA_PREFER_MAMBA
AUTO_HARNESS_TORCH_CUDA_PREFERENCE
AUTO_HARNESS_CONDA_ENVS_DIR
```

## 13. 安全要求

### 13.1 Conda 安全

必须拒绝：

- channel URL
- local path channel
- `--override-channels` 由 LLM 提供
- shell metachar
- post-link shell scripts 不直接执行额外命令
- package spec 中出现 `; && | > < $()`

### 13.2 Secret 安全

不允许：

- conda env vars 中记录 token value
- report 中记录 token value
- memory 中记录 token value
- skill proposal 中记录 token value

只允许记录变量名：

```text
HF_TOKEN
MODELSCOPE_TOKEN
```

### 13.3 LLM 安全

LLM 可以建议：

- backend
- package
- version
- rerun_from
- torch variant

LLM 不可以：

- 直接 shell
- 任意 channel
- 任意 pip index
- 关闭 verify
- 直接修改 source
- 直接写 skill

## 14. 最终验收清单

开发完成后必须能回答：

1. 普通 `deploy` 是否能自动 repair/resume，而不是只靠 live smoke wrapper？
2. repair action 是否经过 policy？
3. repair 后是否重新 verify？
4. verify pass 后是否自动写入 verified success memory？
5. skill proposal 是否来自 verified success，而不是失败建议？
6. conda/mamba 是否真的有 backend 命令生成和执行路径？
7. runner 是否能在 conda 环境中启动？
8. PyTorch CUDA/CPU 方案是否由 solver 生成，而不是 LLM 字符串拼接？
9. LLM 输出 `packages` 数组时是否能被安全规范化？
10. 不跑真实模型部署测试时，是否仍有 fake runner / dry-run / artifact 证明？

## 15. 推荐实施顺序

严格按下面顺序：

```text
1. 主流程 auto-repair-resume
2. action normalizer
3. verified success memory recorder
4. conda/mamba schema + parser
5. conda/mamba backend dry-run plan
6. env_deploy / runner 接入 conda run
7. PyTorch/CUDA conda solver
8. conda repair action
9. skill evolution from verified memory
10. benchmarks and docs
```

不要先写宣传文档。先让普通主流程闭环跑通。

## 16. 最终可接受表述

完成本文档后，可以写：

```text
实现了一个 policy-constrained 自修复部署 Agent：LLM 负责环境策略、失败诊断和修复建议，Python controller 负责 policy gate、conda/venv 环境执行、repair/resume 状态机和 trace-based verify；成功修复会写入 verified long-term memory，并通过 regression-gated skill promotion 形成技能进化。
```

仍不建议写：

```text
生产级大规模 LLMOps 平台
完全自治无人值守 Agent
支持任意模型仓库 100% 自动部署
```

除非后续真的补齐线上指标、多租户、权限、secret manager、分布式调度和真实大规模验证。

## 17. 当前执行状态（2026-07-08）

本轮已完成本计划的本地可验证实现，不依赖真实 GPU 或大模型下载：

- Phase 1：普通 `TaskRunner.run_existing()` 已接入 bounded self-healing auto-resume，默认关闭，通过 `--agent-self-heal` 或配置打开。
- Phase 2：repair action normalizer 已支持 `package` / `packages`，多包逐个 policy check，repair install 可落到 conda/mamba 环境。
- Phase 3：LLM 环境策略 action 已进入 policy gate，支持 `select_environment_backend`、`update_environment_spec`、`select_torch_variant`。
- Phase 4：新增 conda/mamba schema、`environment.yml` parser、prefix command plan、`env_deploy` 接入和 runner command rewrite。
- Phase 5：保留 pip wheel solver，同时新增 conda `pytorch-cuda` 命令方案，并扩展 GPU package matrix 到 `deepspeed` / `accelerate`。
- Phase 6：新增 `VerifiedMemoryRecorder`，自修复 + verify pass 后写入 verified success memory，并在 report 中展示。
- Phase 7：skill promotion apply 现在有 approval、regression gate、history copy、sha256 和 rollback metadata。
- Phase 8：新增 `tests/fixtures/e2e/conda_pytorch_demo/` 与 10 个 benchmark case，覆盖 conda env solve、fake deploy、runner rewrite、self-healing resume、verified memory 和 skill promotion。

已验证命令：

```bash
PYTHONPATH=src python3 -m unittest \
  tests.test_core.CoreTests.test_conda_environment_yml_parser_extracts_channels_and_pip \
  tests.test_core.CoreTests.test_conda_backend_generates_prefix_create_command \
  tests.test_core.CoreTests.test_conda_backend_uses_mamba_when_selected_and_available \
  tests.test_core.CoreTests.test_env_deploy_conda_dry_run_records_effective_commands \
  tests.test_core.CoreTests.test_env_deploy_conda_execute_respects_allowed_commands \
  tests.test_core.CoreTests.test_runner_rewrites_venv_python_to_conda_run \
  tests.test_core.CoreTests.test_repair_normalizes_package_and_packages_payload \
  tests.test_core.CoreTests.test_repair_rejects_mixed_shell_package_string \
  tests.test_core.CoreTests.test_repair_applier_uses_conda_python_when_env_backend_is_conda \
  tests.test_core.CoreTests.test_repair_applier_uses_conda_install_for_conda_package \
  tests.test_core.CoreTests.test_deterministic_environment_yml_selects_conda_without_llm \
  tests.test_core.CoreTests.test_conda_torch_solution_falls_back_to_cpuonly \
  tests.test_core.CoreTests.test_flash_attn_blocks_on_cpu_conda_env \
  tests.test_core.CoreTests.test_bitsandbytes_records_linux_cuda_requirement \
  tests.test_core.CoreTests.test_llm_environment_backend_policy_rejects_unknown_channel \
  tests.test_core.CoreTests.test_llm_update_environment_spec_is_policy_merged \
  tests.test_core.CoreTests.test_task_runner_auto_resumes_after_agent_repair \
  tests.test_core.CoreTests.test_task_runner_does_not_auto_resume_when_config_disabled \
  tests.test_core.CoreTests.test_task_runner_stops_auto_resume_after_max_iterations \
  tests.test_core.CoreTests.test_task_runner_stops_auto_resume_after_verify_pass \
  tests.test_core.CoreTests.test_task_runner_does_not_auto_resume_when_policy_rejected \
  tests.test_core.CoreTests.test_verified_memory_recorded_after_agent_repair_verify_pass \
  tests.test_core.CoreTests.test_verified_memory_not_recorded_when_verify_uncertain \
  tests.test_core.CoreTests.test_verified_memory_not_recorded_when_policy_rejected_high_risk
```

```bash
PYTHONPATH=src python3 -m auto_harness.cli benchmark \
  --case-id conda_backend_environment_yml_plan \
  --case-id conda_backend_pytorch_cuda_plan \
  --case-id conda_runner_command_rewrite \
  --case-id agent_full_self_healing_pipeline \
  --case-id verified_memory_after_self_healing \
  --case-id skill_evolution_from_verified_self_healing \
  --case-id conda_pytorch_env_solve_plan \
  --case-id conda_pytorch_env_deploy_fake_execute \
  --case-id conda_self_healing_missing_package_resume \
  --case-id conda_verified_memory_skill_promotion
```
