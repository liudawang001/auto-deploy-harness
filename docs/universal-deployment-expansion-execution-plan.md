# 通用自动部署扩展与上线执行方案

## 1. 文档信息

- 状态：等待基础闭环完成后执行
- 制定日期：2026-08-30
- 执行范围：原总方案 Phase 5～Phase 8
- 目标版本：建议 `0.5.x`～`0.6.x`
- 前置文档：`docs/universal-deployment-foundation-execution-plan.md`
- 导航索引：`docs/universal-framework-auto-deployment-optimization-plan.md`

## 2. 本文档的执行目标

本文档在基础能力模型、Manifest、Adapter Registry 和 Protocol Verifier 已稳定的前提下，完成：

```text
未知入口证据发现
→ 候选评分与有界回退
→ Grounded LLM unknown fallback
→ JVM/Go/Rust 生态闭环
→ 审计报告与 Readiness
→ legacy/shadow/enforce 灰度
→ 默认控制链切换
```

完成后应达到：

1. 文档完整的 Django 和 Node 项目无需 LLM 即可发现入口。
2. deterministic 无法消歧时，LLM 只能引用或请求 evidence-backed candidate。
3. 新技术栈通过生态 Adapter + Command Policy + Protocol Verify 完整交付。
4. 新旧控制链可以 shadow 比较并安全灰度。
5. Readiness 将通用部署能力绑定到当前 commit 和回归证据。
6. `unknown` 不再直接等于失败，但证据不足时仍 fail closed。

## 3. 执行前置门

开始本文档前必须存在：

```text
docs/evidence/universal-deployment-foundation-handoff.json
```

必须验证：

- handoff commit 与当前实现祖先关系有效。
- capability schema version 受支持。
- deployment contract schema version 受支持。
- Adapter Registry version 受支持。
- Protocol Verifier Registry version 受支持。
- 基础测试全部通过。
- `false_success_count == 0`。
- `unsafe_command_execution_count == 0`。

任一条件不满足时，先回到基础闭环执行文档，不允许直接做 LLM fallback 或默认切换。

## 4. 共享不变量

1. LLM 不是信任根，不直接执行命令。
2. README 只能提供证据，不能授予宿主执行权限。
3. 新生态不能只通过增加全局命令白名单实现。
4. 所有候选必须进入统一 Command Authorization。
5. 未知仓库命令默认使用强化 Docker backend。
6. 人工审批不能覆盖 hard deny。
7. Candidate 排序不能把 rejected/hard denied 变成 executable。
8. Readiness 与 Verify 分离。
9. HTTP 200 和进程存活不能判成功。
10. 默认启用前必须经过 shadow 差异和 readiness 证据门。

## 5. 输入与输出契约

### 5.1 输入

本文档依赖以下稳定模型：

```text
ProjectCapabilities
CapabilityEvidence
CommandRegistry
DeploymentCandidate
DeployabilityAssessment
DeploymentAdapterRegistry
ProtocolVerifierRegistry
LegacyAnalysisCompiler
```

### 5.2 输出

完成后主链应输出：

```json
{
  "capabilities": {},
  "command_registry": {},
  "deployment_candidates": [],
  "deployability": {},
  "candidate_attempts": [],
  "llm_resolution": {},
  "authorization": {},
  "protocol_verify": {},
  "rollout_decision": {},
  "readiness": {}
}
```

## 6. Phase B1：未知入口的仓库证据发现

### 6.1 目标

在不调用 LLM 的情况下，从机器声明和公共文档发现更多安装、构建和启动候选。

### 6.2 发现优先级

```text
operator override
> valid auto-deploy.yaml
> project metadata declaration
> lockfile-backed package script
> README exact command + repository declaration
> framework/entrypoint convention
> low-confidence source signal
```

### 6.3 Python 入口扩展

新增证据来源：

- `manage.py`。
- `asgi.py` / `wsgi.py`。
- PEP 621 `[project.scripts]`。
- Poetry scripts。
- `Procfile` web process。
- README 精确 argv。
- `uv run <cli>`。
- 已安装到 Harness-owned `.venv/bin` 的 console script。

#### Django 规则

检测 evidence：

- 依赖声明包含 `django`。
- 根目录或受控子目录存在 `manage.py`。
- `INSTALLED_APPS`/settings module 只作辅助信号。
- README 若声明 runserver/gunicorn/uvicorn 命令，提高候选分数。

候选示例：

```text
.venv/bin/python manage.py runserver 0.0.0.0:8000
.venv/bin/gunicorn project.wsgi:application --bind 0.0.0.0:8000
.venv/bin/uvicorn project.asgi:application --host 0.0.0.0 --port 8000
```

只有仓库真实声明/安装对应 executable 时才能产生候选。Adapter 不得凭空假设 package 名和 settings module。

#### ASGI/WSGI 规则

- 模块路径必须来自仓库相对路径。
- 文件必须不是 symlink。
- import target 必须满足 Python module name 规则。
- 端口来源必须可解释。
- ASGI/WSGI 文件存在不等于 executable 已安装。

### 6.4 Node 入口扩展

支持：

- `npm run start`
- `npm run serve`
- `pnpm run start`
- `yarn run start`

要求：

- `package.json` 声明同名 script。
- package manager 与 lockfile 匹配。
- 安装使用 frozen/ci 语义。
- run candidate 绑定对应 workspace/cwd。
- 默认 Docker backend。
- script 内容仍视为仓库代码，不因为 lockfile 自动变安全。

`dev` script 默认不是生产首选，除非 Manifest/README/operator 明确指定且 Verify 证明服务稳定。

### 6.5 Procfile

首期只解析：

```text
web: <argv>
```

限制：

- 不能通过 shell 解析任意字符串。
- 只接受可安全 token 化的单命令 argv。
- shell operator 直接拒绝。
- 命令必须能与项目元数据/仓库脚本 evidence 交叉验证。
- Procfile 不能单独授予 local backend 权限。

### 6.6 Dockerfile Evidence

只把 `CMD`/`ENTRYPOINT` JSON form 作为低权限 evidence：

```dockerfile
CMD ["python", "app.py"]
```

首期不执行任意 Dockerfile，不接受 shell form：

```dockerfile
CMD python app.py && curl ...
```

Dockerfile evidence 可用于：

- 入口消歧。
- 默认端口辅助。
- cwd/argv 交叉验证。

不能用于：

- 绕过 command policy。
- 自动开启 host network。
- 自动挂载 Docker socket。
- 自动执行 build instruction。

### 6.7 候选评分

建议统一 0～100：

| Evidence | 分值 |
|---|---:|
| Operator 明确指定 | +100 |
| 合法 Manifest | +90 |
| 机器可读入口声明 | +75 |
| README 精确引用 | +20 |
| Lockfile 匹配 | +15 |
| 高置信 Adapter | +15 |
| 源码端口与命令一致 | +10 |
| 常见文件名 | +5 |
| 端口冲突 | -20 |
| 缺少依赖声明 | -20 |
| README 与声明冲突 | -30 |
| 只有 LLM 推测 | 不能单独执行 |

评分只排序，Policy verdict 独立。

### 6.8 有界候选回退

```text
排序候选
→ 授权首选
→ rejected/hard denied：下一候选
→ approval required：先寻找 auto allowed 备选
→ 启动失败：记录 failure signature，下一授权候选
→ max candidate attempts：进入 LLM replan
→ max strategy attempts：no_safe_plan/human input
```

同一 repository fingerprint + argv + backend + sandbox 的候选不得重复尝试。

### 6.9 Fixture

```text
django_manage_py_documented
django_asgi_uvicorn
django_ambiguous_entrypoints
pep621_custom_cli
poetry_custom_cli
procfile_safe_web
procfile_shell_rejected
node_npm_locked_start
node_pnpm_workspace_start
node_dev_not_preferred
dockerfile_json_entrypoint_evidence
dockerfile_shell_form_rejected
readme_prompt_injection
candidate_fallback_after_start_failure
```

### 6.10 完成标准

- 文档完整 Django 项目无需 LLM 发现正确入口。
- Node locked start fixture 可安装、启动和强验证。
- Procfile/Dockerfile 不成为安全旁路。
- 多候选排序和回退可审计。
- prompt injection 不能产生可执行命令。

### 6.11 Commit

```text
feat(planner): 增强未知入口的仓库命令证据发现
```

如果范围过大，拆为：

```text
feat(planner): 增强 Python 服务入口证据发现
feat(planner): 增强锁定 Node 服务命令发现
feat(policy): 限制 Procfile 与 Dockerfile 命令证据
feat(runtime): 增加授权候选的有界启动回退
```

## 7. Phase B2：Grounded LLM Unknown Fallback

### 7.1 触发条件

满足任一条件才触发：

- Deployability 为 partial。
- 缺少 run entrypoint。
- 多个高分候选无法消歧。
- 端口/cwd/protocol 缺失。
- 已授权候选启动失败。
- Verify uncertain 且仍存在合法 probe 空间。

以下情况不触发：

- Manifest 已提供完整方案。
- deterministic candidate 已 ready 且 strong verified。
- 缺少必需 secret/operator decision。
- 没有任何安全命令证据且继续观察无意义。
- 已达到预算。

### 7.2 LLM 输入

只提供：

- bounded repository inventory。
- redacted selected files。
- Project Capabilities。
- capability gaps。
- Command Registry 摘要。
- Deployment Candidates。
- authorization verdict 摘要。
- 当前 failure signature/log excerpt。
-允许 action schema。

不提供：

- secret value。
- 不必要完整仓库。
- 可写宿主路径。
- 任意 shell executor。

### 7.3 计划 schema

优先引用现有 candidate：

```json
{
  "status": "ok",
  "selected_environment_candidate_id": "env_001",
  "selected_run_candidate_id": "cmd_run_003",
  "selected_verify_candidate_id": "verify_http_002",
  "grounding": [
    {
      "evidence_id": "ev_readme_001",
      "claim": "README declares the Django run command"
    }
  ],
  "missing_capabilities": [],
  "risks": []
}
```

允许状态：

```text
ok
needs_human_input
no_safe_plan
invalid
```

### 7.4 Candidate Request

Registry 没有候选时，LLM 只能请求：

```json
{
  "type": "candidate_request",
  "phase": "run",
  "argv": [".venv/bin/python", "serve.py"],
  "cwd": ".",
  "expected_port": 9000,
  "grounding_evidence_ids": ["ev_readme_010", "ev_script_004"]
}
```

Python 必须执行：

```text
schema parse
→ argv normalization
→ evidence revalidation
→ CommandCandidate creation
→ CommandAuthorizationEngine
→ candidate selection
```

未经上述流程不得执行。

### 7.5 允许动作

- 选择 environment candidate。
- 选择 run candidate。
- 选择 verify candidate。
- 请求更多 bounded observation。
- 提议 expected port/cwd/protocol，需 evidence。
- 返回 no safe plan。

### 7.6 禁止动作

- 直接运行 shell。
- 修改 source/policy/sandbox。
- 添加全局 allowlist。
- 声明 README 命令天然安全。
- 请求外部 Verify URL。
- 以 HTTP 200 判成功。
- 伪造 evidence id。
- 绕过人工审批。

### 7.7 观察与 Replan 预算

建议沿用现有 bounded context，并增加：

```text
initial plan: 1
runner replan: 2
verify replan: 2
same failure signature: 2
candidate request per round: 2
```

每次 replan 必须改变至少一个：

- selected candidate。
- evidence set。
- port/protocol/verify candidate。
- environment candidate。

完全相同计划不得再次执行。

### 7.8 LLM Contribution

记录：

```text
selected_existing_candidate
resolved_ambiguity
filled_port
filled_protocol
filled_verify_candidate
requested_valid_candidate
no_material_contribution
harmful_proposal_rejected
```

`llm_helped=true` 只在状态实际改善时：

```text
partial → ready
wrong candidate → authorized runnable candidate
uncertain verify → strong evidence pass
```

### 7.9 Adversarial Tests

```text
hallucinated_entrypoint_rejected
fabricated_evidence_id_rejected
unregistered_command_rejected
external_verify_url_rejected
shell_wrapper_rejected
policy_override_rejected
same_plan_loop_stopped
prompt_injection_ignored
no_contribution_preserves_deterministic_result
valid_grounded_selection_passes
```

### 7.10 完成标准

- LLM 不能执行无证据命令。
- 至少两个 deterministic 无法消歧的 fixture 被 grounded fallback 解决。
- LLM 无贡献时不改变原结果。
- 有害提议被记录但不执行。
- 达到预算后稳定停止。

### 7.11 Commit

```text
feat(agent): 增加未知框架的受控规划回退
```

建议正文：

```text
- 让部署计划优先引用仓库命令注册表候选
- 支持缺失能力的有界观察、消歧和候选请求
- 对新增候选执行证据复验、归一化和统一策略授权
- 限制同失败签名与重复计划的执行预算
- 增加幻觉入口、越权命令和无贡献回归测试
```

## 8. Phase B3：非 Python 技术栈扩展

### 8.1 完整支持定义

一个生态只有同时交付以下能力才算支持：

```text
dependency detection
+ reproducible build/install
+ run candidate
+ command policy
+ backend/sandbox
+ readiness
+ protocol verify
+ repair boundary
+ offline E2E
```

只识别 `pom.xml/go.mod/Cargo.toml` 不算支持。

### 8.2 执行顺序

1. Spring Boot Maven Wrapper。
2. Spring Boot Gradle Wrapper。
3. Go modules HTTP service。
4. Cargo locked HTTP service。

### 8.3 Spring Boot Maven

#### Evidence

- `pom.xml`。
- `mvnw` 和 `.mvn/wrapper/*`。
- Spring Boot plugin declaration。
- README run/package command。
- `application.properties/yaml` 端口只作配置 evidence。

#### Candidate

优先 wrapper，不依赖宿主 Maven：

```text
./mvnw -DskipTests package
java -jar target/<grounded-artifact>.jar
```

限制：

- wrapper hash 固定。
- artifact 路径必须由 build output 证据确定。
- 不接受任意 profile/system property 注入。
- build 使用 registry-only network。
- run 使用 network none，除非 policy 明确允许。
- 默认 Docker。

#### Verify

复用：

- HTTP Trace Verifier。
- OpenAPI Verifier。
- 如项目只暴露 `/actuator/health`，它只能作为 readiness，不能单独强验证。

### 8.4 Spring Boot Gradle

Evidence：

- `gradlew`、wrapper properties/jar hash。
- `build.gradle`/`build.gradle.kts`。
- Spring Boot plugin。
- README command。

Candidate：

```text
./gradlew bootJar --no-daemon
java -jar build/libs/<grounded-artifact>.jar
```

限制与 Maven 相同，不能仅把 `gradle` 加入全局 allowlist。

### 8.5 Go HTTP

Evidence：

- `go.mod`、`go.sum`。
- `package main`。
- 确定性 main package path。
- README run/build command。

Candidate：

```text
go build -mod=readonly -o .harness/bin/service ./cmd/service
.harness/bin/service
```

限制：

- build output 位于 Harness-owned path。
- `-mod=readonly`。
- module download 使用 registry-only network。
- binary hash 与 build operation 绑定。
- run 默认 network none。

### 8.6 Rust Cargo

Evidence：

- `Cargo.toml`、`Cargo.lock`。
- `[[bin]]` 或确定性 `src/main.rs`。
- README command。

Candidate：

```text
cargo build --locked --release --bin <grounded-bin>
target/release/<grounded-bin>
```

限制：

- 必须存在 `Cargo.lock`。
- `--locked`。
- binary name 来自 metadata。
- build/run backend 分离。
- artifact hash 绑定 operation。

### 8.7 每个生态的 Policy 交付

必须定义：

- argv allow shape。
- required evidence types。
- wrapper/lock checksum。
- executable resolution。
- build network profile。
- run network profile。
- filesystem profile。
- backend。
- approval requirement。
- hard deny inputs。
- execution-time revalidation。

### 8.8 Fixture

每个生态至少：

```text
valid_http_trace_service
missing_lock_or_wrapper
malicious_build_argument
stale_artifact
wrong_port
readiness_only_not_verified
current_trace_verified
```

### 8.9 完成标准

- 每个生态至少一个离线 fixture 达到 strong verified success。
- 缺 wrapper/lock 的候选不会被误认为可复现自动部署。
- 新命令不会通过扩大无条件 local allowlist 获权。
- build artifact 与 run command 可追溯。

### 8.10 Commit

每个生态独立提交：

```text
feat(runtime): 增加 Spring Boot Maven 部署适配器
feat(runtime): 增加 Spring Boot Gradle 部署适配器
feat(runtime): 增加 Go HTTP 服务部署适配器
feat(runtime): 增加 Cargo 锁定服务部署适配器
```

每个提交同时包含 dependency/build/run/policy/readiness/verify 测试。

## 9. Phase B4：审计报告与指标

### 9.1 报告小节

```text
Project Capabilities
Deployment Contract
Adapter Detections
Capability Gaps
Candidate Composition
Command Authorization
Candidate Fallback History
LLM Resolution
Protocol Verification
Rollout Decision
Final Deployability Decision
```

### 9.2 Artifact

```text
runs/<task-id>/reports/project_capabilities.json
runs/<task-id>/reports/deployment_candidates.json
runs/<task-id>/reports/candidate_attempts.jsonl
runs/<task-id>/reports/llm_resolution.json
runs/<task-id>/reports/deployability_assessment.json
runs/<task-id>/reports/protocol_verify_selection.json
runs/<task-id>/reports/rollout_decision.json
```

### 9.3 指标

```text
project_framework_unknown_total
project_deployability_ready_total{source}
deployment_candidate_total{adapter,source}
deployment_candidate_selected_total{adapter,source}
deployment_candidate_rejected_total{reason_code}
deployment_candidate_fallback_total{outcome}
unknown_framework_verified_total{runtime_family}
llm_unknown_fallback_total{outcome}
protocol_verify_attempt_total{verifier}
protocol_verify_pass_total{verifier}
deployment_false_success_total
unsafe_command_execution_total
```

### 9.4 Commit

```text
feat(report): 增加部署能力与候选决策审计
```

修改 Python report 逻辑时不添加 `[skip-python-matrix]`。

## 10. Phase B5：灰度与默认启用

### 10.1 Rollout Modes

```text
legacy
shadow
enforce
```

#### legacy

- 旧 analyzer/runner/verify 决定执行。
- 新模型可关闭。
- 用于紧急兼容回退。

#### shadow

- 旧链决定执行。
- 新链生成 capability/candidate/verifier 建议。
- 写新旧差异。
- 不改变真正命令和 Verify verdict。

#### enforce

- 新链决定 candidate 和 verifier。
- 旧字段由 Legacy Compiler 生成。
- 所有命令仍走统一授权。
- 安全错误 fail closed。

### 10.2 Shadow 差异

必须比较：

```text
framework/capability classification
install plan
run candidate set
top1 candidate
expected port
authorization verdict
verify protocol
final status
```

差异分类：

```text
equivalent
new_more_complete
new_safer
new_less_complete
new_less_safe
incomparable
```

`new_less_safe` 必须阻止 enforce。

### 10.3 Enforce Gate

进入 enforce 前要求：

- 现有回归全通过。
- unknown fixture verified rate 达标。
- false success 为 0。
- unsafe execution 为 0。
- secret leak 为 0。
- 新旧 candidate 差异已评审。
- 所有 Policy/Verify 变更有证据。

### 10.4 安全回退

允许回退到 legacy：

- 新 schema 解析器内部错误，且旧链不更宽松。
- Adapter registry 内部非安全错误。
- Report/telemetry 非关键错误。

不允许静默回退：

- 新 Policy 拒绝的命令。
- stale evidence/approval。
- Verify 强证据失败。
- path/network/security violation。
- 新链发现旧链会产生 false success。

### 10.5 Config

```json
{
  "deployment_capability_mode": "shadow",
  "deployment_contract_enabled": true,
  "deployment_adapter_registry_enabled": true,
  "protocol_verify_registry_enabled": true,
  "unknown_framework_llm_fallback": true,
  "unknown_framework_llm_max_replans": 2
}
```

### 10.6 Controller 接入

修改：

- deploy。
- resume。
- checkpoint state。
- approval resume。
- candidate attempts journal。
- replan state。
- report finalization。

必须保证：

- resume 不重复执行已完成副作用。
- repository fingerprint 变化后旧 Plan 失效。
- candidate attempt history 恢复。
- approval operation id 稳定。
- controller 切换被记录。

### 10.7 Commit

```text
feat(controller): 将通用部署能力接入默认编排主链
```

建议正文：

```text
- 增加 legacy、shadow 和 enforce 部署能力模式
- 比较新旧候选、授权与验证决策差异
- 将新能力模型接入 deploy、resume 和 checkpoint 主链
- 对策略拒绝和强证据失败保持 fail closed
- 增加控制链切换、恢复与灰度门禁测试
```

## 11. Phase B6：Readiness 与发布门禁

### 11.1 Readiness 输入

- Foundation handoff。
- Baseline test artifacts。
- Adapter tests。
- Command Policy tests。
- Unknown framework E2E。
- LLM adversarial tests。
- Non-Python E2E。
- Shadow diff report。
- Verify false-success suite。

### 11.2 证据绑定

每份 readiness evidence 绑定：

```text
commit SHA
working tree cleanliness
config hash
fixture hash
schema versions
execution backend
test command
timestamp
```

### 11.3 Fail-closed 条件

- Evidence 缺失。
- Evidence 过期。
- Evidence hash 不匹配。
- 工作区不满足发布要求。
- false success 非零。
- unsafe execution 非零。
- secret leak 非零。
- shadow 出现未解决 `new_less_safe`。
- unknown fallback 绕过 registry/policy。

### 11.4 Commit

```text
feat(readiness): 将通用部署能力绑定到回归证据
```

## 12. 测试矩阵

### 12.1 Deterministic Unknown Entry

| Case | Expected |
|---|---|
| Django manage.py + README | 正确候选 |
| Django 多入口 | 多候选可解释排序 |
| Node lock + start | 可复现 install/run |
| Procfile safe | 产生受控 candidate |
| Procfile shell | rejected |
| Dockerfile JSON CMD | 低权限 evidence |
| Dockerfile shell CMD | 不产生 executable candidate |

### 12.2 LLM Fallback

| Case | Expected |
|---|---|
| 选择现有候选 | allowed after policy |
| 请求有证据候选 | normalize + authorize |
| 幻觉文件 | rejected |
| 伪造 evidence | rejected |
| 外部 URL | rejected |
| 重复计划 | stopped |
| 无贡献 | 保留 deterministic result |

### 12.3 Ecosystem

| Case | Expected |
|---|---|
| Maven wrapper service | build/run/verify |
| Gradle wrapper service | build/run/verify |
| Go locked service | build/run/verify |
| Cargo locked service | build/run/verify |
| Missing lock/wrapper | capability gap/approval |
| Stale artifact | rejected |

### 12.4 Rollout

| Case | Expected |
|---|---|
| legacy | 旧链决定 |
| shadow equivalent | 可进入下一门 |
| shadow safer | 记录提升 |
| shadow less safe | 阻止 enforce |
| enforce policy reject | fail closed |
| resume | 不重复副作用 |

## 13. 风险与应对

| 风险 | 应对 |
|---|---|
| README 提示注入 | README 只提供 evidence，独立 Policy |
| LLM 幻觉入口 | candidate id + evidence id + revalidation |
| 多候选重复执行 | attempt key + bounded fallback |
| 新生态供应链风险 | wrapper/lock + Docker + registry-only build |
| Adapter 数量膨胀 | 生态/协议复用，框架只表达差异 |
| Shadow 长期不切换 | 明确 readiness 阈值和版本窗口 |
| Legacy 回退变成安全旁路 | 安全错误禁止回退 |
| Verify 通过率低 | 改善协议发现/Manifest，不降低强证据 |

## 14. 建议执行顺序

```text
校验 Foundation Handoff
→ B1 Unknown Entry Discovery
→ B2 Grounded LLM Fallback
→ B4 Report（可与 B1/B2 后半并行评审，但单独提交）
→ B3 Non-Python Ecosystems，逐个交付
→ B5 Shadow/Enforce Rollout
→ B6 Readiness Gate
```

禁止：

- Foundation 未稳定时先接 LLM fallback。
- JVM/Go/Rust 共用一个无差异 Policy。
- 未做 shadow 就默认 enforce。
- Verify/Policy 失败后自动回退到更宽松 legacy。

## 15. Commit 序列

符合仓库近期风格：

```text
<type>(<scope>): <中文动宾短语>
```

推荐顺序：

```text
feat(planner): 增强未知入口的仓库命令证据发现
feat(agent): 增加未知框架的受控规划回退
feat(report): 增加部署能力与候选决策审计
feat(runtime): 增加 Spring Boot Maven 部署适配器
feat(runtime): 增加 Spring Boot Gradle 部署适配器
feat(runtime): 增加 Go HTTP 服务部署适配器
feat(runtime): 增加 Cargo 锁定服务部署适配器
feat(controller): 将通用部署能力接入默认编排主链
feat(readiness): 将通用部署能力绑定到回归证据
```

要求：

- 每个技术栈独立提交。
- 实现与相关测试同提交。
- Report 与 Controller 分开提交。
- Readiness 最后提交。
- Python 逻辑变更不使用 `[skip-python-matrix]`。
- Commit body 说明安全边界、兼容方式和测试证据。

## 16. 最终验收标准

### 16.1 功能

- [ ] 不在原框架集合中的 Python 服务能自动部署。
- [ ] Django、Sanic 或等价未知 Python fixture 至少两类 verified。
- [ ] Node locked start verified。
- [ ] LLM 能消歧但不能越过 registry/policy。
- [ ] 至少一个 JVM 和一个编译型生态 verified；完整版本要求四个生态都通过。
- [ ] 多候选有界回退可审计。

### 16.2 安全

- [ ] 无证据 LLM 命令不执行。
- [ ] README/Procfile/Dockerfile 不成为旁路。
- [ ] 新命令不靠扩大无条件 local allowlist 获权。
- [ ] stale evidence/approval/artifact 被拒绝。
- [ ] hard deny 不可审批覆盖。
- [ ] secret leak 为 0。
- [ ] unsafe execution 为 0。

### 16.3 Verify

- [ ] HTTP 200 不单独通过。
- [ ] readiness 不等于 success。
- [ ] 旧 trace 和错误端口监听者被拒绝。
- [ ] 当前 trace 或等强 artifact evidence 才通过。
- [ ] false success 为 0。

### 16.4 Rollout

- [ ] Shadow 差异报告完成评审。
- [ ] 无未解决 `new_less_safe`。
- [ ] Enforce 模式通过 deploy/resume/checkpoint 测试。
- [ ] Readiness 证据绑定当前 commit/config/fixture。
- [ ] Legacy 仅作为受控兼容模式保留。

## 17. 最终完成定义

只有以下闭环成立，第二份执行文档才算完成：

```text
未知或新技术栈项目
→ deterministic evidence discovery
→ 必要时 grounded LLM resolution
→ candidate authorization
→ isolated install/build/run
→ readiness
→ protocol strong evidence
→ bounded fallback/replan
→ shadow/enforce rollout
→ commit-bound readiness
```

最终不以“识别到更多框架”验收，而以：

```text
unknown_framework_verified_rate 提升
+ false_success_count == 0
+ unsafe_command_execution_count == 0
+ 所有决策可解释和可回滚
```

作为发布标准。
