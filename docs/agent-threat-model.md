# Agent Threat Model

AI-Auto-Harness treats unknown AI demo repositories as untrusted input.

Core boundaries:

- LLM never executes shell directly.
- LLM never edits source directly.
- LLM never reads or stores secret values.
- LLM never decides final success.
- Python runtime owns policy gate, tool execution, state recovery and evidence-based verify.

Primary threats:

- Prompt injection in README or source comments.
- Malicious run command suggestions.
- Secret exfiltration through logs, prompts, reports or memory.
- Unsafe package/channel/index injection.
- False-positive verification from HTTP 200, stale artifacts or generic pages.
- Repair loops that repeatedly execute side effects without evidence.

Controls:

- `AgentInputSanitizer` redacts secrets and marks prompt-injection risk before LLM calls.
- `AgentActionPolicy` and `RepairPolicy` reject shell metacharacters, network executables, unsafe package specs, URL channels and source edit by default.
- `ToolRegistry` marks side-effect tools with risk level and policy requirements.
- `AgentCritic` records reject/revise/approve critique for each runtime step but cannot execute tools.
- `VerifyModule` requires current trace evidence; HTTP 200 alone is not success.
- Verified memory promotion requires final verify pass, repair hash, trace id and regression binding.

Audit artifacts:

- `agent_steps.jsonl`
- `agent_state.json`
- `agent_plan.json`
- `agent_plan_revisions.jsonl`
- `reports/agent_contribution.json`
- `reports/agent_metrics.json`
- `repairs/repair_plan.json`
- `repairs/repair_apply_result.json`
