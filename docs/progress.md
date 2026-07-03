# AI-Auto-Harness Development Progress

## 2026-07-03

### Completed

- Cloned `liudawang001/ai-auto-harness` through authenticated GitHub CLI.
- Initialized a Python package with `pyproject.toml`.
- Added project README, `.gitignore`, `.env.example`, and default config.
- Implemented core data models:
  - `TaskSpec`
  - `TaskState`
  - `StageResult`
  - `VerifyResult`
- Implemented state persistence:
  - `task.json`
  - `state.json`
  - `events.jsonl`
  - stage result files
- Implemented `auto-harness` CLI:
  - `init`
  - `deploy`
  - `resume`
  - `status`
  - `report`
  - `llm-test`
- Implemented LLM provider abstraction:
  - `MockLLMProvider`
  - `XunfeiSparkProvider` with Anthropic-compatible payload construction and environment-based configuration.
- Implemented Agent executor abstraction:
  - `AgentExecutor`
  - `ClaudeCodeExecutor`
- Implemented first-pass modules:
  - `ProjectAnalyzer`
  - `EnvDeployModule`
  - `RunnerModule`
  - `VerifyModule`
  - `ReportGenerator`
- Implemented `TaskRunner` orchestration for dry-run pipeline execution.
- Added initial stdlib unit tests covering:
  - state store roundtrip
  - analyzer framework detection
  - verify no-false-pass behavior without artifact evidence
  - mock LLM provider
- Implemented safe default behavior: deploy is dry-run unless `--execute` is explicitly passed.
- Added explicit execution gates:
  - `--allow-install`
  - `--allow-start`
- Added support for local project paths as `--repo` input by copying them into the isolated run workspace.
- Verified local-path dry-run with a temporary Gradio-style demo:
  - detected `gradio`
  - generated venv/pip install plan
  - generated `app.py` run candidate
  - kept `verify` as `uncertain` because no real trace artifact was produced in dry-run mode
- Implemented first HTTP trace verification path:
  - selects endpoint from `verify_hint.endpoint` or runner endpoint candidate
  - appends `_auto_harness_trace=<trace_id>` to the request
  - stores request/response evidence under `evidence/`
  - only passes when the response body proves the current trace was handled
  - does not treat HTTP 200 alone as success
- Extended HTTP trace verification:
  - supports GET query trace
  - supports POST JSON trace templates
  - generates a Gradio-style default `/api/predict` POST verification hint
- Wired `ClaudeCodeExecutor` into `ProjectAnalyzer` as an optional advisor.
  - disabled by default
  - enabled with `AUTO_HARNESS_USE_AGENT_ANALYZER=1`
  - stores advice as metadata without bypassing deterministic analyzer output
- Added execution command policy checks for `env_deploy` and `runner`.
  - Commands are rejected before execution if their executable name is not in `allowed_commands`.
  - This is required before enabling broader `--execute` use.

### Current Behavior

The system can create a task, scan a repository directory, produce install/run plans, perform dry-run env/runner stages, run evidence-oriented `verify`, and generate a Markdown report.

### Important Design Notes

- `verify` currently returns `uncertain` unless it has actual artifact evidence. This is intentional: false pass is more dangerous than uncertain.
- Xunfei integration is abstracted. The current provider supports an Anthropic-compatible HTTP messages interface via environment variables. Real secrets are intentionally not stored in repository files.
- Claude Code is optional and configured through `CLAUDE_CODE_CMD`. The current pipeline does not depend on it to run the dry-run MVP.

### Next Steps

1. Expand unit tests for provider parsing, command safety, CLI behavior, and report generation.
2. Add a fixture demo project under `tests/fixtures`.
3. Extend Gradio verification with real API discovery and file/download artifact checks.
4. Add JSON schema validation for Agent/LLM outputs.
5. Run a private Xunfei smoke test with local environment variables and verify the exact response format.
6. Expand command policy with argument-level checks and dangerous pattern detection before enabling broad `--execute` usage.
7. Wire `ClaudeCodeExecutor` into analyzer or verify as an optional stage executor.
8. Add benchmark cases:
   - HTTP 200 but no output.
   - historical output file interference.
   - missing dependency.
   - service exits after startup.

### Known Limitations

- Real dependency installation and service startup are disabled by default.
- `VerifyModule` supports GET query trace and POST JSON trace templates, but does not yet support Gradio API discovery, browser/UI actions, file download validation, or CLI trace execution.
- `RunnerModule` does not yet persist process handles for later cleanup.
- `XunfeiSparkProvider` currently assumes an Anthropic-compatible HTTP messages interface; add another transport if a selected Spark API variant requires WebSocket signing.
- The test suite is still minimal and only covers the core dry-run path.
