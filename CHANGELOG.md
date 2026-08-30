# Changelog

All notable changes are documented here.

## 0.3.0 - Unreleased

- Added a read-only performance & cost profiler (`cost-profile`) that aggregates persisted LLM telemetry (`context.usage`, `latency_ms`), events.jsonl stage timing, and terminal outcomes into per-run and cross-run reports with token totals split by `provider_reported`/`estimated` provenance and optional config-provided pricing; usage without pricing is reported as unpriced tokens, never as invented cost.
- Added the `cost_profile` config block (currency, pricing table per model with `pricing_as_of` metadata) and a Performance & Cost section in deployment reports.
- Added opt-in Hybrid Evidence Retrieval with safe repository/memory ingestion, deterministic chunking, SQLite manifests, BM25 fallback, optional embeddings, exact cosine search, and RRF fusion.
- Added a policy-bounded `retrieve_deployment_context` tool for JSON Action and native tool loops; repository hits remain advisory until exact reread and SHA-backed Grounding.
- Added retrieval traces, summaries, causal contribution gates, 30-case offline evaluation fixtures, transactional index recovery, and tiered readiness that never promotes fake embeddings to live evidence.
- Added an independently selectable `native_tools` provider protocol while retaining `json_action` as the backward-compatible default.
- Added strict provider tool-schema projection, normalized tool calls/results, bounded multi-turn execution, safe tool-result messages, and explicit no-fallback protocol routing.
- Implemented native tool transport for DeepSeek and configurable OpenAI-compatible providers, including assistant tool calls and correlated tool-result messages.
- Added a crash-safe Tool Call Ledger, semantic operation identity, seven recovery fault windows, deterministic message reconstruction, and Operation Journal reuse for future side-effect tools.
- Enabled policy-gated internal state-delta tools with before/after state hashes and contribution evidence; native side-effect tools remain blocked until the v0.4 release gate.
- Added protocol/readiness reports, context budgeting for tool schemas, a seven-case evaluation matrix, and an opt-in real-provider read-only smoke test.

## 0.2.0 - Unreleased

- Added layered repository snapshots, bounded JSON observation turns, redacted observation ledgers, checkpoint-safe on-demand reads, and SHA-backed plan grounding.
- Made LangGraph the functional default with explicit automatic, deterministic, and LLM planner selection.
- Added terminal state synchronization and stable CLI exit codes.
- Isolated child-process environments from provider credentials.
- Closed the repair/resume evidence loop and fixed benchmark self-repair behavior.
- Added commit-bound release evidence, fail-closed readiness auditing, packaged defaults and skills, wheel smoke coverage, and public project governance files.
- Classified real provider, network, Docker/GPU, vLLM, and larger-repository validation as external gates rather than local proof.

## 0.1.0

- Initial development release.
