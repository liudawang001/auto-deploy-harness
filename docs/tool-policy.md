# Tool Policy

The agent interacts with the deployment system through named tools, not direct shell access.

Tool risk levels:

- `low`: read-only inspection, parsing, diagnosis or trace verification.
- `medium`: process, filesystem, browser or network side effects.
- `high`: source edit, secret handling or destructive action. High-risk actions are not enabled by default.

Current tool registry includes:

- `inspect_repo_tree`
- `read_selected_files`
- `parse_dependency_files`
- `solve_environment`
- `install_environment`
- `prepare_model_assets`
- `start_service`
- `probe_http`
- `probe_browser_dom`
- `discover_gradio_api`
- `discover_openapi_schema`
- `discover_openai_compatible_model`
- `download_model_asset`
- `inspect_log`
- `classify_failure`
- `propose_repair`
- `apply_repair`
- `resume_from_stage`
- `verify_evidence`

Policy rules:

- LLM can request a tool call, but Python decides whether it runs.
- Side-effect tools require policy approval.
- `install_environment`, `start_service`, `download_model_asset`, `apply_repair` and `resume_from_stage` require `gated_actor` mode.
- Package specs must be atomic and safe; shell strings, URLs, paths and pip index injection are rejected.
- Conda channels must be allowlisted: `defaults`, `conda-forge`, `pytorch`, `nvidia`, `fastai`.
- Source edit remains disabled unless runtime policy explicitly enables it.
- Rejected tool calls are recorded and counted; rejection does not weaken deterministic pipeline safety.
