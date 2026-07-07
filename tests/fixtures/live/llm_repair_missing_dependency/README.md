# LLM Repair Missing Dependency Fixture

This fixture is intentionally small and does not download model weights.

Run:

```bash
python app.py
```

The service echoes `_auto_harness_trace` from the query string or JSON/text body.
`app.py` imports `rich` so the first runner attempt can fail with a missing
dependency when the package is absent. The agent live smoke should diagnose the
missing dependency, propose an `install_package` repair action, and then rerun.
