# Contributing

Python 3.10–3.13 is supported. Create an isolated environment, then install development dependencies:

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m auto_harness.cli benchmark --manifest tests/fixtures/benchmarks/manifest.json
```

Before submitting a change:

1. Add or update tests for behavior changes.
2. Run focused tests, the complete suite, and the benchmark manifest.
3. Build the wheel and verify `auto-deploy-harness init` from the installed wheel.
4. Run `auto-deploy-harness readiness`; it must fail closed if evidence is missing, stale, dirty, or unsuccessful.
5. Do not commit credentials, `.env` files, run artifacts, model weights, private reports, or anything under `docs/`.

Changes should preserve default fail-closed behavior and must not silently fall back from an explicitly requested LLM controller to deterministic behavior.
