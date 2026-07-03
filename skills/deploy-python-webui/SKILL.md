---
name: deploy-python-webui
description: Deploy Python AI web UI projects. Use during env_deploy and runner stages for Gradio, Streamlit, Flask, FastAPI, PyTorch, Transformers, and requirements-based repositories that need virtualenv setup, dependency installation, startup command selection, and port readiness checks.
---

# Deploy Python WebUI

Goal: build an isolated Python runtime and start the most likely service command with bounded permissions.

## Procedure

1. Create `.venv` in the copied run workspace.
2. Install from the strongest dependency source:
   - `requirements.txt`: `.venv/bin/python -m pip install -r requirements.txt`
   - `pyproject.toml`: `.venv/bin/python -m pip install .`
   - `setup.py`: `.venv/bin/python -m pip install .`
3. Select runner command by framework:
   - Gradio: `.venv/bin/python app.py` or another entrypoint containing `.launch`.
   - Streamlit: `.venv/bin/streamlit run app.py`.
   - FastAPI: prefer `uvicorn module:app --host 127.0.0.1 --port 8000` when app object is clear.
4. Capture stdout/stderr to the run log.
5. Treat process exit before readiness as failure, even when the command printed a URL earlier.

## Common failure handling

- Missing module after install: record package name and dependency file, then propose adding a pinned dependency only when source edits are allowed.
- Port occupied: try a configured alternate port only if the framework supports it without source edits.
- GPU or model download failure: record environment requirement; do not fake success with a blank UI.
- API key missing: record required variable names; never persist secret values.

## Evidence

Deployment is not complete until the runner stage has a live process, a ready port when applicable, and a log path for later diagnosis.
