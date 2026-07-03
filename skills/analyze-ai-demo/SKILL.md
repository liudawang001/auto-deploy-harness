---
name: analyze-ai-demo
description: Analyze AI demo repositories before automatic deployment. Use during analyze/project classification to identify framework, entrypoints, dependency files, model assets, service shape, and safe install/run/verify plans for Gradio, Streamlit, FastAPI, Flask, PyTorch, and Transformers projects.
---

# Analyze AI Demo

Goal: produce a conservative deployment plan from repository evidence, not from assumptions.

## Required checks

1. Inspect dependency files first: `requirements.txt`, `pyproject.toml`, `setup.py`, `environment.yml`, `package.json`, `Dockerfile`.
2. Inspect likely entrypoints: `app.py`, `main.py`, `server.py`, `webui.py`, `demo.py`, `api.py`.
3. Classify framework and service shape:
   - Gradio: `gr.Interface`, `gr.Blocks`, `.launch`, default port `7860`.
   - Streamlit: `streamlit run`, default port `8501`.
   - FastAPI: `FastAPI()`, `uvicorn`, default port `8000`.
   - Flask: `Flask(__name__)`, default port `5000`.
4. Detect model/runtime constraints: large weights, GPU-only packages, local checkpoint paths, external API keys, dataset downloads.
5. Prefer no source edits. If source edit is necessary, mark it as a proposed repair, not a default action.

## Output expectations

Return structured advice with:

- `frameworks`: detected framework tokens.
- `install_plan`: ordered commands.
- `run_candidates`: command, expected port, confidence, and reason.
- `verify_hint`: HTTP method/path/body template or artifact verification strategy.
- `risks`: missing files, unknown environment variables, heavyweight model downloads, GPU requirements.

## Safety boundaries

Do not suggest destructive commands. Do not embed secrets. Do not mark a service as deployable unless there is an entrypoint and a plausible verification path.
