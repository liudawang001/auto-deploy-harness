# Missing Dependency Repair Demo

A demo that imports a missing dependency (`requests`).

The runner will fail because `requests` is not installed.
LLM should diagnose the failure and propose installing `requests`.

## Usage

```bash
pip install requests
python app.py
```
