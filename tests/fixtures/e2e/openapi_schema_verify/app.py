"""FastAPI app with POST endpoint for OpenAPI schema verification.

The verify should discover /openapi.json and use POST with trace ID body.
GET / returns HTML but does NOT prove core functionality.
POST /echo with JSON body containing trace_id proves functionality.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="Trace Echo API", version="1.0.0")

PORT = 8920


@app.get("/", response_class=HTMLResponse)
async def root():
    """HTML page - not sufficient for verification."""
    return "<h1>Trace Echo API</h1><p>Use POST /echo for verification.</p>"


@app.post("/echo")
async def echo(request: Request):
    """Echo endpoint - accepts JSON body and echoes trace_id."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    trace_id = body.get("trace_id", body.get("_auto_harness_trace", ""))

    return JSONResponse({
        "status": "ok",
        "trace_id": trace_id,
        "echo": body,
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
