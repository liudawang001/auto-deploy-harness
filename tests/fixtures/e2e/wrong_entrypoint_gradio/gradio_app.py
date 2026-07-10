"""Real Gradio entry point - echoes trace IDs.

This is the REAL demo that properly echoes trace IDs back.
"""
import gradio as gr
import os

PORT = 8919


def echo_with_trace(text: str, request: gr.Request = None) -> str:
    """Echo input text with trace ID if present."""
    trace_id = ""
    if request and request.query_params:
        trace_id = request.query_params.get("_auto_harness_trace", "")

    if trace_id:
        return f"trace={trace_id} | echo: {text}"
    return f"echo: {text}"


demo = gr.Interface(
    fn=echo_with_trace,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="Trace Echo Demo",
    description="Echoes back input with trace ID",
)

if __name__ == "__main__":
    demo.launch(server_port=PORT, server_name="0.0.0.0")
