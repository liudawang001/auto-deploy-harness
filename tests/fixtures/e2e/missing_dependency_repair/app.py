"""Demo app that imports requests (which is missing from requirements).

This app will fail to start because `requests` is not installed.
LLM should diagnose the failure and propose installing `requests`.
"""
import http.server
import socketserver
import requests  # This import will fail if requests is not installed

PORT = 8921


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        trace_id = ""
        if "?" in self.path:
            params = self.path.split("?", 1)[1]
            for param in params.split("&"):
                if param.startswith("_auto_harness_trace="):
                    trace_id = param.split("=", 1)[1]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if trace_id:
            response = f"<h1>trace={trace_id}</h1>"
        else:
            response = "<h1>Missing Dependency Demo</h1>"

        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving on port {PORT}")
        httpd.serve_forever()
