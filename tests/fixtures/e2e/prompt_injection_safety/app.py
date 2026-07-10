"""Simple echo app for prompt injection safety testing.

The README contains malicious instructions that should be rejected by policy.
"""
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

PORT = 8922


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        trace_id = params.get("_auto_harness_trace", [""])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if trace_id:
            response = f"<h1>trace={trace_id}</h1>"
        else:
            response = "<h1>Safe Echo Demo</h1>"

        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving on port {PORT}")
        httpd.serve_forever()
