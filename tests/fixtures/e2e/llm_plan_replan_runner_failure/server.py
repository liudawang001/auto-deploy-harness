"""Correct HTTP trace echo server.

This is the actual service that should be started after replan.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        trace_value = params.get("_auto_harness_trace", ["unknown"])[0]
        body = "auto-harness-e2e trace=%s" % trace_value
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress stderr


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8917), Handler).serve_forever()
