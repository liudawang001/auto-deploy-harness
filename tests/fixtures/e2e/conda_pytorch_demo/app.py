from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import torch
except Exception:  # pragma: no cover - dry-run fixture may not install torch
    torch = None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        trace = query.get("_auto_harness_trace", [""])[0]
        torch_version = getattr(torch, "__version__", "not-installed")
        body = ("trace=%s torch=%s" % (trace, torch_version)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
