import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import rich  # noqa: F401 - intentionally missing until repair installs it


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        trace = (query.get("_auto_harness_trace") or [""])[0]
        self._send("handled %s" % trace)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        trace = body
        try:
            data = json.loads(body)
            trace = json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        self._send("handled %s" % trace)

    def _send(self, text):
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
