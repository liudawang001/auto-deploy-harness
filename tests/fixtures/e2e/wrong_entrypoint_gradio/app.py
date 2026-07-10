"""Placeholder app - NOT the real entry point.

This app starts but does NOT echo trace IDs.
The real entry point is gradio_app.py.
"""
import http.server
import socketserver

PORT = 8919


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>Placeholder App</h1><p>This is not the real demo.</p>")

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving placeholder on port {PORT}")
        httpd.serve_forever()
