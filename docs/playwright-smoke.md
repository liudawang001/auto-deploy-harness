# Playwright Browser Verify Smoke Test

## 目标

该 smoke test 用来验证真实 Python Playwright backend 可以被 `BrowserVerifier` 调用，并且浏览器渲染后的 DOM 能把当前 trace id 作为强证据返回。

它不是默认必跑门禁，因为首次安装 Chromium 会下载浏览器二进制，耗时和网络依赖都比普通单测高。建议在以下场景运行：

- 合并影响 `src/auto_harness/verify/browser.py` 或 `VerifyModule` 的改动前。
- 发布前验证真实浏览器 backend。
- 调试 Gradio / Streamlit 页面在 HTTP trace 不充分时的 DOM 证据。

## 本地运行

```bash
python3 -m pip install -e .
python3 -m pip install playwright
python3 -m playwright install chromium
```

然后运行以下脚本：

```bash
PYTHONPATH=src python3 - <<'PY'
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from auto_harness.verify import BrowserVerifier


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        trace = query.get("_auto_harness_trace", [""])[0]
        body = (
            "<html><head><title>smoke</title></head>"
            "<body><div class='gradio-container'>%s</div></body></html>" % trace
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()

try:
    endpoint = "http://127.0.0.1:%s/" % server.server_port
    check = BrowserVerifier().probe(endpoint, "smoke-trace", frameworks=["gradio"])
    assert check["status"] == "pass", check
    assert check["reason"] == "browser DOM contains current trace id", check
    print("Playwright browser verify smoke passed")
finally:
    server.shutdown()
PY
```

期望输出：

```text
Playwright browser verify smoke passed
```

## GitHub Actions

仓库提供手动触发 workflow：

```text
.github/workflows/playwright-smoke.yml
```

该 workflow 使用 `workflow_dispatch`，不会在普通 push / pull request 上自动下载浏览器。需要验证真实浏览器 backend 时，在 GitHub Actions 页面手动运行即可。

## 判断标准

通过标准：

- Playwright 能启动 headless Chromium。
- 测试页面返回包含 `_auto_harness_trace` 的 DOM。
- `BrowserVerifier.probe(...)` 返回 `browser_dom_probe=pass`。

失败后优先检查：

- Python Playwright 是否安装。
- Chromium 是否安装完成。
- CI 环境是否允许 headless browser sandbox。
- 页面 DOM 是否真的包含当前 trace id。
