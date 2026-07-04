import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, Optional


class PlaywrightBrowserBackend:
    """Optional browser backend. It is active only when Python Playwright exists."""

    def load(self, url: str, timeout_ms: int = 15000, screenshot_path: Optional[Path] = None) -> Dict:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:
            return {
                "status": "unavailable",
                "error": "python playwright is not installed: %s" % exc,
            }

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                html = page.content()
                title = page.title()
                if screenshot_path:
                    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshot_path), full_page=True)
                return {
                    "status": "loaded",
                    "url": page.url,
                    "title": title,
                    "html": html,
                    "status_code": response.status if response else None,
                    "screenshot_path": str(screenshot_path) if screenshot_path and screenshot_path.exists() else None,
                }
            finally:
                browser.close()


class BrowserVerifier:
    ERROR_MARKERS = (
        "Traceback (most recent call last)",
        "ModuleNotFoundError",
        "ImportError",
        "RuntimeError",
        "Exception",
        "Application error",
    )

    UI_MARKERS = {
        "gradio": ("gradio", "gradio-container", "svelte"),
        "streamlit": ("streamlit", "stapp", "data-testid"),
    }

    def __init__(self, browser_backend=None) -> None:
        self.browser_backend = browser_backend or PlaywrightBrowserBackend()

    def probe(self, endpoint: str, trace_id: str, frameworks: Iterable[str] = None, screenshot_path: Path = None) -> Dict:
        url = self._append_trace(endpoint, trace_id)
        loaded = self.browser_backend.load(url, screenshot_path=screenshot_path)
        if loaded.get("status") == "unavailable":
            return {
                "name": "browser_dom_probe",
                "status": "uncertain",
                "evidence": {"url": url, "backend": loaded},
                "reason": "browser backend is unavailable",
            }
        if loaded.get("status") != "loaded":
            return {
                "name": "browser_dom_probe",
                "status": "uncertain",
                "evidence": {"url": url, "backend": loaded},
                "reason": "browser page load did not complete",
            }

        html = loaded.get("html") or ""
        markers = self._markers(html, frameworks or [])
        if markers["error_markers"]:
            status = "fail"
            reason = "browser DOM contains error markers"
        elif trace_id in html:
            status = "pass"
            reason = "browser DOM contains current trace id"
        elif markers["ui_markers"]:
            status = "uncertain"
            reason = "browser DOM loaded but trace was not observed"
        else:
            status = "uncertain"
            reason = "browser DOM loaded without recognizable UI markers"
        return {
            "name": "browser_dom_probe",
            "status": status,
            "evidence": {
                "url": url,
                "final_url": loaded.get("url"),
                "title": loaded.get("title"),
                "status_code": loaded.get("status_code"),
                "screenshot_path": loaded.get("screenshot_path"),
                "body_tail": html[-2000:],
                "markers": markers,
            },
            "reason": reason,
        }

    def _append_trace(self, endpoint: str, trace_id: str) -> str:
        parsed = urllib.parse.urlparse(endpoint)
        query = urllib.parse.parse_qs(parsed.query)
        query["_auto_harness_trace"] = [trace_id]
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))

    def _markers(self, html: str, frameworks: Iterable[str]) -> Dict:
        lower = html.lower()
        ui_markers = []
        framework_set = set(frameworks or [])
        for framework, markers in self.UI_MARKERS.items():
            if framework_set and framework not in framework_set:
                continue
            for marker in markers:
                if marker in lower:
                    ui_markers.append(marker)
        error_markers = [marker for marker in self.ERROR_MARKERS if marker in html]
        return {
            "ui_markers": sorted(set(ui_markers)),
            "error_markers": error_markers,
        }
